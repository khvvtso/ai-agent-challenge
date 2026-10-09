"""Agent tools. Each has a live backend and a mock fallback so the demo never hard-fails.

Live backends: Google Calendar, Gmail (read + send), Companies House, Tavily/DuckDuckGo.
Every call is cached in-process (saves free-tier quota on re-runs) and traced.
"""
from __future__ import annotations

import base64
import json
import re
from email.mime.text import MIMEText
from email.utils import parseaddr
from pathlib import Path
from urllib.parse import urlparse

import requests
from rapidfuzz import fuzz

from core import config
from core.trace import Trace

MOCK = Path(__file__).resolve().parent.parent / "data" / "mock"
_cache: dict = {}


def _cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def clear_cache():
    _cache.clear()


def google_live() -> bool:
    return all(config.get(k) for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"))


def _google(service: str, version: str):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=config.get("GOOGLE_REFRESH_TOKEN"),
        client_id=config.get("GOOGLE_CLIENT_ID"),
        client_secret=config.get("GOOGLE_CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token",
    )
    return build(service, version, credentials=creds, cache_discovery=False)


def _span(trace: Trace | None, name: str, input):
    from contextlib import nullcontext

    return trace.span(name, "tool", input=input) if trace else nullcontext(_NullSpan())


class _NullSpan:
    def update(self, *a, **k):
        pass


# ---------------------------------------------------------------- calendar
def calendar_get_event(query: str, mock: bool = False, trace: Trace | None = None) -> dict:
    live = google_live() and not mock
    with _span(trace, "calendar.get_event", {"query": query, "backend": "google" if live else "mock"}) as s:
        def fetch():
            if live:
                svc = _google("calendar", "v3")
                items = svc.events().list(calendarId="primary", q=query, singleEvents=True,
                                          orderBy="startTime", maxResults=5).execute().get("items", [])
                if not items:
                    return {"found": False, "backend": "google"}
                ev = items[0]
                return {"found": True, "backend": "google", "id": ev["id"], "summary": ev.get("summary"),
                        "location": ev.get("location"), "start": ev.get("start", {}).get("dateTime"),
                        "attendees": [{"email": a.get("email"), "displayName": a.get("displayName"),
                                       "responseStatus": a.get("responseStatus")} for a in ev.get("attendees", [])]}
            events = json.loads((MOCK / "calendar.json").read_text())["events"]
            words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
            best = max(events, key=lambda e: sum(w in e["summary"].lower() for w in words))
            return {"found": True, "backend": "mock", **best}

        out = _cached(("cal", query, live), fetch)
        s.update({"found": out.get("found"), "summary": out.get("summary"), "attendees": len(out.get("attendees", []))})
        return out


# ---------------------------------------------------------------- gmail
def _mock_emails() -> list[dict]:
    return json.loads((MOCK / "emails.json").read_text())["emails"]


def _gmail_body(payload) -> str:
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", "replace")
    for part in payload.get("parts", []) or []:
        text = _gmail_body(part)
        if text:
            return text
    return ""


def gmail_search(query: str, mock: bool = False, trace: Trace | None = None) -> list[dict]:
    """Returns message summaries (id, sender, subject, date)."""
    live = google_live() and not mock
    with _span(trace, "gmail.search", {"query": query, "backend": "gmail" if live else "mock"}) as s:
        def fetch():
            if live:
                svc = _google("gmail", "v1")
                ids = svc.users().messages().list(userId="me", q=query, maxResults=50).execute().get("messages", [])
                return [gmail_read(m["id"], mock=False) for m in ids]
            return _mock_emails()  # dedicated demo inbox: the query is applied by the classifier, not by keyword

        msgs = _cached(("gmail_search", query, live), fetch)
        out = [{k: m[k] for k in ("id", "sender", "subject", "date")} for m in msgs]
        s.update({"count": len(out)})
        return out


def gmail_read(msg_id: str, mock: bool = False, trace: Trace | None = None) -> dict:
    live = google_live() and not mock

    def fetch():
        if live:
            svc = _google("gmail", "v1")
            m = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
            headers = {h["name"].lower(): h["value"] for h in m["payload"].get("headers", [])}
            return {"id": msg_id, "sender": headers.get("from", ""), "to": headers.get("to", ""),
                    "subject": headers.get("subject", ""), "date": headers.get("date", ""),
                    "body": _gmail_body(m["payload"]) or m.get("snippet", "")}
        return next(e for e in _mock_emails() if e["id"] == msg_id)

    return _cached(("gmail_read", msg_id, live), fetch)


def gmail_send(to: str, subject: str, body: str, mock: bool = False, trace: Trace | None = None) -> dict:
    live = google_live() and not mock
    with _span(trace, "gmail.send", {"to": to, "subject": subject, "backend": "gmail" if live else "simulated"}) as s:
        if live:
            msg = MIMEText(body)
            msg["to"], msg["subject"] = to, subject
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            res = _google("gmail", "v1").users().messages().send(userId="me", body={"raw": raw}).execute()
            out = {"status": "sent", "id": res.get("id")}
        else:
            path = MOCK / "outbox.json"
            box = json.loads(path.read_text()) if path.exists() else []
            box.append({"to": to, "subject": subject, "body": body})
            path.write_text(json.dumps(box, indent=2))
            out = {"status": "simulated", "path": str(path.name)}
        s.update(out)
        return out


# ---------------------------------------------------------------- companies house
def _directory() -> list[dict]:
    return json.loads((MOCK / "company_directory.json").read_text())["companies"]


def companies_house_search(name: str, mock: bool = False, trace: Trace | None = None) -> dict:
    """Best match in the UK company register: {found, name, status, number, score, source}."""
    key = config.get("COMPANIES_HOUSE_API_KEY")
    live = bool(key) and not mock
    with _span(trace, "companies_house.search", {"name": name, "backend": "companies_house" if live else "snapshot"}) as s:
        def fetch():
            if live:
                r = requests.get("https://api.company-information.service.gov.uk/search/companies",
                                 params={"q": name, "items_per_page": 5}, auth=(key, ""), timeout=15)
                r.raise_for_status()
                items = r.json().get("items", [])
                scored = [(fuzz.token_set_ratio(_strip_suffix(name), _strip_suffix(i.get("title", ""))), i) for i in items]
                if not scored or max(scored, key=lambda x: x[0])[0] < 80:
                    return {"found": False, "source": "companies_house", "candidates": [i.get("title") for i in items[:3]]}
                score, best = max(scored, key=lambda x: x[0])
                return {"found": True, "name": best.get("title"), "status": best.get("company_status"),
                        "number": best.get("company_number"), "score": score, "source": "companies_house"}
            best, best_score = None, 0
            for c in _directory():
                for cand in [c["name"], *c.get("aliases", [])]:
                    sc = fuzz.token_set_ratio(_strip_suffix(name), _strip_suffix(cand))
                    if sc > best_score:
                        best, best_score = c, sc
            if not best or best_score < 85:
                return {"found": False, "source": "snapshot"}
            return {"found": True, "name": best["name"], "status": best["status"], "number": best["number"],
                    "domain": best.get("domain"), "score": best_score, "source": "snapshot"}

        try:
            out = _cached(("ch", name.lower(), live), fetch)
        except Exception as e:
            out = {"found": None, "error": str(e), "source": "companies_house"}
        s.update(out)
        return out


def _strip_suffix(name: str) -> str:
    return re.sub(r"\b(plc|ltd|limited|llp|group|uk|the)\b\.?", "", name.lower()).strip()


# ---------------------------------------------------------------- web
def web_search(query: str, mock: bool = False, trace: Trace | None = None) -> dict:
    key = config.get("TAVILY_API_KEY")
    with _span(trace, "web.search", {"query": query}) as s:
        def fetch():
            if mock:
                return {"backend": "skipped", "results": []}
            if key:
                r = requests.post("https://api.tavily.com/search",
                                  json={"api_key": key, "query": query, "max_results": 5}, timeout=20)
                r.raise_for_status()
                return {"backend": "tavily", "results": [{"title": x.get("title"), "url": x.get("url")}
                                                          for x in r.json().get("results", [])]}
            try:
                from ddgs import DDGS

                res = DDGS().text(query, max_results=5)
                return {"backend": "duckduckgo", "results": [{"title": x.get("title"), "url": x.get("href")} for x in res]}
            except Exception as e:
                return {"backend": "unavailable", "results": [], "error": str(e)}

        try:
            out = _cached(("web", query, mock), fetch)
        except Exception as e:
            out = {"backend": "error", "results": [], "error": str(e)}
        s.update({"backend": out["backend"], "urls": [r["url"] for r in out["results"]]})
        return out


PERSONAL_DOMAINS = {"gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com", "live.com"}


def _root(domain: str) -> str:
    parts = domain.lower().removeprefix("www.").split(".")
    return ".".join(parts[-3:]) if parts[-2:-1] in (["co"], ["org"], ["ac"]) and parts[-1] == "uk" else ".".join(parts[-2:])


def domain_check(email_domain: str, company: str, known_domain: str | None = None, mock: bool = False,
                 trace: Trace | None = None) -> dict:
    """Does the attendee's email domain belong to the company they claim? Returns match True/False/None."""
    email_domain = email_domain.lower()
    if email_domain in PERSONAL_DOMAINS:
        return {"match": None, "reason": "personal email domain", "evidence": None}
    if known_domain:
        ok = _root(email_domain) == _root(known_domain)
        return {"match": ok, "reason": f"register snapshot domain {known_domain}", "evidence": known_domain}
    res = web_search(f"{company} official website", mock=mock, trace=trace)
    domains = [urlparse(r["url"]).netloc for r in res["results"] if r.get("url")]
    if not domains:
        return {"match": None, "reason": f"no web evidence ({res['backend']})", "evidence": None}
    hit = next((d for d in domains if _root(d) == _root(email_domain)), None)
    return {"match": bool(hit), "reason": "web search result domains", "evidence": hit or domains[:3]}


def parse_sender(sender: str) -> tuple[str, str]:
    name, addr = parseaddr(sender)
    return name, addr
