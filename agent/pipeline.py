"""Event-to-Lead agent: calendar + inbox -> cleansed, verified attendee list.

Division of labour (the core design choice):
  * LLM planner   - decides which tool to call next (ReAct-style JSON loop)
  * LLM extractor - turns messy emails into a typed schema, quoting its source per field
  * Judge         - small calibrated-style judgements (is this an RSVP? attending?)
  * Deterministic - grounding check, normalisation, dedupe, verification, decision table, critic
If no LLM is available (or the planner misbehaves) a fixed plan and a heuristic
extractor take over, and the trace records that the fallback was used.
"""
from __future__ import annotations

import io
import json
import re
from typing import Callable

import pandas as pd
import phonenumbers
from email_validator import EmailNotValidError, validate_email
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from agent import tools
from core import judge, llm
from core.trace import Trace
from decisions import engine

EVENT_QUERY = "AI in Production Roundtable"
EMAIL_QUERY = "roundtable OR RSVP OR registration OR invitation newer_than:90d"
REQUIRED = ["name", "email", "company", "job_title", "phone"]
FIELDS = ["name", "email", "company", "job_title", "phone", "rsvp_status", "plus_one", "dietary"]
NICKNAMES = {"jon": "jonathan", "jonny": "jonathan", "dan": "daniel", "danny": "daniel", "mike": "michael",
             "tom": "thomas", "liz": "elizabeth", "sam": "samuel", "alex": "alexander", "chris": "christopher",
             "kate": "katherine", "rob": "robert", "bob": "robert", "will": "william", "bill": "william"}
COST_OF_BAD_LEAD, REVIEW_COST = 50.0, 5.0  # GBP - drives the human-review routing


class Extracted(BaseModel):
    name: str | None = None
    email: str | None = None
    company: str | None = None
    job_title: str | None = None
    phone: str | None = None
    rsvp_status: str | None = None  # attending | declined | tentative | unclear
    plus_one: str | None = None     # "Name <email>"
    dietary: str | None = None
    source_email_id: str
    quotes: dict[str, str] = Field(default_factory=dict)


# ================================================================= extraction
EXTRACT_SYSTEM = """You extract event RSVP data from emails. Rules:
- Only include emails that are a personal RSVP/registration/attendance request for the event. Skip newsletters, marketing, automated digests.
- For forwarded registration forms, the attendee is the person in the form, not the forwarder.
- Output ONE entry PER EMAIL. Never merge people across emails, even if they look like the same person - deduplication is done downstream by deterministic rules. A plus-one belongs to the email that mentions it.
- Copy values exactly as written. Never guess or infer a value that is not in the email. Use null if absent.
- For EVERY non-null field, give in "quotes" the exact substring of the email that contains it.
- rsvp_status: attending | declined | tentative | unclear. plus_one: "Name <email>" if they ask to bring someone.
Return JSON: {"attendees": [{"name","email","company","job_title","phone","rsvp_status","plus_one","dietary","source_email_id","quotes": {...}}],
 "ignored": [{"email_id","reason"}]}"""


def _email_text(m: dict) -> str:
    return f"From: {m['sender']}\nSubject: {m['subject']}\n\n{m['body']}"


def extract_llm(messages: list[dict], trace: Trace) -> tuple[list[Extracted], list[dict]]:
    payload = "\n\n".join(f"=== EMAIL id={m['id']} date={m['date']} ===\n{_email_text(m)}" for m in messages)
    raw = llm.complete_json(EXTRACT_SYSTEM, payload, trace, "extract.llm")
    out = []
    for a in raw.get("attendees", []):
        try:
            a["quotes"] = {k: str(v) for k, v in (a.get("quotes") or {}).items() if v}
            for f in FIELDS:
                if a.get(f) is not None:
                    a[f] = str(a[f]).strip() or None
            out.append(Extracted(**a))
        except Exception:
            continue
    return out, raw.get("ignored", [])


_PHONE = re.compile(r"(\+?\(?\d[\d\s()]{8,}\d)")
_DECLINE = re.compile(r"won't be able|can't make|cannot attend|unable to attend|unfortunately|have to decline", re.I)
_ATTEND = re.compile(r"i'll|count me in|i'm in|attend|delighted to join|confirm|yes|keen to|join", re.I)
_NOISE = re.compile(r"unsubscribe|manage preferences|newsletter|digest", re.I)


def _heuristic_one(m: dict) -> Extracted | None:
    body, sender = m["body"], m["sender"]
    if _NOISE.search(body) or _NOISE.search(sender):
        return None
    status = "declined" if _DECLINE.search(body) else "attending" if _ATTEND.search(body) else "unclear"
    q: dict[str, str] = {}
    rec: dict = {"source_email_id": m["id"], "rsvp_status": status}
    form = dict(re.findall(r"^(Name|Email|Company|Job title|Phone|Dietary):[ \t]*(.*)$", body, re.M))
    if form.keys() & {"Name", "Email"}:
        for k, f in [("Name", "name"), ("Email", "email"), ("Company", "company"), ("Job title", "job_title"),
                     ("Phone", "phone"), ("Dietary", "dietary")]:
            if form.get(k, "").strip():
                rec[f] = form[k].strip()
                q[f] = f"{k}: {form[k].strip()}"
    else:
        name, addr = tools.parse_sender(sender)
        rec["email"], q["email"] = addr, sender
        if name:
            rec["name"], q["name"] = name, sender
        sig = body.split("\n--\n", 1)[1] if "\n--\n" in body else ""
        lines = [ln.strip() for ln in sig.splitlines() if ln.strip()]
        if len(lines) == 1 and "|" in lines[0]:
            lines = [p.strip() for p in lines[0].split("|")]
        phone_line = next((ln for ln in lines if _PHONE.search(ln)), None)
        text_lines = [ln for ln in lines if ln is not phone_line and ":" not in ln]
        if text_lines:
            rec["name"], q["name"] = text_lines[0], text_lines[0]
        if len(text_lines) > 1:
            rec["job_title"], q["job_title"] = text_lines[1], text_lines[1]
        if len(text_lines) > 2:
            rec["company"], q["company"] = text_lines[2], text_lines[2]
        if phone_line:
            rec["phone"], q["phone"] = _PHONE.search(phone_line).group(1).strip(), phone_line
    diet = re.search(r"Dietary:[ \t]*(\S.*)", body)
    if diet and "dietary" not in rec:
        rec["dietary"], q["dietary"] = diet.group(1).strip(), diet.group(0)
    po = re.search(r"bring (?:my colleague )?([A-Z][\w'-]+ [A-Z][\w'-]+) \(([^)\s]+@[^)\s]+)\)", body)
    if po:
        rec["plus_one"], q["plus_one"] = f"{po.group(1)} <{po.group(2)}>", po.group(0)
    if status != "unclear":
        q["rsvp_status"] = (_DECLINE.search(body) or _ATTEND.search(body)).group(0)
    return Extracted(**rec, quotes=q)


def extract_heuristic(messages: list[dict]) -> tuple[list[Extracted], list[dict]]:
    out, ignored = [], []
    for m in messages:
        r = _heuristic_one(m)
        if r:
            out.append(r)
        else:
            ignored.append({"email_id": m["id"], "reason": "newsletter/marketing markers (unsubscribe, digest)"})
    return out, ignored


def classify_emails(messages: list[dict], trace: Trace) -> dict:
    """One batched judge call: is each email an RSVP, and what is the RSVP status?"""
    questions = {}
    for m in messages:
        questions[f"{m['id']}__is_rsvp"] = {
            "instructions": f"Email {m['id']}: is this a personal RSVP, registration or request to attend the event (not marketing/newsletter)?",
            "options": {"yes": "personal RSVP/registration", "no": "newsletter, marketing or unrelated"}}
        questions[f"{m['id']}__status"] = {
            "instructions": f"Email {m['id']}: what is the sender's (or registrant's) attendance intent?",
            "options": {"attending": "will attend / wants to attend", "declined": "will not attend",
                        "tentative": "maybe", "unclear": "cannot tell"}}
    state = {m["id"]: _email_text(m) for m in messages}
    return judge.decide(state, questions, trace, "judge.classify_emails")


# ================================================================= grounding
def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def grounded(field: str, value: str, source: str) -> bool:
    if field == "phone":
        d = _digits(value)
        return len(d) >= 9 and d[-9:] in _digits(source)
    if field == "email":
        return value.lower() in source.lower()
    if field in ("rsvp_status",):
        return True  # a classification, not a copied value - checked by the judge instead
    return fuzz.partial_ratio(value.lower(), source.lower()) >= 85


def grounding_check(records: list[Extracted], messages: dict[str, dict], audit: list, trace: Trace) -> list[Extracted]:
    with trace.span("grounding_check", "guardrail", input={"records": len(records)}) as s:
        blocked = 0
        for r in records:
            m = messages.get(r.source_email_id)
            if not m:
                continue
            src = _email_text(m)
            for f in FIELDS:
                v = getattr(r, f)
                if v and not grounded(f, v, src):
                    audit.append({"record": r.email or r.name, "field": f, "before": v, "after": None,
                                  "rule": "hallucination_blocked", "source": r.source_email_id})
                    setattr(r, f, None)
                    blocked += 1
        s.update({"blocked_values": blocked})
    return records


# ================================================================= normalisation
def norm_name(name: str | None) -> str | None:
    if not name:
        return None
    name = re.sub(r"\s+", " ", name).strip()
    if name != name.lower() and name != name.upper():
        return name

    def cap(w: str) -> str:
        w = w.lower()
        w = re.sub(r"(^|['-])(\w)", lambda m: m.group(1) + m.group(2).upper(), w)
        return re.sub(r"^Mc(\w)", lambda m: "Mc" + m.group(1).upper(), w)

    return " ".join(cap(w) for w in name.split(" "))


def name_key(name: str | None) -> str:
    parts = (name or "").lower().split()
    if parts:
        parts[0] = NICKNAMES.get(parts[0], parts[0])
    return " ".join(parts)


def norm_phone(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    try:
        n = phonenumbers.parse(raw, "GB")
        if phonenumbers.is_valid_number(n):
            return phonenumbers.format_number(n, phonenumbers.PhoneNumberFormat.E164), None
    except phonenumbers.NumberParseException:
        pass
    return None, f"invalid phone number: {raw}"


def email_ok(email: str | None) -> bool:
    if not email:
        return False
    try:
        validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False


def _domain_company(email: str | None) -> dict | None:
    if not email or "@" not in email:
        return None
    dom = email.split("@")[1].lower()
    return next((c for c in tools._directory() if c.get("domain") and tools._root(c["domain"]) == tools._root(dom)), None)


def _rec(**kw) -> dict:
    base = {f: None for f in FIELDS}
    base.update({"invited": False, "is_plus_one": False, "plus_one_of": None, "calendar_status": None,
                 "sources": [], "flags": [], "conflicts": [], "corrected": False, "date": "", "quotes": {}})
    base.update(kw)
    return base


def normalize(extracted: list[Extracted], event: dict, messages: dict[str, dict], audit: list) -> list[dict]:
    records = []
    for e in extracted:
        r = _rec(**{f: getattr(e, f) for f in FIELDS}, sources=[e.source_email_id], quotes=e.quotes,
                 date=messages.get(e.source_email_id, {}).get("date", ""))
        records.append(r)
        if e.plus_one:
            m = re.match(r"\s*(.*?)\s*<([^>]+)>", e.plus_one)
            if m:
                records.append(_rec(name=m.group(1), email=m.group(2), rsvp_status="attending", is_plus_one=True,
                                    plus_one_of=e.email, sources=[e.source_email_id], date=r["date"],
                                    quotes={"name": e.plus_one}))
    for r in records:
        who = r["email"] or r["name"]
        for f, fn in (("name", norm_name), ("email", lambda v: v.strip().lower() if v else v)):
            new = fn(r[f])
            if new != r[f]:
                audit.append({"record": who, "field": f, "before": r[f], "after": new, "rule": f"normalize_{f}", "source": "rules"})
                r[f] = new
        if r["phone"]:
            e164, err = norm_phone(r["phone"])
            audit.append({"record": who, "field": "phone", "before": r["phone"], "after": e164,
                          "rule": "invalid_phone" if err else "normalize_phone_e164", "source": "phonenumbers"})
            if err:
                r["flags"].append(err)
            r["phone"] = e164
        if r["company"]:
            r["company"] = re.sub(r"\s+", " ", r["company"]).strip()
        elif (c := _domain_company(r["email"])):
            audit.append({"record": who, "field": "company", "before": None, "after": c["name"],
                          "rule": "infer_company_from_email_domain", "source": "reference domain list"})
            r["company"] = c["name"]
            r["corrected"] = True
    return records


# ================================================================= calendar merge + dedupe
def _distinct(vals: list[str], key=lambda v: v.casefold()) -> list[str]:
    seen, out = set(), []
    for v in vals:
        if v and key(v) not in seen:
            seen.add(key(v))
            out.append(v)
    return out


def _merge(group: list[dict], audit: list) -> dict:
    group = sorted(group, key=lambda r: r["date"] or "")
    base = dict(group[-1])
    base["sources"] = sorted({s for r in group for s in r["sources"]})
    base["flags"] = sorted({f for r in group for f in r["flags"]})
    base["conflicts"] = [c for r in group for c in r["conflicts"]]
    base["corrected"] = any(r["corrected"] for r in group)
    base["invited"] = any(r["invited"] for r in group)
    base["is_plus_one"] = all(r["is_plus_one"] for r in group)
    base["quotes"] = {k: v for r in group for k, v in r["quotes"].items()}
    who = base["email"] or base["name"]
    audit.append({"record": who, "field": "*", "before": f"{len(group)} records", "after": "1 record",
                  "rule": "dedupe_merge", "source": ",".join(base["sources"])})
    for f in FIELDS:
        vals = [r[f] for r in group if r[f]]
        key = name_key if f == "name" else (lambda v: v.casefold())
        distinct = _distinct(vals, key)
        if f == "name":
            vals_all = _distinct(vals)
            if len(distinct) == 1 and len(vals_all) > 1:  # nickname variants: keep the fullest form
                full = max(vals_all, key=len)
                audit.append({"record": who, "field": "name", "before": " / ".join(vals_all), "after": full,
                              "rule": "nickname_resolved", "source": "nickname map"})
                base["name"] = full
                continue
        if len(distinct) > 1 and f != "rsvp_status":
            base["conflicts"].append({"field": f, "values": distinct, "kept": vals[-1], "policy": "latest email wins"})
            base[f] = vals[-1]
        elif vals:
            base[f] = vals[-1]
    return base


def merge_calendar(records: list[dict], event: dict, audit: list) -> list[dict]:
    invitees = {a["email"].lower(): a for a in event.get("attendees", []) if a.get("email")}
    by_email = {r["email"]: r for r in records if r["email"]}
    status_map = {"accepted": "attending", "declined": "declined", "tentative": "tentative", "needsAction": "no_response"}
    for em, inv in invitees.items():
        inv_name = norm_name(inv.get("displayName"))
        if em in by_email:
            r = by_email[em]
            r["invited"] = True
            r["calendar_status"] = inv.get("responseStatus")
            if inv_name and r["name"] and name_key(inv_name) != name_key(r["name"]):
                local = em.split("@")[0].replace(".", " ")
                pick = max([r["name"], inv_name], key=lambda n: (n.split()[-1].lower() in local, n == inv_name))
                if fuzz.ratio(inv_name.lower(), r["name"].lower()) >= 80:
                    if pick != r["name"]:
                        audit.append({"record": em, "field": "name", "before": r["name"], "after": pick,
                                      "rule": "typo_corrected_against_invite+email_address", "source": "calendar"})
                        r["name"], r["corrected"] = pick, True
                else:
                    r["conflicts"].append({"field": "name", "values": [r["name"], inv_name], "kept": r["name"],
                                           "policy": "self-reported name kept"})
        else:
            new = _rec(name=inv_name, email=em, invited=True, calendar_status=inv.get("responseStatus"),
                       rsvp_status=status_map.get(inv.get("responseStatus"), "no_response"), sources=["calendar"])
            if (c := _domain_company(em)):
                new["company"], new["corrected"] = c["name"], True
                audit.append({"record": em, "field": "company", "before": None, "after": c["name"],
                              "rule": "infer_company_from_email_domain", "source": "reference domain list"})
            new["flags"].append("no RSVP email found - from calendar invite only")
            records.append(new)
    return records


def dedupe(records: list[dict], audit: list) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["email"] or f"noemail:{name_key(r['name'])}", []).append(r)
    merged = [(_merge(g, audit) if len(g) > 1 else g[0]) for g in groups.values()]
    out: list[dict] = []
    for r in merged:  # fuzzy pass: same person under two different addresses
        dup = next((o for o in out if fuzz.token_sort_ratio(name_key(o["name"]), name_key(r["name"])) >= 90
                    and fuzz.token_set_ratio((o["company"] or "").lower(), (r["company"] or "").lower()) >= 90), None)
        if dup:
            m = _merge([dup, r], audit)
            m["conflicts"].append({"field": "email", "values": [dup["email"], r["email"]], "kept": m["email"],
                                   "policy": "possible duplicate person - confirm"})
            out[out.index(dup)] = m
        else:
            out.append(r)
    return out


# ================================================================= verification + decision
def verify(records: list[dict], mock: bool, trace: Trace) -> list[dict]:
    for r in records:
        if r["company"]:
            ch = tools.companies_house_search(r["company"], mock=mock, trace=trace)
            if ch.get("found"):
                r["company_status"] = (ch.get("status") or "unknown").lower()
                r["registered_name"], r["company_number"] = ch.get("name"), ch.get("number")
            else:
                r["company_status"] = "not_found" if ch.get("found") is False else "unknown"
                r["registered_name"] = r["company_number"] = None
            r["register_source"] = ch.get("source")
            dom = r["email"].split("@")[1] if r["email"] and "@" in r["email"] else ""
            dc = tools.domain_check(dom, r["registered_name"] or r["company"], ch.get("domain"), mock=mock, trace=trace) if dom \
                else {"match": None, "reason": "no email"}
            r["domain_match"], r["domain_evidence"] = dc["match"], dc["reason"]
        else:
            r.update(company_status="unknown", registered_name=None, company_number=None, register_source=None,
                     domain_match=None, domain_evidence="no company")
    return records


def decide(records: list[dict], rsvp_conf: dict[str, float], trace: Trace) -> list[dict]:
    with trace.span("decision_table.record_status", "evaluator", input={"records": len(records)}) as s:
        fired = {}
        for r in records:
            r["missing_fields"] = [f for f in REQUIRED if not r.get(f)]
            inputs = {"rsvp_status": r["rsvp_status"], "email_valid": email_ok(r["email"]),
                      "company_status": r.get("company_status"), "invited": r["invited"], "is_plus_one": r["is_plus_one"],
                      "has_conflict": bool(r["conflicts"]), "domain_match": r.get("domain_match"),
                      "missing_count": len(r["missing_fields"]), "was_corrected": r["corrected"]}
            d = engine.evaluate("record_status", inputs)
            r["status"], r["reason"], r["rule_id"], r["decision_inputs"] = d["status"], d["reason"], d["rule"], inputs
            conf = min((rsvp_conf.get(src, 0.8) for src in r["sources"]), default=0.8)
            p_wrong = 0.02 + (1 - conf) * 0.3
            p_wrong += {"not_found": 0.4, "unknown": 0.2}.get(r.get("company_status"), 0)
            p_wrong += 0.3 if r.get("company_status") in ("dissolved", "liquidation", "closed", "inactive") else 0
            p_wrong += 0.25 if r.get("domain_match") is False else 0
            p_wrong += 0.15 * bool(r["conflicts"]) + 0.08 * len(r["missing_fields"]) + 0.02 * r["corrected"]
            r["p_wrong"] = round(min(p_wrong, 0.95), 2)
            r["human_review"] = r["status"] in ("conflict", "not_invited") or (
                r["status"] != "declined" and engine.needs_human_review(r["p_wrong"], COST_OF_BAD_LEAD, REVIEW_COST))
            fired[r["rule_id"]] = fired.get(r["rule_id"], 0) + 1
        s.update({"rules_fired": fired})
    return records


def critic(records: list[dict], ignored_ids: set[str]) -> list[str]:
    """Deterministic invariants the final output must satisfy."""
    problems = []
    emails = [r["email"] for r in records if r["email"]]
    if len(emails) != len(set(emails)):
        problems.append("duplicate emails in output")
    for r in records:
        if r["phone"] and not re.fullmatch(r"\+\d{10,15}", r["phone"]):
            problems.append(f"{r['email']}: phone not E.164")
        if r["status"] in ("verified", "corrected") and (r.get("company_status") != "active" or r.get("domain_match") is False):
            problems.append(f"{r['email']}: verified without an active company / matching domain")
        if not r["email"]:
            problems.append(f"{r['name']}: record without email")
        if set(r["sources"]) & ignored_ids:
            problems.append(f"{r['email']}: built from an ignored (non-RSVP) email")
    return problems


OUT_COLS = ["status", "human_review", "name", "email", "company", "registered_name", "company_status", "job_title",
            "phone", "rsvp_status", "dietary", "invited", "is_plus_one", "plus_one_of", "missing_fields", "flags",
            "conflicts", "reason", "rule_id", "p_wrong", "domain_match", "sources"]


def to_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {c: r.get(c) for c in OUT_COLS}
        row["missing_fields"] = ", ".join(r.get("missing_fields", []))
        row["flags"] = "; ".join(r.get("flags", []))
        row["conflicts"] = "; ".join(f"{c['field']}: {' vs '.join(map(str, c['values']))}" for c in r.get("conflicts", []))
        row["sources"] = ", ".join(r.get("sources", []))
        rows.append(row)
    order = {"verified": 0, "corrected": 1, "missing_fields": 2, "conflict": 3, "not_invited": 4, "unverified": 5, "declined": 6}
    df = pd.DataFrame(rows, columns=OUT_COLS)
    return df.sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)


# ================================================================= agent
TOOLS_DOC = """get_event(query)            - fetch the calendar event and its invitee list
search_emails(query)        - fetch RSVP/registration emails from the inbox
extract()                   - LLM-extract attendee data from fetched emails (+ classify RSVPs, grounding check). needs: emails
normalize_dedupe()          - deterministic cleanse, merge with calendar invitees, dedupe. needs: extract, event
verify()                    - check companies in the UK register + email-domain match, apply decision table. needs: normalize_dedupe
finalize()                  - run critic invariants and build the output table. needs: verify
send_summary(to)            - email a summary to the organiser. needs: finalize
finish()                    - stop. needs: finalize"""

PLANNER_SYSTEM = f"""You are the orchestrator of a data-pipeline agent. Goal: produce a cleansed, verified attendee list
for the event "{EVENT_QUERY}" by combining calendar invitees with RSVP emails. Tools:
{TOOLS_DOC}
Respond with JSON only: {{"thought": "<one sentence>", "tool": "<tool name>", "args": {{...}}}}"""


class Agent:
    def __init__(self, mock: bool, trace: Trace, on_step: Callable[[str], None] | None, send_to: str | None):
        self.mock, self.trace, self.send_to = mock, trace, send_to
        self.say = on_step or (lambda s: None)
        self.audit: list[dict] = []
        self.steps: list[dict] = []
        self.state: dict = {"event": None, "messages": None, "extracted": None, "records": None,
                            "verified": False, "final": None, "sent": None, "ignored": [], "rsvp_conf": {},
                            "extractor": None, "critic": [], "critic_summary": None}

    # ---- tools exposed to the planner
    def t_get_event(self, query: str = EVENT_QUERY):
        ev = tools.calendar_get_event(query, mock=self.mock, trace=self.trace)
        if not ev.get("found"):
            return "no event found - try a different query"
        self.state["event"] = ev
        return f"event '{ev['summary']}' with {len(ev['attendees'])} invitees (backend={ev['backend']})"

    def t_search_emails(self, query: str = EMAIL_QUERY):
        summaries = tools.gmail_search(query, mock=self.mock, trace=self.trace)
        self.state["messages"] = {m["id"]: tools.gmail_read(m["id"], mock=self.mock) for m in summaries}
        return f"{len(summaries)} emails fetched"

    def t_extract(self):
        msgs = list(self.state["messages"].values())
        with self.trace.span("extract", "chain", input={"emails": len(msgs)}) as s:
            extracted, ignored = None, []
            if llm.available():
                try:
                    extracted, ignored = extract_llm(msgs, self.trace)
                    self.state["extractor"] = "llm"
                    cls = classify_emails(msgs, self.trace)
                    for m in msgs:
                        is_rsvp, st = cls[f"{m['id']}__is_rsvp"], cls[f"{m['id']}__status"]
                        self.state["rsvp_conf"][m["id"]] = st["confidence"]
                        if is_rsvp["choice"] == "no" and is_rsvp["confidence"] >= 0.6:
                            before = len(extracted)
                            extracted = [e for e in extracted if e.source_email_id != m["id"]]
                            if len(extracted) < before or m["id"] not in {i.get("email_id") for i in ignored}:
                                ignored.append({"email_id": m["id"], "reason": f"judge: not an RSVP (p={is_rsvp['confidence']})"})
                        for e in extracted:
                            if e.source_email_id == m["id"] and st["confidence"] >= 0.7 and e.rsvp_status != st["choice"]:
                                self.audit.append({"record": e.email or e.name, "field": "rsvp_status", "before": e.rsvp_status,
                                                   "after": st["choice"], "rule": "judge_override", "source": m["id"]})
                                e.rsvp_status = st["choice"]
                except llm.LLMUnavailable as err:
                    s.update(fallback_reason=str(err)[:300])
                    extracted = None
            if extracted is None:
                extracted, ignored = extract_heuristic(msgs)
                self.state["extractor"] = "heuristic"
                self.state["rsvp_conf"] = {m["id"]: 0.8 for m in msgs}
            extracted = grounding_check(extracted, self.state["messages"], self.audit, self.trace)
            self.state["extracted"], self.state["ignored"] = extracted, ignored
            s.update({"extractor": self.state["extractor"], "records": len(extracted), "ignored": ignored})
        return f"{len(extracted)} attendee mentions extracted ({self.state['extractor']}), {len(ignored)} emails ignored"

    def t_normalize_dedupe(self):
        with self.trace.span("normalize_dedupe", "chain", input={"extracted": len(self.state["extracted"])}) as s:
            recs = normalize(self.state["extracted"], self.state["event"], self.state["messages"], self.audit)
            recs = dedupe(recs, self.audit)
            recs = merge_calendar(recs, self.state["event"], self.audit)
            recs = dedupe(recs, self.audit)
            self.state["records"] = recs
            s.update({"records": len(recs)})
        return f"{len(recs)} unique people after cleansing and dedupe"

    def t_verify(self):
        with self.trace.span("verify", "chain", input={"records": len(self.state["records"])}) as s:
            recs = verify(self.state["records"], self.mock, self.trace)
            recs = decide(recs, self.state["rsvp_conf"], self.trace)
            self.state["records"], self.state["verified"] = recs, True
            counts = pd.Series([r["status"] for r in recs]).value_counts().to_dict()
            s.update(counts)
        return f"statuses: {counts}"

    def t_finalize(self):
        ignored_ids = {i.get("email_id") for i in self.state["ignored"]}
        with self.trace.span("critic", "evaluator", input={"records": len(self.state["records"])}) as s:
            problems = critic(self.state["records"], ignored_ids)
            if problems:  # self-correction: re-run cleanse/dedupe/verify once, then re-check
                self.say(f"Critic found {len(problems)} problem(s) - re-running cleanse + verify")
                self.state["records"] = [r for r in self.state["records"] if not (set(r["sources"]) & ignored_ids) or r["invited"]]
                self.state["records"] = dedupe(self.state["records"], self.audit)
                self.state["records"] = decide(verify(self.state["records"], self.mock, self.trace), self.state["rsvp_conf"], self.trace)
                problems_after = critic(self.state["records"], ignored_ids)
                s.update({"first_pass": problems, "after_retry": problems_after})
                problems = problems_after
            s.update({"problems": problems, "passed": not problems})
            self.state["critic"] = problems
        self.trace.score("critic_passed", 0.0 if problems else 1.0)
        df = to_frame(self.state["records"])
        if llm.available():
            try:
                self.state["critic_summary"] = llm.complete_text(
                    "You review a cleansed attendee list for an event organiser. In 3-5 bullet points, say what needs human "
                    "attention and why. Be specific, use names. No preamble.",
                    df[["status", "name", "company", "reason", "missing_fields", "conflicts", "flags"]].to_csv(index=False),
                    self.trace, "critic.llm_summary")
            except llm.LLMUnavailable:
                pass
        self.state["final"] = df
        return f"final table: {len(df)} rows, critic {'passed' if not problems else 'flagged: ' + '; '.join(problems)}"

    def t_send_summary(self, to: str | None = None):
        to = to or self.send_to
        if not to:
            return "no recipient configured - skipped"
        df = self.state["final"]
        lines = [f"Attendee list for {self.state['event']['summary']}", ""]
        lines += [f"{k}: {v}" for k, v in df["status"].value_counts().items()]
        lines += ["", "Needs review:"] + [f"- {r.name} ({r.company}): {r.reason}" for r in df[df.human_review].itertuples()]
        self.state["sent"] = tools.gmail_send(to, "Roundtable attendee list - verified", "\n".join(lines), mock=self.mock, trace=self.trace)
        return f"summary email {self.state['sent']['status']} to {to}"

    def has(self, key: str) -> bool:
        v = self.state.get(key)
        return v is not None and v is not False and not (isinstance(v, (list, dict)) and not v)

    PRE = {"extract": "messages", "normalize_dedupe": "extracted", "verify": "records", "finalize": "verified",
           "send_summary": "final", "finish": "final"}

    def call(self, tool: str, args: dict, actor: str, thought: str = "") -> str:
        need = self.PRE.get(tool)
        if tool == "normalize_dedupe" and not self.state["event"]:
            need = "event"
        if need and not self.has(need):
            obs = f"ERROR: precondition not met ({need} missing)"
        elif tool == "finish":
            obs = "done"
        else:
            fn = getattr(self, f"t_{tool}", None)
            if not fn:
                obs = f"ERROR: unknown tool {tool}"
            else:
                try:
                    obs = fn(**{k: v for k, v in (args or {}).items() if k in fn.__code__.co_varnames})
                except Exception as e:
                    obs = f"ERROR: {type(e).__name__}: {e}"
        self.steps.append({"step": len(self.steps) + 1, "actor": actor, "thought": thought, "tool": tool,
                           "args": args or {}, "observation": obs})
        self.say(f"**{tool}** → {obs}")
        return obs

    def run_planner(self, max_steps: int = 10) -> bool:
        history, errors = [], 0
        for _ in range(max_steps):
            done = {k: self.has(k) for k in ("event", "messages", "extracted", "records", "verified", "final", "sent")}
            prompt = (f"State: {json.dumps(done)}\nSend summary to: {self.send_to or 'nobody (skip send_summary)'}\n"
                      f"History:\n" + "\n".join(history[-8:]) + "\nWhat next?")
            try:
                d = llm.complete_json(PLANNER_SYSTEM, prompt, self.trace, "planner")
            except llm.LLMUnavailable as e:
                self.say(f"Planner unavailable ({str(e)[:80]}) - switching to fixed plan")
                return False
            tool, args, thought = d.get("tool", ""), d.get("args") or {}, d.get("thought", "")
            self.say(f"🧠 {thought}")
            obs = self.call(tool, args, "planner", thought)
            history.append(f"{tool}({json.dumps(args)}) -> {obs}")
            if tool == "finish" and obs == "done":
                return True
            if obs.startswith("ERROR"):
                errors += 1
                if errors >= 2:
                    self.say("Planner made repeated invalid calls - switching to fixed plan")
                    return False
        return bool(self.state["final"] is not None)

    def run_fixed(self):
        plan = [("get_event", {"query": EVENT_QUERY}, "event"), ("search_emails", {"query": EMAIL_QUERY}, "messages"),
                ("extract", {}, "extracted"), ("normalize_dedupe", {}, "records"), ("verify", {}, "verified"),
                ("finalize", {}, "final")]
        if self.send_to:
            plan.append(("send_summary", {"to": self.send_to}, "sent"))
        for tool, args, produces in plan:
            if not self.has(produces):
                self.call(tool, args, "fixed-plan")


def run(mode: str = "auto", send_summary_to: str | None = None, trace: Trace | None = None,
        on_step: Callable[[str], None] | None = None, use_planner: bool = True) -> dict:
    """mode: 'auto' = live APIs where keys exist, else mock; 'mock' = bundled demo data only."""
    mock = mode == "mock"
    trace = trace or Trace("event_to_lead_agent", input={"mode": mode})
    agent = Agent(mock, trace, on_step, send_summary_to)
    planner_ok = False
    if use_planner and llm.available():
        with trace.span("agent.planner_loop", "agent") as s:
            planner_ok = agent.run_planner()
            s.update({"completed": planner_ok})
    if agent.state["final"] is None or (send_summary_to and not agent.state["sent"]):
        with trace.span("agent.fixed_plan", "chain", input={"reason": "no LLM" if not llm.available() else "planner incomplete"}):
            agent.run_fixed()
    df = agent.state["final"]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    stats = df["status"].value_counts().to_dict()
    stats.update({"total": len(df), "needs_review": int(df["human_review"].sum()), "ignored_emails": len(agent.state["ignored"])})
    for k in ("verified", "corrected"):
        stats.setdefault(k, 0)
    trace.score("verified_ratio", round((stats["verified"] + stats["corrected"]) / max(len(df), 1), 2))
    result = {
        "records": agent.state["records"], "table": df, "csv": buf.getvalue(), "audit": agent.audit,
        "stats": stats, "steps": agent.steps, "ignored": agent.state["ignored"], "critic": agent.state["critic"],
        "critic_summary": agent.state["critic_summary"], "event": agent.state["event"], "sent": agent.state["sent"],
        "orchestration": "llm-planner" if planner_ok else "fixed-plan", "extractor": agent.state["extractor"],
        "backends": {"google": "live" if tools.google_live() and not mock else "mock",
                     "register": "companies_house" if (not mock and tools.config.get("COMPANIES_HOUSE_API_KEY")) else "snapshot",
                     "web": "skipped" if mock else ("tavily" if tools.config.get("TAVILY_API_KEY") else "duckduckgo")},
        "trace": trace,
    }
    trace.end({"stats": stats})
    return result
