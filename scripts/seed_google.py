"""Seed the demo Gmail inbox + Calendar with the sample data in data/mock/.

Messages are *inserted* into the demo mailbox (users.messages.insert) - nothing is
sent to the fictional addresses. The calendar event is created with sendUpdates=none
so no invitations go out. Idempotent: emails carry a fixed Message-ID and the event is
matched by title, so re-running never creates duplicates.
    .venv/bin/python scripts/seed_google.py
"""
import base64
import json
import sys
import time
from email.mime.text import MIMEText
from email.utils import format_datetime
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.tools import _google  # noqa: E402



def call(req, tries=4):
    for i in range(tries):
        try:
            return req.execute()
        except Exception as e:  # transient network / SSL drops
            if i == tries - 1:
                raise
            print(f"  retry after {type(e).__name__}")
            time.sleep(2 * (i + 1))


gmail, cal = _google("gmail", "v1"), _google("calendar", "v3")
me = call(gmail.users().getProfile(userId="me"))["emailAddress"]
print("seeding", me)

for e in json.loads((ROOT / "data/mock/emails.json").read_text())["emails"]:
    mid = f"<seed-{e['id']}@ai-agent-challenge.demo>"
    if call(gmail.users().messages().list(userId="me", q=f"rfc822msgid:{mid}")).get("messages"):
        print("exists  ", e["id"], e["subject"])
        continue
    msg = MIMEText(e["body"])
    msg["Message-ID"] = mid
    msg["From"], msg["To"], msg["Subject"] = e["sender"], me, e["subject"]
    msg["Date"] = format_datetime(datetime.fromisoformat(e["date"]))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    call(gmail.users().messages().insert(userId="me", body={"raw": raw, "labelIds": ["INBOX", "UNREAD"]},
                                         internalDateSource="dateHeader"))
    print("inserted", e["id"], e["subject"])

ev = json.loads((ROOT / "data/mock/calendar.json").read_text())["events"][0]
body = {
    "summary": ev["summary"], "location": ev["location"], "description": ev["description"],
    "start": {"dateTime": ev["start"]}, "end": {"dateTime": ev["end"]},
    "attendees": [{"email": a["email"], "displayName": a["displayName"]} for a in ev["attendees"]],
}
existing = call(cal.events().list(calendarId="primary", q=ev["summary"], singleEvents=True)).get("items", [])
if any(x.get("summary") == ev["summary"] for x in existing):
    print("event exists", ev["summary"])
else:
    created = call(cal.events().insert(calendarId="primary", body=body, sendUpdates="none"))
    print("created event", created["id"], created.get("htmlLink"))
