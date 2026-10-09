"""Seed the demo Gmail inbox + Calendar with the sample data in data/mock/.

Messages are *inserted* into the demo mailbox (users.messages.insert) - nothing is
sent to the fictional addresses. The calendar event is created with sendUpdates=none
so no invitations go out.
    .venv/bin/python scripts/seed_google.py
"""
import base64
import json
import sys
from email.mime.text import MIMEText
from email.utils import format_datetime
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.tools import _google  # noqa: E402

gmail, cal = _google("gmail", "v1"), _google("calendar", "v3")
me = gmail.users().getProfile(userId="me").execute()["emailAddress"]

for e in json.loads((ROOT / "data/mock/emails.json").read_text())["emails"]:
    msg = MIMEText(e["body"])
    msg["From"], msg["To"], msg["Subject"] = e["sender"], me, e["subject"]
    msg["Date"] = format_datetime(datetime.fromisoformat(e["date"]))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    gmail.users().messages().insert(userId="me", body={"raw": raw, "labelIds": ["INBOX", "UNREAD"]},
                                    internalDateSource="dateHeader").execute()
    print("inserted", e["id"], e["subject"])

ev = json.loads((ROOT / "data/mock/calendar.json").read_text())["events"][0]
body = {
    "summary": ev["summary"], "location": ev["location"], "description": ev["description"],
    "start": {"dateTime": ev["start"]}, "end": {"dateTime": ev["end"]},
    "attendees": [{"email": a["email"], "displayName": a["displayName"]} for a in ev["attendees"]],
}
created = cal.events().insert(calendarId="primary", body=body, sendUpdates="none").execute()
print("created event", created["id"], created.get("htmlLink"))
