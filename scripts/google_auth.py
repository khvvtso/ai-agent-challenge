"""One-time local OAuth for the demo Google account.

Usage: download an OAuth *Desktop app* client as credentials.json, then
    .venv/bin/python scripts/google_auth.py
The client id/secret and refresh token are written straight into .env (never printed);
copy them from .env into Streamlit secrets for the deployed app.
"""
import json
import re
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

creds_file = Path(__file__).resolve().parent.parent / "credentials.json"
flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
client = json.loads(creds_file.read_text())
client = client.get("installed") or client.get("web")
values = {"GOOGLE_CLIENT_ID": client["client_id"], "GOOGLE_CLIENT_SECRET": client["client_secret"],
          "GOOGLE_REFRESH_TOKEN": creds.refresh_token}
env = creds_file.parent / ".env"
text = env.read_text() if env.exists() else ""
for k, v in values.items():
    line = f"{k}={v}"
    text = re.sub(rf"^{k}=.*$", lambda _: line, text, flags=re.M) if re.search(rf"^{k}=", text, re.M) else text + f"\n{line}"
env.write_text(text)
print(f"Saved {', '.join(values)} to {env} (values not shown).")
