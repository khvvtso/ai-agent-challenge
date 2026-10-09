"""One-time local OAuth for the demo Google account.

Usage: download an OAuth *Desktop app* client as credentials.json, then
    .venv/bin/python scripts/google_auth.py
Paste the printed values into .env (local) or Streamlit secrets (deployed).
"""
import json
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
print(f'\nGOOGLE_CLIENT_ID="{client["client_id"]}"')
print(f'GOOGLE_CLIENT_SECRET="{client["client_secret"]}"')
print(f'GOOGLE_REFRESH_TOKEN="{creds.refresh_token}"')
