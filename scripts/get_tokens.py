# get_tokens.py
from flask import Flask, redirect, request
import os, requests
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("STRAVA_REDIRECT_URI", "http://localhost:5000/callback")
AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"

app = Flask(__name__)

@app.route("/")
def home():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "approval_prompt": "force",
        "scope": "read,activity:read_all"
    }
    return redirect(f"{AUTH_URL}?{urlencode(params)}")

@app.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return "No 'code' param.", 400
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    }
    r = requests.post(TOKEN_URL, data=data, timeout=20)
    if r.status_code != 200:
        return f"Token exchange failed: {r.status_code} - {r.text}", 400
    td = r.json()
    access = td["access_token"]
    refresh = td["refresh_token"]
    # actualitza .env
    lines = []
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    d = dict(
        STRAVA_ACCESS_TOKEN=access,
        STRAVA_REFRESH_TOKEN=refresh
    )
    def upsert(lines, k, v):
        key = k + "="
        found = False
        for i, line in enumerate(lines):
            if line.startswith(key):
                lines[i] = f"{key}{v}\n"
                found = True
                break
        if not found:
            lines.append(f"{key}{v}\n")
    for k, v in d.items():
        upsert(lines, k, v)
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return (
        "<h3>OK — Tokens guardats al .env</h3>"
        "<p>Ja pots tancar això i executar <code>python strava_to_csv.py</code></p>"
    )

if __name__ == "__main__":
    app.run(port=5000, debug=False)
