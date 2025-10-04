"""Generate and print the Strava OAuth2 authorisation URL.

This script reads the `STRAVA_CLIENT_ID` and `STRAVA_REDIRECT_URI` values
from the local `.env` file and prints the full authorisation link to stdout.
"""
from __future__ import annotations

import os
from urllib.parse import urlencode

from dotenv import load_dotenv


AUTH_URL = "https://www.strava.com/oauth/authorize"
DEFAULT_SCOPE = "read,activity:read_all"


def main() -> None:
    """Load configuration and print the OAuth2 authorisation URL."""
    load_dotenv()

    client_id = os.getenv("STRAVA_CLIENT_ID")
    redirect_uri = os.getenv("STRAVA_REDIRECT_URI")

    if not client_id:
        raise SystemExit("Missing STRAVA_CLIENT_ID in environment or .env file")
    if not redirect_uri:
        raise SystemExit("Missing STRAVA_REDIRECT_URI in environment or .env file")

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": DEFAULT_SCOPE,
        "approval_prompt": "force",
    }

    query = urlencode(params)
    url = f"{AUTH_URL}?{query}"
    print(url)


if __name__ == "__main__":
    main()
