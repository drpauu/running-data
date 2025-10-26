"""
Generate a Strava OAuth2 authorisation link ready to share with teammates.

The script follows Strava's OAuth documentation and produces a URL with the
required query parameters so your teammates can simply click and authorise
the app. By default it loads configuration values from a local `.env` file,
but you can override any of them with command-line options.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from urllib.parse import urlencode

from dotenv import load_dotenv

AUTH_URL = "https://www.strava.com/oauth/authorize"
DEFAULT_SCOPE = ("read", "activity:read_all")


def build_authorisation_url(
    *,
    client_id: str,
    redirect_uri: str,
    scopes: Sequence[str],
    approval_prompt: str,
    state: str | None,
) -> str:
    """Return the fully-qualified Strava authorisation URL."""
    params: dict[str, str] = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": ",".join(scopes),
        "approval_prompt": approval_prompt,
    }

    if state:
        params["state"] = state

    return f"{AUTH_URL}?{urlencode(params)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Strava OAuth authorisation link so teammates can authorise "
            "the application without manually crafting the URL."
        )
    )
    parser.add_argument(
        "--client-id",
        help="Override STRAVA_CLIENT_ID from the environment",
    )
    parser.add_argument(
        "--redirect-uri",
        help="Override STRAVA_REDIRECT_URI from the environment",
    )
    parser.add_argument(
        "--scope",
        nargs="*",
        metavar="SCOPE",
        help=(
            "Scopes to request, space-separated. Defaults to 'read activity:read_all' "
            "(see https://developers.strava.com/docs/authentication/)."
        ),
    )
    parser.add_argument(
        "--approval-prompt",
        choices=["auto", "force"],
        default="force",
        help=(
            "Whether Strava should force the approval screen even if the athlete "
            "already authorised the app."
        ),
    )
    parser.add_argument(
        "--state",
        help="Optional opaque state parameter that will be echoed back by Strava.",
    )
    return parser.parse_args()


def main() -> None:
    """Load configuration and print the OAuth2 authorisation URL."""
    load_dotenv()
    args = parse_args()

    client_id = args.client_id or os.getenv("STRAVA_CLIENT_ID")
    redirect_uri = args.redirect_uri or os.getenv("STRAVA_REDIRECT_URI")
    scopes = tuple(args.scope) if args.scope else DEFAULT_SCOPE

    if not client_id:
        raise SystemExit("Missing STRAVA_CLIENT_ID in environment or .env file")
    if not redirect_uri:
        raise SystemExit("Missing STRAVA_REDIRECT_URI in environment or .env file")
    if not scopes:
        raise SystemExit("At least one scope must be provided")

    url = build_authorisation_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scopes=scopes,
        approval_prompt=args.approval_prompt,
        state=args.state,
    )
    print(url)


if __name__ == "__main__":
    main()
