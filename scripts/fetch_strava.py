#!/usr/bin/env python
"""CLI utility to download Strava streams and dump them to CSV files."""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

from running_data.io import download_streams_to_csv
from running_data.strava_client import StravaClient


def _parse_date(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    return dt.datetime.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-token", dest="access_token", default=None, help="Strava OAuth access token")
    parser.add_argument("--after", type=str, default=None, help="Only download activities after this ISO date (UTC)")
    parser.add_argument("--before", type=str, default=None, help="Only download activities before this ISO date (UTC)")
    parser.add_argument("--streams", type=Path, default=Path("strava_streams.csv"), help="Output CSV for streams")
    parser.add_argument("--athletes", type=Path, default=Path("athletes.csv"), help="Output CSV for athlete metadata")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV files")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum number of activity pages to fetch")
    parser.add_argument("--per-page", type=int, default=50, help="Number of activities per page (max 200)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    token = args.access_token or os.environ.get("STRAVA_ACCESS_TOKEN")
    if not token:
        raise SystemExit("Provide an access token via --access-token or STRAVA_ACCESS_TOKEN")

    client = StravaClient(access_token=token)
    download_streams_to_csv(
        client=client,
        output_streams=args.streams,
        output_athletes=args.athletes,
        after=_parse_date(args.after),
        before=_parse_date(args.before),
        max_pages=args.max_pages,
        per_page=args.per_page,
        force=args.force,
    )


if __name__ == "__main__":
    main()
