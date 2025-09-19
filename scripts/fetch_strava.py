#!/usr/bin/env python
"""Eina de línia d'ordres per descarregar fluxos de Strava i guardar-los en CSV."""
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
    parser.add_argument("--access-token", dest="access_token", default=None, help="Testimoni OAuth de Strava")
    parser.add_argument("--after", type=str, default=None, help="Només descarrega activitats posteriors a aquesta data ISO (UTC)")
    parser.add_argument("--before", type=str, default=None, help="Només descarrega activitats anteriors a aquesta data ISO (UTC)")
    parser.add_argument("--streams", type=Path, default=Path("strava_streams.csv"), help="Fitxer CSV de sortida per als fluxos")
    parser.add_argument("--athletes", type=Path, default=Path("athletes.csv"), help="Fitxer CSV de sortida per a les metadades de l'atleta")
    parser.add_argument("--force", action="store_true", help="Sobreescriu els CSV existents")
    parser.add_argument("--max-pages", type=int, default=10, help="Nombre màxim de pàgines d'activitats a descarregar")
    parser.add_argument("--per-page", type=int, default=50, help="Nombre d'activitats per pàgina (màxim 200)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    token = args.access_token or os.environ.get("STRAVA_ACCESS_TOKEN")
    if not token:
        raise SystemExit("Proporciona un testimoni mitjançant --access-token o STRAVA_ACCESS_TOKEN")

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
