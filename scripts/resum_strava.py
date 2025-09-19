#!/usr/bin/env python
"""Genera un resum de totes les activitats de Strava i del rendiment acumulat."""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from running_data.strava_client import StravaClient


def _parse_date(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    return dt.datetime.fromisoformat(value)


def _pace_from_speed(speed_mps: float | None) -> float | None:
    if speed_mps is None:
        return None
    if speed_mps <= 0:
        return None
    return 1000.0 / float(speed_mps)


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(round(seconds))
    minuts, segons = divmod(total, 60)
    hores, minuts = divmod(minuts, 60)
    if hores:
        return f"{hores:d}h {minuts:02d}m {segons:02d}s"
    return f"{minuts:d}m {segons:02d}s"


def _collect_activities(activities: Iterable[dict]) -> List[dict]:
    registres: List[dict] = []
    for act in activities:
        dist = act.get("distance")
        mov = act.get("moving_time")
        ritme = _pace_from_speed(act.get("average_speed"))
        registre = {
            "id": act.get("id"),
            "nom": act.get("name"),
            "data_inici": act.get("start_date"),
            "esport": act.get("sport_type"),
            "distancia_km": (float(dist) / 1000.0) if dist is not None else None,
            "temps_moviment_s": mov,
            "ritme_mitja_skm": ritme,
            "fc_mitjana": act.get("average_heartrate"),
            "fc_max": act.get("max_heartrate"),
            "desnivell_positiu_m": act.get("total_elevation_gain"),
            "suffer_score": act.get("suffer_score"),
        }
        registres.append(registre)
    return registres


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-token", dest="access_token", default=None, help="Testimoni OAuth de Strava")
    parser.add_argument("--after", type=str, default=None, help="Només inclou activitats posteriors a aquesta data ISO (UTC)")
    parser.add_argument("--before", type=str, default=None, help="Només inclou activitats anteriors a aquesta data ISO (UTC)")
    parser.add_argument("--output", type=Path, default=Path("resum_activitats.csv"), help="Fitxer CSV de sortida amb el resum")
    parser.add_argument("--per-page", type=int, default=200, help="Nombre d'activitats per pàgina (màxim 200)")
    parser.add_argument("--max-pages", type=int, default=200, help="Nombre màxim de pàgines a descarregar")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    token = args.access_token or os.environ.get("STRAVA_ACCESS_TOKEN")
    if not token:
        raise SystemExit("Proporciona un testimoni mitjançant --access-token o STRAVA_ACCESS_TOKEN")

    client = StravaClient(access_token=token)
    activitats = client.iter_activities(
        after=_parse_date(args.after),
        before=_parse_date(args.before),
        per_page=args.per_page,
        max_pages=args.max_pages,
    )
    registres = _collect_activities(activitats)
    if not registres:
        raise SystemExit("No s'han trobat activitats amb els filtres indicats")

    taula = pd.DataFrame(registres)
    taula.to_csv(args.output, index=False)

    distancia_total = taula["distancia_km"].fillna(0).sum()
    temps_total = taula["temps_moviment_s"].fillna(0).sum()
    ritmes_valids = taula["ritme_mitja_skm"].dropna()
    fc_mitjanes = taula["fc_mitjana"].dropna()

    print(f"Activitats exportades: {len(taula)}")
    print(f"Distància total: {distancia_total:.2f} km")
    print(f"Temps en moviment: {_format_seconds(temps_total)}")
    if not ritmes_valids.empty:
        ritme_mig = ritmes_valids.mean()
        print(f"Ritme mitjà: {_format_seconds(ritme_mig)} per km")
    if not fc_mitjanes.empty:
        print(f"Freqüència cardíaca mitjana global: {fc_mitjanes.mean():.0f} bpm")
    print(f"Resum guardat a {args.output}")


if __name__ == "__main__":
    main()
