"""Funcions d'ajuda per convertir fluxos de Strava en fitxers CSV ordenats."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .strava_client import StravaClient, infer_fcmax_from_stream

REQUIRED_STREAM_COLUMNS = [
    "athlete_id",
    "activity_id",
    "ts",
    "hr",
    "cadence",
    "speed_mps",
    "grade",
    "rpe",
]


def _parse_start_date(start_date: str) -> dt.datetime:
    return dt.datetime.fromisoformat(start_date.replace("Z", "+00:00"))


def _stream_to_frame(
    *,
    athlete_id: int,
    activity_id: int,
    activity_start: dt.datetime,
    streams: Dict[str, Dict],
    perceived_exertion: Optional[int],
) -> pd.DataFrame:
    time_stream = streams.get("time", {}).get("data", [])
    if not time_stream:
        raise ValueError("El flux de l'activitat no conté dades de temps")

    def _get(key: str) -> List[float]:
        return streams.get(key, {}).get("data", [])

    hr = _get("heartrate")
    cadence = _get("cadence")
    velocity = _get("velocity_smooth") or _get("velocity")
    grade = _get("grade_smooth") or _get("grade_adjusted")

    length = len(time_stream)
    per_tipus = {
        "heartrate": hr,
        "cadence": cadence,
        "velocity": velocity,
        "grade": grade,
    }
    for key, arr in per_tipus.items():
        if arr and len(arr) != length:
            raise ValueError(f"El flux '{key}' té una longitud no vàlida")

    timestamps = [activity_start + dt.timedelta(seconds=int(t)) for t in time_stream]

    def _ensure_array(values):
        if values:
            return values
        return [np.nan] * length

    frame = pd.DataFrame(
        {
            "athlete_id": athlete_id,
            "activity_id": activity_id,
            "ts": timestamps,
            "hr": _ensure_array(hr),
            "cadence": _ensure_array(cadence),
            "speed_mps": _ensure_array(velocity),
            "grade": _ensure_array(grade),
        }
    )

    if perceived_exertion is None:
        frame["rpe"] = np.nan
    else:
        frame["rpe"] = perceived_exertion

    return frame


def download_streams_to_csv(
    *,
    client: StravaClient,
    output_streams: Path,
    output_athletes: Path,
    athlete_id: Optional[int] = None,
    after: Optional[dt.datetime] = None,
    before: Optional[dt.datetime] = None,
    max_pages: int = 10,
    per_page: int = 50,
    force: bool = False,
) -> None:
    """Descarrega activitats i genera fitxers CSV ordenats llestos per entrenar."""

    output_streams = Path(output_streams)
    output_athletes = Path(output_athletes)
    output_streams.parent.mkdir(parents=True, exist_ok=True)
    output_athletes.parent.mkdir(parents=True, exist_ok=True)

    if output_streams.exists() and not force:
        raise FileExistsError(f"{output_streams} ja existeix. Utilitza force=True per sobreescriure")

    frames: List[pd.DataFrame] = []
    athlete_meta: Dict[str, List] = {
        "athlete_id": [],
        "fcmax": [],
        "critical_speed_mps": [],
    }

    profile = client.get_athlete()
    athlete_identifier = athlete_id or profile.get("id")
    if athlete_identifier is None:
        raise RuntimeError("No ha estat possible determinar l'identificador de l'atleta")

    profile_fcmax = profile.get("max_heartrate")
    athlete_meta["athlete_id"].append(athlete_identifier)
    athlete_meta["fcmax"].append(profile_fcmax if profile_fcmax else np.nan)
    athlete_meta["critical_speed_mps"].append(np.nan)

    activities = client.iter_activities(
        after=after,
        before=before,
        per_page=per_page,
        max_pages=max_pages,
    )

    for activity in activities:
        act_id = activity["id"]
        start = _parse_start_date(activity["start_date"])
        dades_registre = json.dumps({"name": activity.get("name"), "id": act_id})
        streams = client.get_activity_streams(act_id)

        frame = _stream_to_frame(
            athlete_id=athlete_identifier,
            activity_id=act_id,
            activity_start=start,
            streams=streams,
            perceived_exertion=activity.get("perceived_exertion"),
        )

        if frame["speed_mps"].isna().all():
            distance = streams.get("distance", {}).get("data", [])
            time_stream = streams.get("time", {}).get("data", [])
            if distance and time_stream:
                times = np.asarray(time_stream, dtype=float)
                dist = np.asarray(distance, dtype=float)
                delta_dist = np.diff(dist, prepend=dist[0])
                delta_time = np.diff(times, prepend=times[0])
                if delta_time.size:
                    delta_time[0] = 1.0 if len(delta_time) == 1 else max(delta_time[1], 1.0)
                delta_time[delta_time <= 0] = 1.0
                speed = np.divide(delta_dist, delta_time, out=np.zeros_like(delta_dist), where=delta_time > 0)
                frame["speed_mps"] = speed
        frames.append(frame)

        fcmax_stream = infer_fcmax_from_stream(streams)
        current_fcmax = athlete_meta["fcmax"][0]
        if (current_fcmax is None or (isinstance(current_fcmax, float) and np.isnan(current_fcmax))) and fcmax_stream:
            athlete_meta["fcmax"][0] = fcmax_stream

        print(f"Activitat descarregada {dades_registre}")

    if not frames:
        raise RuntimeError("No s'han recuperat activitats. Comprova els permisos de l'API i els filtres.")

    streams_df = pd.concat(frames, ignore_index=True)
    streams_df = streams_df.sort_values(["athlete_id", "activity_id", "ts"])
    streams_df.to_csv(output_streams, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

    pd.DataFrame(athlete_meta).to_csv(output_athletes, index=False)
    print(f"Fluxos desats a {output_streams}")
    print(f"Metadades de l'atleta desades a {output_athletes}")
