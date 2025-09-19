"""Utilitats d'enginyeria de variables per als fluxos de Strava."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SamplingConfig:
    sample_every_s: int = 5
    window_len_s: int = 300
    pred_horizon_s: int = 0


def mps_to_pace_skm(speed_mps: np.ndarray, eps: float = 0.2) -> np.ndarray:
    sp = np.clip(speed_mps, eps, None)
    return 1000.0 / sp


def safe_zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std()
    if pd.isna(sd) or sd < 1e-8:
        return x * 0.0
    return (x - mu) / sd


def rolling_feature(x: pd.Series, win: int, fn: str) -> pd.Series:
    r = x.rolling(win, min_periods=max(1, win // 2))
    if fn == "mean":
        return r.mean()
    if fn == "std":
        return r.std()
    if fn == "p95":
        return r.quantile(0.95)
    if fn == "min":
        return r.min()
    if fn == "max":
        return r.max()
    raise ValueError(f"Funció de finestres no compatible: {fn}")


def load_data(streams_csv: str, athletes_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    streams = pd.read_csv(streams_csv, parse_dates=["ts"])
    athletes = pd.read_csv(athletes_csv)

    required = ["athlete_id", "activity_id", "ts", "hr", "cadence", "speed_mps"]
    for col in required:
        if col not in streams.columns:
            raise ValueError(f"Manca la columna obligatòria: {col}")

    if "grade" not in streams.columns:
        streams["grade"] = 0.0
    if "rpe" not in streams.columns:
        streams["rpe"] = np.nan

    streams = streams.sort_values(["athlete_id", "activity_id", "ts"]).reset_index(drop=True)
    return streams, athletes


def add_relative_features(streams: pd.DataFrame, athletes: pd.DataFrame) -> pd.DataFrame:
    df = streams.copy()
    athletes = athletes.set_index("athlete_id")

    df["fcmax"] = df["athlete_id"].map(athletes["fcmax"])
    if "critical_speed_mps" in athletes.columns:
        df["cs_mps"] = df["athlete_id"].map(athletes["critical_speed_mps"])
    else:
        df["cs_mps"] = np.nan

    df["hr_rel"] = df["hr"] / df["fcmax"].replace(0, np.nan)
    df["pace_skm"] = mps_to_pace_skm(df["speed_mps"].to_numpy())
    df["speed_rel_cs"] = df["speed_mps"] / df["cs_mps"]

    base_speed = df["speed_rel_cs"].where(df["speed_rel_cs"].notna(), df["speed_mps"])
    df["decoupling"] = safe_zscore(df["hr_rel"].fillna(method="ffill")) - safe_zscore(
        base_speed.fillna(method="ffill")
    )
    df["cadence_z"] = safe_zscore(df["cadence"].fillna(method="ffill"))

    df["rpe"] = df.groupby(["athlete_id", "activity_id"])["rpe"].transform(lambda s: s.fillna(s.median()))
    df["rpe"] = df["rpe"].fillna(df["rpe"].median())

    df["grade_smooth"] = df.groupby(["athlete_id", "activity_id"])["grade"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    return df


def resample_and_engineer(streams: pd.DataFrame, sample_every_s: int = 5) -> pd.DataFrame:
    out = []
    cols = [
        "hr",
        "cadence",
        "speed_mps",
        "grade_smooth",
        "hr_rel",
        "speed_rel_cs",
        "cadence_z",
        "decoupling",
        "pace_skm",
        "rpe",
        "fcmax",
        "cs_mps",
    ]
    for (athlete_id, activity_id), group in streams.groupby(["athlete_id", "activity_id"], sort=False):
        g = group.set_index("ts").sort_index()
        resampled = g[cols].resample(f"{sample_every_s}s").mean().interpolate(limit_direction="both")
        resampled["athlete_id"] = athlete_id
        resampled["activity_id"] = activity_id

        steps_1m = max(1, int(60 / sample_every_s))
        steps_5m = max(1, int(300 / sample_every_s))
        for col in ["hr_rel", "cadence_z", "speed_mps", "pace_skm", "decoupling", "grade_smooth"]:
            resampled[f"{col}_mean_1m"] = rolling_feature(resampled[col], steps_1m, "mean")
            resampled[f"{col}_mean_5m"] = rolling_feature(resampled[col], steps_5m, "mean")
            resampled[f"{col}_p95_5m"] = rolling_feature(resampled[col], steps_5m, "p95")

        out.append(resampled.reset_index().rename(columns={"index": "ts"}))

    return pd.concat(out, ignore_index=True)


def build_windows(
    df: pd.DataFrame,
    *,
    window_len_s: int,
    sample_every_s: int,
    pred_horizon_s: int,
    feature_cols: Sequence[str],
    target_col: str = "pace_skm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = max(1, window_len_s // sample_every_s)
    H = max(0, pred_horizon_s // sample_every_s)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    athlete_list: List[int] = []
    act_list: List[int] = []

    for (aid, act), group in df.groupby(["athlete_id", "activity_id"], sort=False):
        g = group.sort_values("ts").reset_index(drop=True)
        if len(g) < T + H + 1:
            continue
        feats = g[list(feature_cols)].to_numpy(dtype=np.float32)
        target = g[target_col].to_numpy(dtype=np.float32)

        for start in range(0, len(g) - (T + H)):
            window = feats[start : start + T]
            if H > 0:
                y = target[start + T + H - 1]
            else:
                y = float(target[start : start + T].mean())
            if not (120 <= y <= 900):
                continue

            X_list.append(window)
            y_list.append(y)
            athlete_list.append(int(aid))
            act_list.append(int(act))

    if not X_list:
        raise RuntimeError("No s'han pogut construir finestres. Comprova la configuració de mostreig.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    athletes = np.array(athlete_list, dtype=np.int64)
    activities = np.array(act_list, dtype=np.int64)
    return X, y, athletes, activities
