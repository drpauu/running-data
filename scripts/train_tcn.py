#!/usr/bin/env python
"""Entrena un model TCN amb incrustacions d'atletes sobre dades de Strava."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from running_data.features import (
    add_relative_features,
    build_windows,
    load_data,
    resample_and_engineer,
)
from running_data.model import PacePredictor, RunWindows, TrainConfig, train_model


DEFAULT_FEATURES: Sequence[str] = (
    "hr_rel",
    "cadence_z",
    "speed_mps",
    "pace_skm",
    "decoupling",
    "grade_smooth",
    "hr_rel_mean_1m",
    "hr_rel_mean_5m",
    "pace_skm_mean_1m",
    "pace_skm_mean_5m",
    "decoupling_mean_5m",
    "grade_smooth_mean_1m",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=Path, default=Path("strava_streams.csv"), help="CSV d'entrada amb fluxos de Strava")
    parser.add_argument("--athletes", type=Path, default=Path("athletes.csv"), help="CSV d'entrada amb metadades d'atletes")
    parser.add_argument("--window", type=int, default=300, help="Longitud de la finestra en segons")
    parser.add_argument("--sample", type=int, default=5, help="Interval de mostreig en segons")
    parser.add_argument("--pred-horizon", type=int, default=0, help="Horitzó de predicció en segons")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'èpoques d'entrenament")
    parser.add_argument("--batch-size", type=int, default=64, help="Mida del lot d'entrenament")
    parser.add_argument("--val-size", type=float, default=0.2, help="Proporció de validació")
    parser.add_argument("--seed", type=int, default=42, help="Llavor aleatòria per reproduïbilitat")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositiu de càlcul")
    parser.add_argument("--model-out", type=Path, default=Path("best_pace_predictor.pt"), help="Ruta per guardar els pesos del millor model")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    streams, athletes = load_data(str(args.streams), str(args.athletes))
    streams = add_relative_features(streams, athletes)
    features = resample_and_engineer(streams, sample_every_s=args.sample)

    for col in DEFAULT_FEATURES:
        if col not in features.columns:
            features[col] = 0.0

    X, y, athlete_ids, activity_ids = build_windows(
        features,
        window_len_s=args.window,
        sample_every_s=args.sample,
        pred_horizon_s=args.pred_horizon,
        feature_cols=DEFAULT_FEATURES,
        target_col="pace_skm",
    )

    unique_activities = np.unique(activity_ids)
    train_acts, val_acts = train_test_split(unique_activities, test_size=args.val_size, random_state=args.seed)
    train_mask = np.isin(activity_ids, train_acts)
    val_mask = np.isin(activity_ids, val_acts)

    unique_athletes = np.unique(athlete_ids)
    athlete_to_index = {aid: idx for idx, aid in enumerate(unique_athletes)}
    mapped_athletes = np.vectorize(athlete_to_index.get)(athlete_ids)

    train_dataset = RunWindows(X[train_mask], mapped_athletes[train_mask], y[train_mask])
    val_dataset = RunWindows(X[val_mask], mapped_athletes[val_mask], y[val_mask])

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("No hi ha prou finestres per crear la partició d'entrenament i validació sol·licitada")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)

    model = PacePredictor(
        n_athletes=len(unique_athletes),
        in_features=len(DEFAULT_FEATURES),
    )
    cfg = TrainConfig(epochs=args.epochs, lr=1e-3, device=args.device)
    best_val = train_model(model, train_loader, val_loader, cfg)
    default_path = Path("best_pace_predictor.pt")
    if args.model_out != default_path:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.model_out.write_bytes(default_path.read_bytes())
        default_path.unlink()
    print(f"MAE de validació: {best_val:.2f} s/km")


if __name__ == "__main__":
    main()
