# -*- coding: utf-8 -*-
"""Entrenament d'una TCN amb embedding d'atleta per predir ritme (s/km) a partir de streams Strava.

Aquest script parteix de dos CSV d'entrada:
    - strava_streams.csv  -> columnes mínimes [athlete_id, activity_id, ts, hr, cadence, speed_mps,
                                             grade?, rpe?]
    - athletes.csv        -> columnes mínimes [athlete_id, fcmax, critical_speed_mps?]

El flux és:
    1. Càrrega i neteja bàsica de les dades (omple grade/rpe si cal).
    2. Crea features relatives per atleta (FC%max, %CS quan es disposa de critical_speed_mps).
    3. Resampleja a finestres fixes de 5 s i calcula agregats rodants.
    4. Construeix finestres temporals de 300 s i objectius de ritme mitjà (s/km).
    5. Entrena una xarxa TCN amb embedding d'atleta i avalua amb MAE en s/km.

Requisits: pip install pandas numpy torch scikit-learn
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ----------------------------- Configuració -----------------------------
@dataclass
class Config:
    """Conté els hiperparàmetres principals de preprocessat i entrenament."""

    sample_every_s: int = 5          # Resampleig dels streams a cada 5 s
    window_len_s: int = 300          # Longitud de les finestres (5 min)
    pred_horizon_s: int = 0          # Seq2one immediat (0 = objectiu dins de la finestra)
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    streams_csv: str = "strava_streams.csv"
    athletes_csv: str = "athletes.csv"
    model_out: str = "best_pace_predictor.pt"


CFG = Config()


# ------------------------------ Utilitats ------------------------------
def mps_to_pace_skm(speed_mps: np.ndarray, eps: float = 0.2) -> np.ndarray:
    """Converteix m/s a segons per quilòmetre controlant valors nuls."""

    speed = np.clip(speed_mps, eps, None)
    return 1000.0 / speed


def safe_zscore(values: pd.Series) -> pd.Series:
    """Càlcul de z-score tolerant a desviacions quasi nul·les."""

    mean = values.mean()
    std = values.std()
    return (values - mean) / (std if std and std > 1e-8 else 1.0)


def rolling_feature(series: pd.Series, window: int, mode: str) -> pd.Series:
    """Calcula agregats rodants amb finestra i mode especificats."""

    rolled = series.rolling(window, min_periods=max(1, window // 2))
    if mode == "mean":
        return rolled.mean()
    if mode == "std":
        return rolled.std()
    if mode == "p95":
        return rolled.quantile(0.95)
    if mode == "min":
        return rolled.min()
    if mode == "max":
        return rolled.max()
    raise ValueError(f"Mode de rolling no suportat: {mode}")


# ------------------- Càrrega i preparació de dades --------------------
def load_data(streams_csv: str, athletes_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega els CSV i assegura les columnes necessàries.

    Si falten grade o rpe es creen amb valors per defecte (0.0 i NaN respectivament).
    Ordena els streams per atleta, activitat i timestamp.
    """

    streams = pd.read_csv(streams_csv)
    athletes = pd.read_csv(athletes_csv)

    required = {"athlete_id", "activity_id", "ts", "hr", "cadence", "speed_mps"}
    missing = required.difference(streams.columns)
    if missing:
        raise ValueError(f"Falten columnes requerides a strava_streams.csv: {sorted(missing)}")

    if "grade" not in streams.columns:
        streams["grade"] = 0.0
    if "rpe" not in streams.columns:
        streams["rpe"] = np.nan

    streams["ts"] = pd.to_datetime(streams["ts"], utc=True)
    streams = streams.sort_values(["athlete_id", "activity_id", "ts"])

    return streams, athletes


def add_relative_features(streams: pd.DataFrame, athletes: pd.DataFrame) -> pd.DataFrame:
    """Afegeix les característiques relatives (hr_rel, speed_rel_cs, etc.)."""

    df = streams.copy()
    athletes_indexed = athletes.set_index("athlete_id")

    df["fcmax"] = df["athlete_id"].map(athletes_indexed["fcmax"])
    if "critical_speed_mps" in athletes_indexed.columns:
        df["cs_mps"] = df["athlete_id"].map(athletes_indexed["critical_speed_mps"])
    else:
        df["cs_mps"] = np.nan

    df["hr_rel"] = df["hr"] / df["fcmax"].replace(0, np.nan)
    df["speed_rel_cs"] = df["speed_mps"] / df["cs_mps"]
    df["pace_skm"] = mps_to_pace_skm(df["speed_mps"].to_numpy())

    base_speed = df["speed_rel_cs"].where(df["speed_rel_cs"].notna(), df["speed_mps"])
    df["decoupling"] = safe_zscore(df["hr_rel"].fillna(method="ffill")) - safe_zscore(
        base_speed.fillna(method="ffill")
    )

    df["cadence_z"] = safe_zscore(df["cadence"].fillna(method="ffill"))

    df["rpe"] = df.groupby(["athlete_id", "activity_id"])["rpe"].transform(
        lambda s: s.fillna(s.median())
    )
    df["rpe"] = df["rpe"].fillna(df["rpe"].median())

    df["grade_smooth"] = df.groupby(["athlete_id", "activity_id"])["grade"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    return df


def resample_and_engineer(streams: pd.DataFrame, sample_every_s: int) -> pd.DataFrame:
    """Resampleja cada activitat a una cadència fixa i crea agregats rodants."""

    windows: List[pd.DataFrame] = []
    feature_cols = [
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
        resampled = g[feature_cols].resample(f"{sample_every_s}s").mean()
        resampled = resampled.interpolate(limit_direction="both")
        resampled["athlete_id"] = athlete_id
        resampled["activity_id"] = activity_id

        steps_1m = max(1, int(60 / sample_every_s))
        steps_5m = max(1, int(300 / sample_every_s))
        for col in ["hr_rel", "cadence_z", "speed_mps", "pace_skm", "decoupling", "grade_smooth"]:
            resampled[f"{col}_mean_1m"] = rolling_feature(resampled[col], steps_1m, "mean")
            resampled[f"{col}_mean_5m"] = rolling_feature(resampled[col], steps_5m, "mean")
            resampled[f"{col}_p95_5m"] = rolling_feature(resampled[col], steps_5m, "p95")

        windows.append(resampled.reset_index().rename(columns={"index": "ts"}))

    return pd.concat(windows, ignore_index=True)


# ------------------------------ Windowing ------------------------------
def build_windows(
    df: pd.DataFrame,
    window_len_s: int,
    sample_every_s: int,
    pred_horizon_s: int,
    feature_cols: Sequence[str],
    target_col: str = "pace_skm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construeix tensors (N, T, F) i objectius seq2one."""

    window_steps = max(1, window_len_s // sample_every_s)
    horizon_steps = max(0, pred_horizon_s // sample_every_s)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    athlete_list: List[int] = []
    activity_list: List[int] = []

    for (athlete_id, activity_id), group in df.groupby(["athlete_id", "activity_id"], sort=False):
        g = group.sort_values("ts").reset_index(drop=True)
        if len(g) < window_steps + horizon_steps + 1:
            continue

        feature_values = g[list(feature_cols)].to_numpy(dtype=np.float32)
        target_values = g[target_col].to_numpy(dtype=np.float32)

        for start in range(0, len(g) - (window_steps + horizon_steps)):
            end = start + window_steps
            window = feature_values[start:end]
            target = target_values[end + horizon_steps - 1] if horizon_steps > 0 else target_values[start:end].mean()

            if not (120.0 <= target <= 900.0):
                continue

            X_list.append(window)
            y_list.append(target)
            athlete_list.append(int(athlete_id))
            activity_list.append(int(activity_id))

    if not X_list:
        raise RuntimeError("No s'han pogut crear finestres vàlides; revisa la longitud de les activitats.")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    athletes_arr = np.asarray(athlete_list, dtype=np.int64)
    activities_arr = np.asarray(activity_list, dtype=np.int64)
    return X, y, athletes_arr, activities_arr


class RunWindows(Dataset):
    """Dataset PyTorch senzill per empaquetar finestres i objectius."""

    def __init__(self, X: np.ndarray, athletes: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.athletes = torch.tensor(athletes, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.athletes[idx], self.y[idx]


# ------------------------------ Model TCN ------------------------------
class Chomp1d(nn.Module):
    """Retalla el padding causal per mantenir la longitud temporal original."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Bloc bàsic d'una TCN amb convolucions dilatades i connexió residual."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, padding: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCN(nn.Module):
    """Apila diversos blocs temporals amb dilacions creixents."""

    def __init__(self, in_channels: int, channel_sizes: Sequence[int], kernel_size: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i, out_channels in enumerate(channel_sizes):
            in_ch = in_channels if i == 0 else channel_sizes[i - 1]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, out_channels, kernel_size, dilation, padding, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PacePredictor(nn.Module):
    """TCN amb embedding d'atleta i capçalera MLP per predir ritme."""

    def __init__(
        self,
        num_athletes: int,
        input_features: int,
        embedding_dim: int = 16,
        tcn_channels: Sequence[int] = (64, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_athletes, embedding_dim=embedding_dim)
        self.tcn = TCN(input_features, tcn_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(tcn_channels[-1] + embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq: torch.Tensor, athlete_ids: torch.Tensor) -> torch.Tensor:
        features = x_seq.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]
        temporal = self.tcn(features)      # [B, C, T]
        pooled = temporal.mean(dim=2)      # Global average pooling
        athlete_vec = self.embedding(athlete_ids)
        combined = torch.cat([pooled, athlete_vec], dim=1)
        return self.head(combined)


# -------------------------- Entrenament i MAE -------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device: str, checkpoint_path: str) -> None:
    """Entrena el model amb MAE i guarda el millor checkpoint segons validació."""

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.to(device)
    best_val = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, athletes, y in train_loader:
            X, athletes, y = X.to(device), athletes.to(device), y.to(device)
            preds = model(X, athletes)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, athletes, y in val_loader:
                X, athletes, y = X.to(device), athletes.to(device), y.to(device)
                preds = model(X, athletes)
                val_loss += criterion(preds, y).item() * X.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch:02d} | train_MAE={train_loss:.2f} s/km | val_MAE={val_loss:.2f} s/km")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)


def mae_to_minsec(value: float) -> str:
    """Converteix segons a format minuts:segons."""

    minutes = int(value // 60)
    seconds = int(round(value - minutes * 60))
    return f"{minutes}:{seconds:02d}"


# --------------------------------- Main --------------------------------
def main() -> None:
    """Executa tot el pipeline de preprocessat, entrenament i avaluació."""

    streams, athletes = load_data(CFG.streams_csv, CFG.athletes_csv)
    enriched = add_relative_features(streams, athletes)
    engineered = resample_and_engineer(enriched, CFG.sample_every_s)

    feature_columns = [
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
    ]
    for column in feature_columns:
        if column not in engineered.columns:
            engineered[column] = 0.0

    X, y, athlete_ids, activity_ids = build_windows(
        engineered,
        window_len_s=CFG.window_len_s,
        sample_every_s=CFG.sample_every_s,
        pred_horizon_s=CFG.pred_horizon_s,
        feature_cols=feature_columns,
        target_col="pace_skm",
    )

    unique_activities = np.unique(activity_ids)
    train_acts, val_acts = train_test_split(unique_activities, test_size=0.2, random_state=42)
    train_mask = np.isin(activity_ids, train_acts)
    val_mask = np.isin(activity_ids, val_acts)

    train_dataset = RunWindows(X[train_mask], athlete_ids[train_mask], y[train_mask])
    val_dataset = RunWindows(X[val_mask], athlete_ids[val_mask], y[val_mask])

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size * 2, shuffle=False)

    num_athletes = int(athlete_ids.max()) + 1
    if len(np.unique(athlete_ids)) != num_athletes:
        num_athletes = len(np.unique(athlete_ids)) + 10

    model = PacePredictor(
        num_athletes=num_athletes,
        input_features=len(feature_columns),
        embedding_dim=16,
        tcn_channels=(64, 64, 64),
        kernel_size=5,
        dropout=0.2,
    )

    train_model(model, train_loader, val_loader, epochs=CFG.epochs, lr=CFG.lr, device=CFG.device, checkpoint_path=CFG.model_out)

    model.eval()
    criterion = nn.L1Loss()
    predictions: List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, athlete_batch, _ in val_loader:
            preds = model(X_batch.to(CFG.device), athlete_batch.to(CFG.device)).cpu().numpy().ravel()
            predictions.append(preds)

    y_pred = np.concatenate(predictions)
    val_targets = val_dataset.y.numpy().ravel()
    val_mae = float(np.mean(np.abs(y_pred - val_targets)))
    print(f"\nValidació MAE: {val_mae:.2f} s/km (~ {mae_to_minsec(val_mae)})")
    print(f"Model guardat a: {CFG.model_out}")

    print(
        """
Resum ràpid del pipeline:
- Càrrega & neteja: llegeix els CSV, omple grade/rpe si cal, converteix timestamps i ordena streams.
- Features relatives: calcula hr_rel (FC%max), speed_rel_cs (%CS) i decoupling (deriva FC vs velocitat).
- Resample & agregats: passa cada activitat a mostres de 5 s i afegeix mitjanes/percentils de 1-5 min.
- Windowing: crea finestres de 300 s amb el ritme mitjà (s/km) com a objectiu per activitat.
- Model TCN: aplica convolucions dilatades sobre la seqüència i concatena un embedding d'atleta.
- Entrenament: optimitza MAE (s/km), guarda el millor checkpoint i mostra el MAE en validació.
"""
    )


if __name__ == "__main__":
    main()
