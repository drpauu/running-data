"""Xarxa convolucional temporal amb incrustacions d'atletes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RunWindows(Dataset):
    def __init__(self, X, athlete_id, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.athlete_id = torch.as_tensor(athlete_id, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index], self.athlete_id[index], self.y[index]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, padding: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, in_channels: int, channels: Tuple[int, ...], kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = in_channels if i == 0 else channels[i - 1]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, padding, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PacePredictor(nn.Module):
    def __init__(
        self,
        n_athletes: int,
        in_features: int,
        emb_dim: int = 16,
        tcn_channels: Tuple[int, ...] = (64, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_athletes, emb_dim)
        self.tcn = TCN(in_features, tcn_channels, kernel_size=kernel_size, dropout=dropout)
        last_channels = tcn_channels[-1] if len(tcn_channels) else in_features
        self.head = nn.Sequential(
            nn.Linear(last_channels + emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, sequence, athlete_id):
        x = sequence.permute(0, 2, 1)
        features = self.tcn(x)
        pooled = torch.mean(features, dim=2)
        athlete = self.embedding(athlete_id)
        combined = torch.cat([pooled, athlete], dim=1)
        return self.head(combined)


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model: PacePredictor, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig) -> float:
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.to(cfg.device)
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for X, aid, y in train_loader:
            X = X.to(cfg.device)
            aid = aid.to(cfg.device)
            y = y.to(cfg.device)

            pred = model(X, aid)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, aid, y in val_loader:
                X = X.to(cfg.device)
                aid = aid.to(cfg.device)
                y = y.to(cfg.device)
                pred = model(X, aid)
                val_loss += criterion(pred, y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_pace_predictor.pt")
        print(f"Època {epoch:02d} | MAE_entrenament={train_loss:.2f} s/km | MAE_validació={val_loss:.2f} s/km")

    return best_val


def evaluate(model: PacePredictor, loader: DataLoader, device: str) -> float:
    criterion = nn.L1Loss()
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, aid, y in loader:
            X = X.to(device)
            aid = aid.to(device)
            y = y.to(device)
            pred = model(X, aid)
            batch_loss = criterion(pred, y).item()
            total_loss += batch_loss * X.size(0)
            total_samples += X.size(0)
    return float(total_loss / total_samples)
