"""
Prediction-history store.

Each forecast the user generates is appended to ``outputs/prediction_history.csv``
so the Riwayat Prediksi page can show a growing log. The page also blends in a
snapshot of today's live per-kecamatan predictions so it is never empty.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

import config

_COLUMNS = ["timestamp", "kabupaten", "kecamatan", "probability", "risk"]


def load_history() -> pd.DataFrame:
    """Load the persisted prediction history (empty frame if none)."""
    path = config.PREDICTION_HISTORY_FILE
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.dropna(subset=["timestamp"])
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=_COLUMNS)


def append_run(kecamatan: str, kabupaten: str, probability: float, risk: str) -> None:
    """Append one prediction record to the history file."""
    path = config.PREDICTION_HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.now().replace(microsecond=0),
        "kabupaten": kabupaten,
        "kecamatan": kecamatan,
        "probability": round(float(probability), 4),
        "risk": risk,
    }])
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)
