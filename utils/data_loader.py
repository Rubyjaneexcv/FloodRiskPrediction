"""
Data loading & schema-normalisation utilities.

Reads the historical prediction CSV and the Jabodetabek GeoJSON, tolerating
minor column-name differences via :data:`config.COLUMN_ALIASES`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

import config
from utils.styling import get_risk_level


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def resolve_column(columns: Iterable[str], key: str) -> Optional[str]:
    """Return the real column matching a canonical *key* (via aliases)."""
    aliases = config.COLUMN_ALIASES.get(key, (key,))
    lookup = {_norm(c): c for c in columns}
    for alias in aliases:
        if _norm(alias) in lookup:
            return lookup[_norm(alias)]
    return None


def clean_name(value) -> str:
    """Human-friendly place name (title-case ALL-CAPS / all-lower sources)."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "-"
    if text.isupper() or text.islower():
        return text.title()
    return text


# --------------------------------------------------------------------------- #
# Prediction CSV
# --------------------------------------------------------------------------- #
def data_available() -> bool:
    return config.PREDICTION_CSV.exists()


@st.cache_data(show_spinner=False)
def load_prediction_csv() -> pd.DataFrame:
    """Load ``Flood_Risk_Prediction.csv`` with canonical columns.

    Guarantees the columns ``kecamatan``, ``kabupaten``, ``probability`` and
    ``category`` exist (category is derived from probability when absent).
    Returns an empty frame if the file is missing.
    """
    if not config.PREDICTION_CSV.exists():
        return pd.DataFrame(
            columns=["kecamatan", "kabupaten", "probability", "category"]
        )

    df = pd.read_csv(config.PREDICTION_CSV)
    df.columns = [str(c).strip() for c in df.columns]

    kec = resolve_column(df.columns, "kecamatan")
    kab = resolve_column(df.columns, "kabupaten")
    prob = resolve_column(df.columns, "probability")
    cat = resolve_column(df.columns, "category")

    out = df.copy()
    out["kecamatan"] = df[kec] if kec else "-"
    out["kabupaten"] = df[kab] if kab else "-"

    if prob:
        out["probability"] = pd.to_numeric(df[prob], errors="coerce").fillna(0.0)
        # Normalise to 0..1 if the column is expressed as a percentage.
        if out["probability"].max() > 1.5:
            out["probability"] = out["probability"] / 100.0
    else:
        out["probability"] = 0.0

    if cat:
        out["category"] = df[cat].astype(str)
    else:
        out["category"] = out["probability"].apply(lambda p: get_risk_level(p).label)

    return out


def get_top_risk(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-*n* kecamatan ordered by flood probability (descending)."""
    if df.empty or "probability" not in df.columns:
        return df
    return df.sort_values("probability", ascending=False).head(n).reset_index(drop=True)


def probability_series(df: pd.DataFrame) -> np.ndarray:
    if df.empty or "probability" not in df.columns:
        return np.array([])
    return df["probability"].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# GeoJSON (GeoPandas imported lazily so the app boots without it installed)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_geojson() -> Optional["object"]:
    """Load the Jabodetabek GeoJSON as a GeoDataFrame (WGS84)."""
    if not config.GEOJSON_FILE.exists():
        return None
    try:
        import geopandas as gpd

        gdf = gpd.read_file(config.GEOJSON_FILE)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        gdf.columns = [str(c).strip() for c in gdf.columns]
        return gdf
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memuat GeoJSON: {exc}")
        return None


def geojson_available() -> bool:
    return config.GEOJSON_FILE.exists()


# --------------------------------------------------------------------------- #
# Training dataset (for the Dataset page)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_training_dataset() -> pd.DataFrame:
    """
    Mencari dataset training di folder data/.

    Prioritas:
    - Nama file mengandung dataset/train/final/historis/history
    - Memiliki kolom label atau banjir
    """

    csvs = list(config.DATA_DIR.glob("*.csv"))

    def _priority(path):
        name = path.name.lower()
        for i, kw in enumerate(("dataset", "train", "final", "historis", "history")):
            if kw in name:
                return i
        return 99

    for path in sorted(csvs, key=lambda x: (_priority(x), x.name)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        df.columns = [str(c).strip() for c in df.columns]

        # Cari kolom target
        label_col = (
            resolve_column(df.columns, "label")
            or resolve_column(df.columns, "banjir")
            or ("banjir" if "banjir" in df.columns else None)
        )

        if label_col is not None:
            return df

    return pd.DataFrame()