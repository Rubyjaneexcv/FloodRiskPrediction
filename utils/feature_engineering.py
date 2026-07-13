"""
Feature engineering.

Transforms a raw Open-Meteo forecast (one row per day) plus a kecamatan's static
geospatial attributes into the exact feature matrix the Random Forest expects.

The engineered superset is reindexed onto ``feature_list`` (the model's own
schema). Names are matched tolerantly through :data:`FEATURE_SYNONYMS`, so
spelling variants such as ``lon``/``long`` or ``maks_hari_hujan_berturut`` /
``maks_hari_hujan_beruntun`` all resolve correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

_WINDOW = 3  # trailing-day window for rolling rainfall statistics

# Accepted spelling variants: canonical name -> alternative source names.
FEATURE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "long": ("lon", "longitude"),
    "lon": ("long", "longitude"),
    "lat": ("latitude",),
    "vegetation_moisture_index": ("vegetation_moisture", "veg_moisture_index"),
    "vegetation_moisture": ("vegetation_moisture_index",),
    "maks_hari_hujan_beruntun": ("maks_hari_hujan_berturut", "max_consecutive_rain"),
    "maks_hari_hujan_berturut": ("maks_hari_hujan_beruntun",),
    "slope": ("kemiringan_lereng", "kemiringan"),
}


def _series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    """Always return a numeric Series aligned to *df* (even if column missing)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _consecutive_rain(precip: pd.Series, wet_mm: float = 1.0) -> np.ndarray:
    """Length of the current run of consecutive wet days ending at each day."""
    run, out = 0, []
    for value in precip:
        run = run + 1 if value >= wet_mm else 0
        out.append(run)
    return np.asarray(out, dtype=float)


def engineer(forecast: pd.DataFrame, static: dict) -> pd.DataFrame:
    """Build the full engineered-feature superset (one row per forecast day)."""
    df = forecast.copy().reset_index(drop=True)
    precip = _series(df, "precipitation_sum", 0.0)
    hours = _series(df, "precipitation_hours", 0.0)
    tmean = _series(df, "temperature_2m_mean", 27.0)
    kum3h = _series(df, "hujan_kumulatif_3h_max", 0.0)
    dates = pd.to_datetime(df["date"])

    roll = precip.rolling(window=_WINDOW, min_periods=1)
    heavy = (precip > config.HEAVY_RAIN_THRESHOLD_MM).astype(int)
    elevation = float(static.get("elevation", 30.0) or 30.0)
    consecutive = _consecutive_rain(precip)

    feats = pd.DataFrame(index=df.index)
    feats["max_rainfall"] = roll.max()
    feats["avg_rainfall"] = roll.mean()
    feats["hujan_harian_max"] = precip
    feats["hujan_kumulatif_3h_max"] = kum3h
    feats["total_hujan_bulan"] = precip.expanding().sum()
    feats["rainfall_intensity"] = (precip / hours.replace(0, np.nan)).fillna(0.0)
    feats["jml_hari_hujan_lebat"] = heavy.expanding().sum()
    feats["maks_hari_hujan_beruntun"] = consecutive
    feats["maks_hari_hujan_berturut"] = consecutive
    feats["avg_temperature"] = tmean
    feats["rainfall_elevation_ratio"] = feats["max_rainfall"] / (elevation + 1.0)
    feats["year"] = dates.dt.year
    feats["month"] = dates.dt.month

    # Static geospatial features (constant across the horizon).
    def stat(*keys: str, default: float = 0.0) -> float:
        for key in keys:
            if key in static and static[key] is not None:
                try:
                    return float(static[key])
                except (TypeError, ValueError):
                    continue
        return default

    lat_v = stat("lat", "latitude", default=-6.2)
    lon_v = stat("lon", "long", "longitude", default=106.8)
    ndvi_v = stat("ndvi", default=0.4)
    feats["elevation"] = elevation
    feats["ndvi"] = ndvi_v
    feats["soil_moisture"] = stat("soil_moisture", default=0.5)
    feats["slope"] = stat("slope", "kemiringan_lereng", default=0.0)
    feats["landcover_class"] = stat("landcover_class", "landcover", default=0.0)
    feats["lat"] = lat_v
    feats["lon"] = lon_v
    feats["long"] = lon_v
    feats["vegetation_moisture_index"] = stat(
        "vegetation_moisture_index", "vegetation_moisture", default=ndvi_v * 0.9)
    feats["vegetation_moisture"] = feats["vegetation_moisture_index"]

    return feats


def build_feature_matrix(
    forecast: pd.DataFrame, static: dict, feature_list: list[str]
) -> pd.DataFrame:
    """Return a matrix whose columns are exactly ``feature_list`` (in order)."""
    engineered = engineer(forecast, static)

    matrix = pd.DataFrame(index=engineered.index)
    for name in feature_list:
        source = _resolve(name, engineered, static)
        matrix[name] = source

    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _resolve(name: str, engineered: pd.DataFrame, static: dict):
    """Find a value for *name* in engineered columns / static / synonyms / 0."""
    if name in engineered.columns:
        return engineered[name]
    if name in static and static[name] is not None:
        return float(_safe(static[name]))
    for alt in FEATURE_SYNONYMS.get(name, ()):  # try known spelling variants
        if alt in engineered.columns:
            return engineered[alt]
        if alt in static and static[alt] is not None:
            return float(_safe(static[alt]))
    return 0.0


def _safe(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
