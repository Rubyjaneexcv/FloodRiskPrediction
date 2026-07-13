"""
Geospatial helpers.

Builds a single ``location index`` — one row per kecamatan with its parent
kabupaten, coordinates and static geospatial features (elevation, slope, NDVI,
soil moisture, land cover). Coordinates and features are read straight from the
GeoJSON properties when available (e.g. ``Lintang``/``Bujur``/``Elevasi``/
``Kemiringan_Lereng``), falling back to polygon centroids otherwise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

import config
from utils.data_loader import (clean_name, load_geojson, load_prediction_csv,
                               resolve_column)

# Sensible physical defaults used only when a value is genuinely unavailable.
_DEFAULTS: dict[str, float] = {
    "elevation": 30.0,
    "slope": 0.0,
    "ndvi": 0.40,
    "soil_moisture": 0.50,
    "vegetation_moisture": 0.40,
    "landcover_class": 0.0,
    "lat": -6.2088,
    "lon": 106.8456,
}

# Minimal fallback so location pickers are never empty (mirrors the mockup).
_FALLBACK_LOCATIONS: list[dict[str, object]] = [
    {"kecamatan": "Bekasi Timur", "kabupaten": "Kota Bekasi", "lat": -6.2615, "lon": 106.9758, "elevation": 28},
    {"kecamatan": "Ciledug", "kabupaten": "Kota Tangerang", "lat": -6.2222, "lon": 106.7080, "elevation": 19},
    {"kecamatan": "Larangan", "kabupaten": "Kota Tangerang", "lat": -6.2350, "lon": 106.7350, "elevation": 21},
    {"kecamatan": "Pancoran Mas", "kabupaten": "Kota Depok", "lat": -6.4025, "lon": 106.8186, "elevation": 95},
    {"kecamatan": "Cimanggis", "kabupaten": "Kota Depok", "lat": -6.3700, "lon": 106.8700, "elevation": 110},
    {"kecamatan": "Cibinong", "kabupaten": "Kab. Bogor", "lat": -6.4817, "lon": 106.8540, "elevation": 140},
    {"kecamatan": "Gunung Putri", "kabupaten": "Kab. Bogor", "lat": -6.4300, "lon": 106.9300, "elevation": 160},
    {"kecamatan": "Cakung", "kabupaten": "Kota Jakarta Timur", "lat": -6.1800, "lon": 106.9500, "elevation": 12},
]


# --------------------------------------------------------------------------- #
# Location index
# --------------------------------------------------------------------------- #
def _centroids(gdf) -> pd.DataFrame:
    """Kecamatan-level centroid lat/lon computed in a metric CRS."""
    projected = gdf.to_crs(epsg=3857)
    centers = projected.geometry.centroid.to_crs(epsg=4326)
    return pd.DataFrame({"lon": centers.x.to_numpy(), "lat": centers.y.to_numpy()})


@st.cache_data(show_spinner=False)
def build_location_index() -> pd.DataFrame:
    """One row per kecamatan with parent kabupaten, coordinates and features."""
    gdf = load_geojson()

    if gdf is not None and len(gdf) > 0:
        cols = gdf.columns
        kec_col = resolve_column(cols, "kecamatan")
        kab_col = resolve_column(cols, "kabupaten")
        lat_col = resolve_column(cols, "latitude")
        lon_col = resolve_column(cols, "longitude")

        frame = pd.DataFrame()
        frame["kecamatan"] = (gdf[kec_col].map(clean_name)
                              if kec_col else [f"Kecamatan {i}" for i in range(len(gdf))])
        frame["kabupaten"] = gdf[kab_col].map(clean_name) if kab_col else "-"

        if lat_col and lon_col:
            frame["lat"] = pd.to_numeric(gdf[lat_col], errors="coerce")
            frame["lon"] = pd.to_numeric(gdf[lon_col], errors="coerce")
        else:
            centers = _centroids(gdf)
            frame["lat"] = centers["lat"].to_numpy()
            frame["lon"] = centers["lon"].to_numpy()

        for feat, target in (("elevation", "elevation"), ("slope", "slope"),
                             ("ndvi", "ndvi"), ("soil_moisture", "soil_moisture"),
                             ("landcover", "landcover_class")):
            col = resolve_column(cols, feat)
            if col is not None:
                frame[target] = pd.to_numeric(gdf[col], errors="coerce")
    else:
        frame = pd.DataFrame(_FALLBACK_LOCATIONS)

    frame = _augment_with_csv(frame)
    frame = _fill_defaults(frame)
    frame = frame.drop_duplicates(subset="kecamatan").reset_index(drop=True)
    return frame


def _augment_with_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill static features from CSV per-kecamatan means where the geojson lacks them."""
    df = load_prediction_csv()
    if df.empty:
        return frame

    kec_col = resolve_column(df.columns, "kecamatan")
    if kec_col is None:
        return frame

    static_map: dict[str, str] = {}
    for feat in ("elevation", "slope", "ndvi", "soil_moisture", "latitude", "longitude"):
        col = resolve_column(df.columns, feat)
        if col is not None:
            static_map[feat] = col
    if not static_map:
        return frame

    keys = df[kec_col].map(clean_name)
    agg = df.groupby(keys).agg({col: "mean" for col in static_map.values()})
    rename = {v: ("lat" if k == "latitude" else "lon" if k == "longitude" else k)
              for k, v in static_map.items()}
    agg = agg.rename(columns=rename)

    frame = frame.set_index("kecamatan")
    for col in agg.columns:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = frame[col].fillna(agg[col])
    return frame.reset_index()


def _fill_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    for col, default in _DEFAULTS.items():
        if col not in frame.columns:
            frame[col] = default
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(default)
    if "vegetation_moisture" not in frame.columns or frame["vegetation_moisture"].isna().all():
        frame["vegetation_moisture"] = (frame["ndvi"] * 0.9).clip(0, 1)
    return frame


# --------------------------------------------------------------------------- #
# Convenience accessors
# --------------------------------------------------------------------------- #
def get_kabupaten_list() -> list[str]:
    frame = build_location_index()
    return sorted(frame["kabupaten"].dropna().astype(str).unique().tolist())


def get_kecamatan_list(kabupaten: Optional[str] = None) -> list[str]:
    frame = build_location_index()
    if kabupaten and kabupaten != "Semua":
        frame = frame[frame["kabupaten"] == kabupaten]
    return sorted(frame["kecamatan"].dropna().astype(str).unique().tolist())


def get_static_features(kecamatan: str) -> dict[str, float]:
    """Static geospatial feature dict for a single kecamatan."""
    frame = build_location_index()
    row = frame[frame["kecamatan"] == kecamatan]
    if row.empty:
        base: dict[str, float] = dict(_DEFAULTS)
        base["kabupaten"] = "-"
    else:
        record = row.iloc[0].to_dict()
        base = {k: float(record.get(k, v)) for k, v in _DEFAULTS.items()}
        base["kabupaten"] = record.get("kabupaten", "-")
    base["long"] = base.get("lon", _DEFAULTS["lon"])
    base["vegetation_moisture_index"] = base.get("vegetation_moisture",
                                                 base.get("ndvi", 0.4) * 0.9)
    return base


def get_centroid(kecamatan: str) -> tuple[float, float]:
    feats = get_static_features(kecamatan)
    return float(feats.get("lat", _DEFAULTS["lat"])), float(feats.get("lon", _DEFAULTS["lon"]))
