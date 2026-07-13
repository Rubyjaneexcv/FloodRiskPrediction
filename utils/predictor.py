"""
Prediction orchestration.

Ties the weather client, feature engineering and the trained Random Forest into
two entry points:

* :func:`predict_kecamatan_14d` — single kecamatan, 14-day forecast (Prediction page).
* :func:`predict_all_kecamatan_14d` — every kecamatan (Flood Risk Map page).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

import config
from utils import weather_api
from utils.feature_engineering import build_feature_matrix
from utils.geo_utils import build_location_index, get_static_features
from utils.model_loader import load_feature_list, load_model, resolve_feature_names
from utils.styling import get_risk_level


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be produced."""


# --------------------------------------------------------------------------- #
# Probability extraction
# --------------------------------------------------------------------------- #
def positive_proba(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    """Probability of the flood (positive) class for each row."""
    proba = model.predict_proba(matrix)
    proba = np.asarray(proba)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, -1]


def _prepare_matrix(forecast: pd.DataFrame, static: dict, model: Any) -> pd.DataFrame:
    feature_list = resolve_feature_names(model, load_feature_list())
    return build_feature_matrix(forecast, static, feature_list)


def _decorate(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["category"] = results["probability"].apply(lambda p: get_risk_level(p).label)
    results["color"] = results["probability"].apply(lambda p: get_risk_level(p).color)
    return results


# --------------------------------------------------------------------------- #
# Single kecamatan
# --------------------------------------------------------------------------- #
def predict_kecamatan_14d(kecamatan: str) -> pd.DataFrame:
    """Return a 14-row frame: day, date, rainfall, temperature, probability, category."""
    model = load_model()
    if model is None:
        raise PredictionError(
            "Model belum tersedia. Letakkan `best_random_forest.pkl` di folder models/."
        )

    static = get_static_features(kecamatan)
    lat, lon = float(static.get("lat")), float(static.get("lon"))
    forecast = weather_api.fetch_forecast(lat, lon)

    matrix = _prepare_matrix(forecast, static, model)
    probability = positive_proba(model, matrix)

    results = pd.DataFrame({
        "day": np.arange(1, len(forecast) + 1),
        "date": pd.to_datetime(forecast["date"]),
        "rainfall": pd.to_numeric(forecast.get("precipitation_sum", 0.0), errors="coerce"),
        "temperature": pd.to_numeric(forecast.get("temperature_2m_mean", 0.0), errors="coerce"),
        "probability": np.round(probability, 4),
    })
    return _decorate(results)


# --------------------------------------------------------------------------- #
# All kecamatan (map)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def predict_all_kecamatan_14d() -> pd.DataFrame:
    """Long-format 14-day predictions for every kecamatan (cached).

    Columns: kecamatan, kabupaten, day, date, probability, category.
    """
    model = load_model()
    index = build_location_index()
    if model is None or index.empty:
        return pd.DataFrame(
            columns=["kecamatan", "kabupaten", "day", "date", "probability", "category"]
        )

    coords = tuple((float(r.lat), float(r.lon)) for r in index.itertuples())
    forecasts = weather_api.fetch_forecast_batch(coords)

    feature_list = resolve_feature_names(model, load_feature_list())
    rows: list[pd.DataFrame] = []

    for pos, row in enumerate(index.itertuples()):
        if pos >= len(forecasts):
            break
        forecast = forecasts[pos]
        static = {
            "elevation": row.elevation, "ndvi": row.ndvi,
            "soil_moisture": row.soil_moisture,
            "vegetation_moisture": getattr(row, "vegetation_moisture", row.ndvi),
            "landcover_class": getattr(row, "landcover_class", 2.0),
            "lat": row.lat, "lon": row.lon,
        }
        matrix = build_feature_matrix(forecast, static, feature_list)
        probability = positive_proba(model, matrix)

        rows.append(pd.DataFrame({
            "kecamatan": row.kecamatan,
            "kabupaten": row.kabupaten,
            "day": np.arange(1, len(forecast) + 1),
            "date": pd.to_datetime(forecast["date"]),
            "probability": np.round(probability, 4),
        }))

    if not rows:
        return pd.DataFrame(
            columns=["kecamatan", "kabupaten", "day", "date", "probability", "category"]
        )

    result = pd.concat(rows, ignore_index=True)
    result["category"] = result["probability"].apply(lambda p: get_risk_level(p).label)
    return result


def predictions_for_day(all_preds: pd.DataFrame, day: int) -> pd.DataFrame:
    """Slice the long prediction frame down to a single forecast day."""
    if all_preds.empty:
        return all_preds
    subset = all_preds[all_preds["day"] == day].copy()
    return subset.reset_index(drop=True)
