"""
Open-Meteo weather client (free, no API key required).

Fetches a 14-day daily forecast plus hourly precipitation, and derives the
maximum 3-hour cumulative rainfall per day. Results are cached to avoid hitting
the API on every rerun. A clearly-labelled offline generator is provided so the
map still renders when the network is unavailable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

import config

_OM = config.OPEN_METEO


class WeatherAPIError(RuntimeError):
    """Raised when the Open-Meteo request fails or returns unusable data."""


def _num_series(values) -> pd.Series:
    """Coerce any list/array/Series into a numeric Series (NaN -> 0)."""
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").fillna(0.0)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _daily_frame(block: dict) -> pd.DataFrame:
    daily = block.get("daily", {})
    if not daily or "time" not in daily:
        raise WeatherAPIError("Respons Open-Meteo tidak memuat data harian.")

    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.drop(columns=["time"])

    # Max 3-hour cumulative rainfall derived from the hourly series.
    kum = _rolling_3h_max(block)
    df["hujan_kumulatif_3h_max"] = (
        df["date"].map(kum.to_dict()).astype(float) if not kum.empty else 0.0
    )

    numeric = [c for c in df.columns if c != "date"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def _rolling_3h_max(block: dict) -> pd.Series:
    """Return a Series mapping each date -> max 3-hour cumulative rainfall."""
    hourly = block.get("hourly", {})
    if not hourly or "precipitation" not in hourly or "time" not in hourly:
        return pd.Series(dtype=float)
    precip = _num_series(hourly["precipitation"])
    times = pd.to_datetime(pd.Series(list(hourly["time"])))
    series = pd.Series(precip.to_numpy(), index=times)
    rolling = series.rolling(window=3, min_periods=1).sum()
    grouped = rolling.groupby(rolling.index.date).max()
    return grouped


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def fetch_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Fetch a 14-day forecast for a single coordinate."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": _OM.daily_param,
        "hourly": _OM.hourly_param,
        "forecast_days": _OM.forecast_days,
        "timezone": _OM.timezone,
    }
    try:
        resp = requests.get(_OM.base_url, params=params, timeout=_OM.timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise WeatherAPIError(f"Koneksi ke Open-Meteo gagal: {exc}") from exc
    except ValueError as exc:
        raise WeatherAPIError("Respons Open-Meteo bukan JSON valid.") from exc
    return _daily_frame(payload)


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def fetch_forecast_batch(coords: tuple[tuple[float, float], ...]) -> list[pd.DataFrame]:
    """Fetch forecasts for many coordinates (chunked multi-location requests)."""
    results: list[pd.DataFrame] = []
    chunk = 40
    for start in range(0, len(coords), chunk):
        subset = coords[start:start + chunk]
        lats = ",".join(str(c[0]) for c in subset)
        lons = ",".join(str(c[1]) for c in subset)
        params = {
            "latitude": lats,
            "longitude": lons,
            "daily": _OM.daily_param,
            "hourly": _OM.hourly_param,
            "forecast_days": _OM.forecast_days,
            "timezone": _OM.timezone,
        }
        try:
            resp = requests.get(_OM.base_url, params=params, timeout=_OM.timeout)
            resp.raise_for_status()
            payload = resp.json()
            blocks = payload if isinstance(payload, list) else [payload]
            for block in blocks:
                results.append(_daily_frame(block))
        except (requests.RequestException, ValueError, WeatherAPIError):
            # Degrade gracefully: synthesise this chunk so the map still renders.
            for lat, lon in subset:
                results.append(offline_forecast(lat, lon))
    return results


def offline_forecast(lat: float, lon: float, days: Optional[int] = None) -> pd.DataFrame:
    """Deterministic synthetic forecast used only when the API is unreachable."""
    n = days or _OM.forecast_days
    rng = np.random.default_rng(int(abs(lat * 1000) + abs(lon * 1000)))
    dates = pd.date_range(pd.Timestamp.today().normalize(), periods=n, freq="D").date
    base = rng.gamma(shape=2.0, scale=12.0, size=n)
    precip = np.clip(base + rng.normal(0, 6, n), 0, None).round(1)
    hours = np.clip((precip > 0) * rng.integers(1, 12, n), 0, 24)
    return pd.DataFrame({
        "date": dates,
        "precipitation_sum": precip,
        "rain_sum": precip,
        "precipitation_hours": hours,
        "temperature_2m_max": (30 + rng.normal(0, 1.2, n)).round(1),
        "temperature_2m_min": (24 + rng.normal(0, 1.0, n)).round(1),
        "temperature_2m_mean": (27 + rng.normal(0, 1.0, n)).round(1),
        "precipitation_probability_max": np.clip(precip * 1.4, 0, 100).round(0),
        "windspeed_10m_max": (12 + rng.normal(0, 3, n)).round(1),
        "hujan_kumulatif_3h_max": (precip * rng.uniform(0.4, 0.7, n)).round(1),
    })
