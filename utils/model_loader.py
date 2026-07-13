"""
Model & metrics loading utilities.

The trained Random Forest and its feature list are loaded once and cached for
the lifetime of the Streamlit process (``st.cache_resource``). Everything here
fails soft: if an artefact is missing the caller receives ``None`` instead of a
crash, so the UI can render a friendly instruction card.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

import config


# --------------------------------------------------------------------------- #
# Low level loaders
# --------------------------------------------------------------------------- #
def _safe_load_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:  # noqa: BLE001 - fall back to std pickle
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal memuat `{path.name}`: {exc}")
            return None


@st.cache_resource(show_spinner=False)
def load_model() -> Optional[Any]:
    """Load and cache the trained Random Forest classifier."""
    return _safe_load_pickle(config.MODEL_FILE)


@st.cache_resource(show_spinner=False)
def load_feature_list() -> list[str]:
    """Load the ordered feature list used at training time.

    Falls back to :data:`config.FALLBACK_FEATURE_LIST` when the pickle is
    unavailable so the pipeline still has a schema to target.
    """
    obj = _safe_load_pickle(config.FEATURE_LIST_FILE)
    if obj is None:
        return list(config.FALLBACK_FEATURE_LIST)
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]
    if isinstance(obj, pd.Index):
        return [str(x) for x in obj.tolist()]
    if hasattr(obj, "tolist"):
        return [str(x) for x in obj.tolist()]
    return list(config.FALLBACK_FEATURE_LIST)


# --------------------------------------------------------------------------- #
# Derived information
# --------------------------------------------------------------------------- #
def model_available() -> bool:
    return config.MODEL_FILE.exists()


def resolve_feature_names(model: Any, feature_list: list[str]) -> list[str]:
    """Prefer the names the estimator was actually fitted on."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    return feature_list


@st.cache_data(show_spinner=False)
def get_feature_importances() -> pd.DataFrame:
    """Return a sorted ``feature / importance`` frame from the model."""
    model = load_model()
    features = load_feature_list()
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = list(model.feature_importances_)
    names = resolve_feature_names(model, features)
    if len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]

    frame = pd.DataFrame({"feature": names, "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, float]:
    """Load evaluation metrics from ``outputs/metrics.json`` or fall back."""
    if config.METRICS_FILE.exists():
        try:
            data = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
            return {k: float(v) for k, v in data.items()
                    if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001
            pass
    return dict(config.DEFAULT_METRICS)


@st.cache_data(show_spinner=False)
def load_confusion_matrix() -> list[list[int]]:
    """Load the confusion matrix from metrics.json if present."""
    if config.METRICS_FILE.exists():
        try:
            data = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
            cm = data.get("confusion_matrix")
            if cm and len(cm) == 2:
                return [[int(v) for v in row] for row in cm]
        except Exception:  # noqa: BLE001
            pass
    return [list(row) for row in config.DEFAULT_CONFUSION_MATRIX]
