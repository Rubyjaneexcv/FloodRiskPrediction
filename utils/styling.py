"""
Styling helpers: CSS injection, risk categorisation and small HTML builders.

Keeping every bit of presentation logic here means the view modules never have
to hand-write colours or repeated markup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

import config
from config import RISK_LEVELS, RiskLevel


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _read_css(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def inject_css() -> None:
    """Inject the custom stylesheet once per session run."""
    css = _read_css(str(config.CSS_FILE))
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("Stylesheet tidak ditemukan di assets/style.css")


# --------------------------------------------------------------------------- #
# Risk categorisation
# --------------------------------------------------------------------------- #
def get_risk_level(probability: float) -> RiskLevel:
    """Map a probability in [0, 1] to its :class:`RiskLevel`."""
    try:
        p = float(probability)
    except (TypeError, ValueError):
        p = 0.0
    p = max(0.0, min(p, 1.0))
    for level in RISK_LEVELS:
        if level.lower <= p < level.upper:
            return level
    return RISK_LEVELS[-1]


def risk_color(probability: float) -> str:
    return get_risk_level(probability).color


def risk_label(probability: float) -> str:
    return get_risk_level(probability).label


def risk_badge(probability: float) -> str:
    """Return an HTML pill badge for the given probability."""
    level = get_risk_level(probability)
    return (
        f"<span class='risk-badge' "
        f"style='background:{level.soft};color:{level.color};'>"
        f"{level.label}</span>"
    )


def label_badge(label: str) -> str:
    """Badge built from a textual risk label (used when reading a CSV column)."""
    match = _level_from_label(label)
    return (
        f"<span class='risk-badge' "
        f"style='background:{match.soft};color:{match.color};'>"
        f"{match.label}</span>"
    )


def _level_from_label(label: str) -> RiskLevel:
    norm = str(label).strip().lower().replace(" ", "_")
    for level in RISK_LEVELS:
        if level.key == norm or level.label.lower() == str(label).strip().lower():
            return level
    return RISK_LEVELS[0]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def fmt(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"
