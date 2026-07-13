"""Reusable presentational cards (pure HTML, injected via markdown)."""

from __future__ import annotations

from typing import Optional, Sequence

import streamlit as st

from config import RISK_LEVELS
from utils.styling import get_risk_level


def _sparkline(color: str = "#16A34A", up: bool = True) -> str:
    """Tiny inline SVG sparkline for KPI cards."""
    pts = "0,18 12,14 24,15 36,9 48,11 60,5 72,6 84,2" if up else \
          "0,4 12,7 24,6 36,11 48,9 60,14 72,13 84,18"
    return (
        f"<svg width='86' height='22' viewBox='0 0 86 22' fill='none'>"
        f"<polyline points='{pts}' stroke='{color}' stroke-width='2.4' "
        f"fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )


def metric_card(
    label: str,
    value: str,
    delta: Optional[float] = None,
    icon: str = "",
    show_spark: bool = True,
) -> None:
    """KPI card with value, trend delta and a sparkline."""
    up = (delta or 0) >= 0
    delta_html = ""
    if delta is not None:
        arrow = "▲" if up else "▼"
        cls = "up" if up else "down"
        delta_html = (f"<div class='metric-delta {cls}'>{arrow} "
                      f"{'+' if up else ''}{delta * 100:.1f}%</div>")
    icon_html = f"<div class='metric-icon'>{icon}</div>" if icon else ""
    spark = f"<div class='metric-spark'>{_sparkline(up=up)}</div>" if show_spark else ""
    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-head'><div class='metric-label'>{label}</div>{icon_html}</div>"
        f"<div class='metric-value'>{value}</div>{delta_html}{spark}</div>",
        unsafe_allow_html=True,
    )


def status_card(
    label: str, value: str, sub: str = "", icon: str = "", ok: bool = False
) -> None:
    """Compact status/info card (e.g. model status, dataset size)."""
    icon_html = f"<div class='metric-icon'>{icon}</div>" if icon else ""
    ok_cls = " ok" if ok else ""
    st.markdown(
        f"<div class='status-card'>"
        f"<div class='metric-head'><div class='status-label'>{label}</div>{icon_html}</div>"
        f"<div class='status-value{ok_cls}'>{value}</div>"
        f"<div class='status-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def risk_card(kecamatan: str, kabupaten: str, probability: float) -> None:
    """Highlighted card for a single high-risk kecamatan (dashboard preview)."""
    level = get_risk_level(probability)
    is_low = level.key == "rendah"
    icon = "✔" if is_low else "⚠"
    badge = (f"<span class='risk-badge' style='background:{level.soft};"
             f"color:{level.color};'>{level.label}</span>")
    st.markdown(
        f"<div class='risk-card'>"
        f"<div class='rc-top'>"
        f"<div class='rc-icon' style='background:{level.soft};color:{level.color};'>{icon}</div>"
        f"<div><div class='rc-name'>{kecamatan}</div>"
        f"<div class='rc-city'>{kabupaten}</div></div></div>"
        f"<div class='rc-bottom'><div class='rc-prob'>Prob. {probability:.2f}</div>{badge}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def info_row(key: str, value_html: str) -> None:
    """Key / value row with a dashed separator (used on About & Map info panels)."""
    st.markdown(
        f"<div class='info-row'><div class='info-key'>{key}</div>"
        f"<div class='info-val'>{value_html}</div></div>",
        unsafe_allow_html=True,
    )


def risk_legend(levels: Sequence = RISK_LEVELS) -> None:
    """Colour legend for the risk categories."""
    rows = []
    for lvl in levels:
        upper = "1.0+" if lvl.upper > 1.0 else f"{lvl.upper:.1f}"
        rows.append(
            f"<div class='legend-row'>"
            f"<span class='legend-dot' style='background:{lvl.color};'></span>"
            f"<span class='legend-label'>{lvl.label}</span>"
            f"<span class='legend-range'>({lvl.lower:.1f} - {upper})</span></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
