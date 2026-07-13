"""Page header component with an optional right-aligned meta block."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st

_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
          "Agustus", "September", "Oktober", "November", "Desember"]


def format_id_date(dt: datetime) -> str:
    """Format a datetime as an Indonesian long date, e.g. 'Senin, 13 Juli 2026'."""
    return f"{_HARI[dt.weekday()]}, {dt.day} {_BULAN[dt.month]} {dt.year}"


def format_id_time(dt: datetime, tz_label: str = "WIB") -> str:
    return f"{dt:%H:%M} {tz_label}"


def render_header(
    title: str,
    subtitle: str = "",
    meta_main: Optional[str] = None,
    meta_time: Optional[str] = None,
) -> None:
    """Render the page header (title + subtitle on the left, meta on the right)."""
    sub_html = f"<div class='header-sub'>{subtitle}</div>" if subtitle else ""
    meta_html = ""
    if meta_main:
        time_html = f"<div class='meta-time'>{meta_time}</div>" if meta_time else ""
        meta_html = f"<div class='header-meta'>{meta_main}{time_html}</div>"
    st.markdown(
        f"<div class='app-header'>"
        f"<div><div class='header-title'>{title}</div>{sub_html}</div>"
        f"{meta_html}</div>",
        unsafe_allow_html=True,
    )
