"""
Flood Risk Prediction System — application entry point.

Run with:

    streamlit run app.py

Uses a single-page-application style router (session state) instead of
Streamlit's built-in multipage feature, so navigation is fully controlled by
the custom sidebar.
"""

from __future__ import annotations

import streamlit as st

from config import APP, DEFAULT_PAGE
from components.sidebar import render_sidebar
from utils.styling import inject_css
from views import PAGES


def _configure_page() -> None:
    st.set_page_config(
        page_title=APP.name,
        page_icon="🌧️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def main() -> None:
    _configure_page()
    inject_css()

    page = render_sidebar()
    render_fn = PAGES.get(page, PAGES[DEFAULT_PAGE])

    try:
        render_fn()
    except Exception as exc:  # noqa: BLE001 - top level guard keeps the UI alive
        st.error(f"Terjadi kesalahan saat memuat halaman: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
