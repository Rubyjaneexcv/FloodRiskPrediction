"""Global footer component."""

from __future__ import annotations

import streamlit as st

from config import APP


def render_footer() -> None:
    st.markdown(
        f"<div class='app-footer'>(c) {APP.year} · {APP.name} v{APP.version} · "
        f"Aplikasi penelitian tugas akhir · {APP.tech_stack}</div>",
        unsafe_allow_html=True,
    )
