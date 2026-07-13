"""Sidebar navigation — a professional left rail driven by session state."""

from __future__ import annotations

import streamlit as st

from config import APP, DEFAULT_PAGE, MENU


def _select_page(page_key: str) -> None:
    st.session_state["page"] = page_key


def _brand() -> None:
    st.markdown(
        f"<div class='sidebar-brand'>"
        f"<div class='logo'>\U0001F327️</div>"
        f"<div class='brand-text'>"
        f"<div class='brand-title'>{APP.short_name}</div>"
        f"<div class='brand-sub'>Jabodetabek</div></div></div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render the sidebar and return the currently selected page key."""
    if "page" not in st.session_state:
        st.session_state["page"] = DEFAULT_PAGE

    with st.sidebar:
        _brand()
        st.markdown("<div class='sidebar-section-label'>Menu</div>",
                    unsafe_allow_html=True)

        current = st.session_state["page"]
        for item in MENU:
            st.button(
                f"{item.icon}  {item.label}",
                key=f"nav_{item.key}",
                type="primary" if item.key == current else "secondary",
                use_container_width=True,
                on_click=_select_page,
                args=(item.key,),
            )

        st.markdown(
            f"<div class='sidebar-footer'>v{APP.version} · (c) {APP.year}<br>"
            f"Random Forest Model</div>",
            unsafe_allow_html=True,
        )

    return st.session_state["page"]
