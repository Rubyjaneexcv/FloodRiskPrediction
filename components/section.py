"""SectionTitle and card-header building blocks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import streamlit as st


def section_title(main: str, sub: Optional[str] = None) -> None:
    """Render a page-section heading with an optional sub-caption."""
    sub_html = f"<div class='section-title-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='section-title'>"
        f"<div class='section-title-main'>{main}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def card_header(title: str, note: Optional[str] = None) -> None:
    """Render a title row (with an optional right-aligned note) inside a card."""
    note_html = f"<div class='card-note'>{note}</div>" if note else ""
    st.markdown(
        f"<div class='card-head'><div class='card-title'>{title}</div>{note_html}</div>",
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: Optional[str] = None, note: Optional[str] = None) -> Iterator[None]:
    """Context manager: a bordered white card that can wrap Streamlit widgets."""
    with st.container(border=True):
        if title:
            card_header(title, note)
        yield
