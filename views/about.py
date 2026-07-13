"""About (Page 5) — thesis, author and application metadata."""

from __future__ import annotations

import streamlit as st

from components import card, info_row, render_footer, render_header, section_title
from config import APP


def render() -> None:
    render_header("About", "Informasi aplikasi dan penelitian")

    with card():
        st.markdown(
            f"<div style='text-align:center;padding:8px 0 4px;'>"
            f"<div style='font-size:52px;'>\U0001F327️</div>"
            f"<div style='font-size:24px;font-weight:800;color:#0F2540;margin-top:6px;'>"
            f"Prediksi Risiko Banjir Tingkat Kecamatan<br>di Wilayah Jabodetabek</div>"
            f"<div style='font-size:16px;color:#64748B;margin-top:10px;'>"
            f"Menggunakan Algoritma Random Forest<br>"
            f"Berbasis Data Meteorologi dan Geospasial</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border:none;border-top:1px solid #E7EDF5;margin:18px 0;'>",
                    unsafe_allow_html=True)

        cols = st.columns(4, gap="large")
        meta = [
            ("DIKEMBANGKAN OLEH", APP.author, f"NIM. {APP.nim}"),
            ("PROGRAM STUDI", APP.study_program, ""),
            ("FAKULTAS", APP.faculty, ""),
            ("UNIVERSITAS", APP.university, ""),
        ]
        for col, (label, value, sub) in zip(cols, meta):
            with col:
                sub_html = (f"<div style='font-size:13px;color:#94A3B8;margin-top:2px;'>"
                            f"{sub}</div>") if sub else ""
                st.markdown(
                    f"<div style='font-size:12px;letter-spacing:0.08em;color:#94A3B8;"
                    f"font-weight:700;'>{label}</div>"
                    f"<div style='font-size:17px;font-weight:800;color:#0F2540;"
                    f"margin-top:4px;'>{value}</div>{sub_html}",
                    unsafe_allow_html=True,
                )

    left, right = st.columns(2, gap="large")
    with left:
        with card("Dosen Pembimbing"):
            for i, advisor in enumerate(APP.advisors, start=1):
                info_row(f"Pembimbing {i}", f"<b>{advisor}</b>")
    with right:
        with card("Sumber Data"):
            for source in APP.data_sources:
                st.markdown(
                    f"<div class='legend-row'>"
                    f"<span class='legend-dot' style='background:#2563EB;'></span>"
                    f"<span class='legend-label' style='font-weight:500;'>{source}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with card("Judul Penelitian"):
        st.markdown(
            f"<div style='font-size:15px;line-height:1.6;color:#0F2540;"
            f"font-weight:600;text-align:justify;'>\"{APP.thesis_title}\"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        info_row("Versi Aplikasi", f"<b>v{APP.version}</b>")
        info_row("Teknologi", APP.tech_stack)

    render_footer()
