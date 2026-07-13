"""
Riwayat Prediksi (Prediction History).

Shows a filterable, paginated log of flood-risk predictions. It blends the
persisted history (forecasts the user has generated) with a snapshot of today's
live per-kecamatan predictions, so the page is informative from first run.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from components import card, render_footer, render_header
from utils.geo_utils import get_kabupaten_list, get_kecamatan_list
from utils.history import load_history
from utils.predictor import predict_all_kecamatan_14d
from utils.styling import get_risk_level, risk_badge

_PER_PAGE = 8


def render() -> None:
    render_header("Riwayat Prediksi", "Riwayat prediksi risiko banjir per kecamatan")

    with st.spinner("Memuat riwayat & prediksi terkini..."):
        data = _history_data()

    kab, kec, start, end = _filter_bar()

    view = data.copy()
    if kab != "Semua":
        view = view[view["kabupaten"] == kab]
    if kec != "Semua":
        view = view[view["kecamatan"] == kec]
    if not view.empty:
        view = view[(view["timestamp"].dt.date >= start) &
                    (view["timestamp"].dt.date <= end)]
    view = view.sort_values("timestamp", ascending=False).reset_index(drop=True)

    _table_with_pager(view)
    _download(view)
    render_footer()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _history_data() -> pd.DataFrame:
    frames = [load_history(), _live_snapshot()]
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame(
            columns=["timestamp", "kabupaten", "kecamatan", "probability", "risk"])
    if data.empty:
        return data
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    return data.drop_duplicates(subset=["timestamp", "kecamatan"]).reset_index(drop=True)


def _live_snapshot() -> pd.DataFrame:
    preds = predict_all_kecamatan_14d()
    if preds.empty:
        return pd.DataFrame(columns=["timestamp", "kabupaten", "kecamatan",
                                     "probability", "risk"])
    grp = preds.groupby(["kecamatan", "kabupaten"], as_index=False)["probability"].max()
    grp["timestamp"] = pd.Timestamp.now().replace(microsecond=0)
    grp["risk"] = grp["probability"].apply(lambda p: get_risk_level(p).label)
    return grp[["timestamp", "kabupaten", "kecamatan", "probability", "risk"]]


# --------------------------------------------------------------------------- #
# Filter bar
# --------------------------------------------------------------------------- #
def _reset_page() -> None:
    st.session_state["hist_page"] = 1


def _filter_bar() -> tuple[str, str, date, date]:
    with card():
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1], gap="medium")
        with c1:
            kab = st.selectbox("Pilih Kabupaten/Kota",
                               ["Semua"] + get_kabupaten_list(), key="hist_kab",
                               on_change=_reset_page)
        with c2:
            kec_opts = ["Semua"] + get_kecamatan_list(None if kab == "Semua" else kab)
            kec = st.selectbox("Pilih Kecamatan", kec_opts, key="hist_kec",
                               on_change=_reset_page)
        with c3:
            start = st.date_input("Tanggal Mulai", value=date.today() - timedelta(days=13),
                                  key="hist_start", on_change=_reset_page)
        with c4:
            end = st.date_input("Tanggal Selesai", value=date.today(),
                                key="hist_end", on_change=_reset_page)
        with c5:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button("🔎 Filter", type="primary", use_container_width=True,
                      on_click=_reset_page)
    return kab, kec, start, end


# --------------------------------------------------------------------------- #
# Table + pagination
# --------------------------------------------------------------------------- #
def _table_with_pager(view: pd.DataFrame) -> None:
    with card():
        if view.empty:
            st.info("Belum ada riwayat prediksi untuk filter ini.")
            return

        total_pages = max(1, math.ceil(len(view) / _PER_PAGE))
        page = min(max(1, st.session_state.get("hist_page", 1)), total_pages)
        st.session_state["hist_page"] = page

        chunk = view.iloc[(page - 1) * _PER_PAGE: page * _PER_PAGE]
        st.markdown(_table_html(chunk), unsafe_allow_html=True)
        _pager(page, total_pages, len(view))


def _table_html(chunk: pd.DataFrame) -> str:
    head = (
        "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
        "<thead><tr style='text-align:left;color:#64748B;"
        "border-bottom:1px solid #E7EDF5;'>"
        "<th style='padding:10px 8px;'>Tanggal Prediksi</th><th>Kabupaten/Kota</th>"
        "<th>Kecamatan</th><th>Probabilitas (Max)</th><th>Risiko (Max)</th></tr>"
        "</thead><tbody>"
    )
    rows = []
    for _, r in chunk.iterrows():
        rows.append(
            f"<tr style='border-bottom:1px solid #F1F5F9;'>"
            f"<td style='padding:10px 8px;color:#0F2540;'>{r['timestamp']:%d %b %Y %H:%M}</td>"
            f"<td style='color:#475569;'>{r['kabupaten']}</td>"
            f"<td style='font-weight:700;color:#0F2540;'>{r['kecamatan']}</td>"
            f"<td style='font-weight:700;'>{float(r['probability']):.2f}</td>"
            f"<td>{risk_badge(float(r['probability']))}</td></tr>"
        )
    return head + "".join(rows) + "</tbody></table>"


def _set_page(value: int) -> None:
    st.session_state["hist_page"] = value


def _pager(page: int, total_pages: int, total_rows: int) -> None:
    left, mid, right = st.columns([1, 3, 1])
    with left:
        st.button("‹ Sebelumnya", disabled=page <= 1, use_container_width=True,
                  on_click=_set_page, args=(page - 1,))
    with mid:
        st.markdown(
            f"<div style='text-align:center;color:#64748B;font-weight:600;"
            f"padding-top:8px;'>Halaman {page} dari {total_pages} · "
            f"{total_rows:,} baris</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.button("Berikutnya ›", disabled=page >= total_pages, use_container_width=True,
                  on_click=_set_page, args=(page + 1,))


def _download(view: pd.DataFrame) -> None:
    if view.empty:
        return
    st.download_button(
        "⬇  Download CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="riwayat_prediksi.csv",
        mime="text/csv",
        type="primary",
    )
