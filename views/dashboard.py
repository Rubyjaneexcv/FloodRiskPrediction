"""
Dashboard (Page 1).

Executive overview: model KPIs, system status, a flood-risk map preview, the
highest-risk kecamatan and the probability distribution.

Risk figures come from the SAME live Random Forest forecast used on the
Prediction and Map pages (day-1 of the 14-day horizon), so the dashboard stays
consistent with the rest of the app. If the model/API are unavailable it falls
back to the historical ``Flood_Risk_Prediction.csv``.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from components import (
    card,
    format_id_date,
    format_id_time,
    metric_card,
    render_footer,
    render_header,
    risk_card,
    risk_legend,
    section_title,
    status_card,
)
from utils.data_loader import (
    data_available,
    geojson_available,
    load_geojson,
    load_prediction_csv,
)
from utils.geo_utils import build_location_index
from utils.model_loader import load_metrics, model_available
from utils.plotting import PRIMARY, style_figure
from utils.predictor import predict_all_kecamatan_14d, predictions_for_day
from utils.styling import risk_badge


def render() -> None:
    now = datetime.now()
    render_header(
        title="Prediksi Risiko Banjir",
        subtitle="Tingkat Kecamatan di Wilayah Jabodetabek · Model Random Forest",
        meta_main=format_id_date(now),
        meta_time=format_id_time(now),
    )

    _kpi_row()
    _status_row()

    with st.spinner("Menghitung risiko banjir terkini untuk seluruh kecamatan..."):
        risk_df, is_live = _risk_today()

    _map_preview(risk_df)
    _top_risk(risk_df, is_live)
    _distribution(risk_df)

    st.caption(f"Last update: {format_id_date(now)} · {format_id_time(now)}")
    render_footer()


# --------------------------------------------------------------------------- #
# Risk source (live model, day 1) with CSV fallback
# --------------------------------------------------------------------------- #
def _risk_today() -> tuple[pd.DataFrame, bool]:
    """Per-kecamatan risk for today (day 1). Returns (frame, is_live)."""
    preds = predict_all_kecamatan_14d()
    if not preds.empty:
        day1 = predictions_for_day(preds, 1)
        return day1[["kecamatan", "kabupaten", "probability"]].reset_index(drop=True), True

    csv = load_prediction_csv()
    if not csv.empty:
        return csv[["kecamatan", "kabupaten", "probability"]].copy(), False
    return pd.DataFrame(columns=["kecamatan", "kabupaten", "probability"]), False


# --------------------------------------------------------------------------- #
# KPI + status
# --------------------------------------------------------------------------- #
def _kpi_row() -> None:
    metrics = load_metrics()
    deltas = config.METRIC_DELTAS
    cards = [
        ("Accuracy", metrics.get("accuracy", 0), deltas.get("accuracy"), "🎯"),
        ("Precision", metrics.get("precision", 0), deltas.get("precision"), "📌"),
        ("Recall", metrics.get("recall", 0), deltas.get("recall"), "🔁"),
        ("ROC-AUC", metrics.get("roc_auc", 0), deltas.get("roc_auc"), "📈"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, delta, icon) in zip(cols, cards):
        with col:
            metric_card(label, f"{value * 100:.1f}%", delta=delta, icon=icon)


def _status_row() -> None:
    n_kec = len(build_location_index())
    n_rows = len(load_prediction_csv())
    cols = st.columns(4, gap="medium")
    with cols[0]:
        status_card("Total Kecamatan", f"{n_kec}", "Kecamatan terpantau", icon="🏙️")
    with cols[1]:
        loaded = model_available()
        status_card("Model", "Random Forest",
                    "✔ Loaded" if loaded else "Belum tersedia",
                    icon="🌲", ok=loaded)
    with cols[2]:
        status_card("Weather API", "Open-Meteo", "Aktif · gratis (14 hari)",
                    icon="🛰️", ok=True)
    with cols[3]:
        sub = f"{n_rows:,} baris historis" if data_available() else "Belum tersedia"
        status_card("Dataset", f"{n_rows:,}" if n_rows else "-", sub, icon="🗄️")


# --------------------------------------------------------------------------- #
# Map preview
# --------------------------------------------------------------------------- #
def _map_preview(risk_df: pd.DataFrame) -> None:
    with card("Peta Risiko Banjir (Preview)",
              "Klik menu Flood Risk Map untuk versi interaktif 14 hari"):
        if not geojson_available():
            st.info("GeoJSON belum tersedia — letakkan `jabodetabek_final.geojson` "
                    "di folder `data/`.")
            return

        gdf = load_geojson()
        prob_lookup = (dict(zip(risk_df["kecamatan"], risk_df["probability"]))
                       if not risk_df.empty else {})

        map_col, legend_col = st.columns([3, 1], gap="large")
        with map_col:
            try:
                from streamlit_folium import st_folium

                from views.mapping import build_risk_map

                fmap = build_risk_map(gdf, prob_lookup, height=420)
                st_folium(fmap, use_container_width=True, height=420,
                          returned_objects=[])
            except ModuleNotFoundError:
                st.warning("Install `streamlit-folium` untuk menampilkan peta.")
        with legend_col:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            risk_legend()


# --------------------------------------------------------------------------- #
# Top risk + distribution
# --------------------------------------------------------------------------- #
def _top_risk(risk_df: pd.DataFrame, is_live: bool) -> None:
    source = "prakiraan model hari ini" if is_live else "data historis (CSV)"
    section_title("Risiko Tertinggi Hari Ini",
                  f"10 kecamatan dengan probabilitas banjir tertinggi · {source}")
    if risk_df.empty:
        with card():
            st.info("Belum ada prediksi. Pastikan model & data tersedia.")
        return

    top = risk_df.sort_values("probability", ascending=False).head(10).reset_index(drop=True)

    cols = st.columns(3, gap="medium")
    for col, (_, row) in zip(cols, top.head(3).iterrows()):
        with col:
            risk_card(str(row["kecamatan"]), str(row["kabupaten"]),
                      float(row["probability"]))

    with card("Top 10 Highest Flood Risk"):
        st.markdown(_top_table_html(top), unsafe_allow_html=True)


def _top_table_html(top: pd.DataFrame) -> str:
    head = (
        "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
        "<thead><tr style='text-align:left;color:#64748B;"
        "border-bottom:1px solid #E7EDF5;'>"
        "<th style='padding:10px 8px;'>#</th><th>Kecamatan</th>"
        "<th>Kabupaten/Kota</th><th>Probabilitas</th><th>Risiko</th></tr></thead><tbody>"
    )
    rows = []
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        rows.append(
            f"<tr style='border-bottom:1px solid #F1F5F9;'>"
            f"<td style='padding:10px 8px;color:#94A3B8;font-weight:700;'>{i}</td>"
            f"<td style='font-weight:700;color:#0F2540;'>{r['kecamatan']}</td>"
            f"<td style='color:#475569;'>{r['kabupaten']}</td>"
            f"<td style='font-weight:700;'>{float(r['probability']):.2f}</td>"
            f"<td>{risk_badge(float(r['probability']))}</td></tr>"
        )
    return head + "".join(rows) + "</tbody></table>"


def _distribution(risk_df: pd.DataFrame) -> None:
    with card("Distribusi Probabilitas Banjir",
              "Sebaran probabilitas seluruh kecamatan (prakiraan hari ini)"):
        if risk_df.empty:
            st.info("Tidak ada data probabilitas untuk ditampilkan.")
            return
        probs = risk_df["probability"].to_numpy(dtype=float)
        fig = px.histogram(x=probs, nbins=20)
        fig.update_traces(marker_color=PRIMARY, marker_line_color="#FFFFFF",
                          marker_line_width=1.2, opacity=0.9)
        fig.update_layout(bargap=0.06, xaxis_title="Flood Probability",
                          yaxis_title="Jumlah Kecamatan")
        style_figure(fig, height=300)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
