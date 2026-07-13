"""
Prediction — 14 Days (Page 2). The core feature.

User picks Kabupaten -> Kecamatan -> Generate Forecast. The app pulls a 14-day
Open-Meteo forecast, engineers the model features, runs the Random Forest and
presents the result as summary cards, a table, a probability chart and a
day-by-day timeline.
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

import config
from components import (
    card,
    info_row,
    metric_card,
    render_footer,
    render_header,
    section_title,
)
from utils.geo_utils import get_kabupaten_list, get_kecamatan_list, get_static_features
from utils.history import append_run
from utils.model_loader import model_available
from utils.plotting import style_figure
from utils.predictor import PredictionError, predict_kecamatan_14d
from utils.styling import get_risk_level, risk_badge
from utils.weather_api import WeatherAPIError


def render() -> None:
    render_header(
        "Prediksi Risiko Banjir 14 Hari",
        "Prakiraan risiko banjir untuk 14 hari ke depan berdasarkan prakiraan cuaca",
    )

    left, right = st.columns([1, 2.4], gap="large")
    with left:
        kecamatan = _location_picker()
        _location_info(kecamatan)
    with right:
        _results_panel(kecamatan)

    render_footer()


# --------------------------------------------------------------------------- #
# Location picker
# --------------------------------------------------------------------------- #
def _location_picker() -> str:
    with card("Pilih Lokasi"):
        kab_options = get_kabupaten_list() or ["-"]
        kabupaten = st.selectbox("Kabupaten / Kota", kab_options, key="pred_kab")

        kec_options = get_kecamatan_list(kabupaten) or ["-"]
        kecamatan = st.selectbox("Kecamatan", kec_options, key="pred_kec")

        if st.button("🔄  Generate Forecast", type="primary",
                     use_container_width=True):
            _run_forecast(kecamatan)
    return kecamatan


def _run_forecast(kecamatan: str) -> None:
    if not model_available():
        st.error("Model belum tersedia. Letakkan `best_random_forest.pkl` "
                 "di folder `models/`.")
        return
    try:
        with st.spinner(f"Menghasilkan prakiraan untuk {kecamatan}..."):
            progress = st.progress(0, text="Mengambil data cuaca Open-Meteo...")
            result = predict_kecamatan_14d(kecamatan)
            progress.progress(70, text="Menjalankan model Random Forest...")
            st.session_state["forecast_result"] = result
            st.session_state["forecast_kecamatan"] = kecamatan
            st.session_state["forecast_time"] = datetime.now()
            _log_history(kecamatan, result)
            progress.progress(100, text="Selesai")
        progress.empty()
        st.toast(f"Prakiraan {kecamatan} berhasil dibuat", icon="✅")
    except (WeatherAPIError, PredictionError) as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Terjadi kesalahan tak terduga: {exc}")


def _log_history(kecamatan: str, result) -> None:
    """Append this forecast's peak risk to the prediction history log."""
    try:
        static = get_static_features(kecamatan)
        peak = result.loc[result["probability"].idxmax()]
        append_run(kecamatan, str(static.get("kabupaten", "-")),
                   float(peak["probability"]), str(peak["category"]))
    except Exception:  # noqa: BLE001 - history logging must never break a forecast
        pass


def _location_info(kecamatan: str) -> None:
    static = get_static_features(kecamatan)
    with card("Informasi Lokasi"):
        info_row("Kecamatan", f"<b>{kecamatan}</b>")
        info_row("Kabupaten/Kota", f"<b>{static.get('kabupaten', '-')}</b>")
        info_row("Koordinat",
                 f"<b>{float(static.get('lat', 0)):.4f}, "
                 f"{float(static.get('lon', 0)):.4f}</b>")
        info_row("Elevasi", f"<b>{float(static.get('elevation', 0)):.0f} m dpl</b>")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _results_panel(current_kecamatan: str) -> None:
    result = st.session_state.get("forecast_result")
    result_kec = st.session_state.get("forecast_kecamatan")

    if result is None or result.empty:
        with card("Hasil Prediksi 14 Hari"):
            st.info("Pilih lokasi lalu klik **Generate Forecast** untuk memulai "
                    "prakiraan risiko banjir 14 hari ke depan.")
        return

    _summary_cards(result)
    _forecast_table(result, result_kec)
    _probability_chart(result)
    _timeline(result)


def _summary_cards(result) -> None:
    peak = result.loc[result["probability"].idxmax()]
    high_days = int((result["probability"] >= 0.70).sum())
    avg_rain = float(result["rainfall"].mean())
    cols = st.columns(3, gap="medium")
    with cols[0]:
        metric_card("Puncak Risiko", f"{peak['probability']:.2f}",
                    icon="⚠️", show_spark=False)
        st.caption(f"{peak['date']:%d %b %Y}")
    with cols[1]:
        metric_card("Hari Risiko Tinggi", f"{high_days} hari",
                    icon="📆", show_spark=False)
    with cols[2]:
        metric_card("Rata-rata Hujan", f"{avg_rain:.1f} mm",
                    icon="🌧️", show_spark=False)


def _forecast_table(result, kecamatan: str) -> None:
    updated = st.session_state.get("forecast_time", datetime.now())
    with card("Hasil Prediksi 14 Hari",
              f"{kecamatan} · Sumber: Open-Meteo · {updated:%d %b %Y %H:%M}"):
        head = (
            "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
            "<thead><tr style='text-align:left;color:#64748B;"
            "border-bottom:1px solid #E7EDF5;'>"
            "<th style='padding:9px 8px;'>Hari</th><th>Tanggal</th>"
            "<th>Curah Hujan (mm)</th><th>Suhu (°C)</th>"
            "<th>Probabilitas</th><th>Risiko</th></tr></thead><tbody>"
        )
        rows = []
        for _, r in result.iterrows():
            rows.append(
                f"<tr style='border-bottom:1px solid #F1F5F9;'>"
                f"<td style='padding:9px 8px;font-weight:700;color:#94A3B8;'>{int(r['day'])}</td>"
                f"<td style='color:#0F2540;'>{r['date']:%d %b %Y}</td>"
                f"<td>{float(r['rainfall']):.1f}</td>"
                f"<td>{float(r['temperature']):.1f}</td>"
                f"<td style='font-weight:700;'>{float(r['probability']):.2f}</td>"
                f"<td>{risk_badge(float(r['probability']))}</td></tr>"
            )
        st.markdown(head + "".join(rows) + "</tbody></table>", unsafe_allow_html=True)


def _probability_chart(result) -> None:
    with card("Grafik Probabilitas Risiko 14 Hari"):
        colors = [get_risk_level(p).color for p in result["probability"]]
        fig = go.Figure()
        # Risk bands
        for lower, upper, colour in [
            (0.0, 0.30, "rgba(34,197,94,0.10)"),
            (0.30, 0.70, "rgba(245,158,11,0.12)"),
            (0.70, 1.00, "rgba(239,68,68,0.10)"),
        ]:
            fig.add_hrect(y0=lower, y1=upper, line_width=0, fillcolor=colour, layer="below")
        fig.add_trace(go.Scatter(
            x=result["day"], y=result["probability"],
            mode="lines+markers",
            line=dict(color="#334155", width=2.4),
            marker=dict(color=colors, size=11, line=dict(color="#FFFFFF", width=1.6)),
            hovertemplate="Hari %{x}<br>Probabilitas %{y:.2f}<extra></extra>",
        ))
        fig.update_yaxes(range=[0, 1.0], title="Probabilitas")
        fig.update_xaxes(title="Hari ke-", dtick=1)
        style_figure(fig, height=320)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})


def _timeline(result) -> None:
    section_title("Timeline Risiko 14 Hari")
    chips = []
    for _, r in result.iterrows():
        level = get_risk_level(float(r["probability"]))
        chips.append(
            f"<div style='flex:1 0 90px;background:#FFFFFF;border:1px solid #E7EDF5;"
            f"border-radius:12px;padding:10px;text-align:center;box-shadow:0 4px 14px "
            f"rgba(15,37,64,0.05);'>"
            f"<div style='font-size:12px;color:#94A3B8;font-weight:600;'>H-{int(r['day'])}</div>"
            f"<div style='font-size:12px;color:#0F2540;font-weight:600;margin:2px 0;'>"
            f"{r['date']:%d %b}</div>"
            f"<div style='width:14px;height:14px;border-radius:50%;background:{level.color};"
            f"margin:6px auto;'></div>"
            f"<div style='font-size:13px;font-weight:800;color:{level.color};'>"
            f"{float(r['probability']):.2f}</div></div>"
        )
    st.markdown(
        "<div style='display:flex;gap:8px;flex-wrap:wrap;'>" + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )
