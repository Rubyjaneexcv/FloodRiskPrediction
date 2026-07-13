"""
Flood Risk Map (Page 3).

An interactive Folium choropleth of per-kecamatan flood risk with a day slider
(Day 1..14). Polygon colours and tooltips update as the selected forecast day
changes. Also exposes :func:`build_risk_map` which the dashboard reuses for its
map preview.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

import config
from components import card, info_row, render_footer, render_header, risk_legend, section_title
from components.cards import _sparkline  # noqa: F401 (kept for parity/imports)
from utils.data_loader import (clean_name, load_geojson, load_prediction_csv,
                               resolve_column)
from utils.predictor import predict_all_kecamatan_14d, predictions_for_day
from utils.styling import get_risk_level, risk_badge


# --------------------------------------------------------------------------- #
# Folium map builder (shared with the dashboard preview)
# --------------------------------------------------------------------------- #
def build_risk_map(gdf, prob_lookup: dict[str, float], height: int = 540):
    """Return a Folium map with polygons coloured by flood probability."""
    import folium

    kec_col = resolve_column(gdf.columns, "kecamatan") or gdf.columns[0]
    kab_col = resolve_column(gdf.columns, "kabupaten")

    enriched = gdf.copy()
    enriched["__kec"] = enriched[kec_col].map(clean_name)
    enriched["__kab"] = enriched[kab_col].map(clean_name) if kab_col else "-"
    enriched["__prob"] = enriched["__kec"].map(prob_lookup).fillna(0.0)
    enriched["__proba_txt"] = enriched["__prob"].map(lambda p: f"{p:.2f}")
    enriched["__kategori"] = enriched["__prob"].map(lambda p: get_risk_level(p).label)

    fmap = folium.Map(
        location=[-6.30, 106.82], zoom_start=9, tiles="cartodbpositron",
        control_scale=True,
    )

    def style_function(feature: dict) -> dict:
        prob = feature["properties"].get("__prob", 0.0) or 0.0
        return {
            "fillColor": get_risk_level(prob).color,
            "color": "#FFFFFF",
            "weight": 1.1,
            "fillOpacity": 0.78,
        }

    def highlight_function(_feature: dict) -> dict:
        return {"weight": 2.6, "color": "#0F2540", "fillOpacity": 0.9}

    folium.GeoJson(
        enriched.__geo_interface__,
        name="Risiko Banjir",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["__kec", "__kab", "__proba_txt", "__kategori"],
            aliases=["Kecamatan", "Kabupaten/Kota", "Probabilitas", "Risiko"],
            sticky=True,
            labels=True,
            style=(
                "background-color:#FFFFFF;color:#0F2540;font-family:Inter;"
                "font-size:12px;padding:8px;border-radius:8px;"
                "box-shadow:0 4px 14px rgba(15,37,64,0.18);"
            ),
        ),
    ).add_to(fmap)
    return fmap


# --------------------------------------------------------------------------- #
# Probability lookup helpers
# --------------------------------------------------------------------------- #
def _lookup_from_predictions(day: int) -> tuple[dict[str, float], bool]:
    """Return {kecamatan: probability} for a given forecast day (+ live flag)."""
    all_preds = predict_all_kecamatan_14d()
    if not all_preds.empty:
        day_df = predictions_for_day(all_preds, day)
        return dict(zip(day_df["kecamatan"], day_df["probability"])), True

    # Fallback: static probabilities from the historical CSV.
    csv = load_prediction_csv()
    if not csv.empty:
        return dict(zip(csv["kecamatan"], csv["probability"])), False
    return {}, False


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render() -> None:
    render_header(
        "Peta Risiko Banjir",
        "Peta sebaran risiko banjir per kecamatan di wilayah Jabodetabek",
    )

    gdf = load_geojson()
    if gdf is None:
        with card("Peta tidak tersedia"):
            st.info(
                "GeoJSON belum tersedia. Letakkan `jabodetabek_final.geojson` di "
                "folder `data/` untuk menampilkan peta interaktif."
            )
        render_footer()
        return

    left, right = st.columns([1, 2.6], gap="large")

    with left:
        with card("Pilih Hari Prediksi"):
            day = st.slider("Hari Prediksi", 1, config.FORECAST_DAYS, 5)
            fdate = date.today() + timedelta(days=day - 1)
            st.markdown(
                f"<div class='info-val' style='margin-top:6px;'>Hari ke-{day} "
                f"· {fdate:%d %b %Y}</div>",
                unsafe_allow_html=True,
            )

        with card("Legenda Risiko"):
            risk_legend()

    with st.spinner("Menghitung risiko banjir untuk seluruh kecamatan..."):
        prob_lookup, is_live = _lookup_from_predictions(day)

    with right:
        with card():
            try:
                from streamlit_folium import st_folium

                fmap = build_risk_map(gdf, prob_lookup)
                state = st_folium(
                    fmap, use_container_width=True, height=560,
                    returned_objects=["last_object_clicked_tooltip"],
                )
            except ModuleNotFoundError:
                state = None
                st.error("Paket `streamlit-folium` belum terpasang. "
                         "Jalankan: pip install streamlit-folium")

    with left:
        with card("Informasi Kecamatan"):
            _info_panel(gdf, prob_lookup, state)
        if not is_live:
            st.caption("Sumber: data historis (CSV). Model/API belum aktif.")

    render_footer()


def _info_panel(gdf, prob_lookup: dict[str, float], state: Optional[dict]) -> None:
    """Detail panel for the kecamatan selected via dropdown or map click."""
    kec_col = resolve_column(gdf.columns, "kecamatan") or gdf.columns[0]
    kab_col = resolve_column(gdf.columns, "kabupaten")
    lookup = gdf.copy()
    lookup["__kec"] = lookup[kec_col].map(clean_name)
    lookup["__kab"] = lookup[kab_col].map(clean_name) if kab_col else "-"
    names = sorted(lookup["__kec"].unique().tolist())

    clicked = None
    if state and state.get("last_object_clicked_tooltip"):
        text = str(state["last_object_clicked_tooltip"])
        clicked = next((n for n in names if n in text), None)

    default_idx = names.index(clicked) if clicked in names else 0
    selected = st.selectbox("Kecamatan", names, index=default_idx)

    prob = float(prob_lookup.get(selected, 0.0))
    row = lookup[lookup["__kec"] == selected]
    kab = str(row.iloc[0]["__kab"]) if not row.empty else "-"

    info_row("Kecamatan", f"<b>{selected}</b>")
    info_row("Kabupaten/Kota", f"<b>{kab}</b>")
    info_row("Probabilitas", f"<b>{prob:.2f}</b>")
    info_row("Risiko", risk_badge(prob))
