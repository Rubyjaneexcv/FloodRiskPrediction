"""
Dataset (Page: Dataset).

Shows the training-dataset summary, a preview table and the feature dictionary,
with a CSV download. Reads the labelled dataset located in ``data/``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import card, render_footer, render_header, status_card
from utils.data_loader import load_training_dataset, resolve_column
from utils.model_loader import load_feature_list


def render() -> None:
    render_header("Dataset", "Data pelatihan model dan deskripsi fitur")

    df = load_training_dataset()
    _summary(df)
    _preview(df)
    _feature_dictionary(df)
    _download(df)

    render_footer()


# --------------------------------------------------------------------------- #
# Summary KPI cards
# --------------------------------------------------------------------------- #
def _summary(df: pd.DataFrame) -> None:
    feature_list = load_feature_list()
    label_col = resolve_column(df.columns, "label") if not df.empty else None
    date_col = resolve_column(df.columns, "date") if not df.empty else None

    total = len(df)
    n_feat = len([c for c in df.columns if c != label_col]) if not df.empty else len(feature_list)

    period = "-"
    if date_col:
        years = pd.to_datetime(df[date_col], errors="coerce").dt.year.dropna()
        if not years.empty:
            period = f"{int(years.min())} - {int(years.max())}"

    n_pos, pct = 0, 0.0
    if label_col:
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
        n_pos = int((labels == 1).sum())
        pct = (n_pos / total * 100) if total else 0.0

    cols = st.columns(4, gap="medium")
    with cols[0]:
        status_card("Total Records", f"{total:,}" if total else "-",
                    "baris data historis", icon="🗄️")
    with cols[1]:
        status_card("Jumlah Fitur", f"{n_feat}", "+ 1 label target", icon="🔢")
    with cols[2]:
        status_card( "Periode Data", " 2020 - 2026", "harian per kecamatan",icon="🗓️")
    with cols[3]:
        status_card("Kejadian Banjir", f"{n_pos:,}" if n_pos else "-",
                    f"label positif ({pct:.1f}%)", icon="⚠️")


# --------------------------------------------------------------------------- #
# Preview table
# --------------------------------------------------------------------------- #
def _preview(df: pd.DataFrame) -> None:
    with card("Preview Dataset",
              f"Menampilkan {min(8, len(df))} dari {len(df):,} baris" if not df.empty else ""):
        if df.empty:
            st.info("Dataset pelatihan belum tersedia. Letakkan file CSV berlabel "
                    "(mis. `dataset.csv` dengan kolom fitur + `label`) di folder "
                    "`data/` untuk menampilkannya di sini.")
            return

        label_col = resolve_column(df.columns, "label")
        curated = ["tanggal", "date", "kecamatan", "kabupaten", "max_rainfall",
                   "avg_rainfall", "elevation", "ndvi", "soil_moisture", "slope"]
        show = [c for c in curated if c in df.columns][:7]
        show += [c for c in df.columns if c not in show and c != label_col][:max(0, 7 - len(show))]
        show = list(dict.fromkeys(show))[:7]
        if label_col:
            show.append(label_col)

        st.markdown(_table_html(df[show].head(8), label_col), unsafe_allow_html=True)


def _label_badge(value) -> str:
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return f"<span>{value}</span>"
    if v == 1:
        return ("<span class='risk-badge' style='background:rgba(239,68,68,0.14);"
                "color:#EF4444;'>1</span>")
    return ("<span class='risk-badge' style='background:rgba(34,197,94,0.14);"
            "color:#16A34A;'>0</span>")


def _table_html(view: pd.DataFrame, label_col: str | None) -> str:
    heads = "".join(f"<th style='padding:9px 8px;'>{c}</th>" for c in view.columns)
    head = ("<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
            "<thead><tr style='text-align:left;color:#64748B;"
            f"border-bottom:1px solid #E7EDF5;'>{heads}</tr></thead><tbody>")
    rows = []
    for _, r in view.iterrows():
        cells = []
        for c in view.columns:
            if c == label_col:
                cells.append(f"<td style='padding:9px 8px;'>{_label_badge(r[c])}</td>")
            else:
                val = r[c]
                if isinstance(val, float):
                    val = f"{val:.2f}"
                cells.append(f"<td style='padding:9px 8px;color:#0F2540;'>{val}</td>")
        rows.append(f"<tr style='border-bottom:1px solid #F1F5F9;'>{''.join(cells)}</tr>")
    return head + "".join(rows) + "</tbody></table>"


# --------------------------------------------------------------------------- #
# Feature dictionary
# --------------------------------------------------------------------------- #
def _feature_dictionary(df: pd.DataFrame) -> None:
    feature_list = load_feature_list()
    with card("Deskripsi Fitur"):
        head = ("<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
                "<thead><tr style='text-align:left;color:#64748B;"
                "border-bottom:1px solid #E7EDF5;'>"
                "<th style='padding:9px 8px;'>Fitur</th><th>Deskripsi</th>"
                "<th>Sumber</th></tr></thead><tbody>")
        rows = []
        for feat in feature_list + ["label"]:
            desc = config.FEATURE_DESCRIPTIONS.get(feat, "-")
            src = config.FEATURE_SOURCES.get(feat, "-")
            rows.append(
                f"<tr style='border-bottom:1px solid #F1F5F9;'>"
                f"<td style='padding:9px 8px;font-weight:700;color:#0F2540;'>{feat}</td>"
                f"<td style='color:#475569;'>{desc}</td>"
                f"<td style='color:#475569;'>{src}</td></tr>"
            )
        st.markdown(head + "".join(rows) + "</tbody></table>", unsafe_allow_html=True)


def _download(df: pd.DataFrame) -> None:
    if df.empty:
        return
    st.download_button(
        "⬇  Download Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="flood_risk_dataset.csv",
        mime="text/csv",
        type="primary",
    )
