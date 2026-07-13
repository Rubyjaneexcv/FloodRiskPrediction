"""
Model Analysis (Page 4).

Tabs: Overview · Feature Importance · Permutation Importance · SHAP ·
ROC Curve · Confusion Matrix. Every chart that can be Plotly is Plotly; SHAP
falls back to its native matplotlib rendering.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from components import card, info_row, metric_card, render_footer, render_header
from utils.data_loader import resolve_column
from utils.model_loader import (
    get_feature_importances,
    load_confusion_matrix,
    load_feature_list,
    load_metrics,
    load_model,
    model_available,
    resolve_feature_names,
)
from utils.plotting import PRIMARY, style_figure


def render() -> None:
    render_header("Analisis Model",
                  "Evaluasi dan interpretasi model Random Forest")

    if not model_available():
        with card("Model belum tersedia"):
            st.info("Letakkan `best_random_forest.pkl` dan `feature_list.pkl` "
                    "di folder `models/` untuk mengaktifkan halaman ini.")
        render_footer()
        return

    tabs = st.tabs([
        "Overview", "Feature Importance", "Permutation Importance",
        "SHAP", "ROC Curve", "Confusion Matrix",
    ])
    with tabs[0]:
        _overview()
    with tabs[1]:
        _feature_importance()
    with tabs[2]:
        _permutation_importance()
    with tabs[3]:
        _shap_summary()
    with tabs[4]:
        _roc_curve()
    with tabs[5]:
        _confusion_matrix()

    render_footer()


# --------------------------------------------------------------------------- #
# Shared data access
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _labeled_frame() -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Mencari dataset berlabel (features + target).

    Mendukung kolom:
    - label
    - banjir
    """

    feature_list = load_feature_list()

    paths = [
        config.PREDICTION_CSV,
        *sorted(config.DATA_DIR.glob("*.csv"))
    ]

    seen = set()

    for path in paths:

        if not path.exists() or str(path) in seen:
            continue

        seen.add(str(path))

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        df.columns = [str(c).strip() for c in df.columns]

        label_col = (
            resolve_column(df.columns, "label")
            or ("banjir" if "banjir" in df.columns else None)
        )

        if label_col is None:
            continue

        present = [f for f in feature_list if f in df.columns]

        if len(present) < max(3, len(feature_list)//2):
            continue

        X = df.reindex(columns=feature_list)

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        y = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        return X, y

    return None, None

def _no_data_notice(what: str) -> None:
    st.info(
        f"{what} memerlukan dataset berlabel (fitur + kolom `label`). "
        "Letakkan dataset pelatihan di folder `data/` untuk mengaktifkannya."
    )


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def _overview() -> None:
    metrics = load_metrics()
    cols = st.columns(5, gap="small")
    labels = [("Accuracy", "accuracy"), ("Precision", "precision"),
              ("Recall", "recall"), ("F1-Score", "f1"), ("ROC-AUC", "roc_auc")]
    for col, (label, key) in zip(cols, labels):
        with col:
            metric_card(label, f"{metrics.get(key, 0) * 100:.1f}%", show_spark=False)

    model = load_model()
    params = getattr(model, "get_params", lambda: {})()
    with card("Konfigurasi Model", "Random Forest Classifier"):
        info_row("Estimators", f"<b>{params.get('n_estimators', '-')}</b>")
        info_row("Max Depth", f"<b>{params.get('max_depth', 'None')}</b>")
        info_row("Min Samples Split", f"<b>{params.get('min_samples_split', '-')}</b>")
        info_row("Criterion", f"<b>{params.get('criterion', '-')}</b>")
        info_row("Jumlah Fitur", f"<b>{len(load_feature_list())}</b>")


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def _importance_bar(frame: pd.DataFrame, title: str) -> None:
    frame = frame.sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=frame["importance"], y=frame["feature"], orientation="h",
        marker=dict(color=PRIMARY, line=dict(color="#FFFFFF", width=0.5)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Importance", yaxis_title="")
    style_figure(fig, height=max(360, 22 * len(frame)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _feature_importance() -> None:
    frame = get_feature_importances()
    with card("Feature Importance (Random Forest)",
              "Impurity-based importance dari model"):
        if frame.empty:
            st.info("Model tidak menyediakan feature importances.")
            return
        _importance_bar(frame, "Feature Importance")


def _permutation_importance() -> None:
    X, y = _labeled_frame()
    with card("Permutation Importance",
              "Penurunan skor saat fitur diacak (lebih andal dari impurity)"):
        if X is None:
            _no_data_notice("Permutation importance")
            return
        try:
            from sklearn.inspection import permutation_importance

            model = load_model()
            with st.spinner("Menghitung permutation importance..."):
                sample = X.sample(min(len(X), 1500), random_state=42)
                y_sample = y.loc[sample.index]
                result = permutation_importance(
                    model, sample, y_sample, n_repeats=8, random_state=42, n_jobs=-1,
                )
            frame = pd.DataFrame({
                "feature": resolve_feature_names(model, load_feature_list()),
                "importance": result.importances_mean,
            })
            _importance_bar(frame, "Permutation Importance")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Gagal menghitung permutation importance: {exc}")


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def _shap_summary() -> None:
    X, _ = _labeled_frame()
    with card("SHAP Summary", "Kontribusi fitur terhadap prediksi"):
        if X is None:
            _no_data_notice("Analisis SHAP")
            return
        try:
            import shap

            model = load_model()
            with st.spinner("Menghitung nilai SHAP..."):
                sample = X.sample(min(len(X), 400), random_state=42)
                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(sample)
                shap_pos = values[1] if isinstance(values, list) else values
                if shap_pos.ndim == 3:      # (n, features, classes)
                    shap_pos = shap_pos[:, :, 1]

            mean_abs = np.abs(shap_pos).mean(axis=0)
            frame = pd.DataFrame({"feature": sample.columns, "importance": mean_abs})
            _importance_bar(frame, "Mean |SHAP value|")

            with st.expander("Lihat SHAP beeswarm plot"):
                import matplotlib.pyplot as plt

                shap.summary_plot(shap_pos, sample, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
        except ModuleNotFoundError:
            st.warning("Paket `shap` belum terpasang. Jalankan: pip install shap")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Gagal menghitung SHAP: {exc}")


# --------------------------------------------------------------------------- #
# ROC curve
# --------------------------------------------------------------------------- #
def _roc_curve() -> None:
    X, y = _labeled_frame()
    metrics = load_metrics()
    with card("ROC Curve", "Receiver Operating Characteristic"):
        if X is None:
            st.info(f"ROC memerlukan data uji berlabel. AUC model saat ini: "
                    f"**{metrics.get('roc_auc', 0):.3f}**.")
            return
        try:
            from sklearn.metrics import auc, roc_curve

            model = load_model()
            classes = list(getattr(model, "classes_", [0, 1]))
            idx = classes.index(1) if 1 in classes else -1
            scores = model.predict_proba(X)[:, idx]
            fpr, tpr, _ = roc_curve(y, scores)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     line=dict(color=PRIMARY, width=3),
                                     name=f"ROC (AUC = {roc_auc:.3f})",
                                     fill="tozeroy",
                                     fillcolor="rgba(37,99,235,0.10)"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="#94A3B8", dash="dash"),
                                     name="Random"))
            fig.update_layout(xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            style_figure(fig, height=420, showlegend=True)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Gagal membuat ROC curve: {exc}")


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
def _confusion_matrix() -> None:
    cm = load_confusion_matrix()
    with card("Confusion Matrix", "Baris = Aktual · Kolom = Prediksi"):
        z = np.array(cm)
        labels = ["Tidak Banjir (0)", "Banjir (1)"]
        annotations = []
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                annotations.append(dict(
                    x=labels[j], y=labels[i], text=f"<b>{z[i, j]}</b>",
                    showarrow=False,
                    font=dict(color="#0F2540" if z[i, j] < z.max() * 0.6 else "#FFFFFF",
                              size=20),
                ))
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels, colorscale="Blues", showscale=True,
            hovertemplate="Aktual %{y}<br>Prediksi %{x}<br>Jumlah %{z}<extra></extra>",
        ))
        fig.update_layout(annotations=annotations, yaxis_autorange="reversed",
                          xaxis_title="Prediksi", yaxis_title="Aktual")
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        tn, fp, fn, tp = z[0, 0], z[0, 1], z[1, 0], z[1, 1]
        total = z.sum()
        st.caption(f"TN={tn} · FP={fp} · FN={fn} · TP={tp} · Total={total}")
