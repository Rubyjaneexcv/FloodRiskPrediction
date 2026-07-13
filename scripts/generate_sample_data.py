"""
Generate synthetic artefacts so the app runs end-to-end without the real files.

Creates model + feature_list + GeoJSON (with the same property names as the real
Jabodetabek file) + prediction CSV + metrics.json. Replace with your real thesis
artefacts for production use.

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RNG = np.random.default_rng(42)
FEATURES = list(config.FALLBACK_FEATURE_LIST)   # real 20-feature schema

KABUPATEN = [
    "Kota Jakarta Pusat", "Kota Jakarta Utara", "Kota Jakarta Barat",
    "Kota Jakarta Selatan", "Kota Jakarta Timur", "Kota Bogor", "Kabupaten Bogor",
    "Kota Depok", "Kota Tangerang", "Kabupaten Tangerang",
    "Kota Tangerang Selatan", "Kota Bekasi", "Kabupaten Bekasi",
]


def _feature_frame(n: int, elevation=None, slope=None, ndvi=None,
                   soil=None, landcover=None, lat=None) -> pd.DataFrame:
    """Assemble the 20 model features from base weather + geospatial inputs."""
    max_rain = RNG.gamma(2.0, 22.0, n)
    hours = RNG.uniform(1, 14, n)
    elevation = RNG.uniform(3, 210, n) if elevation is None else elevation
    slope = RNG.uniform(0, 0.4, n) if slope is None else slope
    ndvi = RNG.uniform(0.1, 0.7, n) if ndvi is None else ndvi
    soil = RNG.uniform(20, 85, n) if soil is None else soil
    landcover = RNG.integers(0, 5, n).astype(float) if landcover is None else landcover
    lat = RNG.uniform(-6.6, -6.0, n) if lat is None else lat

    return pd.DataFrame({
        "avg_rainfall": max_rain * RNG.uniform(0.3, 0.65, n),
        "max_rainfall": max_rain,
        "avg_temperature": RNG.normal(28, 1.4, n),
        "elevation": elevation,
        "landcover_class": landcover,
        "ndvi": ndvi,
        "slope": slope,
        "soil_moisture": soil,
        "year": RNG.integers(2019, 2026, n).astype(float),
        "month": RNG.integers(1, 13, n).astype(float),
        "lat": lat,
        "long": RNG.uniform(106.5, 107.1, n),
        "hujan_harian_max": max_rain * RNG.uniform(0.7, 1.0, n),
        "hujan_kumulatif_3h_max": max_rain * RNG.uniform(0.35, 0.7, n),
        "jml_hari_hujan_lebat": RNG.poisson(2.2, n).astype(float),
        "maks_hari_hujan_beruntun": RNG.integers(0, 8, n).astype(float),
        "total_hujan_bulan": max_rain * RNG.uniform(3, 12, n),
        "rainfall_intensity": max_rain / hours,
        "vegetation_moisture_index": ndvi * RNG.uniform(0.7, 1.0, n),
        "rainfall_elevation_ratio": max_rain / (elevation + 1.0),
    })[FEATURES]


def _train_model() -> RandomForestClassifier:
    df = _feature_frame(6000)
    logit = (0.045 * df["max_rainfall"] + 0.05 * df["hujan_kumulatif_3h_max"]
             + 0.03 * df["soil_moisture"] - 0.02 * df["elevation"]
             + 0.35 * df["jml_hari_hujan_lebat"] - 4.2 + RNG.normal(0, 0.8, len(df)))
    y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.22,
                                              random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, max_depth=14,
                                   min_samples_split=4, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, list(model.classes_).index(1)]
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("  metrics:", {k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return model


def _build_geojson(n: int = 48):
    """GeoJSON grid using the SAME property names as the real dataset."""
    cols, features, records = 8, [], []
    lon0, lat0, step = 106.60, -6.55, 0.09
    for i in range(n):
        r, c = divmod(i, cols)
        lon, lat = lon0 + c * step, lat0 + r * step
        name = f"Kecamatan {i + 1:02d}"
        props = {
            "Objek": name, "Kecamatan_x": name, "Kab_Kota": KABUPATEN[i % len(KABUPATEN)],
            "Kabupaten": KABUPATEN[i % len(KABUPATEN)],
            "Lintang": round(lat + step * 0.46, 5), "Bujur": round(lon + step * 0.46, 5),
            "Elevasi": round(float(RNG.uniform(5, 200)), 1),
            "Kemiringan_Lereng": round(float(RNG.uniform(0, 0.4)), 3),
            "NDVI": round(float(RNG.uniform(0.15, 0.65)), 3),
            "Kelembapan_Tanah": round(float(RNG.uniform(25, 80)), 2),
            "Tutupan_Lahan": float(RNG.integers(0, 5)),
        }
        poly = [[[lon, lat], [lon + step * 0.92, lat],
                 [lon + step * 0.92, lat + step * 0.92], [lon, lat + step * 0.92], [lon, lat]]]
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Polygon", "coordinates": poly}})
        records.append(props)
    return features, records


def _prediction_csv(model, records) -> pd.DataFrame:
    rows = []
    for rec in records:
        feat = _feature_frame(1,
            elevation=np.array([rec["Elevasi"]]), slope=np.array([rec["Kemiringan_Lereng"]]),
            ndvi=np.array([rec["NDVI"]]), soil=np.array([rec["Kelembapan_Tanah"]]),
            landcover=np.array([rec["Tutupan_Lahan"]]), lat=np.array([rec["Lintang"]]))
        prob = float(model.predict_proba(feat)[:, list(model.classes_).index(1)][0])
        rows.append({"kecamatan": rec["Kecamatan_x"], "kabupaten": rec["Kab_Kota"],
                     "flood_probability": round(prob, 4)})
    return pd.DataFrame(rows)


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Training synthetic Random Forest (20 features)...")
    model = _train_model()
    joblib.dump(model, config.MODEL_FILE)
    joblib.dump(FEATURES, config.FEATURE_LIST_FILE)

    print("[2/4] Writing GeoJSON (real property names)...")
    features, records = _build_geojson()
    config.GEOJSON_FILE.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")

    print("[3/4] Writing prediction CSV...")
    _prediction_csv(model, records).to_csv(config.PREDICTION_CSV, index=False)

    print("[4/4] Done. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
