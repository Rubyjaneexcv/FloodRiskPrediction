"""
Central configuration for the Flood Risk Prediction System.

All paths, theme tokens, model metadata, risk thresholds and Open-Meteo
settings live here so the rest of the code base stays free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
DATA_DIR: Final[Path] = BASE_DIR / "data"
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"

CSS_FILE: Final[Path] = ASSETS_DIR / "style.css"
MODEL_FILE: Final[Path] = MODELS_DIR / "best_random_forest.pkl"
FEATURE_LIST_FILE: Final[Path] = MODELS_DIR / "feature_list.pkl"
GEOJSON_FILE: Final[Path] = DATA_DIR / "jabodetabek_final.geojson"
PREDICTION_CSV: Final[Path] = DATA_DIR / "Flood_Risk_Prediction.csv"
METRICS_FILE: Final[Path] = OUTPUTS_DIR / "metrics.json"


# --------------------------------------------------------------------------- #
# Application metadata (shown on About / Footer / Header)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppInfo:
    name: str = "Flood Risk Prediction System"
    short_name: str = "Flood Risk"
    subtitle: str = "Prediksi Risiko Banjir Tingkat Kecamatan - Jabodetabek"
    tagline: str = "Using Random Forest Algorithm"
    description: str = (
        "Berbasis Data Meteorologi dan Geospasial untuk memprediksi risiko "
        "banjir tingkat kecamatan di wilayah Jabodetabek."
    )
    version: str = "1.0.0"
    year: int = 2026
    thesis_title: str = (
        "PREDIKSI RISIKO BANJIR TINGKAT KECAMATAN DI WILAYAH JABODETABEK "
        "MENGGUNAKAN ALGORITMA RANDOM FOREST BERBASIS DATA METEOROLOGI DAN "
        "GEOSPASIAL"
    )
    author: str = "Fikri Faddilah"
    nim: str = "22416255201099"
    study_program: str = "Teknik Informatika"
    faculty: str = "Ilmu Komputer"
    university: str = "Universitas Buana Perjuangan Karawang"
    advisors: tuple[str, ...] = (
        "Dr. Hanny Hikmayanti H, S.Kom., M.Kom.",
        "Rahmat, S.Pd., M.Pd.",
    )
    data_sources: tuple[str, ...] = (
        "BMKG / Open-Meteo - Prakiraan Cuaca 14 Hari",
        "Google Earth Engine - Data Geospasial (batas & elevasi)",
        "Landsat - NDVI / indeks vegetasi",
        "Data Curah Hujan Historis",
    )
    tech_stack: str = "Streamlit - scikit-learn - GeoPandas - Folium - Plotly"


APP: Final[AppInfo] = AppInfo()


# --------------------------------------------------------------------------- #
# Theme tokens (kept in sync with assets/style.css :root variables)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Theme:
    background: str = "#F6F8FC"
    sidebar: str = "#143B69"
    sidebar_accent: str = "#1E4E85"
    primary: str = "#2563EB"
    primary_dark: str = "#1D4ED8"
    card: str = "#FFFFFF"
    text: str = "#0F2540"
    text_muted: str = "#64748B"
    border: str = "#E2E8F0"
    success: str = "#16A34A"
    radius: str = "16px"
    font: str = "Inter"


THEME: Final[Theme] = Theme()


# --------------------------------------------------------------------------- #
# Risk levels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskLevel:
    key: str
    label: str
    color: str
    soft: str          # translucent background for badges
    lower: float       # inclusive lower bound
    upper: float       # exclusive upper bound


# Ordered from lowest to highest. Thresholds are configurable here.
RISK_LEVELS: Final[tuple[RiskLevel, ...]] = (
    RiskLevel("rendah", "Rendah", "#22C55E", "rgba(34,197,94,0.14)", 0.00, 0.30),
    RiskLevel("sedang", "Sedang", "#F59E0B", "rgba(245,158,11,0.16)", 0.30, 0.70),
    RiskLevel("tinggi", "Tinggi", "#F97316", "rgba(249,115,22,0.16)", 0.70, 0.90),
    RiskLevel("sangat_tinggi", "Sangat Tinggi", "#EF4444",
              "rgba(239,68,68,0.16)", 0.90, 1.01),
)


# --------------------------------------------------------------------------- #
# Open-Meteo forecast API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OpenMeteoConfig:
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    forecast_days: int = 14
    timezone: str = "Asia/Jakarta"
    timeout: int = 20
    daily_vars: tuple[str, ...] = (
        "precipitation_sum",
        "rain_sum",
        "precipitation_hours",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_probability_max",
        "windspeed_10m_max",
    )
    hourly_vars: tuple[str, ...] = ("precipitation",)

    @property
    def daily_param(self) -> str:
        return ",".join(self.daily_vars)

    @property
    def hourly_param(self) -> str:
        return ",".join(self.hourly_vars)


OPEN_METEO: Final[OpenMeteoConfig] = OpenMeteoConfig()


# --------------------------------------------------------------------------- #
# Model performance (fallback values, overridable via outputs/metrics.json)
# --------------------------------------------------------------------------- #
DEFAULT_METRICS: Final[dict[str, float]] = {
    "accuracy": 0.925,
    "precision": 0.914,
    "recall": 0.857,
    "f1": 0.884,
    "roc_auc": 0.974,
}

# Trend deltas shown as sparkline captions on the dashboard KPI cards.
METRIC_DELTAS: Final[dict[str, float]] = {
    "accuracy": 0.032,
    "precision": 0.027,
    "recall": 0.019,
    "roc_auc": 0.041,
}

# Confusion matrix fallback: rows = actual, cols = predicted -> [[TN, FP], [FN, TP]]
DEFAULT_CONFUSION_MATRIX: Final[list[list[int]]] = [[704, 98], [59, 560]]


# --------------------------------------------------------------------------- #
# Feature dictionary (human readable descriptions for the Model Analysis page)
# --------------------------------------------------------------------------- #
FEATURE_DESCRIPTIONS: Final[dict[str, str]] = {
    "max_rainfall": "Curah hujan harian maksimum (mm)",
    "avg_rainfall": "Rata-rata curah hujan harian (mm)",
    "hujan_kumulatif_3h_max": "Akumulasi hujan 3 jam maksimum (mm)",
    "hujan_harian_max": "Curah hujan harian tertinggi (mm)",
    "total_hujan_bulan": "Total akumulasi hujan pada jendela prakiraan (mm)",
    "rainfall_intensity": "Intensitas hujan (mm/jam)",
    "jml_hari_hujan_lebat": "Jumlah hari hujan lebat (> 20 mm)",
    "maks_hari_hujan_berturut": "Maksimum hari hujan berturut-turut",
    "avg_temperature": "Rata-rata suhu udara (C)",
    "elevation": "Elevasi rata-rata kecamatan (m dpl)",
    "ndvi": "Indeks vegetasi (NDVI)",
    "vegetation_moisture": "Kelembapan vegetasi",
    "soil_moisture": "Kelembapan tanah",
    "slope": "Kemiringan lereng (derajat)",
    "long": "Bujur (longitude)",
    "maks_hari_hujan_beruntun": "Maksimum hari hujan beruntun",
    "vegetation_moisture_index": "Indeks kelembapan vegetasi",
    "landcover_class": "Kelas tutupan lahan (encoded)",
    "rainfall_elevation_ratio": "Rasio curah hujan terhadap elevasi",
    "lat": "Lintang (latitude)",
    "lon": "Bujur (longitude)",
    "year": "Tahun",
    "month": "Bulan",
    "label": "Kejadian banjir (0 = tidak, 1 = banjir)",
}

# Canonical order of engineered features. The real source of truth remains
# feature_list.pkl; this is only used as a fallback / to seed the pipeline.
FALLBACK_FEATURE_LIST: Final[list[str]] = [
    "avg_rainfall",
    "max_rainfall",
    "avg_temperature",
    "elevation",
    "landcover_class",
    "ndvi",
    "slope",
    "soil_moisture",
    "year",
    "month",
    "lat",
    "long",
    "hujan_harian_max",
    "hujan_kumulatif_3h_max",
    "jml_hari_hujan_lebat",
    "maks_hari_hujan_beruntun",
    "total_hujan_bulan",
    "rainfall_intensity",
    "vegetation_moisture_index",
    "rainfall_elevation_ratio",
]

# Column-name candidates used when reading the prediction CSV / geojson so the
# app tolerates minor schema differences without code changes.
COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "kecamatan": ("kecamatan", "kecamatan_x", "kecamatan_y", "nama_kecamatan",
                  "kecamatan_normal", "objek", "district", "kec", "namobj", "name"),
    "kabupaten": ("kabupaten", "kab_kota", "kabupaten_normal", "kabupaten_kota",
                  "kota", "regency", "wadmkk", "city"),
    "probability": ("flood_probability", "probability", "probabilitas",
                    "proba", "risk_probability", "prob"),
    "category": ("risk_category", "kategori", "category", "risk", "risiko"),
    "latitude": ("lat", "latitude", "lintang", "y", "centroid_lat"),
    "longitude": ("lon", "lng", "longitude", "long", "bujur", "x", "centroid_lon"),
    "elevation": ("elevation", "elevasi", "dem", "altitude"),
    "slope": ("slope", "kemiringan_lereng", "kemiringan", "slope_deg", "lereng"),
    "ndvi": ("ndvi", "vegetation_index"),
    "soil_moisture": ("soil_moisture", "kelembapan_tanah", "soilmoist",
                      "kelembaban_tanah"),
    "landcover": ("landcover_class", "landcover", "tutupan_lahan"),
    "date": ("tanggal", "date", "datetime", "day"),
    "label": ("label", "target", "banjir", "flood", "y", "kejadian", "flood_event"),
}


# --------------------------------------------------------------------------- #
# Navigation menu
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    icon: str          # emoji fallback
    caption: str = ""


MENU: Final[tuple[MenuItem, ...]] = (
    MenuItem("dashboard", "Dashboard", "\U0001F3E0", "Ringkasan sistem"),
    MenuItem("prediction", "Prediksi 14 Hari", "\U0001F327", "Prakiraan risiko"),
    MenuItem("map", "Peta Risiko", "\U0001F4CD", "Peta interaktif"),
    MenuItem("analysis", "Analisis Model", "\U0001F4CA", "Evaluasi model"),
    MenuItem("history", "Riwayat Prediksi", "\U0001F553", "Riwayat prediksi"),
    MenuItem("dataset", "Dataset", "\U0001F5C4", "Data pelatihan"),
    MenuItem("about", "About", "ℹ️", "Informasi aplikasi"),
)

DEFAULT_PAGE: Final[str] = "dashboard"

FEATURE_SOURCES: Final[dict[str, str]] = {
    "avg_rainfall": "Open-Meteo / BMKG",
    "max_rainfall": "Open-Meteo / BMKG",
    "hujan_harian_max": "Open-Meteo / BMKG",
    "hujan_kumulatif_3h_max": "Open-Meteo",
    "total_hujan_bulan": "Open-Meteo / BMKG",
    "rainfall_intensity": "Open-Meteo",
    "jml_hari_hujan_lebat": "Open-Meteo / BMKG",
    "maks_hari_hujan_beruntun": "Open-Meteo / BMKG",
    "avg_temperature": "Open-Meteo",
    "elevation": "BIG / DEMNAS",
    "slope": "BIG / DEMNAS",
    "ndvi": "Landsat",
    "vegetation_moisture_index": "Landsat",
    "soil_moisture": "Citra satelit / model",
    "landcover_class": "Citra satelit / Landsat",
    "rainfall_elevation_ratio": "Fitur turunan",
    "lat": "BIG", "long": "BIG", "year": "Metadata", "month": "Metadata",
    "label": "Data kejadian historis",
}

PREDICTION_HISTORY_FILE: Final[Path] = OUTPUTS_DIR / "prediction_history.csv"

FORECAST_DAYS: Final[int] = OPEN_METEO.forecast_days
HEAVY_RAIN_THRESHOLD_MM: Final[float] = 20.0   # BMKG "hujan lebat" daily threshold
CACHE_TTL_SECONDS: Final[int] = 60 * 30        # 30 min cache for weather calls
