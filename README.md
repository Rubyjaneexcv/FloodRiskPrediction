# Flood Risk Prediction System — Jabodetabek

Prediksi risiko banjir tingkat kecamatan di wilayah Jabodetabek menggunakan
algoritma **Random Forest** berbasis data meteorologi (Open-Meteo) dan
geospasial. Antarmuka profesional bergaya dashboard (ArcGIS / Power BI /
Material Design) dibangun dengan Streamlit + CSS kustom.

## ✨ Fitur

- **Dashboard** — KPI model, status sistem, preview peta, Top-10 risiko, distribusi probabilitas.
- **Prediction (14 Days)** — pilih Kabupaten → Kecamatan → *Generate Forecast*. Menarik prakiraan 14 hari Open-Meteo, feature engineering, prediksi, tabel + grafik + timeline.
- **Flood Risk Map** — peta Folium interaktif dengan slider Hari ke-1..14; warna polygon & tooltip berubah per hari.
- **Model Analysis** — Overview, Feature Importance, Permutation Importance, SHAP, ROC Curve, Confusion Matrix.
- **About** — informasi skripsi, penulis, pembimbing, versi.

## 📁 Struktur Proyek

```
FloodRiskPrediction/
├── app.py                  # Entry point + router
├── config.py               # Konfigurasi terpusat (tema, path, threshold)
├── requirements.txt
├── .streamlit/config.toml
├── assets/
│   └── style.css           # Tema profesional (CSS kustom)
├── components/             # Komponen reusable
│   ├── header.py  sidebar.py  footer.py
│   ├── cards.py            # MetricCard, StatusCard, RiskCard, legend, info row
│   └── section.py          # SectionTitle, card container
├── views/                  # Halaman
│   ├── dashboard.py  prediction.py  mapping.py  analysis.py  about.py
├── utils/                  # Logika
│   ├── model_loader.py  data_loader.py  geo_utils.py
│   ├── weather_api.py   feature_engineering.py  predictor.py  plotting.py  styling.py
├── models/                 # best_random_forest.pkl, feature_list.pkl  (LETAKKAN DI SINI)
├── data/                   # jabodetabek_final.geojson, Flood_Risk_Prediction.csv  (LETAKKAN DI SINI)
├── outputs/                # metrics.json (opsional), ekspor
└── scripts/
    └── generate_sample_data.py   # membuat data sintetis untuk uji coba
```

## 🚀 Menjalankan

```bash
# 1. (opsional) buat virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3a. gunakan artefak asli Anda
#     salin file berikut ke tempatnya:
#       models/best_random_forest.pkl
#       models/feature_list.pkl
#       data/jabodetabek_final.geojson
#       data/Flood_Risk_Prediction.csv
#
# 3b. ATAU buat data sintetis untuk mencoba aplikasi lebih dulu:
python scripts/generate_sample_data.py

# 4. jalankan
streamlit run app.py
```

## 🔧 Menaruh Artefak Model

- `models/best_random_forest.pkl` — model Random Forest terlatih (`joblib`/`pickle`).
- `models/feature_list.pkl` — daftar nama fitur **urut** sesuai saat pelatihan. Ini adalah sumber kebenaran; pipeline feature engineering otomatis menyelaraskan output ke daftar ini.
- `data/jabodetabek_final.geojson` — batas kecamatan (properti: nama kecamatan, kabupaten/kota, dan opsional elevation/ndvi/soil_moisture).
- `data/Flood_Risk_Prediction.csv` — hasil/gabungan probabilitas per kecamatan (untuk Dashboard). Kolom fleksibel (lihat `config.COLUMN_ALIASES`).
- `outputs/metrics.json` *(opsional)* — override metrik & confusion matrix:
  ```json
  {"accuracy":0.925,"precision":0.914,"recall":0.857,"f1":0.884,"roc_auc":0.974,
   "confusion_matrix":[[704,98],[59,560]]}
  ```

## 🧠 Feature Engineering

`utils/feature_engineering.py` menurunkan seluruh fitur model dari prakiraan
Open-Meteo + atribut statis kecamatan, lalu **reindex** ke `feature_list.pkl`
sehingga kolom selalu cocok dengan model. Fitur yang tidak dapat dihitung diisi
dari atribut statis atau 0 — jadi aplikasi tetap berjalan meski daftar fitur
Anda sedikit berbeda.

## 🌐 Open-Meteo

Gratis, tanpa API key. Endpoint `forecast` (harian + hujan per jam), 14 hari,
zona waktu `Asia/Jakarta`. Respons di-cache 30 menit.

## 📝 Catatan

- Nama, NIM, universitas, dan pembimbing dapat diubah di `config.py` (`AppInfo`).
- Ambang kategori risiko dapat diatur di `config.py` (`RISK_LEVELS`).
