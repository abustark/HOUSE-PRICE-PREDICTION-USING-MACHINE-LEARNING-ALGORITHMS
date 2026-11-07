# HOUSE PRICE PREDICTION USING MACHINE LEARNING ALGORITHMS

> 📄 **[`PROJECT-INFO.md`](PROJECT-INFO.md)** — machine-readable project facts (stack + exact versions, architecture, metrics, keyword tags) designed so an AI can generate resume entries and portfolio project summaries from it alone.

> **Live demo (Streamlit):** https://house-price-prediction-using-machine-learning-algorithms-9qccn.streamlit.app/
> **Repo:** https://github.com/abustark/HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS

Final year MCA project — **Basith AbuSyed**, The American College.

Predict the price of a house in Bengaluru from its basic property attributes using
machine learning. Three regression models are trained with strictly fold-local
preprocessing (zero data leakage) and evaluated on a held-out test split. A 4-page
Streamlit web app and a FastAPI REST backend with a custom SPA allow comparing
predictions side-by-side with calibrated historical error ranges and domain warnings.

Models evaluated: **Linear Regression**, **Ridge Regression** and **Random
Forest Regressor**.

---

## Honest Positioning & Scope

- **Case study nature:** This is an academic/portfolio machine learning case study
  demonstrating reproducible regression modeling, leakage prevention, and dual-interface
  serving.
- **Historical snapshot:** Trained on 13,320 listing records from a historical snapshot
  (approximately 2017–2020). Predictions reflect pricing patterns from that era.
- **Not a market quote:** These statistical estimates are not a live real-estate appraisal,
  a market price feed, or a substitute for a licensed valuer's appraisal.

---

## Demo

The main page of the app predicts the price for a house you describe
(location, size, area, bathrooms, balconies, availability, society). The sidebar
lets you switch between the three trained models and displays test-set and cross-validation
scores. The prediction includes an 80% empirical historical error range and domain
extrapolation warnings.

```
Prediction:  Predicted price: ₹90.38 Lakh (≈ ₹9,038,000)
Historical error range (80% coverage): ₹68.40 Lakh – ₹119.50 Lakh

All models on this input:
  Model              Predicted price (Lakh)   R² (log price)
  Linear Regression          61.13                 0.644
  Ridge Regression           61.13                 0.644
  Random Forest              90.38                 0.787
```

## Project structure

```
.
├── api/                        # FastAPI server + custom single-page UI
│   ├── server.py               # REST API (also serves the UI)
│   └── static/                 # index.html, app.js, style.css
├── app/                        # Streamlit web app
│   ├── app.py                  # prediction page (entry point)
│   └── pages/
│       ├── 1_Data_and_EDA.py   # dataset exploration
│       ├── 2_Model_Report.py   # model comparison and diagnostics
│       └── 3_About.py          # project overview and setup
├── data/
│   └── Bengaluru_House_Data.csv
├── src/
│   ├── encoding.py             # shared feature engineering, schema & DataPreprocessor
│   ├── train.py                # leakage-safe training + tuning + evaluation pipeline
│   └── predict.py              # model loading + validation + inference helpers
├── models/                     # trained models + encoders + metrics.json + manifest.json
├── artifacts/                  # charts used by the apps and the notebook
├── notebooks/                  # executed notebook reproducing the pipeline
├── images/                     # static images
├── docs/                       # presentation + commit log
├── tests/                      # pytest suite (parsing, leakage, validation, routes)
└── Dockerfile                  # container for the FastAPI app
```

## Quick start

```bash
# 1. create a virtual environment and install the dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt # everything: FastAPI + Streamlit + plots + pytest + httpx

# 2. (optional) retrain the models from the raw data
python src/train.py

# 3. start the web app you like

# option A - Streamlit (4 pages: predict, EDA, report, about)
streamlit run app/app.py          # opens at http://localhost:8501

# option B - FastAPI + custom UI (also the REST API)
uvicorn api.server:app --host 0.0.0.0 --port 8000
# UI at http://localhost:8000, interactive API docs at http://localhost:8000/docs
```

## Dataset

- **File:** `data/Bengaluru_House_Data.csv`
- **Size:** 13,320 listings, 9 columns
- **Columns:** area type, availability, location, size, society,
  total area, bathrooms, balconies, price (Lakh INR)
- **Source:**
  [Bengaluru House price data](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data)
  on Kaggle (CC0 - public domain)
- **Note:** the data is a historical snapshot of listings from roughly
  2017-2020, so prices reflect that period.

## Data Preparation & Leakage Elimination

The raw data has several quirks handled by the pipeline:

- `total_sqft` is stored as text and mixes square feet with other units
  ("34.46Sq. Meter", "1100Sq. Yards", "5Acres"). Values are converted to
  square feet; ranges become their midpoint.
- **Corrupted rows filtered:** Records with physically impossible values
  (an 11 sqft 3-bedroom flat, a 52,000 sqft house, bath > 12) are filtered
  before modeling.
- **Outer train/test split:** An untouched 30% hold-out test set is split
  **before** fitting any learned preprocessing statistics.
- **Fold-local cross-validation:** In all CV loops (tuning and evaluation),
  BHK median imputation, location frequencies, society frequencies, and
  StandardScaler transformations are fitted **strictly on that fold's training slice**.
  Validation fold records never contaminate preprocessing statistics.
- **Explicit categorical schema:** `area_type` is checked against the 4 canonical
  categories (`Built-up Area`, `Carpet Area`, `Plot Area`, `Super built-up Area`).
  Unsupported categories return explicit validation errors rather than silent fallbacks.
- **High-cardinality frequencies:** `location` (~1,300 values) and `society` (~2,700 values)
  are frequency-encoded based on training prevalence (unseen values map to 0.0).
- **Target transformation:** Prices are right-skewed; models are trained on
  `log(1 + price)` and inverted back to Lakh INR (`expm1`) for evaluation and display.

## Model Results

Metrics on the untouched held-out test set:

| Model | R² (log) | MAE (log) | MAE (Lakh) | MedAE (Lakh) | RMSE (Lakh) | 5-fold CV R² (log) |
|---|---|---|---|---|---|---|
| Linear Regression | 0.633 | 0.315 | ₹41.39 | ₹16.85 | ₹133.67 | 0.630 (±0.014) |
| Ridge Regression (α = 0.1) | 0.633 | 0.315 | ₹41.39 | ₹16.85 | ₹133.67 | 0.630 (±0.014) |
| **Random Forest (200 trees, leaf ≥ 3)** | **0.787** | **0.224** | **₹29.32** | **₹11.07** | **₹80.83** | **0.779 (±0.007)** |

- **Random Forest is the top-performing model:** explains ~79% of log-price variance.
  The cross-validation score (0.779) matches test performance (0.787), confirming strong generalization with zero fold leakage.
- **Scale difference:** MAE (log) is 0.224 (typical relative error ~22–24%). Lakh-scale MAE
  (₹29.32 Lakh) and RMSE (₹80.83 Lakh) reflect skewness from high-value luxury listings.
- **Grouped stress test:** GroupKFold evaluation on unseen locations yields R² = 0.719,
  illustrating the expected penalty when predicting in entirely novel localities.

## REST API

The FastAPI app (`api/server.py`) exposes the models and serves the single-page UI:

| Endpoint | Description |
|---|---|
| `POST /predict` | Predict with `linear`, `ridge`, `random_forest`, or `all` with calibrated intervals and warnings |
| `GET /models` | Names, parameters, test and CV scores for all models |
| `GET /locations` | All known historical training locations |
| `GET /meta` | Supported area types, domain constraints, schema version |
| `GET /manifest` | Reproducibility and provenance manifest (dataset hash, split parameters) |
| `GET /health` | Liveness and loaded models check |
| `GET /docs` | Interactive Swagger OpenAPI documentation |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "area_type": "Super built-up Area",
    "location": "Whitefield",
    "bhk": 3,
    "total_sqft": 1500,
    "bath": 3,
    "balcony": 2,
    "completion_year": 0,
    "model": "all"
  }'
```

## Docker

```bash
docker build -t house-price .
docker run -p 8000:8000 house-price
# UI + API at http://localhost:8000
```

## Tests and CI

```bash
pytest -q
```

The test suite covers:
1. **Data parsing:** range parsing, area unit conversions (sqm, acres, sqyd), malformed inputs.
2. **Leakage safety:** regression tests confirming validation rows do not affect learned frequencies or imputation medians.
3. **Domain validation:** boundary enforcement (100–10,000 sqft, 1–12 bath, 0–5 balcony, 0–14 BHK), extrapolation warnings, unsupported categories.
4. **API behavior:** health check, metadata endpoints, structured 422 error responses on invalid data.
5. **Model invariants:** plausibility, monotonicity (larger house costs more), finite outputs.

GitHub Actions runs the test suite on every push and pull request.

## Tech stack

- Python 3.11
- pandas, numpy
- scikit-learn (LinearRegression, Ridge, RandomForestRegressor, KFold, GroupKFold, StandardScaler)
- joblib (model persistence)
- Streamlit + FastAPI / uvicorn (dual web interfaces)
- matplotlib (diagnostic charts)
- pytest + httpx + GitHub Actions (CI & testing) + Docker (deployment)

## Disclaimer

Predictions are estimates produced by statistical machine learning models trained on
historical 2017–2020 listings. They are not a substitute for a professional property
appraisal or current real-estate market data.
