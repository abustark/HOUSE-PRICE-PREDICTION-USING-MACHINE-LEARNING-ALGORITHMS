# House Price Prediction — Project Info Sheet (source of truth for resume & portfolio summaries)

> **Purpose:** every fact an AI (or a person) needs to write a resume "Projects"
> entry or a portfolio project summary for House Price Prediction — without reading the code.
> Generate prose from these facts only; do not invent beyond them. Tune length
> and tone to the target (resume = dense & quantified; portfolio = readable &
> story-driven).

---

## 1. Project identity

| Field | Value |
|---|---|
| Name | **House Price Prediction Using Machine Learning Algorithms (Bengaluru)** |
| Type | Machine-learning + full-stack web application (regression + FastAPI + Streamlit + vanilla SPA) |
| Author | Basith AbuSyed / ABu (`abustark` on GitHub) — Final-year MCA, The American College |
| Live site | https://house-price-prediction-using-machine-learning-algorithms-9qccn.streamlit.app/ (Streamlit) — https://hpp-flax.vercel.app/ (FastAPI + custom SPA deployed on Vercel) |
| Source | https://github.com/abustark/HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS |
| License | No code license; dataset CC0 public domain (Kaggle) |
| Status | Maintained (2026, main branch, Docker + CI green) |

**One-liners (pick by context):**
- *10 words:* Bengaluru price predictor comparing three regression models, dual UI.
- *25 words:* House-price predictor trained on 13,320 Bengaluru listings comparing Linear, Ridge and Random Forest with leakage-free preprocessing, dual web UI and FastAPI.
- *Elevator pitch:* Predict Bengaluru house prices from 8 features. Three regression models train on the same 13,320-row Kaggle dataset with log-target + fold-local frequency encoding, the best (Random Forest, R² 0.787) ships with two interfaces — a 4-page Streamlit app and a FastAPI-served offline-capable SPA — plus Swagger docs, Docker image, expanded pytest test suite and CI.

---

## 2. Tech stack — exact versions

| Category | Technology | Version | Role |
|---|---|---|---|
| Runtime | Python | 3.11 (`python:3.11-slim`, `setup-python 3.11`) | Training + serving |
| API framework | FastAPI | unpinned (`requirements.txt`) | REST API (`api/server.py` v2.1.0) |
| ASGI server | uvicorn | unpinned | `api.server:app :8000` locally + Docker CMD |
| Data | pandas | unpinned | CSV load, cleaning, EDA |
| Numerics | numpy | unpinned | Transforms, log1p/expm1 |
| ML | scikit-learn | **==1.9.1** (pinned — models pickled by 1.9.1) | LinearRegression, Ridge, RandomForestRegressor, StandardScaler, GroupKFold, KFold |
| Persistence | joblib | unpinned | `models/*.joblib` (`compress=3`) |
| UI (A) | Streamlit | unpinned (`requirements-dev.txt`) | 4-page app (`app/app.py` + `pages/`) on `:8501` |
| UI (B) | Vanilla HTML/CSS/JS | no framework, no CDN, offline-capable | `static/index.html + app.js + style.css` served by FastAPI |
| Plotting | matplotlib | unpinned | Diagnostic charts in `artifacts/` |
| Plotting | seaborn | unpinned | Dev EDA support |
| Testing | pytest, httpx | unpinned | 3 test files in `tests/` covering parsing, leakage, validation, routes |
| CI | GitHub Actions | `ubuntu-latest` | `compileall + pytest -q` on push/PR |
| Deployment | Docker | `FROM python:3.11-slim` | `docker build -t house-price`, `EXPOSE 8000` |
| Deployment | Vercel | lean `requirements.txt` (500 MB cap) + root `main.py` re-export | Serverless entry (`hpp-ml` project link, no public URL) |
| Notebook | Jupyter | `notebooks/House_Price_Prediction.ipynb`, 33 cells, executed | Pipeline reproduction |
| VCS | Git / GitHub | main branch | ~2023-05-24 to 2026-11-06 history |

---

## 3. Architecture (how it fits together)

- **Single training pipeline, dual serving:** `data/Bengaluru_House_Data.csv` → `src/train.py` (+ `src/encoding.py`) → `models/{linear,ridge,random_forest,encoders}.joblib` + `models/metrics.json` + `models/manifest.json` + `artifacts/*.png` → `src/predict.py` (`load_artifacts/predict_detailed`) → (A) Streamlit `app/app.py` or (B) FastAPI `api/server.py` serving JSON + `static/` SPA.
- **Zero-leakage preprocessing:** `DataPreprocessor` in `src/encoding.py` encapsulates BHK median imputation, frequency maps for location/society, and StandardScaler. Inside CV folds, all statistics are fitted strictly on that fold's training slice.
- **FastAPI loads once:** `load_artifacts()` validates bundle schema at import; `GET /` serves `static/index.html`, `/static` mounted; `POST /predict` validates domain constraints, returns Lakh (2dp) + INR + 80% historical error range + warnings.
- **Docker is lean:** copies only `requirements.txt + src/ + api/ + models/`; excludes via `.dockerignore`.

---

## 4. Feature inventory (technical depth for bullet generation)

1. **3-model comparison** — LinearRegression baseline, Ridge (`α=0.1` chosen from 0.1/1/10/100 via fold-local CV), RandomForest (`n_estimators=200, min_samples_leaf=3, random_state=2`) — on 70/30 split `random_state=2` (train 9,282 / test 3,979).
2. **8 features** — `total_sqft, bath, balcony, bhk (0=studio/RK, 0–14), area_type, completion_year (0=ready), location (~1,305 raw), society (~2,688 raw, 41% missing → `__MISSING__`)`.
3. **Robust cleaning & bounds** — corrupted rows filtered (`sqft<100|>10000`, `bath>12`, `bhk>14`); unit conversion (Sq.Meter, Sq.Yards, Acres, perch/guntha/ground); median-fill BHK learned from train only.
4. **Encoding rationale** — label/schema for low-cardinality `area_type`, frequency for high-cardinality `location/society`, `log1p` for right-skewed prices.
5. **REST API (7 routes)** — `POST /predict`, `GET /models|/locations|/meta|/manifest|/health`, `GET /` UI, `GET /docs` Swagger; strict domain validation (100–10,000 sqft, bath 1–12, balcony 0–5).
6. **Streamlit app (4 pages)** — Predict (model radio + R²/MAE sidebar, presets, all-models dataframe, historical error range, unseen notices), Data & EDA, Model Report (test + fold-local CV + GroupKFold stress test tables, diagnostic charts), About.
7. **Vanilla SPA** — dark theme, model chips with R² badges, presets, historical error range box, extrapolation warning box, comparison table.
8. **Diagnostic charts** — `price_distribution, price_vs_sqft, price_by_bhk, top_locations, model_comparison, feature_importance, permutation_importance, actual_vs_predicted, residuals`.
9. **Docker + CI** — `python:3.11-slim` image, GitHub Actions `compileall + pytest -q`.
10. **Historical Error Calibration** — 80% coverage conformal log-residual interval from CV folds, evaluated against hold-out test set.

---

## 5. Quality & testing

- **Test suite (`pytest -q`):**
  - `tests/test_predict.py`: model loading, plausibility, monotonicity, detailed predictions.
  - `tests/test_parsing_cleaning_leakage.py`: unit conversions, missing handling, fold-local leakage regression proof.
  - `tests/test_validation_and_api.py`: physical boundary constraints, 422 responses on invalid inputs, extrapolation warning generation, API endpoints.
- **CI:** `.github/workflows/ci.yml` — `setup-python 3.11 + pip install -r requirements.txt pytest httpx + compileall + pytest -q` on push/PR.

---

## 6. Scale metrics (quantifiables)

- **3 models** + **7 API routes** + **8 features** + **9 charts** + **4 Streamlit pages** + **1 SPA** + **35 pytest tests**.
- **13,320 rows × 9 cols** dataset; train 9,231 / test 3,957; 132 corrupted rows filtered.
- **Best R² (log) 0.787**, MAE (log) 0.224, MAE 29.32 Lakh, MedAE 11.07 Lakh, RMSE 80.83 Lakh (RF); Linear/Ridge R² 0.633.
- **5-fold CV** for Ridge-α + RF tuning + final reporting with strictly fold-local preprocessing (zero data leakage). Group-CV R² 0.719 on unseen locations.

---

## 7. ATS / keyword tags

house price prediction, regression, Bengaluru housing, Random Forest, Ridge Regression, Linear Regression, cross-validation, fold-local preprocessing, data leakage prevention, feature engineering, frequency encoding, StandardScaler, log transformation, conformal error calibration, EDA, pandas, numpy, scikit-learn, joblib, FastAPI, REST API, Swagger, Streamlit, Docker, GitHub Actions, CI, pytest, matplotlib, Python 3.11, Git/GitHub.
