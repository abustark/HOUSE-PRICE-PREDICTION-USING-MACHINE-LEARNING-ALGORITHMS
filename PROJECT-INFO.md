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
| Live site | https://house-price-prediction-using-machine-learning-algorithms-9qccn.streamlit.app/ (Streamlit) — FastAPI + custom UI runs locally at `http://localhost:8000` (docs at `/docs`) |
| Source | https://github.com/abustark/HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS |
| License | No code license; dataset CC0 public domain (Kaggle) |
| Status | Maintained (2026, main branch, Docker + CI green) |

**One-liners (pick by context):**
- *10 words:* Bengaluru price predictor comparing three regression models, dual UI.
- *25 words:* House-price predictor trained on 13,320 Bengaluru listings comparing Linear, Ridge and Random Forest, served via FastAPI REST plus Streamlit and vanilla SPA frontends.
- *Elevator pitch:* Predict Bengaluru house prices from 8 basic features. Three regression models train on the same 13,320-row Kaggle dataset with log-target + frequency encoding, the best (Random Forest, R² 0.787) ships with two interfaces — a 4-page Streamlit app and a FastAPI-served offline-capable SPA — plus Swagger docs, Docker image, 7 pytest tests and CI.

---

## 2. Tech stack — exact versions

| Category | Technology | Version | Role |
|---|---|---|---|
| Runtime | Python | 3.11 (`python:3.11-slim`, `setup-python 3.11`) | Training + serving |
| API framework | FastAPI | unpinned (`requirements.txt`) | REST API (`api/server.py` v2.0.0) |
| ASGI server | uvicorn | unpinned | `api.server:app :8000` locally + Docker CMD |
| Data | pandas | unpinned | CSV load, cleaning, EDA |
| Numerics | numpy | unpinned | Transforms, log1p/expm1 |
| ML | scikit-learn | **==1.9.1** (pinned — models pickled by 1.9.1) | LinearRegression, Ridge, RandomForestRegressor, LabelEncoder, StandardScaler |
| Persistence | joblib | unpinned | `models/*.joblib` (`compress=3`) |
| UI (A) | Streamlit | unpinned (`requirements-dev.txt`) | 4-page app (`app/app.py` + `pages/`) on `:8501` |
| UI (B) | Vanilla HTML/CSS/JS | no framework, no CDN, offline-capable | `static/index.html + app.js + style.css` served by FastAPI |
| Plotting | matplotlib | unpinned | 8 `artifacts/*.png` charts |
| Plotting | seaborn | unpinned | Dev EDA support |
| Testing | pytest | unpinned | `tests/test_predict.py`, 7 tests |
| CI | GitHub Actions | `ubuntu-latest` | `compileall + pytest -q` on push/PR |
| Deployment | Docker | `FROM python:3.11-slim` | `docker build -t house-price`, `EXPOSE 8000` |
| Deployment | Vercel | lean `requirements.txt` (500 MB cap) + root `main.py` re-export | Serverless entry (`hpp-ml` project link, no public URL) |
| Notebook | Jupyter | `notebooks/House_Price_Prediction.ipynb`, 33 cells, executed | Pipeline reproduction |
| VCS | Git / GitHub | main branch | ~2023-05-24 to 2026-10-16 history |

---

## 3. Architecture (how it fits together)

- **Single training pipeline, dual serving:** `data/Bengaluru_House_Data.csv` → `src/train.py` (+ `src/encoding.py`) → `models/{linear,ridge,random_forest,encoders}.joblib` + `models/metrics.json` + 8 `artifacts/*.png` → `src/predict.py` (`load_artifacts/encode_row/predict_all`) → (A) Streamlit `app/app.py` or (B) FastAPI `api/server.py` serving JSON + `static/` SPA.
- **Feature engineering shared:** `src/encoding.py` is the single source — `total_sqft` text→sqft (unit map + range midpoint), `size`→`bhk` (Studio 0 / RK 1 / digits), `availability`→`completion_year` (Ready→0, `DD-Mon`→20YY), `area_type` label-encoded, `location/society` frequency-encoded from train only (unseen→0), `log1p(price)` target, `StandardScaler` for Linear/Ridge only (saved in encoders).
- **FastAPI loads once:** `load_artifacts()` at import; `GET /` serves `static/index.html`, `/static` mounted; `POST /predict` supports `linear/ridge/random_forest/all`, returns Lakh (2dp) + INR + `all_models` + `unseen_society` flag.
- **Docker is lean:** copies only `requirements.txt + src/ + api/ + models/`; excludes via `.dockerignore` (`app, artifacts, data, docs, images, notebooks, tests`).
- **Vercel needs root entry:** `main.py` re-exports `api.server:app` because `app/app.py` (Streamlit) would otherwise shadow Vercel's entrypoint search.

---

## 4. Feature inventory (technical depth for bullet generation)

1. **3-model comparison** — LinearRegression baseline, Ridge (`α=0.1` chosen from 0.1/1/10/100 via 5-fold CV MAElog), RandomForest (`n_estimators=200, min_samples_leaf=3, random_state=2` from 100/200×3/5/10 grid) — all on 70/30 split `random_state=2` (train 9,282 / test 3,979).
2. **8 features** — `total_sqft, bath, balcony, bhk (0=studio/RK, 0–14), area_type, completion_year (0=ready), location (~1,305 raw), society (~2,688 raw, 41% missing → `__MISSING__`)`.
3. **Robust cleaning** — 59 rows (0.4%) dropped (`sqft<100|>10000`, `bath>12`, `bhk>14`); unit conversion (Sq.Meter×10.7639, Sq.Yards×1.19599, Acres×43560, perch/guntha/ground); median-fill BHK, 0-fill bath/balcony.
4. **Encoding rationale** — label for low-cardinality `area_type`, frequency for high-cardinality `location/society` (avoids fake order of label + sparse explosion of one-hot), `log1p` for right-skewed prices.
5. **REST API (6 routes)** — `POST /predict`, `GET /models|/locations|/meta|/health`, `GET /` UI, `GET /docs` Swagger; request caps (`bhk 0–14, sqft>0≤50000, bath 0–20, balcony 0–10, year 0–2035`).
6. **Streamlit app (4 pages)** — Predict (model radio + R²/MAE sidebar, `Whitefield`-style selects, all-models dataframe, unseen-society notice), Data & EDA (13,320/9/1,305/median-72 metrics + 4 charts), Model Report (test + 5-fold CV tables, 4 diagnostic charts), About (MCA context + setup).
7. **Vanilla SPA** — dark theme (`#0b1020/#141c3a`, accent `#4f7cff/#22d3ee`), model chips with R² badges, Lakh (en-IN 2dp) + INR formatting, comparison table, `520px+1fr→1fr @860px` grid, no build step.
8. **8 diagnostic charts** — `price_distribution, price_vs_sqft, price_by_bhk, top_locations, model_comparison, feature_importance, actual_vs_predicted, residuals` — reused in Streamlit + notebook + 23-slide `docs/House Price.pptx`.
9. **Executed notebook** — 33 cells, 7 sections (data → exploration → cleaning → split/encoding → train → diagnostics → save), 0 errors.
10. **Docker + CI** — `python:3.11-slim` image (`PYTHONUNBUFFERED=1`, `:8000`), GitHub Actions `compileall + pytest -q`.
11. **Input validation** — FastAPI Pydantic caps + train-time sane filters + predict-time `max(0,price)` + blank/placeholder (`"", "N/A", "nan"`) → missing handling.
12. **Committed artifacts** — `models/` (RF 13.92 MB, encoders 23 KB, linear/ridge <1 KB, `metrics.json` 84 lines) enables no-secret Streamlit Cloud deploy.

---

## 5. Quality & testing

- **`pytest -q` — 7 tests, 1 file (`tests/test_predict.py`):** artifact shapes (3 models + `best_model` + finite R²/MAE/RMSE), finite-price prediction, plausibility (2BHK 1200 Whitefield within 5–1000 Lakh), unseen-society no-crash, blank/placeholder-as-missing, monotonicity (700/1BHK < 2400/4BHK), API route import (`/, /health, /models, /locations, /meta, /predict`).
- **CI:** `.github/workflows/ci.yml` — `setup-python 3.11 + pip install -r requirements.txt + compileall(src,app,api) + pytest -q` on push/PR.
- **No lint/typecheck:** no ruff/flake8/mypy/black config; only `compileall`.
- **Docs in-repo:** `README.md` (236 lines), `docs/commits.md` (v2.0.x→2.1.x history), `docs/House Price.pptx` (23 slides), `AGENTS.md`, this sheet.

---

## 6. Scale metrics (quantifiables)

- **50** git-tracked files; **~1,382 lines** across **11** Python files (`train.py` 444, `server.py` 149, `app.py` 162, `encoding.py` 119, `predict.py` 95, 3 pages 290, `test_predict.py` 97, `main.py` 25, `__init__.py` 1).
- **3 models** + **6 API routes** + **8 features** + **8 charts** + **4 Streamlit pages** + **1 SPA**.
- **13,320 rows × 9 cols** dataset (~0.88 MB); train 9,282 / test 3,979; 59 rows dropped.
- **7 pytest tests**; **5-fold CV** for Ridge-α + RF grid + final reporting.
- **Best R² (log) 0.787**, MAE (log) 0.231, MAE 31.41 Lakh, RMSE 89.35 Lakh (RF); Linear/Ridge R² 0.644 (identical — small-α Ridge ≈ OLS).
- **33-cell** notebook; **23-slide** deck; **13.92 MB** RF artifact (largest file).

---

## 7. Engineering practices worth mentioning

- Train-only encoder fitting — frequency maps + scaler fit on train, applied to test/predict (no leakage).
- Log-target for skew — `log1p` training, `expm1` inversion for display; dual log/Lakh metrics reported.
- Scale-aware modeling — StandardScaler for Linear/Ridge, raw features for scale-free Random Forest.
- Frequency encoding for high-cardinality categoricals — documented trade-off vs label/one-hot.
- Small, justified grids — Ridge-α (4 values) + RF (2×3) via 5-fold CV log-MAE, best by R²log.
- CV≈test generalisation check — RF CV R² 0.776 vs test 0.787 explicitly reported.
- Committed, versioned artifacts — retrain via `python src/train.py`, serve without retraining.
- Lean runtime split — `requirements.txt` (Vercel/Docker) vs `requirements-dev.txt` (Streamlit/plots/tests).
- Monotonicity + plausibility tests — bigger-house-costs-more invariant in the suite.
- Dual-interface parity — Streamlit and SPA share `src/predict.py`, same 8-feature contract.

---

## 8. ATS / keyword tags

house price prediction, regression, Bengaluru housing, Random Forest, Ridge Regression, Linear Regression, hyperparameter tuning, cross-validation, feature engineering, frequency encoding, label encoding, StandardScaler, log transformation, EDA, pandas, numpy, scikit-learn, joblib, FastAPI, REST API, Swagger, Streamlit, Docker, GitHub Actions, CI, pytest, matplotlib, seaborn, Jupyter, data cleaning, model evaluation (R², MAE, RMSE), Python 3.11, Git/GitHub.

---

## 9. Ready-made examples (regenerate freely — facts only)

**Resume bullets:**
- Built a Bengaluru house-price predictor (Python 3.11, scikit-learn 1.9.1, pandas/numpy) on 13,320 Kaggle listings, comparing Linear, Ridge (α=0.1 via CV) and Random Forest (200 trees, leaf≥3 via 2×3 grid) on a 70/30 split.
- Engineered an 8-feature pipeline (sqft unit conversion + range midpoints, BHK extraction, completion-year parsing, label + frequency encoding, `log1p` target, StandardScaler for linear models) with 59-row sane filtering.
- Achieved best R² (log) 0.787 / MAE (log) 0.231 / RMSE 89.35 Lakh with Random Forest (CV 0.776 ≈ test, generalises); Linear/Ridge both 0.644.
- Served via dual interfaces sharing `src/predict.py` — FastAPI REST (6 routes + Swagger + vanilla offline SPA) and 4-page Streamlit app (predict, EDA, model report, about) — plus Docker image and Vercel-ready lean runtime.
- Wrote 7 pytest tests (artifact shapes, plausibility, unseen-society, monotonicity, route imports) with GitHub Actions CI; produced 8 diagnostic charts, a 33-cell executed notebook and a 23-slide deck.

**Resume project paragraph (3 lines):** ML house-price predictor trained on 13,320 Bengaluru listings (Python, scikit-learn, pandas), comparing Linear, Ridge and tuned Random Forest (best R² 0.787) with log-target + frequency-encoding pipeline. Served via FastAPI REST + vanilla SPA and a 4-page Streamlit app from shared artifacts, with Docker, 7 pytest tests and GitHub Actions CI. Includes 8 diagnostic charts, an executed 33-cell notebook and a 23-slide presentation.

**Portfolio card (title + ~30 words + stack):** Bengaluru House Prices — describe any flat (location, BHK, sqft, baths) and compare three models' predictions side-by-side with R² scores and diagnostics. Dual FastAPI + Streamlit UI, Docker-ready. `Python · scikit-learn · FastAPI · Streamlit · Docker`

**Tone/length guide for the AI:** Frame impact as user capability + the hardest engineering problem solved (no business KPIs). Resume = 3–5 dense bullets or one 3-line paragraph, lead with verbs and numbers, keywords from §8; portfolio = friendly paragraph + model table + UI links + stack chips, mention the 13,320-row dataset, the 0.787 R² winner, and the dual UI; never claim XGBoost or a code license — none exist; the Streamlit site is live, while the FastAPI + SPA face is local-only.
