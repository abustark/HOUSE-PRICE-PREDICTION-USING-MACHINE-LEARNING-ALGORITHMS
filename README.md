# HOUSE PRICE PREDICTION USING MACHINE LEARNING ALGORITHMS

> 🌐 **Primary Live Web App & REST API (Vercel):** https://hpp-flax.vercel.app/  
> 📊 **Secondary Exploratory Dashboard (Streamlit):** https://house-price-prediction-using-machine-learning-algorithms-9qccn.streamlit.app/  
> 📄 **[`PROJECT-INFO.md`](PROJECT-INFO.md)** — Canonical machine-readable knowledge base (stack, metrics, exact versions, architecture, schemas) designed for AI models and resume/portfolio generation.  
> 📦 **GitHub Repository:** https://github.com/abustark/HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS  

Final year MCA project — **Basith AbuSyed** (`ABu`), The American College.

Predict the price of a house in Bengaluru from its basic property attributes using machine learning. Three regression models are trained with strictly fold-local preprocessing (zero data leakage) and evaluated on a held-out test split. 

The primary live production system is deployed on **Vercel Serverless** at **[hpp-flax.vercel.app](https://hpp-flax.vercel.app/)**, featuring a minimalist editorial Single Page Application (SPA) and FastAPI REST backend. An alternative 4-page Streamlit dashboard is also maintained for multi-page exploratory data analysis.

Models evaluated: **Linear Regression**, **Ridge Regression** ($\alpha=0.1$) and **Random Forest Regressor** (200 trees, $R^2 = 0.787$).

---

## Honest Positioning & Scope

- **Case study nature:** This is an academic/portfolio machine learning case study demonstrating reproducible regression modeling, strict leakage prevention, and dual-interface production serving.
- **Historical snapshot:** Trained on 13,320 listing records from a historical Bengaluru snapshot (approximately 2017–2020). Predictions reflect pricing patterns from that era.
- **Not a market quote:** These statistical estimates are not a live real-estate appraisal, a live market feed, or a substitute for a licensed valuer's appraisal.

---

## Primary Application Features ([hpp-flax.vercel.app](https://hpp-flax.vercel.app/))

- **Studio Slate & Cyber Indigo ML Workbench:** Modern dark canvas (`#0a0e17`) with elevated slate-blue cards (`#131c2e`, `#1e293b`), hairline borders (`rgba(99, 102, 241, 0.22)`), radiant Cyber Cyan → Electric Indigo gradient typography, and high-impact buttons.
- **In-App Light / Dark Mode Toggle:** Instant theme switcher (`#theme-toggle`) in the floating navbar, automatically respecting system preferences and persisting across sessions via `localStorage`. Includes a crisp daylight Studio Slate palette (`#f1f5f9`).
- **Universal 4-Tier Responsive Architecture:** Dynamically adapts across widescreen monitors, laptop split-screens, IDE side-panel browsers ($\le 960\text{px}$), tablets ($\le 768\text{px}$), and mobile devices ($\le 520\text{px}$) with zero horizontal overflow and fluid `minmax(0, 1fr)` tracks.
- **Mobile-First Touch Optimizations:** Touch-friendly preset pills ($\ge 38\text{px}$ tap targets), auto-zoom prevention on iOS form fields (`16px` font size), and smooth touch scrolling on data tables.
- **Upgraded Bento Metric Cards:** 4 high-contrast evaluation cards with top gradient accent bars, category badges (`★ Top Pick`, `5-Fold`, `MedAE`, `80% Interval`), 32px tabular monospace numbers, and hover elevation.
- **Automatic Lakh / Crore Scaling:** Values $\ge 100\text{ Lakh}$ scale automatically to **Crores** (e.g., $186.20\text{ Lakh} \to \text{₹1.86 Cr}$) with full rupee breakdown ($\approx \text{₹1,86,20,000}$), while values $< 100\text{ Lakh}$ remain formatted in **Lakhs**.
- **Interactive Developer API Workbench:** Test live `POST /predict`, `GET /models`, `GET /locations`, `GET /meta`, `GET /manifest`, and `GET /health` directly in the browser with real-time response previews, latency metrics (`Status: 200 OK (XXms)`), and clipboard copy.
- **Synchronized Area Control:** Dual-input stepper (10-sqft step) and slider (50-sqft step) across 100 to 10,000 sqft.
- **Calibrated 80% Error Intervals:** Conformal error intervals ($q_{0.10} - q_{0.90}$ log-residuals) derived from 5-fold cross-validation.
- **Domain Extrapolation Alerts:** Identifies unrepresented locations, unseen societies, or extreme property dimensions.
- **Multi-Model Side-by-Side Inference:** Direct comparison between Linear Regression, Ridge, and Random Forest.
- **Branded Footer with Social Icons:** High-contrast SVG icons for Portfolio (`abufolio.vercel.app`), GitHub repository, and LinkedIn, plus author copyright notice (`© 2026 Basith AbuSyed (ABu)`) and Streamlit secondary interface banner.

---

## Project Structure

```text
.
├── api/                        # FastAPI server + custom single-page UI
│   ├── server.py               # REST API v2.2.0 (also serves the UI)
│   └── static/                 # Minimalist SPA (index.html, app.js, style.css)
├── app/                        # Streamlit web app
│   ├── app.py                  # Prediction page (entry point)
│   └── pages/
│       ├── 1_Data_and_EDA.py   # Dataset exploration & correlation analysis
│       ├── 2_Model_Report.py   # Model comparison & diagnostic plots
│       └── 3_About.py          # Project overview & system details
├── data/
│   └── Bengaluru_House_Data.csv # 13,320 historical listings
├── src/
│   ├── encoding.py             # Feature engineering, schema & DataPreprocessor
│   ├── train.py                # Zero-leakage training + tuning + evaluation pipeline
│   └── predict.py              # Serialization loader, validation, and inference helpers
├── models/                     # Trained models + preprocessor bundle + metrics.json + manifest.json
├── artifacts/                  # 9 diagnostic ML charts (price distribution, residuals, etc.)
├── notebooks/                  # Executed 33-cell reproducibility notebook
├── tests/                      # Automated test suite (35 pytest tests + Playwright e2e)
├── vercel.json                 # Vercel serverless deployment routing
└── Dockerfile                  # Production container configuration
```

---

## Quick Start

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt # FastAPI + Streamlit + pytest + httpx + playwright

# 2. (Optional) Retrain models with zero data leakage
python src/train.py

# 3. Launch your preferred application interface:

# Option A: Primary Web App & REST API (FastAPI + SPA)
uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
# UI at http://127.0.0.1:8000, Swagger docs at http://127.0.0.1:8000/docs

# Option B: Secondary Exploratory Dashboard (Streamlit)
streamlit run app/app.py
# Opens at http://localhost:8501
```

---

## Dataset

- **File:** `data/Bengaluru_House_Data.csv`
- **Size:** 13,320 listings, 9 columns
- **Columns:** area type, availability, location, size, society, total area, bathrooms, balconies, price (Lakh INR)
- **Source:** [Bengaluru House price data](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data) on Kaggle (CC0 - public domain)
- **Note:** The data is a historical snapshot of listings from roughly 2017–2020.

---

## Data Preparation & Leakage Elimination

1. **Text Parsing & Unit Conversion:** `total_sqft` mixes square feet with text units (`Sq. Meter`, `Sq. Yards`, `Acres`, `Guntha`, `Ground`, `Perch`). Values are converted to square feet; range strings (`e.g., 2100 - 2850`) become their mathematical midpoint.
2. **Corrupted Row Removal:** 132 listings with physically impossible values (`sqft < 100` or `> 10,000`, `bath > 12`, `bhk > 14`, impossible density ratios) are filtered before modeling.
3. **Outer Holdout Split:** An untouched 30% hold-out test set (3,957 rows) is split **before** fitting any learned preprocessing statistics.
4. **Fold-Local Cross-Validation:** In all CV loops (tuning and evaluation), BHK median imputation, location frequencies, society frequencies, and StandardScaler transformations are fitted **strictly on that fold's training slice**. Validation folds never leak into preprocessing statistics.
5. **Schema Validation:** `area_type` is validated against 4 canonical classes (`Built-up Area`, `Carpet Area`, `Plot Area`, `Super built-up Area`). Invalid values return structured 422 HTTP validation errors.
6. **High-Cardinality Frequency Encoding:** `location` (242 recognized localities) and `society` are encoded based on training prevalence. Novel or missing values are cleanly handled.
7. **Target Transformation:** Prices are right-skewed; models train on $\log(1 + \text{price})$ and invert predictions via $\exp(\hat{y}) - 1$.

---

## Model Benchmark Results

Evaluated on the untouched held-out test set (3,957 listings):

| Model | $R^2$ (log) | MAE (log) | MAE (Lakh) | MedAE (Lakh) | RMSE (Lakh) | 5-Fold CV $R^2$ (log) |
|---|---|---|---|---|---|---|
| Linear Regression | 0.633 | 0.315 | ₹41.39 | ₹16.85 | ₹133.67 | 0.630 (±0.014) |
| Ridge Regression ($\alpha = 0.1$) | 0.633 | 0.315 | ₹41.39 | ₹16.85 | ₹133.67 | 0.630 (±0.014) |
| **Random Forest (200 trees, leaf ≥ 3)** | **0.787** | **0.224** | **₹29.32** | **₹11.07** | **₹80.83** | **0.779 (±0.007)** |

- **Top Model:** Random Forest explains ~79% of log-price variance. 5-fold CV score ($0.779$) closely matches test score ($0.787$).
- **Median Error:** MedAE is ₹11.07 Lakh, showing strong accuracy for typical residential properties.
- **Spatial Stress Test:** GroupKFold spatial cross-validation withholding entire locations achieves $R^2 = 0.719$.

---

## REST API

Served live from `https://hpp-flax.vercel.app` (FastAPI ASGI application):

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the primary minimalist single-page application |
| `/predict` | `POST` | Predict with `linear`, `ridge`, `random_forest`, or `all` with calibrated intervals and warnings |
| `/models` | `GET` | Parameters, test metrics, and CV scores for all models |
| `/locations` | `GET` | 242 recognized historical training locations |
| `/meta` | `GET` | Supported area types, domain constraints, schema version (`2.2.0`) |
| `/manifest` | `GET` | Provenance manifest (dataset hash, split parameters) |
| `/health` | `GET` | Liveness probe and loaded models count |
| `/docs` | `GET` | Interactive Swagger OpenAPI documentation |

### Example Request

```bash
curl -X POST https://hpp-flax.vercel.app/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "area_type": "Super built-up Area",
    "location": "Whitefield",
    "bhk": 3,
    "total_sqft": 1500,
    "bath": 3,
    "balcony": 2,
    "availability": "ready",
    "completion_year": 0,
    "model": "random_forest"
  }'
```

---

## Testing & Quality Gates

```bash
# Run the 35 automated pytest unit and integration tests
pytest -q

# Run Playwright end-to-end verification (in-memory headless Desktop & Mobile)
python tests/run_playwright_inspection.py
```

The test suites verify:
1. **Data parsing:** String cleanup, unit conversions (sqm, acres, sqyd, guntha), range midpoint parsing.
2. **Leakage safety:** Statistical proof confirming validation folds do not alter imputation medians or frequency maps.
3. **Domain validation:** Physical boundary enforcement ($100 \le \text{sqft} \le 10,000$, $1 \le \text{bath} \le 12$, $0 \le \text{bhk} \le 14$), structured 422 errors.
4. **API behavior:** Health check, metadata endpoints, and live response payloads.
5. **Model invariants:** Monotonicity (increasing sqft increases valuation ceteris paribus) and non-negative predictions.
6. **E2E UI & Responsiveness:** Desktop (`1440x900`) and Mobile (`390x844`) layout rendering, Lakh/Crore auto-scaling, and interactive API console queries. Note: Zero persistent screenshots are saved to the repo.

---

## Tech Stack

- **Python:** 3.11 / 3.13
- **Data & ML:** pandas, numpy, scikit-learn (`==1.9.1`), joblib
- **API & Serving:** FastAPI v2.2.0, Uvicorn, Vercel Serverless
- **Frontend:** Vanilla HTML5, CSS3, ES2022 JavaScript (zero framework bloat)
- **Secondary UI:** Streamlit multi-page dashboard
- **Testing:** pytest, HTTPX, Playwright
- **Dev Tooling:** Bun / Bunx (`bun`), Docker, GitHub Actions CI

---

## Disclaimer

Predictions are statistical estimates produced by machine learning models trained on historical 2017–2020 listings. They are designed for educational and portfolio demonstration purposes and are not a substitute for a professional valuation, bank appraisal, or current real-estate market data.
