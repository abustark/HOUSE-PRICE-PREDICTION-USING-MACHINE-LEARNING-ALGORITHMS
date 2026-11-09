# House Price Prediction — Project Knowledge Base & System Specification
# (Canonical source of truth for AI models, resume bullets & portfolio architecture)

> **Purpose:** Comprehensive, machine-readable project facts, engineering decisions, model metrics, schema constraints, and system architecture for the Bengaluru House Price Prediction project.
> AI assistants and humans should reference this document to understand the entire system before inspecting individual source files.

---

## 1. Project Identity & Live Deployments

| Property | Value |
|---|---|
| **Project Name** | **House Price Prediction Using Machine Learning Algorithms (Bengaluru)** |
| **System Type** | Production ML System + REST API + Minimalist Single-Page Application (SPA) + Multi-Page Streamlit Dashboard |
| **Primary Live URL** | **https://hpp-flax.vercel.app/** (FastAPI Serverless + Minimalist Editorial SPA on Vercel) |
| **Secondary Live URL** | **https://house-price-prediction-using-machine-learning-algorithms-9qccn.streamlit.app/** (4-page Streamlit Dashboard) |
| **Source Repository** | **https://github.com/abustark/HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS** |
| **Author** | **Basith AbuSyed** (`ABu`, GitHub: [`abustark`](https://github.com/abustark)) — Final-Year MCA, The American College |
| **Current Version** | **v2.2.0** (FastAPI backend + minimalist editorial UI + Lakh/Crore auto-scaling + instant API console) |
| **Dataset License** | CC0: Public Domain (Amitabha Roy / Kaggle) |
| **Build & CI Status** | Maintained, GitHub Actions CI Green, 35/35 Pytest passing, Playwright E2E verified |

### Canonical Descriptions & Pitches
- **10-word summary:** Leakage-free Bengaluru house price regression system with minimalist SPA and API.
- **25-word summary:** High-precision Bengaluru house price predictor comparing Linear, Ridge, and Random Forest on 13,320 listings with strict fold-local preprocessing, dual-interface serving, and real-time Lakh/Crore scaling.
- **Elevator pitch:** A production-grade regression platform predicting Bengaluru property valuations from 8 structural and geographic features. Trained on 13,320 historical listings with log-transformed targets and strict fold-local preprocessing (zero data leakage), the top-performing Random Forest Regressor ($R^2 = 0.787$) is deployed via a high-performance Vercel serverless FastAPI backend at **https://hpp-flax.vercel.app/**. The primary application features a custom, minimalist editorial UI inspired by modern design systems, featuring automatic Lakh/Crore currency scaling ($\ge 100\text{ Lakh} \to \text{Crore}$), an interactive in-browser REST API workbench, calibrated 80% conformal confidence intervals, physical domain extrapolation warnings, and an alternative 4-page Streamlit exploratory dashboard.

---

## 2. Primary Web Application (`https://hpp-flax.vercel.app`)

The Vercel-deployed application at **https://hpp-flax.vercel.app** serves as the **primary production entry point** of this repository.

### Visual & Interaction Architecture
- **Aesthetic Direction:** Minimalist editorial design language inspired by CoverCraft (`coverltr.vercel.app`) and `abufolio`, built with pure vanilla HTML5, modern CSS3, and ES2022 JavaScript. Zero frontend framework overhead, zero external CDN dependencies, and zero bundle bloat.
- **Color System:** Warm stone and obsidian canvas (`#0c0a09`, `#141210`, `#1a1715`) accented with hairline borders (`rgba(255, 255, 255, 0.08)`), muted typography (`#a8a29e`), and warm gold/bronze badges (`#cfae77`, `#bc9250`).
- **Typography:**
  - `Instrument Sans` — Clean, legible modern sans-serif body.
  - `Newsreader` — Editorial italic serif accents for key headings and titles.
  - `Space Grotesk` — High-contrast monospace-inspired geometric display numerals.
  - `JetBrains Mono` — Code blocks, metric tables, and live API console output.
- **Floating Island Navigation:** Sticky architectural header (`header-floating`) featuring the `HP` monogram, version pill (`v2.2.0`), segmented tab navigation with active dot indicator, and a direct link to the OpenAPI `/docs` specification.
- **Automatic Lakh / Crore Financial Scaling:**
  - If predicted value $\ge 100\text{ Lakh}$, the UI formats the price in **Crores** (e.g., $186.20\text{ Lakh} \to \text{₹1.86 Cr}$) with full rupee equivalent ($\approx \text{₹1,86,20,000}$).
  - If predicted value $< 100\text{ Lakh}$, the UI retains standard **Lakhs** (e.g., $\text{₹50.43 Lakh}$).
- **Dual-Input Synchronized Area Control:** Synchronized numeric stepper (10 sqft granularity) and smooth range slider (50 sqft granularity) from 100 to 10,000 sqft.
- **Instant Developer API Workbench:** A dedicated in-browser tab allowing developers to test `POST /predict` live without external tools like Postman or page reloads. Includes payload preview, one-click execution, live response status and roundtrip latency (`Status: 200 OK (XXms)`), and clipboard copy.
- **Quick Preset Pills:** 1-click test scenarios for major Bengaluru hubs:
  - *Whitefield:* 2 BHK, 1,200 sqft, 2 Bath
  - *Electronic City:* 3 BHK, 1,500 sqft, 3 Bath
  - *HSR Layout:* 1 BHK, 650 sqft, 1 Bath
  - *Indiranagar:* 4 BHK, 3,200 sqft, 4 Bath
- **Multi-Model Side-by-Side Inference:** Computes predictions across all 3 trained models simultaneously, displaying delta differences and variance against the top Random Forest model.
- **Calibrated 80% Error Coverage:** Displays empirical conformal confidence intervals ($q_{0.10} - q_{0.90}$ log-residuals) derived from cross-validation.
- **Domain Extrapolation Warnings:** Real-time visual alerts if user inputs fall outside physical training distribution boundaries or specify unseen locations.
- **Responsive Layout:** Audited across standard Desktop (`1440x900`) and Mobile (`390x844` iPhone/Pixel) viewports with 16px touch inputs preventing iOS Safari auto-zoom.
- **Branded Minimalist Footer:** Author copyright notice (`© 2026 Basith AbuSyed (ABu)`), portfolio link (`abufolio.vercel.app`), GitHub repository, and LinkedIn links.

---

## 3. Technology Stack & Exact Versions

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Backend Runtime** | Python | `3.11` / `3.13` | Core training pipeline, validation logic, and ASGI API |
| **API Framework** | FastAPI | `2.2.0` (`api/server.py`) | REST API engine, JSON schemas, static file mount |
| **ASGI Web Server** | Uvicorn | `0.34.0+` | High-performance asynchronous HTTP server |
| **Production Hosting** | Vercel Serverless | Latest ASGI Bridge | Hosts primary live web app & REST API at `hpp-flax.vercel.app` |
| **Secondary Hosting** | Streamlit Community Cloud | Latest | Hosts secondary 4-page exploratory dashboard |
| **Machine Learning** | scikit-learn | **`==1.9.1`** (Pinned) | `LinearRegression`, `Ridge`, `RandomForestRegressor`, `StandardScaler`, `KFold`, `GroupKFold` |
| **Data Processing** | pandas | `2.2.3+` | Dataframe manipulation, CSV loading, cleaning |
| **Numerical Engine** | numpy | `2.2.3+` | Log transformations (`log1p`, `expm1`), quantile arrays |
| **Serialization** | joblib | `1.4.2+` | Compressed model artifacts (`compress=3`) in `models/` |
| **Data Validation** | Pydantic | `v2.x` | Strict type checking, boundary enforcement, OpenAPI schema |
| **Frontend Stack** | Vanilla HTML5 / CSS3 / ES2022 | No framework | Zero-dependency, lightweight, high-performance UI |
| **Package Manager** | Bun / Bunx (`bun`) | `1.2+` | Preferred JavaScript / TypeScript execution tooling |
| **Unit Testing** | pytest + HTTPX | `8.x` | 35 automated tests for parsing, leakage, validation, endpoints |
| **E2E Testing** | Playwright | `1.50+` | Automated headless Desktop & Mobile UI verification (`tests/run_playwright_inspection.py`) |
| **Containerization** | Docker | `FROM python:3.11-slim` | Production container build (`EXPOSE 8000`) |
| **CI / CD** | GitHub Actions | `ubuntu-latest` | Automated syntax compilation, linting, and pytest execution |

---

## 4. Machine Learning Architecture & Methodology

### 1. Data Cleaning & Leakage Prevention
- **Dataset Source:** Historical Bengaluru housing data (13,320 rows × 9 columns).
- **Corrupted Row Removal:** 132 physically invalid listings filtered prior to split (`sqft < 100` or `> 10,000`, `bath > 12`, `bhk > 14`, impossible sqft-per-bhk ratios).
- **Unit Normalization:** Cleans free-text `total_sqft` fields containing non-standard units (`Sq. Meter`, `Sq. Yards`, `Acres`, `Guntha`, `Ground`, `Perch`) into standardized square feet; converts range strings (`e.g., 2100 - 2850`) to their mathematical midpoint.
- **Outer Holdout Split:** 70% Train (9,231 listings) / 30% Test (3,957 listings) partitioned with fixed seed (`random_state=2`) **before any feature preprocessing or statistical estimation**.
- **Strict Fold-Local Preprocessing:** All learned transformations—including BHK median imputation, location frequencies, society frequencies, and standard feature scaling—are encapsulated in `DataPreprocessor` (`src/encoding.py`). In cross-validation, these statistics are fitted exclusively on that fold's training slice. Validation folds are never exposed to preprocessor fitting.

### 2. Feature Engineering (8 Inputs)
1. `total_sqft` (float, continuous): Standardized property area in square feet (valid range: 100 to 10,000).
2. `bath` (int): Number of bathrooms (valid range: 1 to 12).
3. `balcony` (int): Number of balconies (valid range: 0 to 5; missing values imputed from training median).
4. `bhk` (int): Bedroom count (0 to 14, where 0 denotes Studio / RK).
5. `area_type` (categorical, 4 classes): `Built-up Area`, `Carpet Area`, `Plot Area`, `Super built-up Area`.
6. `completion_year` (int): `0` for Ready-to-Move; four-digit target completion year (2018–2028) for Under Construction.
7. `location` (categorical, 242 recognized localities): High-cardinality frequency encoded by relative prevalence in the training set; novel locations mapped to 0.0.
8. `society` (categorical): High-cardinality frequency encoded; missing society data explicitly tagged as `__MISSING__`.

### 3. Target Variable Transformation
Property prices in Bengaluru exhibit extreme right-skewness due to luxury penthouses and prime plots. Models are trained on log-transformed targets:
$$y = \log(1 + \text{price\_lakh})$$
Predictions are converted back to real-world Lakhs via exponential inversion:
$$\hat{\text{price\_lakh}} = \exp(\hat{y}) - 1$$

### 4. Calibrated 80% Conformal Error Intervals
Rather than assuming theoretical Gaussian error distributions, empirical prediction intervals are calculated from the 10th and 90th percentiles of cross-validation log-residuals:
$$[\hat{y} + q_{0.10}, \hat{y} + q_{0.90}]$$
Inverted back to Lakhs, this guarantees an empirical 80% coverage rate on unseen historical listings.

---

## 5. Benchmarks & Performance Metrics

Evaluated on the untouched 30% hold-out test set (3,957 listings):

| Model | $R^2$ (log) | MAE (log) | MAE (Lakh) | MedAE (Lakh) | RMSE (Lakh) | 5-Fold CV $R^2$ (log) | Status |
|---|---|---|---|---|---|---|---|
| **Random Forest Regressor** | **0.787** | **0.224** | **₹29.32** | **₹11.07** | **₹80.83** | **0.779 (±0.007)** | **Production Best** |
| **Ridge Regression ($\alpha=0.1$)** | 0.633 | 0.315 | ₹41.39 | ₹16.85 | ₹133.67 | 0.630 (±0.014) | Baseline L2 |
| **Linear Regression (OLS)** | 0.633 | 0.315 | ₹41.39 | ₹16.85 | ₹133.67 | 0.630 (±0.014) | Baseline Linear |

### Key Analytical Takeaways
- **Random Forest Superiority:** Random Forest explains **78.7%** of log-price variance, outperforming linear baselines by 15.4 percentage points due to non-linear interactions between square footage, bathroom count, and locality premium.
- **Generalization Integrity:** 5-fold CV score ($0.779$) closely matches test set performance ($0.787$), confirming absence of overfitting or preprocessing data leakage.
- **Median vs. Mean Error:** Median Absolute Error is **₹11.07 Lakh**, demonstrating that for standard residential homes, predictions are tightly clustered around actual values, while Mean Absolute Error (₹29.32 Lakh) is pulled up by a small tail of multi-crore luxury listings.
- **Locality Generalization (GroupKFold):** Spatial cross-validation withholding entire localities yields $R^2 = 0.719$, proving robust generalization even in unseen geographic areas.

---

## 6. REST API Specification (`api/server.py`)

Hosted live at `https://hpp-flax.vercel.app` (and locally on port 8000).

### Endpoints Summary

| Method | Path | Description | Response Highlights |
|---|---|---|---|
| `GET` | `/` | Serves the Minimalist Editorial Web SPA | `text/html` (`static/index.html`) |
| `GET` | `/static/*` | Static file assets (CSS, JS, media) | Stylesheets, client application code |
| `POST` | `/predict` | Primary prediction inference endpoint | Lakh price, INR price, Crore format, 80% CI, comparisons, warnings |
| `GET` | `/models` | Model performance metadata & test metrics | List of models with R², MAE, RMSE, parameters |
| `GET` | `/locations` | Alphabetical list of recognized localities | 242 recognized Bengaluru locations |
| `GET` | `/meta` | Schema constraints, area types, active version | `SCHEMA_VERSION = "2.2.0"`, validation bounds |
| `GET` | `/manifest` | Dataset hash, commit hash, training provenance | Audit trail for reproducibility |
| `GET` | `/health` | Service health probe | Status OK, loaded models count |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation | Swagger UI explorer |
| `GET` | `/redoc` | Interactive ReDoc documentation | Alternative API schema reader |

### Inference Contract: `POST /predict`

#### Request Payload Example
```json
{
  "area_type": "Super built-up Area",
  "location": "Whitefield",
  "society": "",
  "bhk": 3,
  "total_sqft": 1500.0,
  "bath": 3,
  "balcony": 2,
  "availability": "ready",
  "completion_year": 0,
  "model": "random_forest"
}
```

#### Response Payload Example
```json
{
  "price_lakh": 90.38,
  "price_inr": 9038000,
  "formatted_price": "₹90.38 Lakh",
  "formatted_inr": "₹90,38,000",
  "selected_model": "random_forest",
  "selected_model_name": "Random Forest Regressor",
  "lower_bound_lakh": 68.40,
  "upper_bound_lakh": 119.50,
  "all_models": {
    "linear": { "price_lakh": 61.13, "price_inr": 6113000, "r2_log": 0.633 },
    "ridge": { "price_lakh": 61.13, "price_inr": 6113000, "r2_log": 0.633 },
    "random_forest": { "price_lakh": 90.38, "price_inr": 9038000, "r2_log": 0.787 }
  },
  "warnings": [],
  "extrapolations": []
}
```

---

## 7. Quality Assurance, Testing & Gate Enforcement

### 1. Automated Test Suite (35 Pytest Tests)
All tests run via `pytest -q` and pass with 0 failures:
- `tests/test_predict.py`: Model loading, inference plausibility, monotonicity tests (larger square footage produces higher valuation).
- `tests/test_parsing_cleaning_leakage.py`: Area unit conversions (Sq. Meter, Acres, Guntha), range midpoint calculations, and mathematical proofs verifying validation folds never contaminate preprocessing medians.
- `tests/test_validation_and_api.py`: Physical constraint gates (422 response on $<100$ or $>10,000$ sqft, bath $>12$), extrapolation warning triggers, and full REST route status verifications.

### 2. In-Memory Playwright End-to-End Suite (`tests/run_playwright_inspection.py`)
- Launches in-process FastAPI server on a dynamic port with HTTP healthcheck probe.
- Runs cross-platform headless browser checks across both **Desktop (`1440x900`)** and **Mobile (`390x844`)** viewports.
- Verifies interactive prediction calculation, tab switching, live API console query dispatching, and branded footer assertions.
- **Strict Rule Enforced:** Zero persistent screenshot captures or image files are committed to the repository.

---

## 8. Repository Directory Map

```text
.
├── api/
│   ├── server.py               # FastAPI application v2.2.0 (REST endpoints + static SPA serving)
│   └── static/                 # Minimalist single-page application
│       ├── index.html          # Floating nav, dual inputs, live API console, footer
│       ├── style.css           # Warm stone/obsidian palette, responsive breakpoints
│       └── app.js              # Lakh/Crore auto-scaling, live API dispatcher, tabs
├── app/                        # Secondary Streamlit exploratory application
│   ├── app.py                  # Predictor interface with model selection
│   └── pages/                  # Exploratory pages: Data & EDA, Model Report, About
├── artifacts/                  # 9 ML diagnostic regression & evaluation charts
├── data/
│   └── Bengaluru_House_Data.csv # 13,320 historical listing records
├── models/
│   ├── linear.joblib           # Trained Linear Regression model
│   ├── ridge.joblib            # Trained Ridge Regression model (alpha=0.1)
│   ├── random_forest.joblib    # Trained Random Forest Regressor (200 trees)
│   ├── preprocessor.joblib     # Serialized DataPreprocessor bundle
│   ├── metrics.json            # Exact model evaluation & cross-validation metrics
│   └── manifest.json           # Model hash, dataset provenance & training metadata
├── notebooks/
│   └── House_Price_Prediction.ipynb # Fully executed 33-cell reproducibility notebook
├── src/
│   ├── encoding.py             # DataPreprocessor class, categorical schemas, bounds
│   ├── train.py                # Zero-leakage training, tuning, and evaluation script
│   └── predict.py              # Serialization loader, validation, and inference helpers
├── tests/
│   ├── test_predict.py         # Pytest: inference logic and monotonicity
│   ├── test_parsing_cleaning_leakage.py # Pytest: unit parsing and leakage regression tests
│   ├── test_validation_and_api.py       # Pytest: boundary validation & API endpoints
│   └── run_playwright_inspection.py     # Playwright E2E Desktop & Mobile test runner
├── .github/workflows/ci.yml    # GitHub Actions automated test workflow
├── AGENTS.md                   # AI pair programming guidelines & execution rules
├── Dockerfile                  # Production container configuration
├── PROJECT-INFO.md             # This document (canonical system knowledge base)
├── README.md                   # Public repository documentation
├── vercel.json                 # Vercel serverless deployment routing
└── requirements.txt            # Lean production dependencies (under 500MB cap)
```

---

## 9. Local Setup & Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/abustark/HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS.git
cd HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install all development and testing dependencies
pip install -r requirements-dev.txt
```

### 2. Launch the Primary Web App & REST API (FastAPI)
```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
# Access the web application at http://127.0.0.1:8000
# Access interactive OpenAPI documentation at http://127.0.0.1:8000/docs
```

### 3. Launch the Secondary Web App (Streamlit)
```bash
streamlit run app/app.py
# Access the multi-page dashboard at http://localhost:8501
```

### 4. Run Test Suites
```bash
# Run the 35 automated pytest unit tests
pytest -q

# Run end-to-end Playwright Desktop and Mobile verification
python tests/run_playwright_inspection.py
```

---

## 10. Copy-Paste Resume & Portfolio Summaries

### For Machine Learning Engineer Roles
> **Bengaluru Real Estate Valuation Platform | Python, scikit-learn, FastAPI, Vercel**
> - Designed and deployed an end-to-end house price prediction system trained on 13,320 listings, evaluating Linear, Ridge, and Random Forest regressors.
> - Eliminated data leakage across 5-fold cross-validation by building a custom fold-local preprocessing pipeline encapsulating median imputation, categorical frequency mapping, and feature standardization.
> - Achieved $R^2 = 0.787$ and a Median Absolute Error of ₹11.07 Lakh with Random Forest, integrating empirical 80% conformal prediction intervals and physical domain boundary guards.
> - Deployed high-throughput REST API and minimalist SPA to Vercel Serverless ([hpp-flax.vercel.app](https://hpp-flax.vercel.app)) with automated CI and 35 pytest unit tests.

### For Full-Stack Machine Learning Engineer Roles
> **Production Real Estate Prediction Platform | FastAPI, Vanilla JS, scikit-learn, Vercel, Docker**
> - Architected a production dual-interface ML web application serving real-time Bengaluru property valuations via a serverless FastAPI backend ([hpp-flax.vercel.app](https://hpp-flax.vercel.app)).
> - Implemented an ultra-lightweight, editorial frontend featuring real-time Indian financial auto-scaling (converting $\ge 100\text{ Lakh}$ to Crore), synchronized dual-input controls, and an in-browser zero-reload developer API console.
> - Built leakage-free scikit-learn training pipeline ($R^2 = 0.787$) and rigorous input validation handling corrupted rows, spatial extrapolation warnings, and unit conversions.
> - Maintained complete test coverage with 35 unit/integration tests and in-memory Playwright headless testing across desktop and mobile viewports.

---

## 11. ATS & Keyword Index

`Machine Learning`, `Real Estate Valuation`, `Random Forest Regressor`, `Ridge Regression`, `Linear Regression`, `scikit-learn`, `Data Leakage Prevention`, `Fold-Local Preprocessing`, `Cross-Validation`, `GroupKFold`, `Conformal Prediction`, `Confidence Intervals`, `Feature Engineering`, `Frequency Encoding`, `Log Transformations`, `FastAPI`, `Uvicorn`, `REST API`, `OpenAPI`, `Swagger`, `Vercel Serverless`, `Streamlit`, `Pydantic`, `Docker`, `GitHub Actions`, `Pytest`, `Playwright E2E`, `Bun`, `Vanilla JavaScript`, `Responsive Design`, `Python 3.11`, `Basith AbuSyed`, `Bengaluru Real Estate`.
