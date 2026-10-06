# HOUSE PRICE PREDICTION USING MACHINE LEARNING ALGORITHMS

Final year MCA project - **Basith AbuSyed**, The American College.

Predict the price of a house in Bengaluru from its basic details using
machine learning. Three regression models are trained on the same dataset
and compared, and a Streamlit web app lets you get an instant price
estimate with any of the three models.

Models used: **Linear Regression**, **Ridge Regression** and **Random
Forest Regressor**.

---

## Demo

The main page of the app predicts the price for a house you describe
(location, size, area, bathrooms, availability, society). The sidebar
lets you switch between the three trained models and shows each model's
test-set score. A comparison table shows what all three models predict
for the same input.

```
Prediction:  Predicted price: ₹90.38 Lakh (≈ ₹9,038,000)

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
│   ├── encoding.py             # shared feature engineering rules
│   ├── train.py                # training + tuning + evaluation pipeline
│   └── predict.py              # model loading + input encoding for the apps
├── models/                     # trained models + encoders + metrics.json
├── artifacts/                  # charts used by the apps and the notebook
├── notebooks/                  # executed notebook reproducing the pipeline
├── images/                     # static images
├── docs/                       # presentation + commit log
├── tests/                      # pytest suite
└── Dockerfile                  # container for the FastAPI app
```

## Quick start

```bash
# 1. create a virtual environment and install the dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt # everything: FastAPI + Streamlit + plots + pytest
# `pip install -r requirements.txt` installs the lean runtime set only,
# which is enough for option B below and is what Vercel deploys.

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
  2017-2020, so the prices reflect that period.

## Data preparation

The raw data has several quirks that the pipeline handles:

- `total_sqft` is stored as text and mixes square feet with other units
  ("34.46Sq. Meter", "1100Sq. Yards", "5Acres"). Values are converted to
  square feet; ranges become their midpoint.
- **59 rows (0.4%)** have physically impossible measurements (an 11 sqft
  3-bedroom flat, a 52,000 sqft house, a 40-bathroom listing) and are
  removed.
- `size` labels ("2 BHK", "4 Bedroom", "Studio") become a bedroom count.
- `availability` mixes "Ready To Move" with dates like "18-Dec"; it
  becomes a single completion-year feature (0 = ready to move).
- Missing locations/societies become a dedicated "not available" category
  instead of 0; missing baths/balconies become 0.

Eight features are used in the end: total area, bathrooms, balconies, BHK
count, area type, completion year, location and society.

- `area_type` (3 categories) is label encoded.
- `location` (~1,100 values) and `society` (~2,700 values) are
  **frequency encoded** - the number of times the value appears in the
  training split. Label encoding would impose a fake order on the values,
  and one-hot encoding would create thousands of sparse columns.
- Prices are heavily right-skewed (a few thousand-Lakh listings), so all
  models are trained on `log(1 + price)`; predictions are inverted back to
  Lakh INR for display.
- Linear and Ridge are fitted on standardized features (the standardizer is
  saved with the encoders and applied to new inputs). Random Forest is
  scale-free and uses the raw encoded features.
- Train/test split: 70/30, random state 2.

## Model results

All metrics below are computed on the held-out test set (3,979 rows).
R² and MAE (log) are in the log-price space the models were trained in;
MAE/RMSE (Lakh) are on the original price scale for readability.

| Model | R² (log) | MAE (log) | MAE (Lakh) | RMSE (Lakh) | 5-fold CV R² (log) |
|---|---|---|---|---|---|
| Linear Regression | 0.644 | 0.317 | 44.80 | 173.03 | 0.623 |
| Ridge Regression (α = 0.1) | 0.644 | 0.317 | 44.80 | 173.02 | 0.623 |
| **Random Forest (200 trees, leaf ≥ 3)** | **0.787** | **0.231** | **31.41** | **89.35** | **0.776** |

**Random Forest is the best model**: it explains ~79% of the (log) price
variation and its cross-validation score matches the test score, so it
generalizes rather than memorizing. An MAE (log) of 0.231 corresponds to a
typical relative error of roughly a quarter of the price; the Lakh-scale
MAE is inflated by a small number of very expensive listings. Its
hyper-parameters (200 trees, min_samples_leaf=3) were chosen by a small
5-fold CV grid search over (trees x leaf size).

Ridge with small α is essentially plain linear regression here, which is
why the two sit at the same score. Full charts (model comparison, feature
importance, actual vs predicted, residuals) are available on the
**Model Report** page of the app.

## How a prediction is made

1. The inputs are cleaned and turned into the 8 training features
   (frequency encoding uses the counts saved during training, so an unseen
   society or location gets 0).
2. For Linear/Ridge the standardized features are fed to the model; for
   Random Forest the raw encoded features are used.
3. The model predicts `log(1 + price)`, which is inverted to Lakh INR.

## REST API

The FastAPI app (`api/server.py`) exposes the same three models as JSON
endpoints and serves the custom single-page UI:

| Endpoint | Description |
|---|---|
| `POST /predict` | predict with `linear`, `ridge` or `random_forest` (or `all`) |
| `GET /models` | names + test/CV scores of all models |
| `GET /locations` | all locations known to the model |
| `GET /meta` | area types, best model, dataset info |
| `GET /health` | liveness check |
| `GET /docs` | interactive Swagger documentation |

Example:

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

### Deploy the Streamlit app (Streamlit Cloud)

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io): **New app** → pick the
   repository and branch, main file `app/app.py`.
3. No secrets are needed — the trained models and encoders are committed in
   `models/`.

The FastAPI face is best run as the Docker image above (any container host:
Render, Railway, Fly.io, a VPS).

## Tests and CI

```bash
pytest -q
```

The suite covers the artifact shapes, the prediction path (known/unknown
inputs, plausibility, monotonicity) and the API routes. GitHub Actions runs
it on every push and pull request (the workflow file ships in a follow-up commit - see `docs/commits.md`).

## Tech stack

- Python 3.11
- pandas, numpy
- scikit-learn (LinearRegression, Ridge, RandomForestRegressor)
- joblib (model persistence)
- Streamlit + FastAPI/uvicorn (web interfaces)
- matplotlib (charts)
- pytest + GitHub Actions (tests / CI) + Docker (deployment)

## Disclaimer

Predictions are estimates produced by statistical models trained on
historical listings. They are not a substitute for a professional
valuation or current market data.
