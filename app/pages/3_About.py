"""About page - project overview, dataset source and setup instructions."""

import streamlit as st

st.set_page_config(page_title="About | Bengaluru House Price Prediction", page_icon="🏡", layout="wide")

st.title("About this project")

st.markdown(
    """
    **Final year MCA project** by **Basith AbuSyed** — The American College.

    This project predicts the price of a house in Bengaluru from its basic
    details using machine learning. Three regression models are trained and
    compared with strictly fold-local preprocessing to ensure zero data leakage:

    1. **Linear Regression** — baseline weighted sum of standardized features.
    2. **Ridge Regression** — linear regression with L2 regularization to
       control multicollinearity.
    3. **Random Forest Regressor** — ensemble of 200 decision trees (min samples leaf ≥ 3)
       capturing non-linear interactions across spatial and structural features.

    The Streamlit app lets you compare all three models side-by-side with
    calibrated historical error ranges and domain extrapolation warnings.
    """
)

st.subheader("Dataset & Historical Positioning")
st.markdown(
    """
    - **File:** `data/Bengaluru_House_Data.csv`
    - **Size:** 13,320 listings, 9 columns
    - **Columns:** area type, availability, location, size, society,
      total area (sqft), bathrooms, balconies, price (Lakh INR)
    - **Source:** [Bengaluru House price data on Kaggle](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data) (CC0 — public domain)
    - **Historical Context:** This dataset represents a historical snapshot of listings
      (approximately 2017–2020). Predictions are statistical estimates for educational and portfolio
      demonstration, not current market quotes or bank valuations.
    """
)

st.subheader("Project structure")
st.code(
    """
    .
    ├── api/                  # FastAPI server + custom single-page UI
    │   ├── server.py         # REST API (routes: /predict, /models, /locations, /meta, /manifest)
    │   └── static/           # index.html, app.js, style.css
    ├── app/                  # Streamlit web app
    │   ├── app.py            # prediction page (entry point)
    │   └── pages/            # 1_Data_and_EDA, 2_Model_Report, 3_About
    ├── data/
    │   └── Bengaluru_House_Data.csv
    ├── src/
    │   ├── encoding.py       # shared feature engineering, validation & DataPreprocessor
    │   ├── train.py          # leakage-free training + tuning + evaluation pipeline
    │   └── predict.py        # artifact loader, validation, and inference helpers
    ├── models/               # trained models, preprocessor bundle, metrics.json, manifest.json
    ├── artifacts/            # diagnostic plots and figures
    ├── tests/                # expanded pytest test suite
    └── Dockerfile            # containerization for the FastAPI service
    """,
    language="text",
)

st.subheader("Run it locally")
st.code(
    """
    # 1. create a virtual environment and install dependencies
    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
    pip install -r requirements-dev.txt

    # 2. (optional) retrain the models with zero leakage
    python src/train.py

    # 3. start the web app
    streamlit run app/app.py          # Streamlit UI at http://localhost:8501
    # or
    uvicorn api.server:app --port 8000 # FastAPI + custom UI at http://localhost:8000
    """,
    language="bash",
)

st.subheader("Tech stack")
st.markdown(
    """
    - **Python 3.11**
    - **pandas / numpy** — data cleaning and transformations
    - **scikit-learn** — Linear Regression, Ridge, Random Forest, KFold, GroupKFold, StandardScaler
    - **joblib** — model and preprocessor persistence
    - **Streamlit** & **FastAPI / uvicorn** — dual prediction interfaces
    - **matplotlib** — diagnostic charts and EDA plots
    - **pytest** & **GitHub Actions** — CI validation suite
    """
)

st.divider()
st.caption(
    "Predictions are educational estimates produced by statistical models on historical data. "
    "They are not a substitute for a professional valuation or current market appraisal."
)
