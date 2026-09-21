"""About page - project overview, dataset source and setup instructions."""

import streamlit as st

st.set_page_config(page_title="About | Bengaluru House Price Prediction", page_icon="\U0001F3E0")

st.title("About this project")

st.markdown(
    """
    **Final year MCA project** by **Basith AbuSyed** - The American College.

    This project predicts the price of a house in Bengaluru from its basic
    details using machine learning. Three regression models are trained and
    compared:

    1. **Linear Regression** - baseline, learns a weighted sum of the features.
    2. **Ridge Regression** - linear regression with L2 regularization to
       keep the weights stable.
    3. **Random Forest** - an ensemble of 100 decision trees; captures
       non-linear relationships and interactions between features.

    The Streamlit app lets you pick any of the three models and get an
    instant price estimate.
    """
)

st.subheader("Dataset")
st.markdown(
    """
    - **File:** `data/Bengaluru_House_Data.csv`
    - **Size:** 13,320 listings, 9 columns
    - **Columns:** area type, availability, location, size, society,
      total area (sqft), bathrooms, balconies, price (Lakh INR)
    - **Source:** [Bengaluru House price data on Kaggle](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data) (CC0 - public domain)
    - **Note:** the data is a historical snapshot of listings from roughly
      2017-2020, so prices reflect that period, not today's market.
    """
)

st.subheader("Project structure")
st.code(
    """
    .
    ├── app/                  # Streamlit web app
    │   ├── app.py            # prediction page (entry point)
    │   └── pages/            # Data & EDA, Model Report, About
    ├── data/
    │   └── Bengaluru_House_Data.csv
    ├── src/
    │   ├── train.py          # training + evaluation pipeline
    │   └── predict.py        # model loading + input encoding for the app
    ├── models/               # trained models, encoders, metrics.json
    ├── artifacts/            # charts used by the app
    ├── notebooks/            # exploratory notebook
    ├── images/               # static images
    └── docs/                 # presentation
    """,
    language="text",
)

st.subheader("Run it locally")
st.code(
    """
    # 1. create a virtual environment and install dependencies
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # 2. (optional) retrain the models from the raw data
    python src/train.py

    # 3. start the web app
    streamlit run app/app.py
    """,
    language="bash",
)

st.subheader("Tech stack")
st.markdown(
    """
    - **Python 3.11**
    - **pandas / numpy** - data handling
    - **scikit-learn** - models and evaluation
    - **joblib** - model persistence
    - **Streamlit** - web interface
    - **matplotlib** - charts
    """
)

st.divider()
st.caption(
    "Predictions are estimates produced by statistical models on historical "
    "data. They are not a substitute for a professional valuation."
)
