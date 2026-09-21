"""
Bengaluru House Price Prediction - main prediction page.

Run with:
    streamlit run app/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.predict import MISSING, load_artifacts, predict_all  # noqa: E402

st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="\U0001F3E0",
    layout="wide",
)

artifacts = load_artifacts()
bundle = artifacts["bundle"]
metrics = artifacts["metrics"]
model_meta = metrics["models"]
model_keys = list(model_meta)

st.title("\U0001F3E0 Bengaluru House Price Prediction")
st.caption(
    "Predict the price of a house in Bengaluru. Three models were trained on "
    f"{metrics['dataset']['rows']:,} listings - switch between them in the sidebar."
)

location_options = sorted(
    loc for loc in bundle["location_freq"] if loc != MISSING
)
area_type_options = list(bundle["le_area_type"].classes_)

with st.sidebar:
    st.header("Model")
    selected = st.radio(
        "Choose a model",
        options=model_keys,
        format_func=lambda k: model_meta[k]["name"],
        index=model_keys.index(metrics["best_model"]),
        help="All three models were trained on the same train/test split.",
    )
    chosen = model_meta[selected]
    st.markdown(
        f"**R\u00b2 (log price):** {chosen['test']['r2_log']:.3f}  \n"
        f"**MAE (test set):** {chosen['test']['mae']:.2f} Lakh"
    )
    st.divider()
    with st.expander("How the prediction works"):
        st.markdown(
            """
            1. The house details are cleaned and turned into 8 features
               (area, bathrooms, balconies, BHK, area type, completion
               year, location and society).
            2. Location and society are converted into how frequently they
               appear in the training data.
            3. The selected model predicts the price in **Lakh INR**.
            """
        )
    st.divider()
    st.caption(
        "Trained on historical listings (2017-2020). Predictions are estimates, "
        "not market quotes."
    )

col1, col2 = st.columns(2)

with col1:
    area_type = st.selectbox("Area type", options=area_type_options)
    location = st.selectbox(
        "Location",
        options=location_options,
        help="All locations present in the training data.",
    )
    size = st.selectbox(
        "Size (BHK)",
        options=list(range(0, 11)),
        format_func=lambda b: "Studio / RK" if b == 0 else f"{b} BHK",
        index=2,
    )
    availability_choice = st.selectbox(
        "Availability",
        options=["Ready to move", "Under construction"],
    )

with col2:
    total_sqft = st.number_input(
        "Total area (sqft)", min_value=100, max_value=20000, value=1200, step=10
    )
    bath = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
    balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
    society = st.text_input(
        "Society name (optional)",
        placeholder="Leave blank if unknown",
    )

    expected_year = None
    if availability_choice == "Under construction":
        expected_year = st.number_input(
            "Expected completion year", min_value=2013, max_value=2030, value=2020
        )

if st.button("Predict price", type="primary"):
    # completion_year = 0 means the house is ready to move.
    year = (
        0
        if availability_choice == "Ready to move"
        else int(expected_year or 2020)
    )
    row = {
        "area_type": area_type,
        "location": location,
        "society": society,
        "bhk": size,
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "completion_year": year,
    }
    results = predict_all(row, artifacts)

    st.divider()
    pred = max(0.0, results[selected])
    st.success(
        f"**Predicted price: \u20b9{pred:,.2f} Lakh** "
        f"(\u2248 \u20b9{pred * 100_000:,.0f})",
        icon="\U0001F4B0",
    )

    st.markdown("**All models on this input**")
    comparison = pd.DataFrame(
        [
            {
                "Model": model_meta[k]["name"],
                "Predicted price (Lakh INR)": round(max(0.0, results[k]), 2),
                "R\u00b2 (log price)": model_meta[k]["test"]["r2_log"],
            }
            for k in model_keys
        ]
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    if society.strip() and not society.strip().upper().startswith(MISSING) \
            and society.strip() not in bundle["society_freq"]:
        st.info(
            "The society name was not in the training data, so it was treated "
            "as an unseen society. The prediction relies on the other features."
        )

st.divider()
st.caption(
    "Predictions come from machine learning models trained on historical "
    "listings and may not reflect current market values."
)
