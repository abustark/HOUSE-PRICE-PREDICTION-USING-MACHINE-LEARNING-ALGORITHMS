"""
Bengaluru House Price Prediction - main prediction page.

Interactive Streamlit interface comparing Linear Regression, Ridge Regression,
and Random Forest Regressor models trained on 13,320 historical Bengaluru listings.

Run with:
    streamlit run app/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.encoding import (  # noqa: E402
    MISSING,
    SUPPORTED_AREA_TYPES,
    VALIDATION_CONSTRAINTS,
    validate_input,
)
from src.predict import (  # noqa: E402
    load_artifacts,
    predict_detailed,
)


def format_price_inr(lakh: float) -> str:
    """Format price in Lakh or Crore depending on magnitude (>= 100 Lakh -> Crore)."""
    if lakh >= 100:
        cr = lakh / 100.0
        return f"₹{cr:,.2f} Cr"
    return f"₹{lakh:,.2f} Lakh"

st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏡",
    layout="wide",
)

# Custom Modern Editorial Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0d13;
    }
    
    .primary-site-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .banner-pill {
        background: #f59e0b;
        color: #090a0f;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 9px;
        border-radius: 9999px;
    }
    
    .banner-text {
        color: #f1f5f9;
        font-size: 0.95rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .banner-btn {
        background: #f8fafc;
        color: #090d16 !important;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 8px 16px;
        border-radius: 8px;
        text-decoration: none;
        transition: all 0.15s ease;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .banner-btn:hover {
        background: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(255, 255, 255, 0.2);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    </style>
    
    <div class="primary-site-banner">
        <div>
            <span class="banner-pill">Primary Production Site</span>
            <span class="banner-text">Looking for the Minimalist Vercel Web App & REST API Workbench?</span>
        </div>
        <a class="banner-btn" href="https://hpp-flax.vercel.app/" target="_blank" rel="noopener noreferrer">
            <span>Open hpp-flax.vercel.app</span>
            <span>↗</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

artifacts = load_artifacts()
bundle = artifacts["bundle"]
metrics = artifacts["metrics"]
model_meta = metrics["models"]
model_keys = list(model_meta)

st.title("🏡 Bengaluru House Price Prediction")
st.caption(
    "Predict the price of a house in Bengaluru from basic property attributes. "
    f"Three regression models were evaluated on {metrics['dataset']['rows']:,} historical listings "
    "(approx. 2017–2020 snapshot) using strictly fold-local preprocessing to prevent data leakage."
)

location_options = sorted(
    loc for loc in bundle["location_freq"] if loc != MISSING
)
area_type_options = list(SUPPORTED_AREA_TYPES)

with st.sidebar:
    st.header("Model Selection")
    selected = st.radio(
        "Choose a model",
        options=model_keys,
        format_func=lambda k: model_meta[k]["name"],
        index=model_keys.index(metrics.get("best_model", "random_forest")),
        help="All models share the same training and hold-out test split with fold-local preprocessing.",
    )
    chosen = model_meta[selected]
    st.markdown(
        f"**Test R² (log price):** {chosen['test']['r2_log']:.3f}  \n"
        f"**Test MAE:** ₹{chosen['test']['mae']:.2f} Lakh  \n"
        f"**5-fold CV R²:** {chosen['cv']['r2_log_mean']:.3f} (±{chosen['cv']['r2_log_std']:.3f})"
    )
    st.divider()
    with st.expander("How the prediction works"):
        st.markdown(
            """
            1. Property inputs are validated against the training support domain.
            2. Features are cleaned and turned into 8 numerical features.
            3. `location` and `society` use frequency encoding learned solely from training data.
            4. The chosen model predicts `log(1 + price)`, which is inverted back to **Lakh INR**.
            5. An empirical historical error range (80% coverage on hold-out tests) provides estimation context.
            """
        )
    st.divider()
    st.info(
        "**Historical Snapshot Notice:** Trained on listings from 2017–2020. "
        "Predictions are historical ML estimates for educational evaluation, "
        "not a live valuation or bank appraisal.",
        icon="ℹ️",
    )

# Sample historical presets
st.subheader("Property Details")
with st.expander("💡 Load a typical historical example", expanded=False):
    preset_cols = st.columns(4)
    preset_row = None
    if preset_cols[0].button("2 BHK Whitefield (1,200 sqft)"):
        st.session_state["p_loc"] = "Whitefield"
        st.session_state["p_sqft"] = 1200
        st.session_state["p_bhk"] = 2
        st.session_state["p_bath"] = 2
        st.session_state["p_balc"] = 1
        st.session_state["p_area"] = "Super built-up Area"
    if preset_cols[1].button("3 BHK Electronic City (1,500 sqft)"):
        st.session_state["p_loc"] = "Electronic City"
        st.session_state["p_sqft"] = 1500
        st.session_state["p_bhk"] = 3
        st.session_state["p_bath"] = 3
        st.session_state["p_balc"] = 2
        st.session_state["p_area"] = "Super built-up Area"
    if preset_cols[2].button("1 BHK HSR Layout (650 sqft)"):
        st.session_state["p_loc"] = "HSR Layout"
        st.session_state["p_sqft"] = 650
        st.session_state["p_bhk"] = 1
        st.session_state["p_bath"] = 1
        st.session_state["p_balc"] = 1
        st.session_state["p_area"] = "Built-up Area"
    if preset_cols[3].button("4 BHK Indiranagar (3,200 sqft)"):
        st.session_state["p_loc"] = "Indiranagar"
        st.session_state["p_sqft"] = 3200
        st.session_state["p_bhk"] = 4
        st.session_state["p_bath"] = 4
        st.session_state["p_balc"] = 2
        st.session_state["p_area"] = "Plot Area"

default_loc = st.session_state.get("p_loc", "Whitefield")
loc_idx = location_options.index(default_loc) if default_loc in location_options else 0
default_sqft = st.session_state.get("p_sqft", 1200)
default_bhk = st.session_state.get("p_bhk", 2)
default_bath = st.session_state.get("p_bath", 2)
default_balc = st.session_state.get("p_balc", 1)
default_area = st.session_state.get("p_area", "Super built-up Area")
area_idx = area_type_options.index(default_area) if default_area in area_type_options else 0

col1, col2 = st.columns(2)

with col1:
    area_type = st.selectbox("Area type", options=area_type_options, index=area_idx)
    location = st.selectbox(
        "Location",
        options=location_options,
        index=loc_idx,
        help="Locations present in the historical training data.",
    )
    size = st.selectbox(
        "Size (BHK)",
        options=list(range(0, 11)),
        format_func=lambda b: "Studio / RK" if b == 0 else f"{b} BHK",
        index=default_bhk,
    )
    availability_choice = st.selectbox(
        "Availability",
        options=["Ready to move", "Under construction"],
    )

with col2:
    total_sqft = st.number_input(
        "Total area (sqft)",
        min_value=int(VALIDATION_CONSTRAINTS["total_sqft"]["min"]),
        max_value=int(VALIDATION_CONSTRAINTS["total_sqft"]["max"]),
        value=int(default_sqft),
        step=10,
        help="Constrained to 100 – 10,000 sqft (the historical training support domain).",
    )
    bath = st.number_input(
        "Bathrooms",
        min_value=int(VALIDATION_CONSTRAINTS["bath"]["min"]),
        max_value=int(VALIDATION_CONSTRAINTS["bath"]["max"]),
        value=int(default_bath),
    )
    balcony = st.number_input(
        "Balconies",
        min_value=int(VALIDATION_CONSTRAINTS["balcony"]["min"]),
        max_value=int(VALIDATION_CONSTRAINTS["balcony"]["max"]),
        value=int(default_balc),
    )
    society = st.text_input(
        "Society name (optional)",
        placeholder="Leave blank if independent house or unknown",
    )

    expected_year = 0
    if availability_choice == "Under construction":
        expected_year = st.number_input(
            "Expected completion year", min_value=2013, max_value=2030, value=2020
        )

if st.button("Predict price", type="primary"):
    year = 0 if availability_choice == "Ready to move" else int(expected_year)
    input_row = {
        "area_type": area_type,
        "location": location,
        "society": society,
        "bhk": size,
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "completion_year": year,
    }

    result = predict_detailed(input_row, artifacts, selected_model=selected)

    if not result.get("is_valid", True):
        st.error("Validation failed:\n" + "\n".join(f"- {e}" for e in result["errors"]))
    else:
        st.divider()
        pred_lakh = result["predicted_price_lakh"]
        pred_inr = result["predicted_price_inr"]
        err_range = result["historical_error_range"]

        formatted_price = format_price_inr(pred_lakh)
        alt_text = f" / ₹{pred_lakh:,.2f} Lakh" if pred_lakh >= 100 else ""
        st.success(
            f"**Predicted Price ({model_meta[selected]['name']}): {formatted_price}** "
            f"(≈ ₹{pred_inr:,.0f}{alt_text})",
            icon="💰",
        )

        lower_fmt = format_price_inr(err_range["lower_lakh"])
        upper_fmt = format_price_inr(err_range["upper_lakh"])
        st.info(
            f"**Estimated Historical Error Range (80% coverage):** "
            f"**{lower_fmt} – {upper_fmt}**  \n"
            f"Based on out-of-fold cross-validation residuals on the historical 2017–2020 dataset "
            f"(verified {err_range['historical_test_coverage'] * 100:.1f}% test set coverage).",
            icon="📊",
        )

        # Domain & Extrapolation Warnings
        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(f"⚠️ **Extrapolation Notice:** {w}")

        if result.get("unseen_society"):
            st.caption(
                "ℹ️ **Note:** The entered society name was not in the training dataset, "
                "so frequency encoding mapped it to 0.0. The prediction relies on the remaining features."
            )

        st.subheader("Model Comparison on This Property")
        comparison = pd.DataFrame(
            [
                {
                    "Model": model_meta[k]["name"],
                    "Predicted Price": format_price_inr(result["all_models"][k]["price_lakh"]),
                    "Predicted Price (Lakh INR)": round(result["all_models"][k]["price_lakh"], 2),
                    "Predicted Price (INR)": f"₹{result['all_models'][k]['price_inr']:,}",
                    "Test R² (log price)": model_meta[k]["test"]["r2_log"],
                    "Test MAE (Lakh INR)": model_meta[k]["test"]["mae"],
                }
                for k in model_keys
            ]
        )
        st.dataframe(comparison, hide_index=True, width="stretch")

st.divider()
st.caption(
    "⚠️ **Educational and Portfolio Case Study:** Predictions are produced by statistical models "
    "fitted to historical real-estate listings (2017–2020). Property values have evolved significantly; "
    "these estimates are not valid for current buying, selling, or bank appraisal decisions."
)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1.5rem 0; line-height: 1.8;">
        <p style="margin: 0; font-weight: 500;">
            © 2026 <strong>Basith AbuSyed</strong> (ABu). All rights reserved.
        </p>
        <p style="margin: 0.2rem 0; font-size: 0.8rem; color: #64748b;">
            Bengaluru Real Estate Valuation Intelligence · Historical Model Benchmark (2017–2020)
        </p>
        <p style="margin-top: 0.5rem;">
            <a href="https://abufolio.vercel.app/" target="_blank" style="color: #10b981; text-decoration: none; margin: 0 10px;">💼 Portfolio</a> •
            <a href="https://github.com/abustark/HOUSE-PRICE-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS" target="_blank" style="color: #38bdf8; text-decoration: none; margin: 0 10px;">🐙 GitHub Repo</a> •
            <a href="https://www.linkedin.com/in/bas-abu/" target="_blank" style="color: #10b981; text-decoration: none; margin: 0 10px;">🔗 LinkedIn</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
