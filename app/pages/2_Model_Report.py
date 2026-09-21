"""Model report page - how each model performed and what drives the prices."""

import json
import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

st.set_page_config(page_title="Model Report | Bengaluru House Price Prediction", page_icon="\U0001F4C8")

st.title("Model Report")
st.caption("Three regression models, one dataset, same train/test split.")

with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as fh:
    metrics = json.load(fh)

model_meta = metrics["models"]
best_key = metrics["best_model"]
best = model_meta[best_key]

st.success(
    f"**Best model: {best['name']}** - R\u00b2 {best['test']['r2_log']:.3f} (log price), "
    f"MAE {best['test']['mae']:.2f} Lakh on the held-out test set.",
    icon="\U0001F3C6",
)

st.subheader("Test set performance")
test_table = pd.DataFrame(
    [
        {
            "Model": model_meta[k]["name"],
            "R\u00b2 (log price)": model_meta[k]["test"]["r2_log"],
            "MAE (log price)": model_meta[k]["test"]["mae_log"],
            "MAE (Lakh INR)": model_meta[k]["test"]["mae"],
            "RMSE (Lakh INR)": model_meta[k]["test"]["rmse"],
            "Params": str(model_meta[k]["params"]),
        }
        for k in model_meta
    ]
)
st.dataframe(test_table, hide_index=True, width="stretch")

st.subheader("5-fold cross-validation (training split)")
cv_table = pd.DataFrame(
    [
        {
            "Model": model_meta[k]["name"],
            "R\u00b2 (log) mean": model_meta[k]["cv"]["r2_log_mean"],
            "R\u00b2 (log) std": model_meta[k]["cv"]["r2_log_std"],
            "MAE (log) mean": model_meta[k]["cv"]["mae_log_mean"],
            "MAE (log) std": model_meta[k]["cv"]["mae_log_std"],
        }
        for k in model_meta
    ]
)
st.dataframe(cv_table, hide_index=True, width="stretch")

st.markdown(
    """
    **How to read this**
    - The models are trained on `log(1 + price)`, so **R\u00b2 (log)** and
      **MAE (log)** are computed in that space - they are the fair,
      like-for-like comparison. An MAE (log) of 0.3 means the typical
      prediction is off by about 35% multiplicatively.
    - **MAE / RMSE (Lakh)** show the same errors on the original price
      scale for readability. A few thousand-Lakh listings dominate the
      price spread, which is why the raw-scale R\u00b2 of any model on this
      data looks poor - the log-scale numbers are the ones to compare.
    - Cross-validation numbers close to the test numbers mean the models
      generalise rather than memorise the training rows.
    """
)

st.subheader("Model comparison")
st.image(os.path.join(ARTIFACTS_DIR, "model_comparison.png"), width="stretch")

st.subheader("Feature importance (Random Forest)")
st.image(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), width="stretch")
st.markdown(
    """
    Total area and BHK dominate - exactly what a human would expect. Location
    frequency captures how active a market is, and society frequency adds a
    weak but real signal.
    """
)

st.subheader("Actual vs predicted (best model)")
st.image(os.path.join(ARTIFACTS_DIR, "actual_vs_predicted.png"), width="stretch")

st.subheader("Residuals (best model)")
st.image(os.path.join(ARTIFACTS_DIR, "residuals.png"), width="stretch")

st.divider()
st.subheader("Data preparation notes")
n_dropped = metrics.get("rows_dropped", 0)
pct_dropped = (n_dropped / metrics["dataset"]["rows"]) * 100
st.markdown(
    f"""
    - **8 features** are used: total area (sqft), bathrooms, balconies, BHK
      count, area type, expected completion year, location and society.
    - **{n_dropped} rows ({pct_dropped:.1f}%)** with physically impossible
      measurements (e.g. an 11 sqft 3-bedroom flat or a 52,000 sqft house)
      were removed before training.
    - `total_sqft` is parsed to a number (ranges like "1050-1100" become the
      midpoint) instead of being dropped.
    - `size` labels such as "2 BHK" / "4 Bedroom" / "Studio" are converted to
      a bedroom count.
    - `availability` mixes "Ready To Move" with dates like "18-Dec", so it is
      converted to a single numeric completion year (0 = ready to move).
    - `area_type` is label encoded (only 3 categories).
    - `location` (~1,100 values) and `society` (~2,700 values) are
      **frequency encoded** - the number of times the value appears in the
      training split. Label encoding would impose a fake order, and one-hot
      encoding would create thousands of sparse columns.
    - Missing locations/societies become a dedicated "not available" category;
      missing baths/balconies become 0.
    - Linear and Ridge models are trained on standardized features; the
      standardizer is saved with the encoders and applied to new inputs.
    - Prices are heavily right-skewed, so all three models are trained on
      `log(1 + price)`. Headline metrics (R\u00b2, MAE) are computed in that
      log space; MAE/RMSE in Lakh INR are also reported for readability.
    - Train/test split: 70/30, random state 2.
    """
)
