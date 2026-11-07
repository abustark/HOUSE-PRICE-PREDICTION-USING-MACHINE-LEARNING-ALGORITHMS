"""Model report page - evaluation metrics, diagnostics, and methodology analysis."""

import json
import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

st.set_page_config(page_title="Model Report | Bengaluru House Price Prediction", page_icon="📈", layout="wide")

st.title("Model Evaluation & Methodology Report")
st.caption("Rigorous evaluation of Linear, Ridge, and Random Forest models with zero preprocessing leakage.")

with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as fh:
    metrics = json.load(fh)

model_meta = metrics["models"]
best_key = metrics.get("best_model", "random_forest")
best = model_meta[best_key]

st.success(
    f"**Selected Model: {best['name']}** — Held-out Test R² {best['test']['r2_log']:.3f} (log price), "
    f"Test MAE ₹{best['test']['mae']:.2f} Lakh, MedAE ₹{best['test'].get('medae', 0):.2f} Lakh.",
    icon="🏆",
)

st.subheader("1. Held-out Test Set Performance (30% Split)")
test_table = pd.DataFrame(
    [
        {
            "Model": model_meta[k]["name"],
            "R² (log price)": model_meta[k]["test"]["r2_log"],
            "MAE (log price)": model_meta[k]["test"]["mae_log"],
            "MAE (Lakh INR)": f"₹{model_meta[k]['test']['mae']:.2f} Lakh",
            "MedAE (Lakh INR)": f"₹{model_meta[k]['test'].get('medae', 0.0):.2f} Lakh",
            "RMSE (Lakh INR)": f"₹{model_meta[k]['test']['rmse']:.2f} Lakh",
            "Parameters": str(model_meta[k]["params"]),
        }
        for k in model_meta
    ]
)
st.dataframe(test_table, hide_index=True, width="stretch")

st.subheader("2. 5-Fold Cross-Validation (Training Split, Fold-Local Preprocessing)")
cv_table = pd.DataFrame(
    [
        {
            "Model": model_meta[k]["name"],
            "R² (log) Mean": model_meta[k]["cv"]["r2_log_mean"],
            "R² (log) Std": f"±{model_meta[k]['cv']['r2_log_std']:.3f}",
            "MAE (log) Mean": model_meta[k]["cv"]["mae_log_mean"],
            "MAE (log) Std": f"±{model_meta[k]['cv']['mae_log_std']:.3f}",
        }
        for k in model_meta
    ]
)
st.dataframe(cv_table, hide_index=True, width="stretch")

st.markdown(
    """
    **Metric Interpretation:**
    - **Log-space metrics:** Because property prices are heavily right-skewed, models are fitted on `log(1 + price)`.
      R² and MAE in log space represent relative multiplicative error (an MAE (log) of 0.23 indicates typical prediction
      error around 23–25%).
    - **Original scale (Lakh INR):** MAE and MedAE represent absolute errors. Notice that RMSE is notably higher
      than MAE because a small number of ultra-luxury listings (>₹1,000 Lakh) square large residuals.
    - **CV vs Test Consistency:** CV scores closely match the held-out test scores, confirming the pipeline does
      not suffer from memorization or data leakage.
    """
)

# Optional Location-Grouped Stress Test
if "location_grouped_stress_test" in metrics:
    st.subheader("3. Generalization Stress Test: Location-Grouped Cross-Validation")
    st.caption("GroupKFold by location: evaluates model performance when predicting properties in entirely unseen localities.")
    grp_data = metrics["location_grouped_stress_test"]["models"]
    grp_table = pd.DataFrame(
        [
            {
                "Model": model_meta[k]["name"],
                "Group-CV R² (log) Mean": grp_data[k]["r2_log_mean"],
                "Group-CV R² (log) Std": f"±{grp_data[k]['r2_log_std']:.3f}",
                "Group-CV MAE (log) Mean": grp_data[k]["mae_log_mean"],
                "Group-CV MAE (log) Std": f"±{grp_data[k]['mae_log_std']:.3f}",
            }
            for k in grp_data
        ]
    )
    st.dataframe(grp_table, hide_index=True, width="stretch")
    st.markdown(
        """
        > **Insight:** When an entire locality is absent from training, frequency encoding falls back to 0.0,
        > and performance drops across all models. This stress test confirms that location is a major driver of
        > property valuation in Bengaluru, and explains why predictions for unseen localities carry an explicit warning.
        """
    )

st.subheader("4. Model Comparison & Diagnostics")
st.image(os.path.join(ARTIFACTS_DIR, "model_comparison.png"), width="stretch")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Random Forest Feature Importance (MDI)")
    st.image(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), width="stretch")
    st.caption(
        "MDI (Mean Decrease in Impurity) measures how much each feature decreases tree variance. "
        "It is a model-level summary of sensitivity, not a causal economic pricing law."
    )

with col_right:
    perm_path = os.path.join(ARTIFACTS_DIR, "permutation_importance.png")
    if os.path.exists(perm_path):
        st.subheader("Permutation Importance (Held-out Test)")
        st.image(perm_path, width="stretch")
        st.caption(
            "Permutation importance measures the drop in test R² when feature values are randomly shuffled. "
            "Total area and BHK remain the strongest predictors."
        )

st.subheader("5. Actual vs Predicted & Residual Diagnostics")
col_a, col_b = st.columns(2)
with col_a:
    st.image(os.path.join(ARTIFACTS_DIR, "actual_vs_predicted.png"), width="stretch")
with col_b:
    st.image(os.path.join(ARTIFACTS_DIR, "residuals.png"), width="stretch")

st.divider()
st.subheader("6. Preprocessing & Leakage Elimination Audit")
n_dropped = metrics.get("rows_dropped", 0)
total_rows = metrics["dataset"]["rows"]
pct_dropped = (n_dropped / total_rows) * 100
st.markdown(
    f"""
    - **No Preprocessing Leakage:** The raw dataset is cleaned only of corrupt scraper errors ({n_dropped} rows, {pct_dropped:.1f}%).
      The 70/30 train/test split is performed **before** computing any learned imputation or encoding statistics.
    - **Fold-Local Encodings:** In every CV fold, BHK median imputation, location frequencies, society frequencies,
      and StandardScaler statistics are fitted **strictly on that fold's training slice**, preventing validation fold contamination.
    - **Explicit Category Schema:** `area_type` is validated against the 4 known canonical categories
      (`{', '.join(metrics.get('supported_area_types', ['Built-up Area', 'Carpet Area', 'Plot Area', 'Super built-up Area']))}`).
      Unknown types are never silently mapped to an arbitrary category.
    - **Frequency Encoding Semantics:** High-cardinality `location` (~1,300 values) and `society` (~2,700 values)
      use frequency counts. This reflects listing prevalence in this historical Kaggle sample, not market liquidity.
    - **Conformal Error Calibration:** Historical prediction intervals are calibrated using the 80th percentile of
      out-of-fold log residuals from cross-validation, verified against the untouched hold-out test set.
    """
)
