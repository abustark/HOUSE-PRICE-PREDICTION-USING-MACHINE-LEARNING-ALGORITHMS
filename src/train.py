"""
Training pipeline for the Bengaluru house price prediction project.

Reads raw listings from data/, cleans invalid records, performs a leakage-safe
70/30 split, evaluates three regression models using 5-fold cross-validation with
strictly fold-local preprocessing (imputation, frequency maps, scaling), performs
a location-grouped stress test, and evaluates on the untouched hold-out test set.

Saves:
    models/linear.joblib          - Linear Regression model
    models/ridge.joblib           - Ridge Regression model
    models/random_forest.joblib   - Random Forest Regressor
    models/encoders.joblib        - fitted DataPreprocessor bundle
    models/metrics.json           - comprehensive performance metrics
    models/manifest.json          - reproducibility and provenance manifest
    artifacts/*.png               - updated diagnostic and EDA charts

Run with:
    python src/train.py
"""

import hashlib
import json
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold, train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.encoding import (  # noqa: E402
    FEATURES,
    MISSING,
    SCHEMA_VERSION,
    SUPPORTED_AREA_TYPES,
    DataPreprocessor,
    clean_area_type,
    clean_category,
    extract_bhk,
    parse_availability,
    parse_sqft,
)

DATA_PATH = os.path.join(BASE_DIR, "data", "Bengaluru_House_Data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

TARGET = "price"
RANDOM_STATE = 2
TEST_SIZE = 0.3
MODEL_NAMES = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
}


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of the dataset for provenance tracking."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Data Cleaning (No leakage: only record-level parsing & sane bounds)
# ---------------------------------------------------------------------------
def clean_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Parse raw records and filter physically impossible scraper errors.

    Crucially, NO dataset-wide imputation (such as BHK median or frequency maps)
    is performed here. Imputation and encodings are learned strictly within
    training folds.
    """
    data = raw_df.copy()

    data["total_sqft"] = data["total_sqft"].map(parse_sqft)
    data["bhk"] = data["size"].map(extract_bhk)
    data["bath"] = pd.to_numeric(data["bath"], errors="coerce")
    data["balcony"] = pd.to_numeric(data["balcony"], errors="coerce").fillna(0)
    data["completion_year"] = data["availability"].map(parse_availability).astype(int)

    # Filter corrupted or physically impossible records:
    # area < 100 sqft or > 10,000 sqft, bath > 12, bhk > 14, non-positive price
    sane = (
        data["total_sqft"].notna()
        & (data["total_sqft"] >= 100)
        & (data["total_sqft"] <= 10000)
        & data["bath"].notna()
        & (data["bath"] >= 1)
        & (data["bath"] <= 12)
        & (data["bhk"].isna() | (data["bhk"] <= 14))
        & data[TARGET].notna()
        & (data[TARGET] > 0)
    )
    n_dropped = int((~sane).sum())
    cleaned = data[sane].copy()

    # Normalize category strings
    cleaned["location"] = cleaned["location"].map(clean_category)
    cleaned["society"] = cleaned["society"].map(clean_category)
    cleaned["area_type"] = cleaned["area_type"].map(clean_area_type).fillna("Super built-up Area")

    return cleaned, n_dropped


# ---------------------------------------------------------------------------
# Modelling helpers
# ---------------------------------------------------------------------------
def make_model(key: str, ridge_alpha: float = 1.0, rf_n_est: int = 200, rf_leaf: int = 3):
    if key == "linear":
        return LinearRegression()
    if key == "ridge":
        return Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)
    return RandomForestRegressor(
        n_estimators=rf_n_est,
        min_samples_leaf=rf_leaf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def choose_ridge_alpha_fold_local(train_df: pd.DataFrame, y_train_log: pd.Series) -> float:
    """
    Select Ridge alpha using 5-fold CV on training split.
    Fold preprocessor is fitted strictly inside each fold.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_alpha, best_score = 1.0, np.inf
    candidate_alphas = (0.1, 1.0, 10.0, 100.0)

    for alpha in candidate_alphas:
        fold_scores = []
        for tr_idx, va_idx in kf.split(train_df):
            tr_fold, va_fold = train_df.iloc[tr_idx], train_df.iloc[va_idx]
            y_tr_fold = y_train_log.iloc[tr_idx]
            y_va_fold = y_train_log.iloc[va_idx]

            # Fold-local preprocessor
            prep = DataPreprocessor().fit(tr_fold)
            X_tr = prep.transform(tr_fold, scale_linear=True)
            X_va = prep.transform(va_fold, scale_linear=True)

            model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
            model.fit(X_tr, y_tr_fold)
            pred = model.predict(X_va)
            fold_scores.append(mean_absolute_error(y_va_fold, pred))

        mean_mae = float(np.mean(fold_scores))
        if mean_mae < best_score:
            best_alpha, best_score = alpha, mean_mae

    return best_alpha


def tune_random_forest_fold_local(
    train_df: pd.DataFrame, y_train_log: pd.Series
) -> Tuple[int, int]:
    """
    Tune Random Forest hyper-parameters using 5-fold CV on training split.
    Fold preprocessor is fitted strictly inside each fold.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_score = np.inf
    best_n_est, best_leaf = 100, 5

    for n_est in (100, 200):
        for leaf in (3, 5):
            fold_scores = []
            for tr_idx, va_idx in kf.split(train_df):
                tr_fold, va_fold = train_df.iloc[tr_idx], train_df.iloc[va_idx]
                y_tr_fold = y_train_log.iloc[tr_idx]
                y_va_fold = y_train_log.iloc[va_idx]

                prep = DataPreprocessor().fit(tr_fold)
                X_tr = prep.transform(tr_fold, scale_linear=False)
                X_va = prep.transform(va_fold, scale_linear=False)

                model = RandomForestRegressor(
                    n_estimators=n_est,
                    min_samples_leaf=leaf,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                model.fit(X_tr, y_tr_fold)
                pred = model.predict(X_va)
                fold_scores.append(mean_absolute_error(y_va_fold, pred))

            mean_mae = float(np.mean(fold_scores))
            print(f"  RF n_estimators={n_est} min_samples_leaf={leaf} -> CV MAE(log): {mean_mae:.4f}")
            if mean_mae < best_score:
                best_score = mean_mae
                best_n_est, best_leaf = n_est, leaf

    return best_n_est, best_leaf


def cross_validate_fold_local(
    key: str,
    train_df: pd.DataFrame,
    y_train_log: pd.Series,
    ridge_alpha: float = 1.0,
    rf_n_est: int = 200,
    rf_leaf: int = 3,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    5-fold CV with strictly fold-local preprocessing.
    Returns CV score metrics and out-of-fold log predictions for calibration.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    maes_log, r2s_log = [], []
    oof_preds_log = np.zeros(len(train_df))
    scale_linear = key in ("linear", "ridge")

    for tr_idx, va_idx in kf.split(train_df):
        tr_fold, va_fold = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        y_tr_fold = y_train_log.iloc[tr_idx]
        y_va_fold = y_train_log.iloc[va_idx]

        prep = DataPreprocessor().fit(tr_fold)
        X_tr = prep.transform(tr_fold, scale_linear=scale_linear)
        X_va = prep.transform(va_fold, scale_linear=scale_linear)

        model = make_model(key, ridge_alpha, rf_n_est, rf_leaf)
        model.fit(X_tr, y_tr_fold)
        pred_log = model.predict(X_va)

        oof_preds_log[va_idx] = pred_log
        maes_log.append(mean_absolute_error(y_va_fold, pred_log))
        r2s_log.append(r2_score(y_va_fold, pred_log))

    return (
        {
            "r2_log_mean": round(float(np.mean(r2s_log)), 3),
            "r2_log_std": round(float(np.std(r2s_log)), 3),
            "mae_log_mean": round(float(np.mean(maes_log)), 3),
            "mae_log_std": round(float(np.std(maes_log)), 3),
        },
        oof_preds_log,
    )


def stress_test_grouped_locations(
    key: str,
    train_df: pd.DataFrame,
    y_train_log: pd.Series,
    ridge_alpha: float = 1.0,
    rf_n_est: int = 200,
    rf_leaf: int = 3,
) -> Dict[str, float]:
    """
    Stress-test evaluation on unseen locations using GroupKFold.
    Measures generalization difficulty when properties belong to entirely new localities.
    """
    gkf = GroupKFold(n_splits=5)
    groups = train_df["location"].values
    maes_log, r2s_log = [], []
    scale_linear = key in ("linear", "ridge")

    for tr_idx, va_idx in gkf.split(train_df, groups=groups):
        tr_fold, va_fold = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        y_tr_fold = y_train_log.iloc[tr_idx]
        y_va_fold = y_train_log.iloc[va_idx]

        prep = DataPreprocessor().fit(tr_fold)
        X_tr = prep.transform(tr_fold, scale_linear=scale_linear)
        X_va = prep.transform(va_fold, scale_linear=scale_linear)

        model = make_model(key, ridge_alpha, rf_n_est, rf_leaf)
        model.fit(X_tr, y_tr_fold)
        pred_log = model.predict(X_va)

        maes_log.append(mean_absolute_error(y_va_fold, pred_log))
        r2s_log.append(r2_score(y_va_fold, pred_log))

    return {
        "r2_log_mean": round(float(np.mean(r2s_log)), 3),
        "r2_log_std": round(float(np.std(r2s_log)), 3),
        "mae_log_mean": round(float(np.mean(maes_log)), 3),
        "mae_log_std": round(float(np.std(maes_log)), 3),
    }


def evaluate_test(model, X_test: pd.DataFrame, y_test: pd.Series, y_test_log: pd.Series) -> Dict[str, float]:
    """Score model on the hold-out test set in both log and original Lakh INR space."""
    pred_log = model.predict(X_test)
    pred_lakh = np.expm1(pred_log)

    return {
        "r2_log": round(float(r2_score(y_test_log, pred_log)), 3),
        "mae_log": round(float(mean_absolute_error(y_test_log, pred_log)), 3),
        "mae": round(float(mean_absolute_error(y_test, pred_lakh)), 2),
        "medae": round(float(median_absolute_error(y_test, pred_lakh)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred_lakh))), 2),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def save_chart(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, name), dpi=110)
    plt.close(fig)


def generate_eda_charts(df: pd.DataFrame) -> None:
    # 1. Price distribution
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(df[TARGET], bins=60, color="#4c72b0", edgecolor="white")
    ax.set_title("Distribution of House Prices (Lakh INR)")
    ax.set_xlabel("Price (Lakh INR)")
    ax.set_ylabel("Listings")
    save_chart(fig, "price_distribution.png")

    # 2. Price vs Area
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(df["total_sqft"], df[TARGET], s=6, alpha=0.35, color="#4c72b0")
    coeffs = np.polyfit(df["total_sqft"], df[TARGET], 1)
    xs = np.linspace(df["total_sqft"].min(), df["total_sqft"].max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="#c44e52", lw=2, label="linear fit")
    ax.set_title("Price vs Total Area")
    ax.set_xlabel("Total area (sqft)")
    ax.set_ylabel("Price (Lakh INR)")
    ax.legend()
    save_chart(fig, "price_vs_sqft.png")

    # 3. Price by BHK
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bhk_data = df.dropna(subset=["bhk"])
    common_bhks = sorted([b for b in bhk_data["bhk"].unique() if b <= 8])
    ax.boxplot(
        [bhk_data.loc[bhk_data["bhk"] == b, TARGET] for b in common_bhks],
        tick_labels=[f"{int(b)} BHK" for b in common_bhks],
        patch_artist=True,
    )
    ax.set_title("Price by BHK (1 to 8 BHK)")
    ax.set_ylabel("Price (Lakh INR)")
    save_chart(fig, "price_by_bhk.png")

    # 4. Top Locations
    fig, ax = plt.subplots(figsize=(7, 4.8))
    counts = df["location"].value_counts().head(15)
    ax.barh(range(len(counts)), counts.values, color="#55a868")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Most Common Locations in Historical Dataset")
    ax.set_xlabel("Listings")
    save_chart(fig, "top_locations.png")


def generate_model_charts(
    models: Dict[str, Any],
    test_metrics_dict: Dict[str, Dict[str, float]],
    X_test_rf: pd.DataFrame,
    y_test: pd.Series,
    y_test_log: pd.Series,
    best_key: str,
    perm_importances: Dict[str, float],
) -> None:
    # 1. Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = [MODEL_NAMES[k] for k in models]
    axes[0].bar(names, [test_metrics_dict[k]["mae_log"] for k in models], color=["#4c72b0", "#55a868", "#c44e52"])
    for i, k in enumerate(models):
        axes[0].text(i, test_metrics_dict[k]["mae_log"] + 0.005, f"{test_metrics_dict[k]['mae_log']:.3f}", ha="center", fontsize=9)
    axes[0].set_title("Mean Absolute Error (log price)")
    axes[0].set_ylabel("MAE (log)")

    axes[1].bar(names, [test_metrics_dict[k]["r2_log"] for k in models], color=["#4c72b0", "#55a868", "#c44e52"])
    for i, k in enumerate(models):
        axes[1].text(i, test_metrics_dict[k]["r2_log"] + 0.01, f"{test_metrics_dict[k]['r2_log']:.3f}", ha="center", fontsize=9)
    axes[1].set_title("R$^2$ (log price)")
    axes[1].set_ylabel("R$^2$")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save_chart(fig, "model_comparison.png")

    # 2. Feature Importance (Random Forest MDI & Permutation)
    rf = models["random_forest"]
    importances = dict(zip(FEATURES, rf.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(importances), list(importances.values()), color="#4c72b0")
    ax.set_title("Feature Importance (Random Forest MDI)")
    ax.set_xlabel("Mean Decrease in Impurity (Variance Reduction)")
    save_chart(fig, "feature_importance.png")

    # Permutation importance chart
    perm_sorted = dict(sorted(perm_importances.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(perm_sorted), list(perm_sorted.values()), color="#55a868")
    ax.set_title("Permutation Feature Importance (Held-out Test Set)")
    ax.set_xlabel("Drop in R$^2$ on Feature Shuffle")
    save_chart(fig, "permutation_importance.png")

    # 3. Actual vs Predicted (Best Model)
    best_model = models[best_key]
    pred_log = best_model.predict(X_test_rf)
    pred_lakh = np.expm1(pred_log)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred_lakh, s=8, alpha=0.35, color="#4c72b0")
    lim = [0, max(y_test.max(), pred_lakh.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1.5, label="y = x (Perfect fit)")
    ax.set_title(f"Actual vs Predicted ({MODEL_NAMES[best_key]})")
    ax.set_xlabel("Actual price (Lakh INR)")
    ax.set_ylabel("Predicted price (Lakh INR)")
    ax.legend()
    save_chart(fig, "actual_vs_predicted.png")

    # 4. Residuals
    residuals = y_test - pred_lakh
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(residuals, bins=60, color="#4c72b0", edgecolor="white")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("Residual Distribution (Lakh INR)")
    axes[0].set_xlabel("Residual (Actual - Predicted)")

    axes[1].scatter(pred_lakh, residuals, s=6, alpha=0.35, color="#55a868")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted price (Lakh INR)")
    axes[1].set_ylabel("Residual (Lakh INR)")
    save_chart(fig, "residuals.png")


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading raw dataset from {DATA_PATH}...")
    dataset_hash = compute_file_sha256(DATA_PATH)
    raw_df = pd.read_csv(DATA_PATH)
    total_raw_rows = len(raw_df)
    print(f"Loaded {total_raw_rows} listings (SHA-256: {dataset_hash[:12]}...).")

    cleaned_df, n_dropped = clean_dataset(raw_df)
    print(f"Filtered {n_dropped} impossible/corrupted rows ({n_dropped / total_raw_rows * 100:.2f}%). Cleaned: {len(cleaned_df)} rows.")

    print("Generating EDA charts...")
    generate_eda_charts(cleaned_df)

    # Outer split: Split BEFORE any learned imputation or encodings!
    # The holdout test set is kept completely untouched until final evaluation.
    train_df, test_df = train_test_split(
        cleaned_df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    print(f"Outer split: Train = {len(train_df):,} listings, Test = {len(test_df):,} listings.")

    # 1. Hyperparameter Tuning using strictly fold-local cross-validation
    print("\n--- Model Tuning & Cross-Validation (Strictly Fold-Local Preprocessing) ---")
    ridge_alpha = choose_ridge_alpha_fold_local(train_df, y_train_log)
    print(f"Chosen Ridge alpha: {ridge_alpha}")

    print("Tuning Random Forest parameters via fold-local 5-fold CV...")
    rf_n_est, rf_leaf = tune_random_forest_fold_local(train_df, y_train_log)
    print(f"Chosen RF parameters: n_estimators={rf_n_est}, min_samples_leaf={rf_leaf}")

    # 2. Cross-Validation on training split
    cv_results = {}
    oof_predictions = {}
    for key in ("linear", "ridge", "random_forest"):
        alpha = ridge_alpha if key == "ridge" else 1.0
        cv_metric, oof_preds = cross_validate_fold_local(
            key, train_df, y_train_log, alpha, rf_n_est, rf_leaf
        )
        cv_results[key] = cv_metric
        oof_predictions[key] = oof_preds
        print(
            f"{MODEL_NAMES[key]}: CV R2(log)={cv_metric['r2_log_mean']:.3f} (+/-{cv_metric['r2_log_std']:.3f}) | "
            f"CV MAE(log)={cv_metric['mae_log_mean']:.3f} (+/-{cv_metric['mae_log_std']:.3f})"
        )

    # 3. Location-Grouped Stress Test (Harder Generalization Scenario)
    print("\n--- Location-Grouped Validation Stress Test (Unseen Locality Evaluation) ---")
    grouped_stress_results = {}
    for key in ("linear", "ridge", "random_forest"):
        alpha = ridge_alpha if key == "ridge" else 1.0
        grp_score = stress_test_grouped_locations(
            key, train_df, y_train_log, alpha, rf_n_est, rf_leaf
        )
        grouped_stress_results[key] = grp_score
        print(f"{MODEL_NAMES[key]}: Group-CV R2(log)={grp_score['r2_log_mean']:.3f} | MAE(log)={grp_score['mae_log_mean']:.3f}")

    # 4. Fit Final Production Preprocessor on FULL Training Split Only
    print("\nFitting final DataPreprocessor bundle on full training split...")
    final_preprocessor = DataPreprocessor().fit(train_df)
    X_train_raw = final_preprocessor.transform(train_df, scale_linear=False)
    X_train_scaled = final_preprocessor.transform(train_df, scale_linear=True)

    # Transform hold-out test set using the fitted preprocessor
    X_test_raw = final_preprocessor.transform(test_df, scale_linear=False)
    X_test_scaled = final_preprocessor.transform(test_df, scale_linear=True)

    # 5. Train Final Models and Evaluate on Held-out Test Set
    print("\n--- Final Model Training and Hold-out Test Set Evaluation ---")
    models = {}
    test_metrics_dict = {}
    for key in ("linear", "ridge", "random_forest"):
        alpha = ridge_alpha if key == "ridge" else 1.0
        X_tr = X_train_scaled if key in ("linear", "ridge") else X_train_raw
        X_te = X_test_scaled if key in ("linear", "ridge") else X_test_raw

        model = make_model(key, alpha, rf_n_est, rf_leaf)
        model.fit(X_tr, y_train_log)
        tm = evaluate_test(model, X_te, y_test, y_test_log)

        models[key] = model
        test_metrics_dict[key] = tm
        print(
            f"{MODEL_NAMES[key]}: Test R2(log)={tm['r2_log']:.3f} | MAE(log)={tm['mae_log']:.3f} | "
            f"MAE={tm['mae']:.2f} Lakh | MedAE={tm['medae']:.2f} Lakh | RMSE={tm['rmse']:.2f} Lakh"
        )

    best_key = max(test_metrics_dict, key=lambda k: test_metrics_dict[k]["r2_log"])
    print(f"\nBest model selected: {MODEL_NAMES[best_key]}")

    # 6. Conformal Error Interval Calibration
    # Uses 80% quantile of out-of-fold log residuals on training set
    rf_oof_residuals = np.abs(y_train_log.values - oof_predictions[best_key])
    q_80 = float(np.quantile(rf_oof_residuals, 0.80))

    # Evaluate test set coverage
    rf_test_pred_log = models[best_key].predict(X_test_raw)
    test_log_errors = np.abs(y_test_log.values - rf_test_pred_log)
    actual_test_coverage = float(np.mean(test_log_errors <= q_80))
    print(f"Error Calibration (80% nominal coverage): log_margin = {q_80:.4f}, empirical test coverage = {actual_test_coverage * 100:.1f}%.")

    # 7. Permutation Feature Importance on Held-out Test Set
    print("Computing permutation importance for Random Forest on hold-out test set...")
    perm = permutation_importance(
        models["random_forest"],
        X_test_raw,
        y_test_log,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_importances = {
        feat: round(float(perm.importances_mean[i]), 4)
        for i, feat in enumerate(FEATURES)
    }

    # 8. Generate Updated Model Performance Charts
    print("Generating diagnostic charts in artifacts/...")
    generate_model_charts(
        models,
        test_metrics_dict,
        X_test_raw,
        y_test,
        y_test_log,
        best_key,
        perm_importances,
    )

    # 9. Build Comprehensive Metrics and Provenance Manifest
    metrics_payload = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "file": "data/Bengaluru_House_Data.csv",
            "sha256": dataset_hash,
            "source": "https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data",
            "snapshot_years": "2017-2020",
            "rows": total_raw_rows,
            "columns": len(raw_df.columns),
        },
        "split": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "method": "Leakage-free train_test_split; fold-local preprocessing in CV",
        },
        "target": "price (Lakh INR)",
        "target_transform": "log1p (predictions inverted back to Lakh INR via expm1)",
        "rows_dropped": n_dropped,
        "features": list(FEATURES),
        "supported_area_types": list(SUPPORTED_AREA_TYPES),
        "best_model": best_key,
        "error_calibration": {
            "method": "Empirical quantile of out-of-fold CV log residuals",
            "nominal_coverage": 0.80,
            "log_margin": round(q_80, 4),
            "test_coverage": round(actual_test_coverage, 4),
            "interpretation": (
                "Reflects historical error dispersion on 2017-2020 listing records. "
                "Not a current market uncertainty estimate."
            ),
        },
        "permutation_importance": perm_importances,
        "location_grouped_stress_test": {
            "description": "5-fold GroupKFold by location to evaluate generalization to entirely unseen localities",
            "models": grouped_stress_results,
        },
        "models": {
            key: {
                "name": MODEL_NAMES[key],
                "params": (
                    {"alpha": ridge_alpha}
                    if key == "ridge"
                    else {"n_estimators": rf_n_est, "min_samples_leaf": rf_leaf}
                    if key == "random_forest"
                    else {}
                ),
                "test": test_metrics_dict[key],
                "cv": cv_results[key],
            }
            for key in ("linear", "ridge", "random_forest")
        },
    }

    manifest_payload = {
        "timestamp": datetime.now().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": dataset_hash,
        "split_seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_dropped": n_dropped,
        "bhk_median": final_preprocessor.bhk_median_,
        "feature_order": list(FEATURES),
        "models_summary": {
            k: {
                "name": MODEL_NAMES[k],
                "test_r2_log": test_metrics_dict[k]["r2_log"],
                "test_mae_lakh": test_metrics_dict[k]["mae"],
            }
            for k in models
        },
    }

    # 10. Persist Artifacts
    print("Saving trained model joblibs, preprocessor bundle, metrics and manifest...")
    for key, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{key}.joblib"), compress=3)

    joblib.dump(final_preprocessor.to_bundle(), os.path.join(MODELS_DIR, "encoders.joblib"), compress=3)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    with open(os.path.join(MODELS_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest_payload, fh, indent=2)

    print("\nTraining completed successfully with zero preprocessing leakage!")


if __name__ == "__main__":
    main()
