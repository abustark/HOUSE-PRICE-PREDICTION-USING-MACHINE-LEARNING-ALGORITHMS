"""
Training pipeline for the Bengaluru house price prediction project.

Reads the raw dataset from data/, cleans and engineers the features, trains
three regression models (Linear Regression, Ridge Regression and Random
Forest), evaluates them with held-out test metrics and 5-fold cross
validation, and saves everything the web app needs:

    models/linear.joblib          - Linear Regression model
    models/ridge.joblib           - Ridge Regression model
    models/random_forest.joblib   - Random Forest Regressor
    models/encoders.joblib        - encoders + frequency maps + scaler
    models/metrics.json           - evaluation metrics for the report page

Charts used by the web app are written to artifacts/.

Run with:
    python src/train.py
"""

import json
import os
import sys
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.encoding import FEATURES, MISSING, extract_bhk, parse_availability, parse_sqft  # noqa: E402

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


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def prepare(df):
    """Clean the raw data and add the engineered columns."""
    data = df.copy()

    data["total_sqft"] = data["total_sqft"].map(parse_sqft)
    data["bhk"] = data["size"].map(extract_bhk)

    for col in ("bath", "balcony"):
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    data["bhk"] = data["bhk"].fillna(data["bhk"].median()).astype(int)

    # Drop rows with missing or physically impossible measurements
    # (scraper errors such as a 3-bedroom flat of 11 sqft, a 52,000 sqft
    # house or a 30-acre listing). These rows would wreck the linear models.
    sane = (
        data["total_sqft"].notna()
        & (data["total_sqft"] >= 100)
        & (data["total_sqft"] <= 10000)
        & (data["bath"] <= 12)
        & (data["bhk"] <= 14)
    )
    n_dropped = int((~sane).sum())
    data = data[sane].copy()

    data["total_sqft"] = data["total_sqft"].fillna(data["total_sqft"].median())

    # 'availability' mixes 'Ready To Move' with expected completion dates
    # such as '18-Dec'. It becomes a single numeric feature: the expected
    # completion year, where 0 means ready to move.
    data["completion_year"] = data["availability"].map(parse_availability).astype(int)

    # Missing locations/societies get a dedicated category instead of 0.
    # Whitespace is stripped so ' Anekal' and 'Anekal' are the same value.
    data["location"] = data["location"].astype("string").str.strip().fillna(MISSING).astype(str)
    data["society"] = data["society"].astype("string").str.strip().fillna(MISSING).astype(str)
    data["area_type"] = data["area_type"].fillna("Other").astype(str)

    return data, n_dropped


def build_encoders(train_df, all_area_types):
    """
    area_type is label encoded (fitted on the full data: only 3 stable
    categories). Location and society have thousands of unique values, so
    they are frequency encoded (how often the value appears in the training
    split). Frequency maps are built from the training split only, so no
    information from the test set leaks in.
    """
    return {
        "le_area_type": LabelEncoder().fit(sorted(all_area_types)),
        "location_freq": train_df["location"].value_counts().to_dict(),
        "society_freq": train_df["society"].value_counts().to_dict(),
        "feature_order": list(FEATURES),
    }


def encode(df, encoders):
    """Apply the saved encoding rules to a prepared DataFrame."""
    out = pd.DataFrame(index=df.index)
    out["total_sqft"] = df["total_sqft"].astype(float)
    out["bath"] = df["bath"].astype(float)
    out["balcony"] = df["balcony"].astype(float)
    out["bhk"] = df["bhk"].astype(int)
    out["area_type"] = encoders["le_area_type"].transform(df["area_type"])
    out["completion_year"] = df["completion_year"].astype(int)
    out["location"] = df["location"].map(encoders["location_freq"]).fillna(0).astype(float)
    out["society"] = df["society"].map(encoders["society_freq"]).fillna(0).astype(float)
    return out[encoders["feature_order"]]


# ---------------------------------------------------------------------------
# Modelling helpers
# ---------------------------------------------------------------------------
def make_model(key, ridge_alpha=1.0, rf_n_est=100, rf_leaf=5):
    if key == "linear":
        return LinearRegression()
    if key == "ridge":
        return Ridge(alpha=ridge_alpha)
    # min_samples_leaf keeps the trees (and the saved file) a reasonable
    # size; it also acts as a light regularizer.
    return RandomForestRegressor(
        n_estimators=rf_n_est, min_samples_leaf=rf_leaf, random_state=RANDOM_STATE
    )


def choose_ridge_alpha(X, y_log):
    """Pick the Ridge alpha with the lowest mean CV MAE in log price."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_alpha, best_score = 1.0, np.inf
    for alpha in (0.1, 1.0, 10.0, 100.0):
        scores = []
        for tr_idx, va_idx in kf.split(X):
            model = Ridge(alpha=alpha)
            model.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            scores.append(mean_absolute_error(y_log.iloc[va_idx], pred))
        if np.mean(scores) < best_score:
            best_alpha, best_score = alpha, np.mean(scores)
    return best_alpha


def cross_validate(key, X, y_log, ridge_alpha=1.0, rf_n_est=100, rf_leaf=5):
    """5-fold CV on the training split, scored in log price."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    maes, r2s = [], []
    for tr_idx, va_idx in kf.split(X):
        model = make_model(key, ridge_alpha, rf_n_est, rf_leaf)
        model.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
        pred = model.predict(X.iloc[va_idx])
        maes.append(mean_absolute_error(y_log.iloc[va_idx], pred))
        r2s.append(r2_score(y_log.iloc[va_idx], pred))
    return {
        "mae_log_mean": round(float(np.mean(maes)), 3),
        "mae_log_std": round(float(np.std(maes)), 3),
        "r2_log_mean": round(float(np.mean(r2s)), 3),
        "r2_log_std": round(float(np.std(r2s)), 3),
    }


def tune_random_forest(X, y_log):
    """
    Small deterministic grid search on the training split (5-fold CV,
    log-MAE objective). The grid is kept small on purpose so a full
    retrain stays fast.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_score, best_n_est, best_leaf = np.inf, 100, 5
    for n_est in (100, 200):
        for leaf in (3, 5, 10):
            scores = []
            for tr_idx, va_idx in kf.split(X):
                model = RandomForestRegressor(
                    n_estimators=n_est, min_samples_leaf=leaf, random_state=RANDOM_STATE
                )
                model.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
                pred = model.predict(X.iloc[va_idx])
                scores.append(mean_absolute_error(y_log.iloc[va_idx], pred))
            print(f"  RF n_estimators={n_est} min_samples_leaf={leaf} -> MAE(log) {np.mean(scores):.4f}")
            if np.mean(scores) < best_score:
                best_score, best_n_est, best_leaf = np.mean(scores), n_est, leaf
    return best_n_est, best_leaf


def test_metrics(model, X, y, y_log):
    """
    Score a model on the test set.

    R2 and MAE (log) are computed in log price, which is the space the
    models were trained in. MAE/RMSE (Lakh) are reported on the original
    price scale for readability; a handful of very expensive listings
    inflates those numbers, so the log-score metrics are the fair ones.
    """
    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    return {
        "r2_log": round(float(r2_score(y_log, pred_log)), 3),
        "mae_log": round(float(mean_absolute_error(y_log, pred_log)), 3),
        "mae": round(float(mean_absolute_error(y, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, name), dpi=110)
    plt.close(fig)


def eda_charts(data):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(data["price"], bins=60, color="#4c72b0", edgecolor="white")
    ax.set_title("Distribution of House Prices (Lakh INR)")
    ax.set_xlabel("Price (Lakh INR)")
    ax.set_ylabel("Listings")
    save(fig, "price_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(data["total_sqft"], data["price"], s=6, alpha=0.35, color="#4c72b0")
    coeffs = np.polyfit(data["total_sqft"], data["price"], 1)
    xs = np.linspace(data["total_sqft"].min(), data["total_sqft"].max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="#c44e52", lw=2, label="linear fit")
    ax.set_title("Price vs Total Area")
    ax.set_xlabel("Total area (sqft)")
    ax.set_ylabel("Price (Lakh INR)")
    ax.legend()
    save(fig, "price_vs_sqft.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bhk_values = sorted(data["bhk"].unique())
    ax.boxplot(
        [data.loc[data["bhk"] == b, "price"] for b in bhk_values],
        tick_labels=[f"{b} BHK" for b in bhk_values],
        patch_artist=True,
    )
    ax.set_title("Price by BHK")
    ax.set_ylabel("Price (Lakh INR)")
    save(fig, "price_by_bhk.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    counts = data["location"].value_counts().head(15)
    ax.barh(range(len(counts)), counts.values, color="#55a868")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Most Common Locations")
    ax.set_xlabel("Listings")
    save(fig, "top_locations.png")


def model_charts(models, X_test, X_test_scaled, y_test, y_test_log, best_key, encoders):
    def _X(key):
        # Linear models were fitted on standardized features.
        return X_test if key == "random_forest" else X_test_scaled

    test = {
        key: test_metrics(model, _X(key), y_test, y_test_log)
        for key, model in models.items()
    }
    pred_best = np.expm1(models[best_key].predict(_X(best_key)))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = [MODEL_NAMES[k] for k in models]
    axes[0].bar(names, [test[k]["mae_log"] for k in models], color=["#4c72b0", "#55a868", "#c44e52"])
    for i, k in enumerate(models):
        axes[0].text(i, test[k]["mae_log"] + 0.01, f"{test[k]['mae_log']:.3f}", ha="center", fontsize=9)
    axes[0].set_title("Mean Absolute Error (log price)")
    axes[0].set_ylabel("MAE (log)")
    axes[1].bar(names, [test[k]["r2_log"] for k in models], color=["#4c72b0", "#55a868", "#c44e52"])
    for i, k in enumerate(models):
        axes[1].text(i, test[k]["r2_log"] + 0.01, f"{test[k]['r2_log']:.3f}", ha="center", fontsize=9)
    axes[1].set_title("R$^2$ (log price)")
    axes[1].set_ylabel("R$^2$")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "model_comparison.png")

    rf = models["random_forest"]
    importances = dict(zip(encoders["feature_order"], rf.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(importances), list(importances.values()), color="#4c72b0")
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    save(fig, "feature_importance.png")

    pred = pred_best
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, s=8, alpha=0.4, color="#4c72b0")
    lims = [0, max(y_test.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1.5, label="y = x")
    ax.set_title(f"Actual vs Predicted ({MODEL_NAMES[best_key]})")
    ax.set_xlabel("Actual price (Lakh INR)")
    ax.set_ylabel("Predicted price (Lakh INR)")
    ax.legend()
    save(fig, "actual_vs_predicted.png")

    residuals = y_test - pred
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(residuals, bins=60, color="#4c72b0", edgecolor="white")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (Lakh INR)")
    axes[1].scatter(pred, residuals, s=6, alpha=0.35, color="#55a868")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted price (Lakh INR)")
    axes[1].set_ylabel("Residual (Lakh INR)")
    save(fig, "residuals.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(raw)} rows from data/Bengaluru_House_Data.csv")

    data, n_dropped = prepare(raw)
    print(f"Dropped {n_dropped} rows with impossible area/bathroom values")
    eda_charts(data)

    X_prepared, y = data.drop(columns=[TARGET]), data[TARGET]
    X_train_prepared, X_test_prepared, y_train, y_test = train_test_split(
        X_prepared, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    encoders = build_encoders(X_train_prepared, data["area_type"].unique())
    X_train = encode(X_train_prepared, encoders)
    X_test = encode(X_test_prepared, encoders)

    # House prices are heavily right-skewed (a few thousand-Lakh outliers),
    # which makes ordinary linear fits unstable. Models are therefore trained
    # on log(1 + price) and predictions are inverted back to Lakh INR.
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Standardize features for the linear models (Random Forest is scale-free).
    scaler = StandardScaler().fit(X_train)
    encoders["scaler"] = scaler
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), index=X_train.index, columns=FEATURES
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=FEATURES
    )

    ridge_alpha = choose_ridge_alpha(X_train_scaled, y_train_log)
    print(f"Chosen Ridge alpha: {ridge_alpha}")

    print("\nTuning Random Forest (5-fold CV on the training split, log-MAE objective)")
    rf_n_est, rf_leaf = tune_random_forest(X_train, y_train_log)
    print(f"Chosen RF params: n_estimators={rf_n_est}, min_samples_leaf={rf_leaf}")

    metrics = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": {
            "file": "data/Bengaluru_House_Data.csv",
            "source": "https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data",
            "rows": len(raw),
            "columns": len(raw.columns),
        },
        "split": {
            "train": int(X_train.shape[0]),
            "test": int(X_test.shape[0]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "target": "price (Lakh INR)",
        "target_transform": "log1p (predictions inverted before scoring)",
        "rows_dropped": n_dropped,
        "features": list(FEATURES),
        "models": {},
    }

    models = {}
    for key in ("linear", "ridge", "random_forest"):
        alpha = ridge_alpha if key == "ridge" else 1.0
        X_tr = X_train_scaled if key in ("linear", "ridge") else X_train
        X_te = X_test_scaled if key in ("linear", "ridge") else X_test

        print(f"\n=== {MODEL_NAMES[key]} ===")
        cv = cross_validate(key, X_tr, y_train_log, alpha, rf_n_est, rf_leaf)
        model = make_model(key, alpha, rf_n_est, rf_leaf)
        model.fit(X_tr, y_train_log)
        tm = test_metrics(model, X_te, y_test, y_test_log)

        models[key] = model
        metrics["models"][key] = {
            "name": MODEL_NAMES[key],
            "params": (
                {"alpha": alpha} if key == "ridge"
                else {"n_estimators": rf_n_est, "min_samples_leaf": rf_leaf} if key == "random_forest"
                else {}
            ),
            "test": tm,
            "cv": cv,
        }
        print(f"  test  R2(log)={tm['r2_log']:.3f}  MAE(log)={tm['mae_log']:.3f}  MAE={tm['mae']:.2f} Lakh  RMSE={tm['rmse']:.2f} Lakh")
        print(f"  cv    R2(log)={cv['r2_log_mean']:.3f} (+/-{cv['r2_log_std']:.3f})  MAE(log)={cv['mae_log_mean']:.3f} (+/-{cv['mae_log_std']:.3f})")

    best_key = max(metrics["models"], key=lambda k: metrics["models"][k]["test"]["r2_log"])
    metrics["best_model"] = best_key
    print(f"\nBest model by test R2 (log price): {MODEL_NAMES[best_key]}")

    model_charts(models, X_test, X_test_scaled, y_test, y_test_log, best_key, encoders)

    for key, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{key}.joblib"), compress=3)
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"), compress=3)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\nSaved models, encoders, metrics and charts to artifacts/.")


if __name__ == "__main__":
    main()
