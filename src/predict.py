"""
Shared helpers for the web apps and API.

Loads the trained models, preprocessor bundle, and metrics from models/.
Performs startup schema validation, input validation, and inference with
honest historical error ranges and domain diagnostics.
"""

import json
import os
import sys
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.encoding import (  # noqa: E402
    FEATURES,
    MISSING,
    SCHEMA_VERSION,
    SUPPORTED_AREA_TYPES,
    DataPreprocessor,
    clean_category,
    validate_input,
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "linear": "linear.joblib",
    "ridge": "ridge.joblib",
    "random_forest": "random_forest.joblib",
}

SCALED_MODELS = {"linear", "ridge"}


def validate_artifacts(artifacts: Dict[str, Any]) -> None:
    """Validate model bundle compatibility at startup. Fail fast if incompatible."""
    bundle = artifacts.get("bundle")
    if not isinstance(bundle, dict):
        raise RuntimeError("Corrupted or missing preprocessor bundle in models/encoders.joblib")

    # Check feature ordering
    bundle_features = bundle.get("feature_order")
    if bundle_features != FEATURES:
        raise RuntimeError(
            f"Feature schema mismatch. Expected {FEATURES}, but loaded bundle has {bundle_features}."
        )

    # Check models
    models = artifacts.get("models", {})
    missing_models = set(MODEL_FILES) - set(models)
    if missing_models:
        raise RuntimeError(f"Missing model artifacts for: {', '.join(missing_models)}")

    # Check metrics
    metrics = artifacts.get("metrics", {})
    if "best_model" not in metrics or metrics["best_model"] not in models:
        raise RuntimeError("Invalid or missing 'best_model' specification in models/metrics.json")


def load_artifacts() -> Dict[str, Any]:
    """Load and validate the saved models, preprocessor bundle, and evaluation metrics."""
    encoders_path = os.path.join(MODELS_DIR, "encoders.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    manifest_path = os.path.join(MODELS_DIR, "manifest.json")

    if not os.path.exists(encoders_path):
        raise RuntimeError(f"Required artifact not found: {encoders_path}")
    if not os.path.exists(metrics_path):
        raise RuntimeError(f"Required artifact not found: {metrics_path}")

    bundle = joblib.load(encoders_path)
    with open(metrics_path, encoding="utf-8") as fh:
        metrics = json.load(fh)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    models = {}
    for key, fname in MODEL_FILES.items():
        fpath = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(fpath):
            raise RuntimeError(f"Model artifact not found: {fpath}")
        models[key] = joblib.load(fpath)

    artifacts = {
        "models": models,
        "bundle": bundle,
        "metrics": metrics,
        "manifest": manifest,
        "preprocessor": DataPreprocessor.from_bundle(bundle),
    }
    validate_artifacts(artifacts)
    return artifacts


def encode_row(row: Dict[str, Any], bundle: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert one raw input row (dict) into a one-row feature DataFrame.
    Validates input and applies the saved preprocessor rules.
    """
    v_res = validate_input(row)
    if not v_res.is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(v_res.errors)}")

    preprocessor = DataPreprocessor.from_bundle(bundle)
    return preprocessor.transform_row(v_res.sanitized, scale_linear=False)


def predict_all(row: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, float]:
    """
    Run the input row through every saved model; returns {model_key: price_in_lakh}.
    Preserves exact backward-compatibility with existing tests and scripts.
    """
    v_res = validate_input(row)
    if not v_res.is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(v_res.errors)}")

    preprocessor = artifacts.get("preprocessor") or DataPreprocessor.from_bundle(artifacts["bundle"])
    features_raw = preprocessor.transform_row(v_res.sanitized, scale_linear=False)
    features_scaled = preprocessor.transform_row(v_res.sanitized, scale_linear=True)

    results: Dict[str, float] = {}
    for key, model in artifacts["models"].items():
        X = features_scaled if key in SCALED_MODELS else features_raw
        pred_log = float(model.predict(X)[0])
        results[key] = max(0.0, float(np.expm1(pred_log)))
    return results


def predict_detailed(
    row: Dict[str, Any],
    artifacts: Dict[str, Any],
    selected_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive prediction output including historical error ranges,
    unseen category flags, and out-of-domain extrapolation warnings.
    """
    v_res = validate_input(row)
    if not v_res.is_valid:
        return {
            "is_valid": False,
            "errors": v_res.errors,
            "warnings": v_res.warnings,
        }

    bundle = artifacts["bundle"]
    metrics = artifacts["metrics"]
    best_key = metrics.get("best_model", "random_forest")
    chosen_key = (selected_model or best_key).lower()
    if chosen_key not in artifacts["models"] and chosen_key != "all":
        chosen_key = best_key

    preprocessor = artifacts.get("preprocessor") or DataPreprocessor.from_bundle(bundle)
    features_raw = preprocessor.transform_row(v_res.sanitized, scale_linear=False)
    features_scaled = preprocessor.transform_row(v_res.sanitized, scale_linear=True)

    all_prices = {}
    for key, model in artifacts["models"].items():
        X = features_scaled if key in SCALED_MODELS else features_raw
        pred_log = float(model.predict(X)[0])
        price = max(0.0, float(np.expm1(pred_log)))
        all_prices[key] = {
            "name": metrics["models"][key]["name"],
            "price_lakh": round(price, 2),
            "price_inr": int(round(price * 100000)),
        }

    eval_key = chosen_key if chosen_key != "all" else best_key
    target_price = all_prices[eval_key]["price_lakh"]

    # Historical error calibration:
    # Uses empirical 80% out-of-fold log-residual quantile from CV
    calib = metrics.get("error_calibration", {})
    log_margin = calib.get("log_margin", 0.28)
    X_chosen = features_scaled if eval_key in SCALED_MODELS else features_raw
    chosen_pred_log = float(artifacts["models"][eval_key].predict(X_chosen)[0])

    lower_lakh = max(0.0, float(np.expm1(chosen_pred_log - log_margin)))
    upper_lakh = float(np.expm1(chosen_pred_log + log_margin))

    loc_name = v_res.sanitized.get("location", "")
    soc_name = v_res.sanitized.get("society", "")
    unseen_loc = bool(loc_name and loc_name != MISSING and loc_name not in bundle["location_freq"])
    unseen_soc = bool(soc_name and soc_name != MISSING and soc_name not in bundle["society_freq"])

    return {
        "is_valid": True,
        "model_key": eval_key,
        "model_name": metrics["models"][eval_key]["name"],
        "predicted_price_lakh": round(target_price, 2),
        "predicted_price_inr": int(round(target_price * 100000)),
        "historical_error_range": {
            "lower_lakh": round(lower_lakh, 2),
            "upper_lakh": round(upper_lakh, 2),
            "confidence_label": "80% estimated historical error range",
            "historical_test_coverage": calib.get("test_coverage", 0.80),
        },
        "all_models": all_prices,
        "unseen_location": unseen_loc,
        "unseen_society": unseen_soc,
        "warnings": v_res.warnings,
        "disclaimer": (
            "Educational estimate based on historical 2017–2020 listings. "
            "Not a real-estate appraisal or current market quote."
        ),
    }
