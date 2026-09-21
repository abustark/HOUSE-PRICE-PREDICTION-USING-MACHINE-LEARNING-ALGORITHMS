"""
Shared helpers for the web app.

Loads the trained models and encoders saved under models/ and converts a
single raw input row into the feature matrix the models were trained on.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.encoding import FEATURES, MISSING, clean_category  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "linear": "linear.joblib",
    "ridge": "ridge.joblib",
    "random_forest": "random_forest.joblib",
}
# Linear models are trained on standardized features, so the scaler has to
# be applied before predicting with them. Random Forest does not care about
# feature scale.
SCALED_MODELS = {"linear", "ridge"}


def load_artifacts():
    """Load the saved models, encoders and evaluation metrics."""
    bundle = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    models = {
        key: joblib.load(os.path.join(MODELS_DIR, fname))
        for key, fname in MODEL_FILES.items()
    }
    return {"models": models, "bundle": bundle, "metrics": metrics}


def encode_row(row, bundle):
    """
    Convert one raw input row (dict) into a one-row feature DataFrame
    using the same rules as the training pipeline.
    """
    encoded = {
        "total_sqft": float(row["total_sqft"]),
        "bath": float(row["bath"]),
        "balcony": float(row["balcony"]),
        "bhk": int(row["bhk"]),
        "area_type": str(row["area_type"]).strip(),
        "completion_year": int(row["completion_year"]),
        "location": clean_category(row.get("location")),
        "society": clean_category(row.get("society")),
    }

    # area_type: label encoding with a safe fallback for unseen values.
    le = bundle["le_area_type"]
    if encoded["area_type"] not in le.classes_:
        encoded["area_type"] = le.classes_[0]
    encoded["area_type"] = int(le.transform([encoded["area_type"]])[0])

    # High-cardinality categories: frequency encoding. A location or society
    # that was not in the training data gets a frequency of 0.
    encoded["location"] = float(bundle["location_freq"].get(encoded["location"], 0.0))
    encoded["society"] = float(bundle["society_freq"].get(encoded["society"], 0.0))

    return pd.DataFrame([encoded])[bundle["feature_order"]]


def predict_all(row, artifacts):
    """Run the input row through every saved model; returns {key: price}."""
    bundle = artifacts["bundle"]
    features = encode_row(row, bundle)
    scaled = None
    results = {}
    for key, model in artifacts["models"].items():
        if key in SCALED_MODELS:
            if scaled is None:
                scaled = pd.DataFrame(
                    bundle["scaler"].transform(features),
                    columns=bundle["feature_order"],
                    index=features.index,
                )
            X = scaled
        else:
            X = features
        # Models are trained on log(1 + price); invert back to Lakh INR.
        results[key] = float(np.expm1(model.predict(X)[0]))
    return results
