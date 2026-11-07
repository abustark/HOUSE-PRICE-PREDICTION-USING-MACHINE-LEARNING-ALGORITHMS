"""
Sanity tests for model artifacts, consistency, and inference predictions.

Run from repository root:
    pytest -q
"""

import json
import math
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.encoding import FEATURES, SCHEMA_VERSION, SUPPORTED_AREA_TYPES  # noqa: E402
from src.predict import (  # noqa: E402
    load_artifacts,
    predict_all,
    predict_detailed,
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

VALID_ROW = {
    "area_type": "Super built-up Area",
    "location": "Whitefield",
    "society": "",
    "bhk": 2,
    "total_sqft": 1200,
    "bath": 2,
    "balcony": 1,
    "completion_year": 0,
}


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_artifacts_exist(artifacts):
    assert set(artifacts["models"]) == {"linear", "ridge", "random_forest"}
    metrics_file = os.path.join(MODELS_DIR, "metrics.json")
    assert os.path.exists(metrics_file)

    with open(metrics_file, encoding="utf-8") as fh:
        metrics = json.load(fh)

    assert metrics["best_model"] in metrics["models"]
    for key, meta in metrics["models"].items():
        for field in ("r2_log", "mae_log", "mae", "rmse"):
            assert field in meta["test"]
            assert math.isfinite(meta["test"][field])


def test_artifact_schema_consistency(artifacts):
    bundle = artifacts["bundle"]
    assert bundle["feature_order"] == FEATURES
    for area in SUPPORTED_AREA_TYPES:
        assert area in bundle["supported_area_types"] or hasattr(bundle.get("le_area_type"), "classes_")


def test_predict_all_returns_finite_prices(artifacts):
    results = predict_all(VALID_ROW, artifacts)
    assert set(results) == {"linear", "ridge", "random_forest"}
    for value in results.values():
        assert math.isfinite(value)
        assert value > 0


def test_predictions_are_plausible(artifacts):
    results = predict_all(VALID_ROW, artifacts)
    # A typical 2BHK, 1200 sqft listing in Whitefield sits between
    # 20 and 200 Lakh; all models should land in a sane band.
    for value in results.values():
        assert 5 < value < 1000


def test_unseen_society_does_not_crash(artifacts):
    row = dict(VALID_ROW, society="Society That Does Not Exist 123")
    results = predict_all(row, artifacts)
    for value in results.values():
        assert math.isfinite(value)


def test_unseen_location_does_not_crash(artifacts):
    row = dict(VALID_ROW, location="Completely New Unseen Locality 999")
    results = predict_all(row, artifacts)
    for value in results.values():
        assert math.isfinite(value)
        assert value > 0


def test_blank_and_placeholder_inputs_treated_as_missing(artifacts):
    for society in ("", "   ", "N/A", "nan"):
        row = dict(VALID_ROW, society=society)
        results = predict_all(row, artifacts)
        for value in results.values():
            assert math.isfinite(value)


def test_bigger_house_costs_more(artifacts):
    small = dict(VALID_ROW, total_sqft=700, bhk=1)
    large = dict(VALID_ROW, total_sqft=2400, bhk=4, bath=3)
    small_prices = predict_all(small, artifacts)
    large_prices = predict_all(large, artifacts)
    for key in small_prices:
        assert large_prices[key] > small_prices[key]


def test_predict_detailed_contains_historical_intervals_and_context(artifacts):
    detail = predict_detailed(VALID_ROW, artifacts)
    assert detail["is_valid"] is True
    assert "historical_error_range" in detail
    assert "lower_lakh" in detail["historical_error_range"]
    assert "upper_lakh" in detail["historical_error_range"]
    assert detail["historical_error_range"]["lower_lakh"] < detail["historical_error_range"]["upper_lakh"]
    assert "disclaimer" in detail
    assert "warnings" in detail


def test_api_server_imports():
    """The FastAPI app must import and expose the expected routes."""
    from api.server import app

    routes = {route.path for route in app.routes}
    for path in ("/", "/health", "/models", "/locations", "/meta", "/manifest", "/predict"):
        assert path in routes
