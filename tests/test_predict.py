"""
Sanity tests for the model artifacts and the prediction path.

Run from the repository root:
    pytest -q
"""

import json
import math
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.predict import load_artifacts, predict_all  # noqa: E402

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
    assert os.path.exists(os.path.join(MODELS_DIR, "metrics.json"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert metrics["best_model"] in metrics["models"]
    for key, meta in metrics["models"].items():
        for field in ("r2_log", "mae_log", "mae", "rmse"):
            assert field in meta["test"]
            assert math.isfinite(meta["test"][field])


def test_predict_all_returns_finite_prices(artifacts):
    results = predict_all(VALID_ROW, artifacts)
    assert set(results) == {"linear", "ridge", "random_forest"}
    for value in results.values():
        assert math.isfinite(value)
        assert value > 0


def test_predictions_are_plausible(artifacts):
    results = predict_all(VALID_ROW, artifacts)
    # A typical 2BHK, 1200 sqft listing in this dataset sits between
    # 20 and 200 Lakh; all models should land in a sane band.
    for value in results.values():
        assert 5 < value < 1000


def test_unseen_society_does_not_crash(artifacts):
    row = dict(VALID_ROW, society="Society That Does Not Exist 123")
    results = predict_all(row, artifacts)
    for value in results.values():
        assert math.isfinite(value)


def test_blank_and_placeholder_inputs_treated_as_missing(artifacts):
    for society in ("", "   ", "N/A", "nan"):
        row = dict(VALID_ROW, society=society)
        results = predict_all(row, artifacts)
        for value in results.values():
            assert math.isfinite(value)


def test_bigger_house_costs_more(artifacts):
    small = dict(VALID_ROW, total_sqft=700, bhk=1)
    large = dict(VALID_ROW, total_sqft=2400, bhk=4)
    small_prices = predict_all(small, artifacts)
    large_prices = predict_all(large, artifacts)
    for key in small_prices:
        assert large_prices[key] > small_prices[key]


def test_api_server_imports():
    """The FastAPI app must import and expose the expected routes."""
    from api.server import app

    routes = {route.path for route in app.routes}
    for path in ("/", "/health", "/models", "/locations", "/meta", "/predict"):
        assert path in routes
