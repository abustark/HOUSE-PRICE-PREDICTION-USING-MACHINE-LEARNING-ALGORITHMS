"""
Tests for domain boundary validation, extrapolation diagnostics, and FastAPI endpoints.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
from fastapi.testclient import TestClient

from api.server import app
from src.encoding import SUPPORTED_AREA_TYPES, validate_input

client = TestClient(app)

VALID_PAYLOAD = {
    "area_type": "Super built-up Area",
    "location": "Whitefield",
    "society": "Goted U",
    "bhk": 2,
    "total_sqft": 1200.0,
    "bath": 2,
    "balcony": 1,
    "completion_year": 0,
    "model": "random_forest",
}


# ---------------------------------------------------------------------------
# Domain validation tests
# ---------------------------------------------------------------------------
def test_valid_input_passes():
    res = validate_input(VALID_PAYLOAD)
    assert res.is_valid is True
    assert len(res.errors) == 0


def test_area_type_validation():
    # Unsupported area type must fail validation
    bad_row = dict(VALID_PAYLOAD, area_type="Underwater Dome")
    res = validate_input(bad_row)
    assert res.is_valid is False
    assert any("Unsupported area type" in err for err in res.errors)

    # Whitespace variants of supported types should normalize and pass
    for area in ("Plot  Area", " Carpet Area ", "Built-up  Area"):
        norm_row = dict(VALID_PAYLOAD, area_type=area)
        assert validate_input(norm_row).is_valid is True


def test_boundary_sqft_constraints():
    # Exactly at bounds (100 and 10000) -> valid
    assert validate_input(dict(VALID_PAYLOAD, total_sqft=100.0)).is_valid is True
    assert validate_input(dict(VALID_PAYLOAD, total_sqft=10000.0)).is_valid is True

    # Below minimum (<100) -> invalid
    res_low = validate_input(dict(VALID_PAYLOAD, total_sqft=99.0))
    assert res_low.is_valid is False
    assert any("Total area must be between" in err for err in res_low.errors)

    # Above maximum (>10000) -> invalid
    res_high = validate_input(dict(VALID_PAYLOAD, total_sqft=10001.0))
    assert res_high.is_valid is False
    assert any("Total area must be between" in err for err in res_high.errors)


def test_boundary_bathroom_constraints():
    assert validate_input(dict(VALID_PAYLOAD, bath=1)).is_valid is True
    assert validate_input(dict(VALID_PAYLOAD, bath=12)).is_valid is True

    # bath 0 or 13 -> invalid
    assert validate_input(dict(VALID_PAYLOAD, bath=0)).is_valid is False
    assert validate_input(dict(VALID_PAYLOAD, bath=13)).is_valid is False


def test_boundary_balcony_and_bhk_constraints():
    assert validate_input(dict(VALID_PAYLOAD, balcony=0)).is_valid is True
    assert validate_input(dict(VALID_PAYLOAD, balcony=5)).is_valid is True
    assert validate_input(dict(VALID_PAYLOAD, balcony=6)).is_valid is False

    assert validate_input(dict(VALID_PAYLOAD, bhk=0)).is_valid is True  # Studio
    assert validate_input(dict(VALID_PAYLOAD, bhk=14)).is_valid is True
    assert validate_input(dict(VALID_PAYLOAD, bhk=15)).is_valid is False


def test_extrapolation_warnings_generated():
    # Large area warning (>4000 sqft)
    res_large = validate_input(dict(VALID_PAYLOAD, total_sqft=5000.0, bhk=4))
    assert res_large.is_valid is True
    assert any("top 2% of historical listings" in w for w in res_large.warnings)

    # High BHK warning (>=6)
    res_bhk = validate_input(dict(VALID_PAYLOAD, total_sqft=3500.0, bhk=8))
    assert res_bhk.is_valid is True
    assert any("very rare in this dataset" in w for w in res_bhk.warnings)

    # Unusually compact area per bedroom (<200 sqft/BHK)
    res_cramped = validate_input(dict(VALID_PAYLOAD, total_sqft=300.0, bhk=3))
    assert res_cramped.is_valid is True
    assert any("compact" in w for w in res_cramped.warnings)

    # Unusually high bathrooms relative to BHK (bath > bhk + 2)
    res_bath = validate_input(dict(VALID_PAYLOAD, total_sqft=1200.0, bhk=1, bath=4))
    assert res_bath.is_valid is True
    assert any("significantly exceed bedrooms" in w for w in res_bath.warnings)


# ---------------------------------------------------------------------------
# FastAPI Route Tests
# ---------------------------------------------------------------------------
def test_api_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "linear" in data["models"]
    assert "random_forest" in data["models"]


def test_api_models_endpoint():
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "best_model" in data
    assert len(data["models"]) == 3
    # Check fields present
    for m in data["models"]:
        for field in ("key", "name", "r2_log", "mae_log", "mae_lakh", "rmse_lakh"):
            assert field in m


def test_api_locations_endpoint():
    res = client.get("/locations")
    assert res.status_code == 200
    data = res.json()
    assert "locations" in data
    assert len(data["locations"]) > 500
    assert "Whitefield" in data["locations"]


def test_api_meta_endpoint():
    res = client.get("/meta")
    assert res.status_code == 200
    data = res.json()
    assert "supported_area_types" in data
    assert "validation_constraints" in data


def test_api_predict_valid_request():
    res = client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert data["predicted_price_lakh"] > 0
    assert data["predicted_price_inr"] > 0
    assert "historical_error_range" in data
    assert "all_models" in data
    assert "disclaimer" in data


def test_api_predict_invalid_model():
    bad_model_req = dict(VALID_PAYLOAD, model="quantum_neural_net")
    res = client.post("/predict", json=bad_model_req)
    assert res.status_code == 400


def test_api_predict_out_of_bounds_returns_422():
    # sqft exceeds training maximum (e.g. 25,000 sqft)
    bad_sqft_req = dict(VALID_PAYLOAD, total_sqft=25000.0)
    res = client.post("/predict", json=bad_sqft_req)
    assert res.status_code == 422


def test_api_predict_unsupported_area_type_returns_422():
    bad_area_req = dict(VALID_PAYLOAD, area_type="Palace in the Sky")
    res = client.post("/predict", json=bad_area_req)
    assert res.status_code == 422
