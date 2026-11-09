"""
FastAPI server for the Bengaluru house price prediction project.

Serves the JSON API and the single-page web UI that lives in static/.
Loads the trained models and preprocessor bundle once at startup and
enforces strict training-domain validation and structured error handling.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException, status  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from src.encoding import (  # noqa: E402
    SCHEMA_VERSION,
    SUPPORTED_AREA_TYPES,
    VALIDATION_CONSTRAINTS,
    clean_area_type,
    validate_input,
)
from src.predict import (  # noqa: E402
    load_artifacts,
    predict_all,
    predict_detailed,
)

STATIC_DIR = os.path.join(BASE_DIR, "static")

artifacts = load_artifacts()
metrics = artifacts["metrics"]
bundle = artifacts["bundle"]
manifest = artifacts.get("manifest", {})

APP_VERSION = "2.3.3"

app = FastAPI(
    title="Bengaluru House Price API",
    description=(
        "Predict the price of a house in Bengaluru from historical listings (2017–2020). "
        "Three regression models (Linear, Ridge, Random Forest) are trained with fold-local "
        "preprocessing to prevent data leakage."
    ),
    version=APP_VERSION,
)


class PredictionRequest(BaseModel):
    area_type: str = Field(..., examples=["Super built-up Area"])
    location: str = Field(..., examples=["Whitefield"])
    society: str = Field("", description="Society name, empty if unknown")
    bhk: int = Field(2, ge=0, le=14, description="0 = studio/RK, 1–14 BHK")
    total_sqft: float = Field(1200.0, ge=100.0, le=10000.0, description="Total area in square feet")
    bath: int = Field(2, ge=1, le=12, description="Bathroom count (1–12)")
    balcony: int = Field(1, ge=0, le=5, description="Balcony count (0–5)")
    completion_year: int = Field(
        0, ge=0, le=2030, description="0 = ready to move, or expected year (2013–2030)"
    )
    model: str = Field(
        "random_forest",
        description="One of: linear, ridge, random_forest (or 'all' for every model)",
    )

    @field_validator("area_type")
    @classmethod
    def check_area_type(cls, v: str) -> str:
        norm = clean_area_type(v)
        if not norm or norm not in SUPPORTED_AREA_TYPES:
            raise ValueError(
                f"Unsupported area type '{v}'. Supported: {', '.join(SUPPORTED_AREA_TYPES)}"
            )
        return norm


def _extract_dict(payload: PredictionRequest) -> Dict[str, Any]:
    return {
        "area_type": payload.area_type,
        "location": payload.location,
        "society": payload.society,
        "bhk": payload.bhk,
        "total_sqft": payload.total_sqft,
        "bath": payload.bath,
        "balcony": payload.balcony,
        "completion_year": payload.completion_year,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "models": list(artifacts["models"]),
    }


@app.get("/models")
def models():
    """Scores of all trained models, best model first."""
    rows = []
    for key, meta in metrics["models"].items():
        rows.append(
            {
                "key": key,
                "name": meta["name"],
                "params": meta["params"],
                "r2_log": meta["test"]["r2_log"],
                "mae_log": meta["test"]["mae_log"],
                "mae_lakh": meta["test"]["mae"],
                "medae_lakh": meta["test"].get("medae", 0.0),
                "rmse_lakh": meta["test"]["rmse"],
                "cv_r2_log_mean": meta["cv"]["r2_log_mean"],
                "cv_r2_log_std": meta["cv"]["r2_log_std"],
            }
        )
    rows.sort(key=lambda r: -r["r2_log"])
    return {"best_model": metrics["best_model"], "models": rows}


@app.get("/locations")
def locations():
    return {
        "locations": sorted(k for k in bundle["location_freq"] if k != "__MISSING__")
    }


@app.get("/version")
def version():
    return {
        "version": APP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
    }


@app.get("/meta")
def meta():
    return {
        "version": APP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "supported_area_types": list(SUPPORTED_AREA_TYPES),
        "validation_constraints": VALIDATION_CONSTRAINTS,
        "best_model": metrics["best_model"],
        "dataset": metrics["dataset"],
    }


@app.get("/manifest")
def get_manifest():
    return manifest or {"schema_version": SCHEMA_VERSION, "status": "loaded"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    key = payload.model.lower()
    available_models = list(artifacts["models"]) + ["all"]
    if key not in available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model '{payload.model}'. Use one of: {', '.join(available_models)}",
        )

    row_dict = _extract_dict(payload)
    v_res = validate_input(row_dict)
    if not v_res.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Input validation failed", "errors": v_res.errors},
        )

    detailed = predict_detailed(row_dict, artifacts, selected_model=key)
    return detailed


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
