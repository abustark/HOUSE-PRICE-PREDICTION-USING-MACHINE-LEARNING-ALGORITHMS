"""
FastAPI server for the Bengaluru house price prediction project.

Serves the JSON API and the single-page web UI that lives in
api/static/. The same three models used by the Streamlit app are loaded
once at startup.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.predict import load_artifacts, predict_all  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

artifacts = load_artifacts()
metrics = artifacts["metrics"]
bundle = artifacts["bundle"]

app = FastAPI(
    title="Bengaluru House Price API",
    description=(
        "Predict the price of a house in Bengaluru. Three regression models "
        "(Linear, Ridge, Random Forest) were trained on 13,320 historical "
        "listings; pick any of them per request."
    ),
    version="2.0.0",
)


class PredictionRequest(BaseModel):
    area_type: str = Field(..., examples=["Super built-up Area"])
    location: str = Field(..., examples=["Whitefield"])
    society: str = Field("", description="Society name, empty if unknown")
    bhk: int = Field(2, ge=0, le=14, description="0 = studio/RK")
    total_sqft: float = Field(1200, gt=0, le=50000)
    bath: int = Field(2, ge=0, le=20)
    balcony: int = Field(1, ge=0, le=10)
    completion_year: int = Field(
        0, ge=0, le=2035, description="0 = ready to move, otherwise expected year"
    )
    model: str = Field(
        "random_forest",
        description="One of: linear, ridge, random_forest (or 'all' for every model)",
    )


def _row(payload: PredictionRequest) -> dict:
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
    return {"status": "ok", "models": list(artifacts["models"])}


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
                "rmse_lakh": meta["test"]["rmse"],
            }
        )
    rows.sort(key=lambda r: -r["r2_log"])
    return {"best_model": metrics["best_model"], "models": rows}


@app.get("/locations")
def locations():
    return {"locations": sorted(k for k in bundle["location_freq"] if k != "__MISSING__")}


@app.get("/meta")
def meta():
    return {
        "area_types": list(bundle["le_area_type"].classes_),
        "best_model": metrics["best_model"],
        "dataset": metrics["dataset"],
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    key = payload.model.lower()
    if key not in artifacts["models"] and key != "all":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{payload.model}'. Use one of: "
            + ", ".join(list(artifacts["models"]) + ["all"]),
        )

    results = predict_all(_row(payload), artifacts)
    chosen = key if key != "all" else metrics["best_model"]
    price = max(0.0, results[chosen])

    return {
        "model": metrics["models"][chosen]["name"],
        "model_key": chosen,
        "predicted_price_lakh": round(price, 2),
        "predicted_price_inr": int(round(price * 100000)),
        "all_models": {
            k: {
                "name": metrics["models"][k]["name"],
                "price_lakh": round(max(0.0, results[k]), 2),
            }
            for k in artifacts["models"]
        },
        "unseen_society": bool(
            payload.society.strip()
            and payload.society.strip() not in bundle["society_freq"]
        ),
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
