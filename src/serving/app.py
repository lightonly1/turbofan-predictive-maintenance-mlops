"""
Module 4 - FastAPI Application
Endpoints:
  GET  /health        - Health check
  POST /predict       - Single prediction
  POST /predict/batch - Batch prediction
  GET  /docs          - Swagger UI
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.serving.schemas import (
    SensorReading, PredictionResponse,
    BatchRequest, BatchResponse, HealthResponse
)
from src.serving.predictor import RULPredictor

app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predicts Remaining Useful Life of turbofan engines",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = RULPredictor()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        api_version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(reading: SensorReading):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        features = reading.model_dump()
        result = predictor.predict(features)
        logger.info(f"Prediction - Engine: {result['engine_id']} | RUL: {result['predicted_rul']} | Risk: {result['risk_level']}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        readings = [r.model_dump() for r in request.readings]
        predictions = predictor.predict_batch(readings)
        critical = [p["engine_id"] for p in predictions if p["risk_level"] == "CRITICAL"]
        logger.info(f"Batch - {len(predictions)} engines | {len(critical)} critical")
        return BatchResponse(
            predictions=[PredictionResponse(**p) for p in predictions],
            total_engines=len(predictions),
            critical_engines=critical
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Turbofan RUL Prediction API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }
