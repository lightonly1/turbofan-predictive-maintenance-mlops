"""
Module 4 - API Schemas
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List


class SensorReading(BaseModel):
    engine_id: int = Field(..., example=1)
    cycle: int = Field(..., example=150)
    setting_1: float = Field(..., example=-0.0007)
    setting_2: float = Field(..., example=-0.0004)
    setting_3: float = Field(..., example=100.0)
    sensor_2: float = Field(..., example=641.82)
    sensor_3: float = Field(..., example=1589.70)
    sensor_4: float = Field(..., example=1400.60)
    sensor_6: float = Field(..., example=21.61)
    sensor_7: float = Field(..., example=554.36)
    sensor_8: float = Field(..., example=2388.06)
    sensor_9: float = Field(..., example=9046.19)
    sensor_11: float = Field(..., example=47.47)
    sensor_12: float = Field(..., example=521.66)
    sensor_13: float = Field(..., example=2388.02)
    sensor_14: float = Field(..., example=8138.62)
    sensor_15: float = Field(..., example=8.4195)
    sensor_17: float = Field(..., example=391.0)
    sensor_20: float = Field(..., example=38.86)
    sensor_21: float = Field(..., example=23.3735)
    sensor_2_rm: float = Field(..., example=641.50)
    sensor_3_rm: float = Field(..., example=1589.50)
    sensor_4_rm: float = Field(..., example=1400.50)
    sensor_6_rm: float = Field(..., example=21.60)
    sensor_7_rm: float = Field(..., example=554.30)
    sensor_8_rm: float = Field(..., example=2388.00)
    sensor_9_rm: float = Field(..., example=9046.00)
    sensor_11_rm: float = Field(..., example=47.40)
    sensor_12_rm: float = Field(..., example=521.60)
    sensor_13_rm: float = Field(..., example=2388.00)
    sensor_14_rm: float = Field(..., example=8138.00)
    sensor_15_rm: float = Field(..., example=8.40)
    sensor_17_rm: float = Field(..., example=391.0)
    sensor_20_rm: float = Field(..., example=38.80)
    sensor_21_rm: float = Field(..., example=23.37)


class PredictionResponse(BaseModel):
    engine_id: int
    cycle: int
    predicted_rul: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchRequest(BaseModel):
    readings: List[SensorReading]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_engines: int
    critical_engines: List[int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    api_version: str
