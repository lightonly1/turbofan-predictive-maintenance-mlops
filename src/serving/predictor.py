"""
Module 4 - Model Predictor
Loads trained XGBoost model and serves predictions.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import xgboost as xgb
from loguru import logger


def get_risk_level(rul: float) -> tuple:
    if rul <= 10:
        return "CRITICAL", "Immediate maintenance required! Engine failure imminent."
    elif rul <= 30:
        return "HIGH", "Schedule maintenance within 30 cycles. Monitor closely."
    elif rul <= 60:
        return "MEDIUM", "Plan maintenance within 60 cycles. Continue monitoring."
    else:
        return "LOW", "Engine healthy. Continue normal operations."


class RULPredictor:
    def __init__(self):
        self.model = None
        self.model_version = "unknown"
        self._load_model()

    def _load_model(self):
        try:
            mlruns_path = ROOT_DIR / "mlruns"
            if not mlruns_path.exists():
                logger.warning("mlruns not found. Train model first!")
                return
            model_files = list(mlruns_path.rglob("model.xgb"))
            if not model_files:
                model_files = list(mlruns_path.rglob("model.json"))
            if model_files:
                latest = max(model_files, key=lambda p: p.stat().st_mtime)
                self.model = xgb.XGBRegressor()
                self.model.load_model(str(latest))
                self.model_version = latest.parent.name
                logger.info(f"Model loaded: {latest}")
            else:
                logger.warning("No model file found. Run training first!")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model = None

    def predict(self, features: dict) -> dict:
        if self.model is None:
            raise ValueError("Model not loaded.")
        exclude = ["engine_id", "cycle"]
        feature_data = {k: v for k, v in features.items() if k not in exclude}
        df = pd.DataFrame([feature_data])
        rul_pred = float(self.model.predict(df)[0])
        rul_pred = max(0.0, min(125.0, rul_pred))
        risk_level, recommendation = get_risk_level(rul_pred)
        return {
            "engine_id": features.get("engine_id", 0),
            "cycle": features.get("cycle", 0),
            "predicted_rul": round(rul_pred, 2),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "model_version": self.model_version
        }

    def predict_batch(self, readings: list) -> list:
        return [self.predict(r) for r in readings]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
