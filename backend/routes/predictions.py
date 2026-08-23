"""
Machine Learning Traffic Forecasting Routes
"""

import datetime
from fastapi import APIRouter, Query
from typing import Optional
from ai_engine.traffic_predictor.predictor_service import TrafficPredictorService

router = APIRouter(prefix="/api/predictions", tags=["ML Predictions"])

predictor_service = None
try:
    predictor_service = TrafficPredictorService()
except Exception as e:
    print(f"Warning: Could not initialize predictor service ({e})")

@router.get("/forecast")
def get_traffic_forecast(
    datetime_str: Optional[str] = Query(None, description="ISO format datetime (e.g. 2026-08-25T08:00:00)"),
    weather: Optional[str] = Query("Clear", description="Weather condition (Clear, Rain, Heavy Rain)")
):
    if predictor_service is None:
        return {"error": "ML model not initialized. Run train_predictor.py first."}

    if datetime_str:
        try:
            target_dt = datetime.datetime.fromisoformat(datetime_str)
        except Exception:
            target_dt = datetime.datetime.now()
    else:
        target_dt = datetime.datetime.now()

    prediction = predictor_service.predict(target_dt=target_dt, weather=weather)
    return prediction

@router.get("/model-info")
def get_model_metadata():
    if predictor_service is None:
        return {"status": "NOT_LOADED"}
    
    return {
        "model_type": "Random Forest Multi-Output Regressor + Classifier",
        "training_framework": "Scikit-Learn",
        "features": predictor_service.feature_cols,
        "metrics": predictor_service.metrics,
        "status": "ONLINE"
    }
