# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
import os
import sys

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.schema import PredictionRequest, PredictionResponse
from src.predict import predict_outcome, get_active_lists, load_predictor

app = FastAPI(
    title="F1 Race Predictor API",
    description="Inference service for predicting Formula 1 Grand Prix finishing positions.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    payload = load_predictor()
    model_status = "Ready" if payload is not None else "Demonstration Heuristic (Model Untrained)"
    model_type = payload['model_type'] if payload is not None else "N/A"
    
    return {
        "status": "Online",
        "model_status": model_status,
        "model_type": model_type,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(req: PredictionRequest):
    try:
        res = predict_outcome(
            driver=req.driver,
            gp=req.gp,
            team=req.team,
            grid_position=req.grid_position,
            qualifying_position=req.qualifying_position,
            air_temp=req.air_temp,
            track_temp=req.track_temp,
            rainfall=req.rainfall
        )
        
        # Round the continuous prediction to find the closest finishing position rank (1 to 20)
        rank = int(round(res['predicted_position']))
        rank = max(1, min(20, rank))
        
        return PredictionResponse(
            predicted_position=res['predicted_position'],
            predicted_finish_rank=rank,
            podium_probability=res['podium_probability'],
            is_fallback=res['is_fallback'],
            model_type=res['model_type']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/meta")
def get_meta_lists():
    lists = get_active_lists()
    return {
        "drivers": lists['drivers'],
        "teams": lists['teams'],
        "gps": lists['gps']
    }