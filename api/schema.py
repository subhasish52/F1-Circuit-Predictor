# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    driver: str = Field(..., description="Driver abbreviation (e.g. 'VER', 'HAM')", examples=["VER"])
    gp: str = Field(..., description="Grand Prix Event Name (e.g. 'Monaco Grand Prix')", examples=["Monaco Grand Prix"])
    team: str = Field(..., description="Constructor/Team Name (e.g. 'Red Bull Racing')", examples=["Red Bull Racing"])
    grid_position: int = Field(..., ge=1, le=20, description="Starting grid position", examples=[1])
    qualifying_position: Optional[int] = Field(None, ge=1, le=20, description="Qualifying position (defaults to grid position)", examples=[1])
    air_temp: Optional[float] = Field(25.0, description="Mean air temperature in °C", examples=[25.0])
    track_temp: Optional[float] = Field(35.0, description="Mean track temperature in °C", examples=[35.0])
    rainfall: Optional[float] = Field(0.0, description="Percentage of session with rainfall (0.0 to 1.0)", examples=[0.0])

class PredictionResponse(BaseModel):
    predicted_position: float = Field(..., description="Predicted continuous finish position value")
    predicted_finish_rank: int = Field(..., description="Rounded finishing position rank (e.g. P1)")
    podium_probability: float = Field(..., description="Estimated probability of finishing in the top 3")
    is_fallback: bool = Field(..., description="Indicates if model baseline heuristic was used due to untrained model")
    model_type: str = Field(..., description="Type of model used for predicting the outcome")