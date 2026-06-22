# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_FILE_PATH, FEATURES

_predictor_cache = None

def load_predictor():
    """
    Loads the trained model and lookup tables from the pickle file.
    Caches the payload in memory for faster sub-sequent calls.
    """
    global _predictor_cache
    if _predictor_cache is not None:
        return _predictor_cache
        
    if not os.path.exists(MODEL_FILE_PATH):
        return None
        
    try:
        with open(MODEL_FILE_PATH, 'rb') as f:
            _predictor_cache = pickle.load(f)
        return _predictor_cache
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return None

def predict_outcome(driver, gp, team, grid_position, qualifying_position=None, 
                    air_temp=25.0, track_temp=35.0, rainfall=0.0):
    """
    Predicts the final finishing position of a driver in a race.
    Falls back to a grid-position baseline if the model is not trained yet.
    """
    payload = load_predictor()
    
    # 1. Fallback Heuristic if model is not trained yet
    if payload is None:
        # Simple baseline: start grid position is heavily correlated with finish position
        # Add a tiny random variance and clamp to valid range
        pred = float(grid_position)
        pred = max(1.0, min(20.0, pred))
        return {
            'predicted_position': round(pred, 2),
            'podium_probability': max(0.01, min(0.99, 1.0 - (grid_position / 10.0))),
            'is_fallback': True,
            'model_type': "Baseline Heuristic (Model not trained yet)"
        }
        
    model = payload['model']
    driver_stats = payload['driver_stats']
    team_stats = payload['team_stats']
    gp_history = payload['gp_history']
    
    # 2. Lookup rolling driver statistics
    d_stats = driver_stats.get(driver, {
        'Driver_Avg_Position_Season': 10.5,
        'Driver_DNF_Rate_Season': 0.15
    })
    
    # 3. Lookup rolling constructor statistics
    t_stats = team_stats.get(team, {
        'Team_Avg_Position_Season': 10.5
    })
    
    # 4. Lookup GP historical performance
    gp_driver_perf = gp_history.get((gp, driver), 10.5)
    
    # 5. Handle missing qualifying position
    if qualifying_position is None:
        qualifying_position = grid_position
        
    # 6. Build the feature mapping in correct order
    feature_dict = {
        'GridPosition': grid_position,
        'QualifyingPosition': qualifying_position,
        'Driver_Avg_Position_Season': d_stats['Driver_Avg_Position_Season'],
        'Driver_DNF_Rate_Season': d_stats['Driver_DNF_Rate_Season'],
        'Team_Avg_Position_Season': t_stats['Team_Avg_Position_Season'],
        'GP_Driver_Avg_Position_Hist': gp_driver_perf,
        'AirTemp_Mean': air_temp,
        'TrackTemp_Mean': track_temp,
        'Rainfall_Pct': rainfall
    }
    
    # Format features as a dataframe with 1 row in the exact same column order
    X = pd.DataFrame([feature_dict])[FEATURES]
    
    # 7. Predict
    pred_val = model.predict(X)[0]
    
    # Clamp the predicted position to valid F1 limits [1, 20]
    clamped_pred = max(1.0, min(20.0, pred_val))
    
    # Calculate a simple podium probability based on distance of prediction from podium
    # Closer to 1.0 -> higher probability
    podium_prob = np.exp(-0.4 * max(0, clamped_pred - 1.0))
    podium_prob = max(0.01, min(0.99, podium_prob))
    
    return {
        'predicted_position': round(clamped_pred, 2),
        'podium_probability': round(podium_prob, 2),
        'is_fallback': False,
        'model_type': payload['model_type']
    }

def get_active_lists():
    """Returns list of active drivers, teams, and GPs for the UI."""
    payload = load_predictor()
    if payload is not None:
        return {
            'drivers': payload['active_drivers'],
            'teams': payload['active_constructors'],
            'gps': payload['active_gps']
        }
        
    # Standard hardcoded defaults if model is not trained yet (to ensure UI works immediately)
    default_drivers = ['VER', 'HAM', 'LEC', 'NOR', 'SAI', 'PER', 'RUS', 'PIA', 'ALO', 'STR', 'TSU', 'RIC', 'ALB', 'SAR', 'MAG', 'HUL', 'OCO', 'GAS', 'BOT', 'ZHO']
    default_teams = ['Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren', 'Aston Martin', 'RB', 'Williams', 'Haas F1 Team', 'Alpine', 'Kick Sauber']
    default_gps = ['Australian Grand Prix', 'Bahrain Grand Prix', 'Chinese Grand Prix', 'Azerbaijan Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Canadian Grand Prix', 'French Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'German Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Italian Grand Prix']
    
    return {
        'drivers': sorted(default_drivers),
        'teams': sorted(default_teams),
        'gps': sorted(default_gps)
    }

if __name__ == '__main__':
    # Dry run prediction
    res = predict_outcome(driver='VER', gp='Monaco Grand Prix', team='Red Bull Racing', grid_position=1)
    print("Test Prediction:")
    print(res)