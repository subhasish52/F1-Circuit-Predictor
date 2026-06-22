# -*- coding: utf-8 -*-
import os

# Base paths
ROOT_DIR = "."
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# File paths
ALL_LAPS_PATH = os.path.join(PROCESSED_DATA_DIR, "all_laps.csv")
ALL_RESULTS_PATH = os.path.join(PROCESSED_DATA_DIR, "all_results.csv")
ALL_WEATHER_PATH = os.path.join(PROCESSED_DATA_DIR, "all_weather.csv")
ACTIVE_DRIVERS_PATH = os.path.join(PROCESSED_DATA_DIR, "active_drivers.csv")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "f1_predictor_model.pkl")

# Feature definition
FEATURES = [
    'GridPosition',
    'QualifyingPosition',
    'Driver_Avg_Position_Season',
    'Driver_DNF_Rate_Season',
    'Team_Avg_Position_Season',
    'GP_Driver_Avg_Position_Hist',
    'AirTemp_Mean',
    'TrackTemp_Mean',
    'Rainfall_Pct'
]

TARGET = 'Position'