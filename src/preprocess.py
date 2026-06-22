# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ALL_RESULTS_PATH, ALL_WEATHER_PATH, ALL_LAPS_PATH

def load_data():
    """Load results, weather, and lap data from the processed directory."""
    if not os.path.exists(ALL_RESULTS_PATH):
        raise FileNotFoundError(f"Missing results file: {ALL_RESULTS_PATH}. Please run data ingestion first.")
    
    results = pd.read_csv(ALL_RESULTS_PATH)
    
    weather = None
    if os.path.exists(ALL_WEATHER_PATH):
        weather = pd.read_csv(ALL_WEATHER_PATH)
        
    return results, weather

def preprocess_results(results):
    """
    Clean and filter F1 results data.
    - Filters for Race ('R') and Qualifying ('Q') sessions.
    - Handles GridPosition and final Position.
    - Identifies DNFs (Did Not Finish).
    """
    print(f"Initial results shape: {results.shape}")
    
    # Fill missing values and convert types
    results['GridPosition'] = pd.to_numeric(results['GridPosition'], errors='coerce').fillna(20).astype(int)
    results['Position'] = pd.to_numeric(results['Position'], errors='coerce').fillna(20).astype(int)
    
    # Identify DNF from Status column (Finished or +Laps are classified, others are typically DNFs)
    # Typical finished statuses: 'Finished', '+1 Lap', '+2 Laps', etc.
    results['Is_DNF'] = ~results['Status'].astype(str).str.contains('Finished|Lap', case=False, na=False)
    results['Is_DNF'] = results['Is_DNF'].astype(int)
    
    # Standardize driver and team names
    results['Driver'] = results['Abbreviation'].astype(str)
    results['Team'] = results['TeamName'].astype(str)
    
    return results

def preprocess_weather(weather):
    """
    Process weather logs into aggregated session weather features.
    Extracts mean air temperature, mean track temperature, and rainfall percentage.
    """
    if weather is None or weather.empty:
        print("⚠️ No weather data provided. Using default fallback weather values.")
        return pd.DataFrame(columns=['Year', 'GP', 'Session', 'AirTemp_Mean', 'TrackTemp_Mean', 'Rainfall_Pct'])
        
    print(f"Initial weather shape: {weather.shape}")
    
    # Handle cases where weather data from ingestion doesn't have metadata
    # (data_ingestion.py might have skipped adding metadata to weather before saving raw)
    if 'Year' not in weather.columns or 'GP' not in weather.columns:
        print("⚠️ Weather dataset is missing metadata columns. Attempting to infer or using fallback.")
        # Fallback to empty if we cannot align
        return pd.DataFrame(columns=['Year', 'GP', 'Session', 'AirTemp_Mean', 'TrackTemp_Mean', 'Rainfall_Pct'])

    # Fill NaNs in numeric weather columns
    weather['AirTemp'] = pd.to_numeric(weather['AirTemp'], errors='coerce').fillna(25.0)
    weather['TrackTemp'] = pd.to_numeric(weather['TrackTemp'], errors='coerce').fillna(35.0)
    weather['Rainfall'] = weather['Rainfall'].astype(bool).astype(float)
    
    # Group by event session and aggregate
    weather_agg = weather.groupby(['Year', 'GP', 'Session']).agg(
        AirTemp_Mean=('AirTemp', 'mean'),
        TrackTemp_Mean=('TrackTemp', 'mean'),
        Rainfall_Pct=('Rainfall', 'mean')
    ).reset_index()
    
    return weather_agg

def merge_race_and_qualifying(results, weather_agg):
    """
    Splits results into qualifying and race, merges them together, and joins weather.
    Every row in the returned dataframe represents a driver's race session.
    """
    # Split qualifying and race results
    q_results = results[results['Session'] == 'Q'].copy()
    r_results = results[results['Session'] == 'R'].copy()
    
    print(f"Qualifying records: {len(q_results)}, Race records: {len(r_results)}")
    
    # Merge qualifying position to the race results
    q_slim = q_results[['Year', 'GP', 'Driver', 'Position']].rename(columns={'Position': 'QualifyingPosition'})
    merged = pd.merge(r_results, q_slim, on=['Year', 'GP', 'Driver'], how='left')
    
    # If qualifying data is missing for a driver, default it to their grid position
    merged['QualifyingPosition'] = merged['QualifyingPosition'].fillna(merged['GridPosition']).astype(int)
    
    # Merge race weather data
    r_weather = weather_agg[weather_agg['Session'] == 'R'].copy()
    if not r_weather.empty:
        merged = pd.merge(merged, r_weather.drop(columns=['Session']), on=['Year', 'GP'], how='left')
    
    # Fill weather NaNs with general defaults if no weather matches
    merged['AirTemp_Mean'] = merged['AirTemp_Mean'].fillna(25.0)
    merged['TrackTemp_Mean'] = merged['TrackTemp_Mean'].fillna(35.0)
    merged['Rainfall_Pct'] = merged['Rainfall_Pct'].fillna(0.0)
    
    # Sort chronologically by year and event
    merged = merged.sort_values(by=['Year', 'GP', 'Position']).reset_index(drop=True)
    
    return merged

def get_preprocessed_dataset():
    """Runs the complete preprocessing pipeline and returns the combined dataset."""
    results, weather = load_data()
    cleaned_results = preprocess_results(results)
    cleaned_weather = preprocess_weather(weather)
    dataset = merge_race_and_qualifying(cleaned_results, cleaned_weather)
    print(f"Preprocessed dataset shape: {dataset.shape}")
    return dataset

if __name__ == '__main__':
    try:
        df = get_preprocessed_dataset()
        print("✅ Preprocessing completed successfully.")
        print(df.head())
    except Exception as e:
        print(f"❌ Error during preprocessing test: {e}")