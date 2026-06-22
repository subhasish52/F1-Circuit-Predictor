# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import fastf1

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FEATURES, TARGET

def add_round_numbers(df):
    """
    Retrieves RoundNumber from FastF1 schedules to sort the races chronologically.
    This ensures rolling stats are computed in the correct temporal order.
    """
    print("Mapping RoundNumbers from FastF1 schedules...")
    years = df['Year'].unique()
    schedule_maps = []
    
    for year in years:
        try:
            # get_event_schedule is cached and fast
            schedule = fastf1.get_event_schedule(year)
            # Map EventName (GP) to RoundNumber
            sched_df = schedule[['RoundNumber', 'EventName']].copy()
            sched_df['Year'] = year
            sched_df = sched_df.rename(columns={'EventName': 'GP'})
            schedule_maps.append(sched_df)
        except Exception as e:
            print(f"⚠️ Could not fetch schedule for year {year}: {e}")
            
    if schedule_maps:
        all_schedules = pd.concat(schedule_maps, ignore_index=True)
        # Standardize GP names to match safe_filename style in results
        all_schedules['GP'] = all_schedules['GP'].astype(str)
        # Merge onto our dataset
        df = pd.merge(df, all_schedules, on=['Year', 'GP'], how='left')
        df['RoundNumber'] = df['RoundNumber'].fillna(0).astype(int)
    else:
        df['RoundNumber'] = 0
        
    # Sort chronologically
    df = df.sort_values(by=['Year', 'RoundNumber', 'Position']).reset_index(drop=True)
    return df

def generate_features(df):
    """
    Engineer predictive features for the F1 model.
    CRITICAL: Avoid lookahead bias (data leakage) by shifting all rolling windows.
    """
    # 1. Sort dataset chronologically to prevent leakages
    df = add_round_numbers(df)
    
    print("Engineering historical and rolling performance features...")
    
    # 2. Driver Season Rolling Stats (expanding mean shifted by 1)
    # Average finish position in the current season before the current race
    df['Driver_Avg_Position_Season'] = df.groupby(['Year', 'Driver'])['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    # DNF rate in the current season before the current race
    df['Driver_DNF_Rate_Season'] = df.groupby(['Year', 'Driver'])['Is_DNF'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # 3. Team Season Rolling Stats
    # Average team finish position in the current season before the current race
    df['Team_Avg_Position_Season'] = df.groupby(['Year', 'Team'])['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # 4. Driver GP Historical Performance
    # Average finish position of this driver at this specific GP in previous years
    df['GP_Driver_Avg_Position_Hist'] = df.groupby(['GP', 'Driver'])['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # 5. Fill NaNs created by shift(1) with sensible baseline defaults (representing midfield or average values)
    df['Driver_Avg_Position_Season'] = df['Driver_Avg_Position_Season'].fillna(10.5)
    df['Driver_DNF_Rate_Season'] = df['Driver_DNF_Rate_Season'].fillna(0.15)
    df['Team_Avg_Position_Season'] = df['Team_Avg_Position_Season'].fillna(10.5)
    df['GP_Driver_Avg_Position_Hist'] = df['GP_Driver_Avg_Position_Hist'].fillna(10.5)
    
    # Ensure all required features are present and numeric
    for feat in FEATURES:
        if feat not in df.columns:
            print(f"⚠️ Warning: feature {feat} not generated. Filling with 0.")
            df[feat] = 0.0
        else:
            df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0.0)
            
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').fillna(20.0)
    
    return df

if __name__ == '__main__':
    from src.preprocess import get_preprocessed_dataset
    try:
        raw_df = get_preprocessed_dataset()
        feat_df = generate_features(raw_df)
        print("✅ Feature engineering completed successfully.")
        print(feat_df[['Year', 'GP', 'Driver', 'Position'] + FEATURES].head())
    except Exception as e:
        print(f"❌ Error during feature engineering test: {e}")