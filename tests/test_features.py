# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytest
from src.features import generate_features, add_round_numbers
from src.config import FEATURES

def test_generate_features_output_shape():
    """Verify features DataFrame has expected columns and types."""
    # Build a small dummy results dataset
    dummy_data = pd.DataFrame({
        'Year': [2021, 2021, 2021, 2021],
        'GP': ['Monaco Grand Prix', 'Monaco Grand Prix', 'Monaco Grand Prix', 'Monaco Grand Prix'],
        'Session': ['R', 'R', 'R', 'R'],
        'Driver': ['VER', 'HAM', 'LEC', 'NOR'],
        'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'McLaren'],
        'GridPosition': [1, 2, 3, 4],
        'Position': [1, 2, 3, 4],
        'Is_DNF': [0, 0, 0, 0],
        'QualifyingPosition': [1, 2, 3, 4],
        'AirTemp_Mean': [25.0, 25.0, 25.0, 25.0],
        'TrackTemp_Mean': [35.0, 35.0, 35.0, 35.0],
        'Rainfall_Pct': [0.0, 0.0, 0.0, 0.0]
    })
    
    # We mock add_round_numbers because fastf1 network call isn't available in tests
    # We can patch it or run it. Let's patch add_round_numbers to just append RoundNumber
    def mock_add_round_numbers(df):
        df['RoundNumber'] = 1
        return df.sort_values(by=['Year', 'RoundNumber', 'Position']).reset_index(drop=True)
        
    # Apply mock patch
    original_add_round_numbers = add_round_numbers
    import src.features
    src.features.add_round_numbers = mock_add_round_numbers
    
    try:
        output_df = generate_features(dummy_data)
        
        # Verify columns exist
        for feat in FEATURES:
            assert feat in output_df.columns
            assert not output_df[feat].isnull().any()
            
        assert 'Position' in output_df.columns
    finally:
        # Restore original function
        src.features.add_round_numbers = original_add_round_numbers