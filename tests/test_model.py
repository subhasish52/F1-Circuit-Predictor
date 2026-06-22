# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytest
from src.model import evaluate_f1_metrics

def test_evaluate_f1_metrics():
    """Verify F1 accuracy metric values are within expected boundaries [0, 1]."""
    # Create test results where predicted matches actual exactly
    df_eval = pd.DataFrame({
        'Year': [2021, 2021, 2021],
        'GP': ['Monaco GP', 'Monaco GP', 'Monaco GP'],
        'Driver': ['VER', 'HAM', 'NOR'],
        'Position': [1, 2, 3]
    })
    
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 1.9, 3.2])  # Ranks align perfectly
    
    metrics = evaluate_f1_metrics(df_eval, y_true, y_pred)
    
    assert metrics['Winner_Accuracy'] == 1.0
    assert metrics['Podium_Precision'] == 1.0
    assert metrics['MAE'] < 0.5
    assert metrics['MAE'] >= 0.0