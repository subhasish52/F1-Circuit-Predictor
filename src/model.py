# -*- coding: utf-8 -*-
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FEATURES, TARGET, MODEL_FILE_PATH, MODELS_DIR
from src.preprocess import get_preprocessed_dataset
from src.features import generate_features

def evaluate_f1_metrics(df_eval, y_true, y_pred):
    """
    Evaluates regression and F1-specific ranking metrics:
    - MAE, RMSE, R2
    - Winner Accuracy (Top-1): Percent of races where predicted #1 finished #1.
    - Podium Accuracy (Top-3): Precision of predicted podium finishers.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Add predictions to copy of df to calculate ranking metrics
    df_temp = df_eval.copy()
    df_temp['Pred_Position'] = y_pred
    
    # Rank drivers within each race (Year, GP) based on predicted position
    # Lower predicted position value (e.g. 1.2 vs 5.4) means better finishing position
    df_temp['Predicted_Rank'] = df_temp.groupby(['Year', 'GP'])['Pred_Position'].rank(method='min')
    
    races = df_temp.groupby(['Year', 'GP'])
    
    total_races = 0
    correct_winners = 0
    correct_podiums = 0
    total_podiums = 0
    
    for name, group in races:
        total_races += 1
        
        # Get actual winner & podium finishers
        actual_winner = group[group['Position'] == 1]['Driver'].values
        actual_podium = set(group[group['Position'] <= 3]['Driver'].values)
        
        # Get predicted winner (lowest Predicted_Rank)
        predicted_winner = group[group['Predicted_Rank'] == group['Predicted_Rank'].min()]['Driver'].values
        predicted_podium = set(group[group['Predicted_Rank'] <= 3]['Driver'].values)
        
        # Check if actual winner is among predicted winners (handles ties)
        if len(actual_winner) > 0 and len(predicted_winner) > 0:
            if actual_winner[0] in predicted_winner:
                correct_winners += 1
                
        # Intersect predicted and actual podium sets
        correct_podiums += len(actual_podium.intersection(predicted_podium))
        total_podiums += 3
        
    winner_accuracy = correct_winners / total_races if total_races > 0 else 0.0
    podium_precision = correct_podiums / total_podiums if total_podiums > 0 else 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Winner_Accuracy': winner_accuracy,
        'Podium_Precision': podium_precision
    }

def hyperparameter_tuning_rf(X, y, groups):
    """Run RandomizedSearchCV to find best parameters for Random Forest."""
    print("\n--- Tuning Random Forest Regressor ---")
    rf = RandomForestRegressor(random_state=42)
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Use GroupKFold grouped by Year to avoid time-series correlation leakage
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=gkf,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X, y, groups=groups)
    print(f"Best RF Parameters: {search.best_params_}")
    print(f"Best RF Score (Neg MAE): {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_

def hyperparameter_tuning_gb(X, y, groups):
    """Run RandomizedSearchCV to find best parameters for Gradient Boosting."""
    print("\n--- Tuning Gradient Boosting Regressor ---")
    gb = GradientBoostingRegressor(random_state=42)
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_leaf': [1, 2, 4]
    }
    
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_dist,
        n_iter=10,
        cv=gkf,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X, y, groups=groups)
    print(f"Best GB Parameters: {search.best_params_}")
    print(f"Best GB Score (Neg MAE): {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_

def main():
    print("🚀 Starting F1 Predictor Model Training Pipeline...")
    
    # 1. Load and engineer features
    try:
        raw_df = get_preprocessed_dataset()
        df = generate_features(raw_df)
    except Exception as e:
        print(f"❌ Failed to load or process data: {e}")
        print("💡 Suggestion: Ensure data is downloaded by running data_ingestion.py first.")
        return
        
    print(f"Dataset compiled. Rows: {len(df)}, Columns: {df.shape[1]}")
    
    # 2. Chronological Split (Train: 2018-2024, Test: 2025/2026)
    # If test set is empty due to early download state, fallback to 80/20 train/test split
    unique_years = sorted(df['Year'].unique())
    print(f"Available years: {unique_years}")
    
    if len(unique_years) >= 3:
        test_years = [unique_years[-1]]
        train_years = unique_years[:-1]
        
        train_mask = df['Year'].isin(train_years)
        test_mask = df['Year'].isin(test_years)
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
    else:
        # Fallback split
        print("⚠️ Insufficient years for chronological split. Falling back to 80/20 split.")
        df_train = df.sample(frac=0.8, random_state=42)
        df_test = df.drop(df_train.index)
        
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    groups_train = df_train['Year']
    
    X_test = df_test[FEATURES]
    y_test = df_test[TARGET]
    
    print(f"Train set: {len(X_train)} rows. Test set: {len(X_test)} rows.")
    
    # 3. Model Tuning & Optimization
    best_rf, best_rf_params = hyperparameter_tuning_rf(X_train, y_train, groups_train)
    best_gb, best_gb_params = hyperparameter_tuning_gb(X_train, y_train, groups_train)
    
    # 4. Evaluation
    print("\n--- Model Evaluation ---")
    
    # RF predictions
    rf_pred_train = best_rf.predict(X_train)
    rf_pred_test = best_rf.predict(X_test) if len(X_test) > 0 else np.array([])
    rf_train_metrics = evaluate_f1_metrics(df_train, y_train, rf_pred_train)
    rf_test_metrics = evaluate_f1_metrics(df_test, y_test, rf_pred_test) if len(X_test) > 0 else {}
    
    # GB predictions
    gb_pred_train = best_gb.predict(X_train)
    gb_pred_test = best_gb.predict(X_test) if len(X_test) > 0 else np.array([])
    gb_train_metrics = evaluate_f1_metrics(df_train, y_train, gb_pred_train)
    gb_test_metrics = evaluate_f1_metrics(df_test, y_test, gb_pred_test) if len(X_test) > 0 else {}
    
    # Print metrics
    print("\nRandom Forest Results:")
    print(f"  Train MAE: {rf_train_metrics['MAE']:.3f} | Test MAE: {rf_test_metrics.get('MAE', 0.0):.3f}")
    print(f"  Train Winner Accuracy: {rf_train_metrics['Winner_Accuracy']:.1%} | Test Winner Accuracy: {rf_test_metrics.get('Winner_Accuracy', 0.0):.1%}")
    
    print("\nGradient Boosting Results:")
    print(f"  Train MAE: {gb_train_metrics['MAE']:.3f} | Test MAE: {gb_test_metrics.get('MAE', 0.0):.3f}")
    print(f"  Train Winner Accuracy: {gb_train_metrics['Winner_Accuracy']:.1%} | Test Winner Accuracy: {gb_test_metrics.get('Winner_Accuracy', 0.0):.1%}")
    
    # 5. Model Selection
    # Choose best model based on Test MAE (or Train MAE if test is empty)
    metric_to_compare = 'MAE'
    rf_score = rf_test_metrics.get(metric_to_compare, rf_train_metrics[metric_to_compare])
    gb_score = gb_test_metrics.get(metric_to_compare, gb_train_metrics[metric_to_compare])
    
    if gb_score < rf_score:
        best_model = best_gb
        model_type = "Gradient Boosting Regressor"
        best_params = best_gb_params
        final_test_metrics = gb_test_metrics
        final_train_metrics = gb_train_metrics
    else:
        best_model = best_rf
        model_type = "Random Forest Regressor"
        best_params = best_rf_params
        final_test_metrics = rf_test_metrics
        final_train_metrics = rf_train_metrics
        
    print(f"\n🏆 Best Model Selected: {model_type}")
    
    # 6. Save Best Model and Metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Extract feature importances
    importances = best_model.feature_importances_
    feature_importance_map = dict(zip(FEATURES, importances))
    
    model_payload = {
        'model': best_model,
        'model_type': model_type,
        'features': FEATURES,
        'best_params': best_params,
        'train_metrics': final_train_metrics,
        'test_metrics': final_test_metrics,
        'feature_importances': feature_importance_map,
        # Lookup tables for predict.py
        'driver_stats': df.groupby('Driver').last()[['Driver_Avg_Position_Season', 'Driver_DNF_Rate_Season']].to_dict(orient='index'),
        'team_stats': df.groupby('Team').last()[['Team_Avg_Position_Season']].to_dict(orient='index'),
        'gp_history': df.groupby(['GP', 'Driver'])['Position'].mean().to_dict(),
        'active_drivers': sorted(df['Driver'].unique().tolist()),
        'active_constructors': sorted(df['Team'].unique().tolist()),
        'active_gps': sorted(df['GP'].unique().tolist())
    }

    
    with open(MODEL_FILE_PATH, 'wb') as f:
        pickle.dump(model_payload, f)
        
    print(f"💾 Best model and metadata saved to {MODEL_FILE_PATH}")
    
    # 7. Write Model Evaluation Report
    report_content = f"""# 🏁 F1 Circuit Predictor Model Evaluation Report

This report outlines the metrics and findings from training and tuning the machine learning models.

## Executive Summary
We trained two models—**Random Forest Regressor** and **Gradient Boosting Regressor**—on F1 race results data. The models were evaluated using standard regression metrics alongside F1-specific ranking accuracy metrics.

**Best Model Selected:** `{model_type}`

### Training and Testing Split
- **Train Seasons:** {", ".join(map(str, train_years))} ({len(df_train)} records)
- **Test Season (Out-of-Time Validation):** {", ".join(map(str, test_years))} ({len(df_test)} records)

---

## Model Performance

| Metric | Random Forest (Train / Test) | Gradient Boosting (Train / Test) |
|---|---|---|
| **Mean Absolute Error (MAE)** | {rf_train_metrics['MAE']:.3f} / {rf_test_metrics.get('MAE', 0.0):.3f} | {gb_train_metrics['MAE']:.3f} / {gb_test_metrics.get('MAE', 0.0):.3f} |
| **Root Mean Squared Error (RMSE)** | {rf_train_metrics['RMSE']:.3f} / {rf_test_metrics.get('RMSE', 0.0):.3f} | {gb_train_metrics['RMSE']:.3f} / {gb_test_metrics.get('RMSE', 0.0):.3f} |
| **R-squared ($R^2$)** | {rf_train_metrics['R2']:.3f} / {rf_test_metrics.get('R2', 0.0):.3f} | {gb_train_metrics['R2']:.3f} / {gb_test_metrics.get('R2', 0.0):.3f} |
| **Winner (Top-1) Accuracy** | {rf_train_metrics['Winner_Accuracy']:.1%} / {rf_test_metrics.get('Winner_Accuracy', 0.0):.1%} | {gb_train_metrics['Winner_Accuracy']:.1%} / {gb_test_metrics.get('Winner_Accuracy', 0.0):.1%} |
| **Podium (Top-3) Precision** | {rf_train_metrics['Podium_Precision']:.1%} / {rf_test_metrics.get('Podium_Precision', 0.0):.1%} | {gb_train_metrics['Podium_Precision']:.1%} / {gb_test_metrics.get('Podium_Precision', 0.0):.1%} |

*Note: Winner Accuracy refers to the percentage of races where the driver ranked #1 by our model's predictions won the actual race.*

---

## Best Hyperparameters
```python
{best_params}
```

---

## Feature Importances
Below is the relative weight of each feature in the selected `{model_type}`:

"""
    for feat, imp in sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True):
        report_content += f"- **{feat}**: {imp:.4f} ({imp*100:.1f}%)\n"
        
    report_content += "\n*Generated dynamically by src/model.py.*"
    
    report_path = "reports/model_eval.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"📝 Evaluation report written to {report_path}")

if __name__ == '__main__':
    main()