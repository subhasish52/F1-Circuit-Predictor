# 🏁 F1 Race Outcome Predictor - Model Evaluation Report

This report outlines the methodology, validation strategy, and machine learning models implemented for predicting Formula 1 race results.

---

## 1. Project Objective & Task Formulation
Predicting the outcome of a Formula 1 Grand Prix represents a **ranking regression problem**. While traditional models simply predict finishing positions as continuous numbers, we treat it as an ordering problem. Within each race, drivers are ranked relative to one another based on their predicted finish values.

---

## 2. Feature Engineering & Temporal Data Integrity
To ensure explainability to a technical panel and avoid **data leakage (lookahead bias)**, all rolling features are computed strictly using historical data *up to but not including the target race session*.

### Feature Set
1. **GridPosition (Starting Grid):** The driver's starting grid slot (highly correlated with final position).
2. **QualifyingPosition:** The driver's finishing rank in the corresponding Qualifying session.
3. **Driver_Avg_Position_Season:** The driver's rolling mean finish position in the current season prior to this GP.
4. **Driver_DNF_Rate_Season:** The driver's rolling DNF rate in the current season prior to this GP.
5. **Team_Avg_Position_Season:** The rolling mean finish position of the constructor's cars in the current season prior to this GP.
6. **GP_Driver_Avg_Position_Hist:** The driver's historical average finish position at this specific track in previous years.
7. **Weather Features (AirTemp_Mean, TrackTemp_Mean, Rainfall_Pct):** Aggregated metrics tracking track conditions during the session.

---

## 3. Machine Learning Algorithms
We train and tune two distinct algorithms to compare their performance:

### A. Random Forest Regressor
* **Why it fits F1:** Highly robust to outlier results (like multi-car accidents or sudden weather shifts) due to ensemble bagging.
* **Hyperparameter Grid:**
  * `n_estimators`: `[50, 100, 200]`
  * `max_depth`: `[3, 5, 8, 10, None]`
  * `min_samples_split`: `[2, 5, 10]`
  * `min_samples_leaf`: `[1, 2, 4]`

### B. Gradient Boosting Regressor
* **Why it fits F1:** Iteratively optimizes for regression errors, focusing on harder-to-predict races by minimizing boosting residuals.
* **Hyperparameter Grid:**
  * `n_estimators`: `[50, 100, 150]`
  * `learning_rate`: `[0.01, 0.05, 0.1, 0.2]`
  * `max_depth`: `[3, 4, 5, 6]`
  * `subsample`: `[0.8, 0.9, 1.0]`

---

## 4. Validation & Evaluation Strategy
We employ **GroupKFold cross-validation grouped by Year** for tuning. This mimics the temporal deployment scenario: training on historical seasons and predicting future seasons, rather than randomly shuffling races which would leak seasonal car performance.

### Domain-Specific Evaluation Metrics
Standard metrics like Mean Absolute Error (MAE) and RMSE are complemented by:
* **Winner Accuracy (Top-1):** The percentage of races where the driver we predicted to finish first actually won the race.
* **Podium Precision (Top-3):** The percentage of predicted top-3 finishers who actually finished on the podium.

---

*Note: Once the data ingestion finishes downloading all seasons, trigger model training from the Streamlit UI (under the "Model Training" tab) to dynamically write the comparative metric tables and feature importances to this report.*