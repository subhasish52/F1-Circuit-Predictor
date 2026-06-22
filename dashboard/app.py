# -*- coding: utf-8 -*-
import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_outcome, get_active_lists, load_predictor
from src.config import ALL_RESULTS_PATH, MODELS_DIR

# Set page configuration
st.set_page_config(
    page_title="F1 Race Outcome Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for a premium F1 look (Dark Theme, Neon Red/Gray Accents)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .f1-title {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #FF1801;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .card {
        background-color: #1e2630;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #FF1801;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #FF1801;
    }
    .metric-label {
        font-size: 1rem;
        color: #a0aab5;
    }
</style>
""", unsafe_allow_html=True)

def check_datasets():
    """Checks if processed dataset exists."""
    return os.path.exists(ALL_RESULTS_PATH)

def main_app():
    st.markdown('<h1 class="f1-title">🏁 F1 Race Outcome Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict the final finish position of Formula 1 drivers using machine learning (Random Forest & Gradient Boosting).")
    st.write("---")

    # Retrieve lists for UI selectors
    lists = get_active_lists()
    
    # Sidebar: Input Panel
    st.sidebar.markdown("### 🏎️ Race parameters")
    
    selected_gp = st.sidebar.selectbox("Grand Prix Location", lists['gps'])
    selected_driver = st.sidebar.selectbox("Driver Abbreviation", lists['drivers'])
    selected_team = st.sidebar.selectbox("Team / Constructor", lists['teams'])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚦 Session Grid Positions")
    grid_pos = st.sidebar.slider("Starting Grid Position", 1, 20, 1)
    qual_pos = st.sidebar.slider("Qualifying Session Position", 1, 20, grid_pos)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌦️ Weather Conditions")
    air_temp = st.sidebar.slider("Air Temp (°C)", 10.0, 45.0, 25.0, 0.5)
    track_temp = st.sidebar.slider("Track Temp (°C)", 15.0, 65.0, 35.0, 0.5)
    rain = st.sidebar.checkbox("Rain during session?", value=False)
    rain_pct = 1.0 if rain else 0.0

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Race Prediction", "📊 Model Training & Insights", "📂 Data Exploration"])
    
    # TAB 1: Prediction
    with tab1:
        st.subheader("Race Prediction Analysis")
        
        # Calculate prediction
        prediction = predict_outcome(
            driver=selected_driver,
            gp=selected_gp,
            team=selected_team,
            grid_position=grid_pos,
            qualifying_position=qual_pos,
            air_temp=air_temp,
            track_temp=track_temp,
            rainfall=rain_pct
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Predicted Finishing Position</div>
                <div class="metric-value">P{int(round(prediction['predicted_position']))}</div>
                <div class="metric-label">Exact regression value: {prediction['predicted_position']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            prob_pct = int(prediction['podium_probability'] * 100)
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid #00D2BE;">
                <div class="metric-label">Podium Finish Probability (Top 3)</div>
                <div class="metric-value">{prob_pct}%</div>
                <div class="metric-label">Based on historical finish probability distributions</div>
            </div>
            """, unsafe_allow_html=True)
            
        if prediction['is_fallback']:
            st.warning("⚠️ **Demonstration Mode:** The machine learning model has not been trained yet. Showing predictions using starting grid baseline heuristic. Go to the **Model Training** tab to train the machine learning pipeline!")
            
        # Insights & Heuristics Explanation
        st.markdown("### 💡 Explanation for Interview Panel")
        st.info(f"""
        **Model Feature Mapping:**
        - **Model Architecture:** The current model in use is **{prediction['model_type']}**.
        - **Drivers & Teams:** Standardizes performance indices. For example, `{selected_driver}` driving for `{selected_team}` will reference their latest seasonal rolling average position.
        - **Weather Aggregations:** Temperature gradients (`{air_temp}°C` air, `{track_temp}°C` track) and rain indicators influence lap tire performance and grid attrition rate.
        - **Temporal Integrity:** Rolling windows do not include the session being predicted, preventing lookahead bias.
        """)
        
        # Grid impact visualization
        st.markdown("### 🚥 Impact of Starting Grid Position")
        grid_impacts = []
        for i in range(1, 21):
            pred_res = predict_outcome(
                driver=selected_driver,
                gp=selected_gp,
                team=selected_team,
                grid_position=i,
                qualifying_position=i,
                air_temp=air_temp,
                track_temp=track_temp,
                rainfall=rain_pct
            )
            grid_impacts.append(pred_res['predicted_position'])
            
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1e2630')
        
        ax.plot(range(1, 21), grid_impacts, marker='o', color='#FF1801', linewidth=2, label="Predicted Finish Position")
        ax.plot(range(1, 21), range(1, 21), linestyle='--', color='#a0aab5', alpha=0.5, label="Grid = Finish Baseline")
        
        ax.set_title("Starting Grid Position vs. Predicted Race Finish", color='white', fontsize=12, pad=10)
        ax.set_xlabel("Starting Grid Position", color='white')
        ax.set_ylabel("Predicted Finish Position", color='white')
        ax.tick_params(colors='white')
        ax.set_xticks(range(1, 21))
        ax.set_yticks(range(1, 22, 2))
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(facecolor='#1e2630', edgecolor='none', labelcolor='white')
        
        st.pyplot(fig)
        
    # TAB 2: Model Training and Metrics
    with tab2:
        st.subheader("ML Model Tuning & Pipeline")
        
        payload = load_predictor()
        
        if payload is not None:
            st.success(f"✅ Trained model is loaded: **{payload['model_type']}**")
            
            # Model details & metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("### Model Details")
                st.write(f"**Selected Model:** {payload['model_type']}")
                st.write("**Hyperparameters:**")
                st.json(payload['best_params'])
                
            with m_col2:
                st.markdown("### Out-of-Time Test Metrics")
                t_metrics = payload['test_metrics']
                if t_metrics:
                    st.metric("Test Mean Absolute Error (MAE)", f"{t_metrics['MAE']:.3f} positions")
                    st.metric("Winner Prediction Accuracy (Top-1)", f"{t_metrics['Winner_Accuracy']:.1%}")
                    st.metric("Podium Prediction Accuracy (Top-3)", f"{t_metrics['Podium_Precision']:.1%}")
                else:
                    st.write("Cross-validation scores used (insufficient validation seasons).")
                    st.metric("Train MAE", f"{payload['train_metrics']['MAE']:.3f}")
            
            # Feature importance chart
            st.markdown("### Feature Importance Analysis")
            importances = payload['feature_importances']
            df_imp = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#1e2630')
            
            ax2.barh(df_imp['Feature'], df_imp['Importance'], color='#FF1801')
            ax2.set_title("Model Feature Importances", color='white', fontsize=12)
            ax2.tick_params(colors='white')
            ax2.grid(True, linestyle=':', alpha=0.3, axis='x')
            
            st.pyplot(fig2)
        else:
            st.warning("⚠️ No trained model found on disk. Please run the model training script below.")
            
        # On-demand training panel
        st.markdown("---")
        st.markdown("### 🛠️ Train Machine Learning Pipeline")
        st.write("Trigger hyperparameter tuning and model optimization across Random Forest and Gradient Boosting models directly from the UI.")
        
        if not check_datasets():
            st.error("❌ Processed datasets are missing from `data/processed`. You cannot train the model until the fastf1 ingestion script finishes downloading the datasets.")
        else:
            if st.button("Train Model Now"):
                with st.spinner("Running GridSearch and optimizing model parameters..."):
                    try:
                        # Import and run model main
                        from src.model import main as train_pipeline
                        train_pipeline()
                        st.success("🎉 Model trained successfully! Please reload the page to apply.")
                    except Exception as ex:
                        st.error(f"Error during training: {ex}")
                        
    # TAB 3: Data Exploration
    with tab3:
        st.subheader("F1 Dataset Statistics")
        
        if not check_datasets():
            st.warning("⚠️ No processed F1 datasets found on disk. FastF1 download might still be in progress.")
        else:
            try:
                df_res = pd.read_csv(ALL_RESULTS_PATH)
                st.write(f"**Total Records Loaded:** {df_res.shape[0]} rows across {df_res['Year'].nunique()} seasons.")
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("#### Sample Results Data")
                    st.dataframe(df_res[['Year', 'GP', 'Session', 'Abbreviation', 'TeamName', 'Position', 'GridPosition']].head(10))
                with col_d2:
                    st.markdown("#### Top Drivers by Appearances")
                    st.bar_chart(df_res['Abbreviation'].value_counts().head(10))
            except Exception as ex:
                st.error(f"Error reading datasets: {ex}")

if __name__ == '__main__':
    main_app()