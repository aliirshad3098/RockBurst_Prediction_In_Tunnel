# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model & preprocessing objects
model, features, scaler = joblib.load("trained_models/xgboost_model_V3.pkl")

# Label mapping
label_map = {0: "Low", 1: "Moderate", 2: "Intense", 3: "Extreme"}

st.title("RockBurst Prediction UI")
st.markdown("Enter geological parameters to get prediction.")

# Input form
with st.form("prediction_form"):
    Energy_Unit = st.number_input("Energy Unit", min_value=0.0, format="%.4f")
    Event_freq_unit_per_day = st.number_input("Event Frequency (per day)", min_value=0.0, format="%.4f")
    Duration_days = st.number_input("Duration (days)", min_value=0.0, format="%.4f")
    Energy_per_Volume = st.number_input("Energy per Volume", min_value=0.0, format="%.4f")
    Energy_density_Joule = st.number_input("Energy Density (Joule)", min_value=0.0, format="%.4f")
    Volume_m3 = st.number_input("Volume (m³)", min_value=0.0, format="%.4f")
    Energy_Joule_per_day = st.number_input("Energy Joule per Day", min_value=0.0, format="%.4f")
    Volume_m3_per_day = st.number_input("Volume m³ per Day", min_value=0.0, format="%.4f")

    submit = st.form_submit_button("Predict")

if submit:
    # Raw input DataFrame
    input_df = pd.DataFrame([{
        "Energy_Unit": Energy_Unit,
        "Event_freq_unit_per_day": Event_freq_unit_per_day,
        "Duration_days": Duration_days,
        "Energy_per_Volume": Energy_per_Volume,
        "Energy_density_Joule": Energy_density_Joule,
        "Volume_m3": Volume_m3,
        "Energy_Joule_per_day": Energy_Joule_per_day,
        "Volume_m3_per_day": Volume_m3_per_day
    }])

    # Log transforms (same as API)
    input_df["Energy_Unit_log"] = np.log(input_df["Energy_Unit"])
    input_df["Event_freq_unit_per_day_log"] = np.log(input_df["Event_freq_unit_per_day"])
    input_df["Duration_days_log"] = np.log(input_df["Duration_days"])
    input_df["Energy_per_Volume_log"] = np.log(input_df["Energy_per_Volume"])

    # Scaling
    scaled_input = scaler.transform(input_df[features])

    # Predict
    pred_class = model.predict(scaled_input)[0]
    pred_proba = model.predict_proba(scaled_input)[0]

    # Output
    st.subheader(f"Prediction: **{label_map[int(pred_class)]}**")
    st.write("Probabilities:")
    st.bar_chart(pd.Series(pred_proba, index=[label_map[i] for i in range(len(pred_proba))]))
