# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load Model & Preprocessing

model, features = joblib.load("trained_models/Best_model/Log_Reg_model.pkl")
scaler = joblib.load("trained_models/scaler_V3.pkl")

# Label mapping
label_map = {0: "Low", 1: "Moderate", 2: "Intense", 3: "Extreme"}


# UI Title

st.title("🚧 RockBurst Prediction UI")
st.markdown("Enter the geological parameters below or adjust sliders to get a risk prediction.")


# Helper function for slider + number input

def slider_with_number(label, min_val, max_val, default):
    col1, col2 = st.columns([4, 1])
    with col1:
        slider_val = st.slider(label, min_val, max_val, default, step=0.01)
    with col2:
        num_val = st.number_input(f"{label} (manual)", min_val, max_val, slider_val, step=0.01)
    return num_val


# Sliders for all parameters

energy_unit = slider_with_number("Energy Unit", 0.0, 80.0, 12.34)
energy_density = slider_with_number("Energy Density (Joule)", 0.0, 8.0, 4.36)
volume_m3 = slider_with_number("Volume (m³)", 0.0, 6.0, 4.13)
event_freq = slider_with_number("Event Frequency (per day)", 0.0, 15.0, 1.66)
energy_joule_per_day = slider_with_number("Energy Joule per Day", 0.0, 7.0, 3.54)
volume_m3_per_day = slider_with_number("Volume m³ per Day", 0.0, 5.0, 3.32)
duration_days = slider_with_number("Duration (days)", 0.0, 30.0, 7.77)

# Energy per Volume (calculated)
energy_per_volume = 0.0
if volume_m3 > 0:
    energy_per_volume = energy_unit / volume_m3


# Prediction Button

if st.button("🔍 Predict"):
    # Prepare dataframe
    input_df = pd.DataFrame([{
        "Energy_Unit": energy_unit,
        "Event_freq_unit_per_day": event_freq,
        "Duration_days": duration_days,
        "Energy_per_Volume": energy_per_volume,
        "Energy_density_Joule": energy_density,
        "Volume_m3": volume_m3,
        "Energy_Joule_per_day": energy_joule_per_day,
        "Volume_m3_per_day": volume_m3_per_day
    }])

    # Apply log transforms
    input_df['Energy_Unit_log'] = np.log(input_df['Energy_Unit'] + 1)
    input_df['Event_freq_unit_per_day_log'] = np.log(input_df['Event_freq_unit_per_day'] + 1)
    input_df['Duration_days_log'] = np.log(input_df['Duration_days'] + 1)
    input_df['Energy_per_Volume_log'] = np.log(input_df['Energy_per_Volume'] + 1)

    # Scale
    scaled_input = scaler.transform(input_df[features])

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    # Output
    st.subheader(f"Prediction: **{label_map[prediction]}**")
    st.markdown("### Probabilities")
    st.bar_chart(pd.Series(probability, index=[label_map[i] for i in range(len(probability))]))
