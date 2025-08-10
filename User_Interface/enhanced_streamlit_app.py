# enhanced_streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import io


# Page config & styling

st.set_page_config(page_title="RockBurst Dashboard", layout="wide", initial_sidebar_state="expanded")

# small CSS for badge
BADGE_CSS = """
<style>
.badge {
  display:inline-block;
  padding:8px 14px;
  border-radius:999px;
  color:white;
  font-weight:600;
  font-size:16px;
}
.card {background-color: rgba(255,255,255,0.03); padding:12px; border-radius:8px;}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)


# Load model & scaler (cached)

@st.cache_resource
def load_artifacts():
    model_obj = joblib.load("trained_models/Best_model/Log_Reg_model.pkl")  # expecting (model, features)
    scaler = joblib.load("trained_models/scaler_V3.pkl")
    return model_obj, scaler

(model, features), scaler = load_artifacts()

# Label mapping and colors
label_map = {0: "Low", 1: "Moderate", 2: "Intense", 3: "Extreme"}
label_color = {"Low": "#16a34a", "Moderate": "#f59e0b", "Intense": "#f97316", "Extreme": "#dc2626"}


# Sidebar: settings + history management

st.sidebar.header("⚙️ Settings")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=1)
save_history = st.sidebar.checkbox("Save prediction history (local CSV)", value=True)
history_file = "prediction_history.csv"
st.sidebar.markdown("---")
st.sidebar.info("Use sliders or manual numbers. Predictions update immediately when values change.")


# Helper: slider + number input combo

def slider_number(label, min_v, max_v, default, step=0.01, fmt="%.3f"):
    col1, col2 = st.columns([4,1])
    with col1:
        s = st.slider(label, min_value=float(min_v), max_value=float(max_v), value=float(default), step=step, key=f"{label}_slider")
    with col2:
        n = st.number_input(label + " (manual)", min_value=float(min_v), max_value=float(max_v), value=float(s), step=step, format=fmt, key=f"{label}_num")
    return n


# Main layout

st.header("🚀 RockBurst Prediction Dashboard — Realtime")

left, right = st.columns([2,1])

with left:
    st.subheader("Input Parameters (sliders + manual)")
    energy_unit = slider_number("Energy Unit", 0.0, 80.0, 12.34)
    event_freq = slider_number("Event Frequency (per day)", 0.0, 15.0, 1.66)
    duration_days = slider_number("Duration (days)", 0.0, 30.0, 7.77)
    energy_per_volume_manual = slider_number("Energy per Volume (optional, leave 0 to auto-calc)", 0.0, 80.0, 0.0)
    energy_density = slider_number("Energy Density (Joule)", 0.0, 8.0, 4.36)
    volume_m3 = slider_number("Volume (m³)", 0.0, 6.0, 4.13)
    energy_joule_per_day = slider_number("Energy Joule per Day", 0.0, 7.0, 3.54)
    volume_m3_per_day = slider_number("Volume m³ per Day", 0.0, 5.0, 3.32)

    # if user left energy_per_volume = 0, compute from energy_unit / volume
    if energy_per_volume_manual and energy_per_volume_manual > 0:
        energy_per_volume = energy_per_volume_manual
    else:
        energy_per_volume = energy_unit / (volume_m3 if volume_m3 > 0 else 1.0)

with right:
    st.subheader("Prediction")
    # Prepare input row exactly matching features order
    raw_input = {
        "Energy_Unit": energy_unit,
        "Event_freq_unit_per_day": event_freq,
        "Duration_days": duration_days,
        "Energy_per_Volume": energy_per_volume,
        "Energy_density_Joule": energy_density,
        "Volume_m3": volume_m3,
        "Energy_Joule_per_day": energy_joule_per_day,
        "Volume_m3_per_day": volume_m3_per_day
    }

    df_raw = pd.DataFrame([raw_input])

    # Apply preprocessing: log transforms (use log1p to handle zeros) - matches training pipeline
    df_raw["Energy_Unit_log"] = np.log1p(df_raw["Energy_Unit"])
    df_raw["Event_freq_unit_per_day_log"] = np.log1p(df_raw["Event_freq_unit_per_day"])
    df_raw["Duration_days_log"] = np.log1p(df_raw["Duration_days"])
    df_raw["Energy_per_Volume_log"] = np.log1p(df_raw["Energy_per_Volume"])

    # Ensure order and apply scaler
    X = df_raw[features]
    X_scaled = scaler.transform(X)

    # Predict
    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    pred_label = label_map[int(pred_class)]
    top_prob = float(pred_proba[int(pred_class)])

    # Colored badge
    color = label_color[pred_label]
    badge_html = f'<div class="card"><span class="badge" style="background:{color}">{pred_label}</span></div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # Top-class gauge + numeric
    st.markdown(f"**Top prediction:** {pred_label}  \n**Confidence:** {top_prob*100:.2f}%")
    st.progress(top_prob)

    # Probability dataframe + bar chart
    proba_df = pd.DataFrame({
        "class": [label_map[i] for i in range(len(pred_proba))],
        "prob": pred_proba
    }).set_index("class")
    st.markdown("**Probability distribution**")
    st.dataframe(proba_df.style.format({"prob":"{:.4f}"}).background_gradient(cmap="Blues"), height=200)
    st.bar_chart(proba_df["prob"])


# Save history (local CSV) and provide export

if save_history:
    hist_row = {
        "timestamp": datetime.utcnow().isoformat(),
        **raw_input,
        "pred_class": int(pred_class),
        "pred_label": pred_label,
        "pred_confidence": top_prob,
        "probabilities": "|".join([f"{p:.6f}" for p in pred_proba])
    }

    # Append to CSV (create if missing)
    try:
        hist_df = pd.read_csv(history_file)
        hist_df = pd.concat([hist_df, pd.DataFrame([hist_row])], ignore_index=True)
    except FileNotFoundError:
        hist_df = pd.DataFrame([hist_row])
    hist_df.to_csv(history_file, index=False)

    st.sidebar.markdown("### Prediction history")
    st.sidebar.write(f"Total records: {len(hist_df)}")
    if st.sidebar.button("Show last 10"):
        st.sidebar.dataframe(hist_df.tail(10))

    # Download button
    csv_bytes = hist_df.to_csv(index=False).encode()
    st.sidebar.download_button("Download history CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")


# Footer / help

st.markdown("---")
st.caption("Model inference uses same preprocessing as training: log1p on selected features + scaler. Local history stored only if enabled.")
