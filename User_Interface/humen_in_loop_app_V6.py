import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "Intense", 3: "Extreme"}

# --- API Utility functions ---

def call_prediction_api(data: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

def get_all_predictions():
    try:
        response = requests.get(f"{API_URL}/predictions")
        response.raise_for_status()
        return response.json()  # Expecting list of dicts
    except Exception as e:
        st.error(f"Failed to fetch prediction history: {e}")
        return []

def update_prediction_record(record_id: int, final_label: str, review_comment: str):
    try:
        payload = {
            "final_label_reviewed": final_label,
            "review_comment": review_comment
        }
        response = requests.put(f"{API_URL}/predictions/{record_id}", json=payload)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to update record: {e}")
        return False

def format_float_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)
    return df

# --- Streamlit UI setup ---
st.set_page_config(page_title="RockBurst Prediction & Review Dashboard", layout="wide")
st.title("🚀 RockBurst Prediction & Review Dashboard")

# --- Initialize session state ---
if "has_prediction" not in st.session_state:
    st.session_state.update({
        "has_prediction": False,
        "last_prediction_id": None,
        "pred_label": None,
        "probabilities": {},
        "final_label_index": 0,
        "Review_comment": "",
    })

# --- Layout ---
left_col, right_col = st.columns([2, 3])

with left_col:
    st.header("Input Parameters")
    energy_unit = st.slider("Energy Unit", 0.0, 80.0, 12.34, step=0.01)
    event_freq = st.slider("Event Frequency (per day)", 0.0, 15.0, 1.66, step=0.01)
    duration_days = st.slider("Duration (days)", 0.0, 30.0, 7.77, step=0.01)
    energy_density = st.slider("Energy Density (Joule)", 0.0, 8.0, 4.36, step=0.01)
    volume_m3 = st.slider("Volume (m³)", 0.0, 6.0, 4.13, step=0.01)
    energy_joule_per_day = st.slider("Energy Joule per Day", 0.0, 7.0, 3.54, step=0.01)
    volume_m3_per_day = st.slider("Volume m³ per Day", 0.0, 5.0, 3.32, step=0.01)

    energy_per_volume = energy_unit / volume_m3 if volume_m3 > 0 else 0.0
    st.markdown(f"**Calculated Energy per Volume:** `{energy_per_volume:.4f}`")

    st.markdown("---")

    input_data = {
        "Energy_Unit": energy_unit,
        "Event_freq_unit_per_day": event_freq,
        "Duration_days": duration_days,
        "Energy_per_Volume": energy_per_volume,
        "Energy_density_Joule": energy_density,
        "Volume_m3": volume_m3,
        "Energy_Joule_per_day": energy_joule_per_day,
        "Volume_m3_per_day": volume_m3_per_day,
    }

    predict_btn = st.button("🔍 Predict Risk Level")

with right_col:
    st.header("Prediction & Review")

    if predict_btn:
        with st.spinner("Requesting prediction and saving data..."):
            api_response = call_prediction_api(input_data)

        if api_response:
            pred_label = api_response.get("predicted_class", "N/A")
            probabilities = api_response.get("probabilities", {})

            # Refresh prediction state
            st.session_state["has_prediction"] = True
            st.session_state["last_prediction_id"] = api_response.get("id")
            st.session_state["pred_label"] = pred_label
            st.session_state["probabilities"] = probabilities
            st.session_state["final_label_index"] = list(LABEL_MAP.values()).index(pred_label) if pred_label in LABEL_MAP.values() else 0
            st.session_state["Review_comment"] = ""

            st.success("Prediction and input data saved to database successfully.")

    if st.session_state["has_prediction"]:
        st.markdown(f"### Prediction: **{st.session_state['pred_label']}**")

        proba_df = pd.DataFrame(
            list(st.session_state["probabilities"].items()),
            columns=["Risk Level", "Probability"]
        )
        proba_df["Probability"] = proba_df["Probability"].astype(float)
        st.bar_chart(proba_df.set_index("Risk Level"))

        st.markdown("---")
        st.subheader("Review & Edit Prediction")

        final_label = st.selectbox(
            "Final Label (edit if needed):",
            options=list(LABEL_MAP.values()),
            index=st.session_state.get("final_label_index", 0),
            key="edit_final_label"
        )
        review_comment = st.text_area(
            "Review Comment (optional):",
            value=st.session_state.get("Review_comment", ""),
            height=80,
            key="edit_review_comment"
        )

        # Update session state immediately
        st.session_state["final_label_index"] = list(LABEL_MAP.values()).index(final_label)
        st.session_state["Review_comment"] = review_comment

        if st.button("💾 Save Edited Review"):
            if st.session_state["last_prediction_id"] is None:
                st.error("No prediction record ID found for updating.")
            else:
                success = update_prediction_record(
                    record_id=st.session_state["last_prediction_id"],
                    final_label=final_label,
                    review_comment=review_comment
                )
                if success:
                    st.success("Edited review saved successfully!")
                    st.session_state["update_done"] = True  # or toggle a boolean flag


    else:
        st.info("Prediction & Review panel will appear here after you make your first prediction.")

# --- Visualization Section ---
st.markdown("---")
st.header("Model Predictions Analytics & Visualization")

history_data = get_all_predictions()
pred_df = pd.DataFrame(history_data)

if pred_df.empty:
    st.info("No model predictions available. Perform predictions to generate data.")
else:
    # Parse timestamp column safely
    pred_df["Timestamp_utc"] = pd.to_datetime(pred_df.get("Timestamp", pd.Series()), errors='coerce')

    # Prepare probability columns with JSON parsing
    prob_cols = [f"Prob_{lbl}" for lbl in LABEL_MAP.values()]

    def parse_probabilities(prob_str):
        try:
            return json.loads(prob_str)
        except Exception:
            return {}

    if "probabilities" in pred_df.columns:
        proba_dicts = pred_df["probabilities"].apply(parse_probabilities)
        proba_df = pd.json_normalize(proba_dicts)
        proba_df = proba_df.reindex(columns=LABEL_MAP.values()).fillna(0)
        proba_df.columns = prob_cols
        pred_df = pd.concat([pred_df.reset_index(drop=True), proba_df], axis=1)
    else:
        for col in prob_cols:
            pred_df[col] = np.nan

    # Select label column for visualization
    if (
        "Final_label_reviewed" in pred_df.columns
        and pred_df["Final_label_reviewed"].notna().any()
        and pred_df["Final_label_reviewed"].astype(str).str.strip().astype(bool).any()
    ):
        label_col = "Final_label_reviewed"
    elif "Model_pred_label" in pred_df.columns:
        label_col = "Model_pred_label"
    else:
        label_col = None
        st.error("No valid label column found in prediction history for visualization.")

    if label_col:
        col1, col2, col3 = st.columns(3)

        # 1. Counts bar chart
        with col1:
            st.subheader("Prediction Counts per Risk Level")
            pred_counts = pred_df[label_col].value_counts().reindex(LABEL_MAP.values(), fill_value=0)
            fig_bar = px.bar(
                x=pred_counts.index,
                y=pred_counts.values,
                labels={"x": "Risk Level", "y": "Count"},
                title="Counts of Predicted Risk Levels",
                color=pred_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 2. Probability distribution boxplot
        with col2:
            st.subheader("Prediction Probability Distributions")
            prob_melt = pred_df.melt(
                id_vars=[label_col],
                value_vars=prob_cols,
                var_name="Risk Level Probability",
                value_name="Probability"
            )
            prob_melt["Risk Level Probability"] = prob_melt["Risk Level Probability"].str.replace("Prob_", "", regex=False)

            fig_box = px.box(
                prob_melt,
                x="Risk Level Probability",
                y="Probability",
                color="Risk Level Probability",
                title="Prediction Probabilities Distribution",
                labels={"Probability": "Probability", "Risk Level Probability": "Risk Level"},
                points="outliers",
                category_orders={"Risk Level Probability": list(LABEL_MAP.values())},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # 3. Pie chart of final label distribution
        with col3:
            st.subheader("Final Label Distribution (Pie Chart)")
            label_counts = pred_df[label_col].value_counts().reindex(LABEL_MAP.values(), fill_value=0)
            fig_pie = go.Figure(data=[go.Pie(
                labels=label_counts.index,
                values=label_counts.values,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig_pie.update_layout(title_text="Risk Level Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Recent predictions table
        st.markdown("---")
        st.subheader("Recent Predictions & Reviews (Last 10)")

        recent_cols = [
            "Timestamp", "Energy_Unit", "Event_freq_unit_per_day", "Duration_days",
            "Energy_per_Volume", "Energy_density_Joule", "Volume_m3",
            "Energy_Joule_per_day", "Volume_m3_per_day",
            "Model_pred_label", "Final_label_reviewed", "Review_comment"
        ]
        recent_df = pred_df.sort_values("Timestamp_utc", ascending=False).head(10)
        recent_df_display = recent_df[recent_cols].copy()
        recent_df_display = format_float_columns(recent_df_display, recent_cols[1:9])
        st.dataframe(recent_df_display.style.set_properties(**{'white-space': 'pre-wrap'}))

    else:
        st.warning("Visualization skipped due to missing label column.")

# --- Section: Edit Previous Predictions ---
st.markdown("---")
st.header("Edit Previous Predictions")

if pred_df.empty:
    st.info("No previous predictions found.")
else:
    history_df = pred_df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

    options = history_df.apply(
        lambda row: f"{row['Timestamp']} | Predicted: {row.get('Model_pred_label', '')} | Final: {row.get('Final_label_reviewed', '')}",
        axis=1
    ).tolist()

    selected_index = st.selectbox("Select a prediction record to edit:", options, index=0)

    record_idx = options.index(selected_index)
    selected_record = history_df.loc[record_idx]

    st.markdown("### Selected Prediction Details")

    # Exclude unwanted columns
    cols_to_show = selected_record.index.difference(["probabilities", "Review_comment", "Final_label_reviewed"])
    
    # Prepare data as key-value pairs
    data = {
        "Data": cols_to_show,
        "Value": [round(selected_record[col], 3) if isinstance(selected_record[col], float) else selected_record[col] for col in cols_to_show]
    }
    
    # Create DataFrame for display
    df_display = pd.DataFrame(data)
    
    # Show table with two columns: 'Data' and 'Value'
    st.table(df_display)

    new_final_label = st.selectbox(
        "Update Final Label:",
        options=list(LABEL_MAP.values()),
        index=list(LABEL_MAP.values()).index(selected_record.get("Final_label_reviewed") or selected_record.get("Model_pred_label", LABEL_MAP[0]))
    )
    new_comment = st.text_area(
        "Add/Edit Review Comment:",
        value=selected_record.get("Review_comment") or "",
        height=100
    )

    if st.button("Save Changes"):
        update_success = update_prediction_record(
            record_id=int(selected_record["id"]),
            final_label=new_final_label,
            review_comment=new_comment
        )
        if update_success:
            st.success("Record updated successfully.")
            st.session_state["update_done"] = True  # or toggle a boolean flag

