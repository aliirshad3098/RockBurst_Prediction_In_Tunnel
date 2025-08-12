import joblib
import numpy as np
import pandas as pd

model, features = joblib.load("trained_models/Best_model/Log_Reg_model.pkl")
scaler = joblib.load("trained_models/scaler_V3.pkl")

label_map = {0: "Low", 1: "Moderate", 2: "Intense", 3: "Extreme"}

def predict(data: dict):
    df = pd.DataFrame([data])

    # Apply log transforms safely
    df["Energy_Unit_log"] = np.log1p(df["Energy_Unit"])
    df["Event_freq_unit_per_day_log"] = np.log1p(df["Event_freq_unit_per_day"])
    df["Duration_days_log"] = np.log1p(df["Duration_days"])
    df["Energy_per_Volume_log"] = np.log1p(df["Energy_per_Volume"])

    # Scale features based on preloaded scaler and features list
    scaled_input = scaler.transform(df[features])

    pred_class = model.predict(scaled_input)[0]
    pred_proba = model.predict_proba(scaled_input)[0]

    return {
        "predicted_class": label_map[int(pred_class)],
        "probabilities": {label_map[i]: float(pred_proba[i]) for i in range(len(pred_proba))}
    }
