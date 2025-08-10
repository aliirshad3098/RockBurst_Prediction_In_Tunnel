import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and scaler
model,features = joblib.load("trained_models/Best_model/Log_Reg_model.pkl")
scaler = joblib.load("trained_models/scaler_V3.pkl")  # Make sure you saved it during training

# Label mapping
label_map = {
    0: "Low",
    1: "Moderate",
    2: "Intense",
    3: "Extreme"
}

app = FastAPI()

# Input schema
class RockBurstInput(BaseModel):
    Energy_Unit: float
    Event_freq_unit_per_day: float
    Duration_days: float
    Energy_per_Volume: float
    Energy_density_Joule: float
    Volume_m3: float
    Energy_Joule_per_day: float
    Volume_m3_per_day: float

@app.post("/predict")
def predict(data: RockBurstInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Apply log transforms
        input_df['Energy_Unit_log'] = np.log(input_df['Energy_Unit'])
        input_df['Event_freq_unit_per_day_log'] = np.log(input_df['Event_freq_unit_per_day'])
        input_df['Duration_days_log'] = np.log(input_df['Duration_days'])
        input_df['Energy_per_Volume_log'] = np.log(input_df['Energy_per_Volume'])

        # Apply scaling
        scaled_input = scaler.transform(input_df[features])

        # Predict
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input).tolist()

        return {
            "prediction": int(prediction),
            "label": label_map[int(prediction)],
            "probabilities": probability
        }

    except Exception as e:
        return {"error": str(e)}
