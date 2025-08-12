from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class RockBurstFeatures(BaseModel):
    Energy_Unit: float
    Event_freq_unit_per_day: float
    Duration_days: float
    Energy_per_Volume: float
    Energy_density_Joule: float
    Volume_m3: float
    Energy_Joule_per_day: float
    Volume_m3_per_day: float

class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    id: int  # Added to reflect backend response

class PredictionRecord(BaseModel):
    id: int
    Timestamp: datetime
    Energy_Unit: float
    Event_freq_unit_per_day: float
    Duration_days: float
    Energy_per_Volume: float
    Energy_density_Joule: float
    Volume_m3: float
    Energy_Joule_per_day: float
    Volume_m3_per_day: float
    Model_pred_label: str
    Final_label_reviewed: Optional[str] = None
    Review_comment: Optional[str] = None
    
    class Config:
        orm_mode = True  # enables parsing from ORM models

class PredictionUpdate(BaseModel):
    final_label_reviewed: Optional[str] = None
    review_comment: Optional[str] = None
