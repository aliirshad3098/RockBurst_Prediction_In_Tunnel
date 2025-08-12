from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from .database import Base
from datetime import datetime

class RockBurstEntry(Base):
    __tablename__ = "rockburst_predictions"

    id = Column(Integer, primary_key=True, index=True)
    Timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    Energy_Unit = Column(Float)
    Event_freq_unit_per_day = Column(Float)
    Duration_days = Column(Float)
    Energy_per_Volume = Column(Float)
    Energy_density_Joule = Column(Float)
    Volume_m3 = Column(Float)
    Energy_Joule_per_day = Column(Float)
    Volume_m3_per_day = Column(Float)
    Model_pred_label = Column(String(20))
    Final_label_reviewed = Column(String(20), nullable=True)
    Review_comment = Column(Text, nullable=True)
    probabilities = Column("Probabilities", Text, nullable=True)  # Note the quoted column name here