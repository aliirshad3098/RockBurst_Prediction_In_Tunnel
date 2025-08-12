from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json
from datetime import datetime

from app import models, schemas, ml_model, database
from app.database import get_db

app = FastAPI(title="RockBurst Prediction API")

@app.on_event("startup")
def on_startup():
    models.Base.metadata.create_all(bind=database.engine)

@app.post("/predict", response_model=schemas.PredictionResponse)
def predict_rockburst(input_data: schemas.RockBurstFeatures, db: Session = Depends(database.get_db)):
    # Perform ML prediction
    result = ml_model.predict(input_data.dict())
    
    # Store probabilities as JSON string (lowercase field name)
    prob_str = json.dumps(result["probabilities"])

    # Create DB entry
    entry = models.RockBurstEntry(
        **input_data.dict(),
        Model_pred_label=result["predicted_class"],
        probabilities=prob_str,
        Timestamp=datetime.utcnow()
    )

    try:
        db.add(entry)
        db.commit()
        db.refresh(entry)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database commit failed: {str(e)}")

    # Return the prediction result including saved record ID
    return {
        "predicted_class": result["predicted_class"],
        "probabilities": result["probabilities"],
        "id": entry.id
    }


@app.get("/predictions", response_model=List[schemas.PredictionRecord])
def get_all_predictions(db: Session = Depends(get_db)):
    records = db.query(models.RockBurstEntry).order_by(models.RockBurstEntry.Timestamp.desc()).all()
    if not records:
        raise HTTPException(status_code=404, detail="No prediction records found.")
    return records

@app.put("/predictions/{record_id}")
def update_prediction(record_id: int, update_data: schemas.PredictionUpdate, db: Session = Depends(get_db)):
    record = db.query(models.RockBurstEntry).filter(models.RockBurstEntry.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    if update_data.final_label_reviewed is not None:
        record.Final_label_reviewed = update_data.final_label_reviewed
    if update_data.review_comment is not None:
        record.Review_comment = update_data.review_comment

    try:
        db.commit()
        db.refresh(record)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update record: {str(e)}")

    return {"message": "Record updated successfully"}
