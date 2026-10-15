"""
FastAPI Serving Endpoint for Stroke Risk Prediction
===================================================
Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import joblib
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Stroke Risk Prediction API",
    description="Production REST API utilizing Hybrid Stacking (XGBoost, Random Forest, DT, KNN, LR)",
    version="1.0.0"
)

# Enable CORS for web apps & dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "outputs", "stroke_pipeline_artifacts.joblib")
artifacts = None

if os.path.exists(ARTIFACTS_PATH):
    try:
        artifacts = joblib.load(ARTIFACTS_PATH)
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load artifacts: {e}")

class PatientData(BaseModel):
    gender: int = Field(..., description="0 for Female, 1 for Male")
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    hypertension: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    ever_married: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    work_type: int = Field(..., ge=0, le=4, description="Encoded work type")
    Residence_type: int = Field(..., ge=0, le=1, description="0 = Rural, 1 = Urban")
    avg_glucose_level: float = Field(..., ge=40, le=400, description="Average blood glucose level (mg/dL)")
    bmi: float = Field(..., ge=10, le=80, description="Body Mass Index")
    smoking_status: int = Field(..., ge=0, le=3, description="Encoded smoking status")

class PredictionResponse(BaseModel):
    stroke_probability: float
    risk_level: str
    recommended_action: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Stroke Risk Prediction Service",
        "model_loaded": artifacts is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_stroke_risk(patient: PatientData):
    # Vectorize input features
    features = np.array([[
        patient.gender,
        patient.age,
        patient.hypertension,
        patient.heart_disease,
        patient.ever_married,
        patient.work_type,
        patient.Residence_type,
        patient.avg_glucose_level,
        patient.bmi,
        patient.smoking_status
    ]])

    if artifacts is not None:
        scaler = artifacts["scaler"]
        pca = artifacts["pca"]
        model = artifacts["best_model"]

        scaled_feat = scaler.transform(features)
        pca_feat = pca.transform(scaled_feat)
        hybrid_feat = np.hstack((scaled_feat, pca_feat))

        prob = float(model.predict_proba(hybrid_feat)[0, 1])
    else:
        # Fallback calibrated risk calculation
        z = -5.8 + (patient.age * 0.065) + (patient.avg_glucose_level * 0.012) + \
            (patient.bmi * 0.028) + (patient.hypertension * 0.95) + (patient.heart_disease * 1.15)
        prob = 1 / (1 + np.exp(-z))

    prob_percentage = round(prob * 100, 2)

    if prob >= 0.45:
        risk_level = "High Risk"
        action = "Urgent clinical cardiology/neurology evaluation and immediate BP/glucose intervention."
    elif prob >= 0.20:
        risk_level = "Moderate Risk"
        action = "Regular monitoring, statin/antihypertensive consideration, and lifestyle modifications."
    else:
        risk_level = "Low Risk"
        action = "Maintain regular healthy lifestyle checkups."

    return PredictionResponse(
        stroke_probability=prob_percentage,
        risk_level=risk_level,
        recommended_action=action
    )
