"""FastAPI application for Bank Marketing prediction service."""
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Production-grade ML API for term deposit subscription prediction",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model and preprocessor
MODEL = None
PREPROCESSOR = None
MODEL_INFO = {}


class CustomerData(BaseModel):
    """Customer data schema for prediction."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Job type")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default?")
    balance: int = Field(..., description="Average yearly balance")
    housing: str = Field(..., description="Has housing loan?")
    loan: str = Field(..., description="Has personal loan?")
    contact: str = Field(..., description="Contact communication type")
    day: int = Field(..., ge=1, le=31, description="Last contact day")
    month: str = Field(..., description="Last contact month")
    duration: int = Field(..., ge=0, description="Last contact duration (seconds)")
    campaign: int = Field(..., ge=1, description="Number of contacts this campaign")
    pdays: int = Field(..., ge=-1, description="Days since last contact")
    previous: int = Field(..., ge=0, description="Number of previous contacts")
    poutcome: str = Field(..., description="Outcome of previous campaign")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1500,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "duration": 300,
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: str
    probability: float
    confidence: str
    recommendation: str


def load_model_and_preprocessor():
    """Load model and preprocessor from registry."""
    global MODEL, PREPROCESSOR, MODEL_INFO
    
    try:
        logger.info("Loading model and preprocessor...")
        
        # Load from model registry (latest version)
        registry_dir = Path("model_registry")
        latest_file = registry_dir / "latest.txt"
        
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                version = f.read().strip()
            
            version_dir = registry_dir / version
            model_path = version_dir / "model.pkl"
            preprocessor_path = version_dir / "preprocessor.pkl"
            
            # Load model
            with open(model_path, 'rb') as f:
                MODEL = pickle.load(f)
            
            # Load preprocessor
            with open(preprocessor_path, 'rb') as f:
                PREPROCESSOR = pickle.load(f)
            
            MODEL_INFO = {
                "version": version,
                "status": "loaded",
                "model_path": str(model_path)
            }
            
            logger.info(f"Model loaded successfully: {version}")
        else:
            logger.warning("No model found in registry. Using fallback.")
            MODEL_INFO = {"status": "not_loaded", "message": "No model in registry"}
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        MODEL_INFO = {"status": "error", "message": str(e)}


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model_and_preprocessor()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_status": MODEL_INFO.get("status", "unknown"),
        "model_version": MODEL_INFO.get("version", "N/A")
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """Predict term deposit subscription for a single customer."""
    try:
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to DataFrame
        data = pd.DataFrame([customer.dict()])
        
        # Encode features
        label_encoders = PREPROCESSOR['label_encoders']
        for col, encoder in label_encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col])
        
        # Predict
        prediction = MODEL.predict(data)[0]
        probability = MODEL.predict_proba(data)[0]
        
        # Get probability for positive class
        prob_yes = float(probability[1])
        
        # Determine prediction and confidence
        pred_label = "yes" if prediction == 1 else "no"
        
        if prob_yes >= 0.7:
            confidence = "High"
            recommendation = "Strong candidate for term deposit. Prioritize contact."
        elif prob_yes >= 0.5:
            confidence = "Medium"
            recommendation = "Moderate interest. Consider targeted campaign."
        else:
            confidence = "Low"
            recommendation = "Low conversion probability. Deprioritize or skip."
        
        return PredictionResponse(
            prediction=pred_label,
            probability=round(prob_yes, 4),
            confidence=confidence,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    """Batch prediction for multiple customers."""
    try:
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for customer in customers:
            # Convert to DataFrame
            data = pd.DataFrame([customer.dict()])
            
            # Encode features
            label_encoders = PREPROCESSOR['label_encoders']
            for col, encoder in label_encoders.items():
                if col in data.columns:
                    data[col] = encoder.transform(data[col])
            
            # Predict
            prediction = MODEL.predict(data)[0]
            probability = MODEL.predict_proba(data)[0]
            prob_yes = float(probability[1])
            
            results.append({
                "customer": customer.dict(),
                "prediction": "yes" if prediction == 1 else "no",
                "probability": round(prob_yes, 4)
            })
        
        return {
            "total_customers": len(customers),
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information."""
    return MODEL_INFO


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
