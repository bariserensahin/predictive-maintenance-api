from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os
from typing import Dict, Any

# Pydantic model for input data
class SensorData(BaseModel):
    Machine_ID: int
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float
    TWF: int = 0
    HDF: int = 0
    PWF: int = 0
    OSF: int = 0
    RNF: int = 0

# Pydantic model for response
class PredictionResponse(BaseModel):
    failure_probability: float
    failure_percentage: float
    prediction: str
    status: str

# FastAPI app initialization
app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction using CatBoost model",
    version="1.0.0"
)

# Global model variable
model = None
feature_names = None

@app.on_event("startup")
async def load_model():
    """Load the CatBoost model into memory on startup"""
    global model, feature_names
    
    model_path = "models/catboost_model.cbm"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        
        # Define feature names based on training data
        feature_names = [
            'UDI', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
        ]
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Feature names: {feature_names}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict - POST endpoint for failure prediction",
            "health": "/health - Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(sensor_data: SensorData):
    """
    Predict machine failure probability based on sensor data
    
    Args:
        sensor_data: JSON object containing sensor readings
        
    Returns:
        PredictionResponse: Failure probability and prediction
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input data to DataFrame with correct column names
        input_data = {
            'UDI': sensor_data.Machine_ID,
            'Air temperature [K]': sensor_data.Air_temperature,
            'Process temperature [K]': sensor_data.Process_temperature,
            'Rotational speed [rpm]': sensor_data.Rotational_speed,
            'Torque [Nm]': sensor_data.Torque,
            'Tool wear [min]': sensor_data.Tool_wear,
            'TWF': sensor_data.TWF,
            'HDF': sensor_data.HDF,
            'PWF': sensor_data.PWF,
            'OSF': sensor_data.OSF,
            'RNF': sensor_data.RNF
        }
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([input_data], columns=feature_names)
        
        # Make prediction
        prediction_proba = model.predict_proba(df)[:, 1][0]
        prediction_class = model.predict(df)[0]
        
        # Calculate percentage
        failure_percentage = prediction_proba * 100
        
        # Determine prediction label
        if prediction_class == 1:
            prediction_label = "Failure"
            status = "Warning: Machine failure predicted"
        else:
            prediction_label = "No Failure"
            status = "Normal operation"
        
        return PredictionResponse(
            failure_probability=float(prediction_proba),
            failure_percentage=float(failure_percentage),
            prediction=prediction_label,
            status=status
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(sensor_data_list: list[SensorData]):
    """
    Predict machine failure for multiple sensor readings
    
    Args:
        sensor_data_list: List of sensor data objects
        
    Returns:
        List of predictions
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        predictions = []
        
        for i, sensor_data in enumerate(sensor_data_list):
            # Convert input data to DataFrame
            input_data = {
                'UDI': sensor_data.Machine_ID,
                'Air temperature [K]': sensor_data.Air_temperature,
                'Process temperature [K]': sensor_data.Process_temperature,
                'Rotational speed [rpm]': sensor_data.Rotational_speed,
                'Torque [Nm]': sensor_data.Torque,
                'Tool wear [min]': sensor_data.Tool_wear,
                'TWF': sensor_data.TWF,
                'HDF': sensor_data.HDF,
                'PWF': sensor_data.PWF,
                'OSF': sensor_data.OSF,
                'RNF': sensor_data.RNF
            }
            
            df = pd.DataFrame([input_data], columns=feature_names)
            
            # Make prediction
            prediction_proba = model.predict_proba(df)[:, 1][0]
            prediction_class = model.predict(df)[0]
            
            predictions.append({
                "machine_id": sensor_data.Machine_ID,
                "failure_probability": float(prediction_proba),
                "failure_percentage": float(prediction_proba * 100),
                "prediction": "Failure" if prediction_class == 1 else "No Failure"
            })
        
        return {"predictions": predictions, "total_count": len(predictions)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
