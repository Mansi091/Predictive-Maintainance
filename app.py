import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import io

app = FastAPI(
    title="Predictive Maintenance REST API",
    description="Production-grade API for predicting machine failure with SHAP explainability and telemetry validation.",
    version="1.0.0"
)

MODEL_PATH = "model.keras"
SCALER_PATH = "scaler.pkl"
DATASET_PATH = "rui-dataset.csv"

model = None
scaler = None
explainer = None

class TelemetryData(BaseModel):
    air_temp: float = Field(..., alias="air_temp", description="Air temperature [K]", ge=0)
    process_temp: float = Field(..., alias="process_temp", description="Process temperature [K]", ge=0)
    rotational_speed: float = Field(..., alias="rotational_speed", description="Rotational speed [rpm]", ge=0)
    torque: float = Field(..., alias="torque", description="Torque [Nm]", ge=0)
    tool_wear: float = Field(..., alias="tool_wear", description="Tool wear [min]", ge=0)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "air_temp": 298.1,
                "process_temp": 308.6,
                "rotational_speed": 1500.0,
                "torque": 40.0,
                "tool_wear": 50.0
            }
        }

@app.on_event("startup")
def startup_event():
    global model, scaler, explainer
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Missing required model files ({MODEL_PATH} or {SCALER_PATH}).")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        feature_cols = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        
        np.random.seed(42)
        bg_indices = np.random.choice(X_scaled.shape[0], 50, replace=False)
        bg_data = X_scaled[bg_indices]
        
        explainer = shap.KernelExplainer(lambda x: model.predict(x, verbose=0), bg_data)
    else:
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH} for SHAP background data initialization.")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "description": "Predictive Maintenance REST API with SHAP explainability. Go to /docs for interactive documentation."
    }

@app.post("/predict")
def predict_telemetry(data: TelemetryData):
    if model is None or scaler is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model or explainer not initialized.")
        
    try:
        features = np.array([[
            data.air_temp,
            data.process_temp,
            data.rotational_speed,
            data.torque,
            data.tool_wear
        ]])
        
        features_scaled = scaler.transform(features)
        
        prob = float(model.predict(features_scaled, verbose=0)[0][0])
        prediction = int(prob >= 0.5)
        
        shap_vals = explainer.shap_values(features_scaled, nsamples=100)
        shap_arr = np.array(shap_vals)
        
        if len(shap_arr.shape) == 3:
            shap_contributions = shap_arr[0, :, 0].tolist()
        elif len(shap_arr.shape) == 2:
            shap_contributions = shap_arr[0].tolist()
        else:
            shap_contributions = shap_arr.flatten().tolist()
            
        feature_names = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        
        explanations = {
            name: contrib for name, contrib in zip(feature_names, shap_contributions)
        }
        
        recommendations = []
        if prediction == 1:
            max_contrib_feature = max(explanations, key=explanations.get)
            if max_contrib_feature == 'Tool wear [min]' and data.tool_wear > 150:
                recommendations.append("Tool wear is high. Schedule a cutting tool replacement immediately.")
            elif max_contrib_feature == 'Torque [Nm]' and data.torque > 50:
                recommendations.append("Torque is excessively high. Reduce operational load or speed.")
            elif max_contrib_feature == 'Rotational speed [rpm]' and data.rotational_speed > 2000:
                recommendations.append("Rotational speed is dangerously high. Slow down spindle rotation.")
            elif max_contrib_feature in ['Air temperature [K]', 'Process temperature [K]']:
                recommendations.append("Overheating detected. Verify cooling fluid level or pause machine operation.")
            else:
                recommendations.append("General maintenance check required. Telemetry values exceed safety bounds.")
        else:
            recommendations.append("Machine is operating within safe parameters. Continue standard operations.")

        return {
            "failure_prediction": prediction,
            "failure_probability": prob,
            "risk_level": "High" if prob >= 0.5 else ("Medium" if prob >= 0.2 else "Low"),
            "feature_contributions": explanations,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
    required_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400, 
            detail=f"CSV is missing required columns: {missing_cols}"
        )
        
    try:
        X_batch = df_input[required_cols]
        X_batch_scaled = scaler.transform(X_batch)
        
        probs = model.predict(X_batch_scaled, verbose=0).flatten().tolist()
        predictions = [int(p >= 0.5) for p in probs]
        
        df_results = df_input.copy()
        df_results['Failure_Probability'] = probs
        df_results['Failure_Prediction'] = predictions
        
        results = df_results.to_dict(orient='records')
        
        return {
            "total_records": len(results),
            "failures_detected": sum(predictions),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
