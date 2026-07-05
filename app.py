import os
import pickle
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import io

app = FastAPI(
    title="Predictive Maintenance REST API",
    version="2.0.0"
)

MODEL_PKL_PATH = "model.pkl"

model = None
scaler = None
threshold = 0.5
feature_cols = []
explainer = None

class TelemetryData(BaseModel):
    air_temp: float = Field(..., alias="air_temp", ge=0)
    process_temp: float = Field(..., alias="process_temp", ge=0)
    rotational_speed: float = Field(..., alias="rotational_speed", ge=0)
    torque: float = Field(..., alias="torque", ge=0)
    tool_wear: float = Field(..., alias="tool_wear", ge=0)

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
    global model, scaler, threshold, feature_cols, explainer
    
    if not os.path.exists(MODEL_PKL_PATH):
        raise RuntimeError(f"Missing model file: {MODEL_PKL_PATH}")

    with open(MODEL_PKL_PATH, "rb") as f:
        artifacts = pickle.load(f)
        
    model = artifacts['model']
    scaler = artifacts['scaler']
    threshold = artifacts['threshold']
    feature_cols = artifacts['feature_cols']
    explainer = shap.TreeExplainer(model)

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

@app.post("/predict")
def predict_telemetry(data: TelemetryData):
    if model is None or scaler is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")
        
    try:
        power = data.torque * data.rotational_speed
        temp_diff = data.process_temp - data.air_temp
        
        features = np.array([[
            data.air_temp,
            data.process_temp,
            data.rotational_speed,
            data.torque,
            data.tool_wear,
            power,
            temp_diff
        ]])
        
        features_scaled = scaler.transform(features)
        prob = float(model.predict_proba(features_scaled)[0][1])
        prediction = int(prob >= threshold)
        
        shap_vals = explainer.shap_values(features_scaled)
        shap_arr = np.array(shap_vals)
        
        if len(shap_arr.shape) == 2:
            shap_contributions = shap_arr[0].tolist()
        elif len(shap_arr.shape) == 3:
            shap_contributions = shap_arr[0, :, 1].tolist()
        else:
            shap_contributions = shap_arr.flatten().tolist()
            
        explanations = {
            name: contrib for name, contrib in zip(feature_cols, shap_contributions)
        }
        
        recommendations = []
        if prediction == 1:
            max_contrib_feature = max(explanations, key=explanations.get)
            if max_contrib_feature == 'Tool wear [min]' and data.tool_wear > 150:
                recommendations.append("Tool wear is high. Schedule a cutting tool replacement immediately.")
            elif max_contrib_feature == 'Torque [Nm]' or max_contrib_feature == 'Power_Nm_RPM':
                recommendations.append("Mechanical workload/power is excessively high. Reduce operation load.")
            elif max_contrib_feature == 'Rotational speed [rpm]':
                recommendations.append("Rotational speed is dangerously high. Slow down spindle rotation.")
            elif max_contrib_feature in ['Air temperature [K]', 'Process temperature [K]', 'Temp_Difference_K']:
                recommendations.append("Overheating detected. Verify cooling fluid level or pause machine operation.")
            else:
                recommendations.append("General maintenance check required. Telemetry values exceed safety bounds.")
        else:
            recommendations.append("Machine is operating within safe parameters. Continue standard operations.")

        return {
            "failure_prediction": prediction,
            "failure_probability": prob,
            "decision_threshold": threshold,
            "risk_level": "High" if prob >= threshold else ("Medium" if prob >= (threshold * 0.5) else "Low"),
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
        df_engineered = df_input.copy()
        df_engineered['Power_Nm_RPM'] = df_engineered['Torque [Nm]'] * df_engineered['Rotational speed [rpm]']
        df_engineered['Temp_Difference_K'] = df_engineered['Process temperature [K]'] - df_engineered['Air temperature [K]']
        
        X_batch = df_engineered[feature_cols]
        X_batch_scaled = scaler.transform(X_batch)
        
        probs = model.predict_proba(X_batch_scaled)[:, 1].tolist()
        predictions = [int(p >= threshold) for p in probs]
        
        df_results = df_input.copy()
        df_results['Failure_Probability'] = probs
        df_results['Failure_Prediction'] = predictions
        
        results = df_results.to_dict(orient='records')
        
        return {
            "total_records": len(results),
            "failures_detected": sum(predictions),
            "decision_threshold": threshold,
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
