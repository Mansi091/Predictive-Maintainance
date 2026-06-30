# Predictive Maintenance System & Explainable AI (XAI) Engine

This repository contains an end-to-end Machine Learning and REST API system designed to monitor industrial machine telemetry, predict imminent mechanical failures, and provide transparent explanations for the predictions using Explainable AI (SHAP).

## 🚀 Key Highlights

* **Deep Learning Model:** A 4-layer TensorFlow/Keras neural network trained to detect machine failures under severe class imbalance (~3% failure rate) using custom class weighting.
* **Explainable AI (XAI):** Uses **SHAP (SHapley Additive exPlanations)** on the backend to output real-time feature contributions, telling engineers *why* a machine is predicted to fail (e.g., speed overload vs. high tool wear).
* **Production-Grade API:** Deployed via FastAPI with robust request validation (Pydantic) and support for single-machine diagnostics and bulk CSV batch predictions.

---

## 📊 Performance Metrics

* **Recall:** **91%** (successfully detects 91% of all actual failures)
* **Precision:** **28%** (minimizes missed failures while managing false alarms)
* **Avg API Latency:** **213ms** (single-prediction latency measured on a standard CPU)
* **Dataset Size:** **10,000** machine telemetry records

---

## 🛠️ Project Architecture

```
                 +-----------------------+
                 |  rui-dataset.csv      |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  train.py (DL Model)  |
                 +-----+-----------+-----+
                       |           |
                       v           v
           +-------------+       +-------------+
           | model.keras |       | scaler.pkl  |
           +------+------+       +------+------+
                  |                     |
                  +----------+----------+
                             |
                             v
                 +-----------------------+
                 |  app.py (FastAPI)     | <--- Client Request
                 +-----------+-----------+
                             |
                             v
              +-----------------------------+
              | - Failure Probability       |
              | - SHAP Feature Importance   |
              | - Troubleshooting Guidance  |
              +-----------------------------+
```

---

## 📋 Telemetry Features

The system monitors 5 core parameters from machine sensors:
1. **Air temperature [K]**
2. **Process temperature [K]**
3. **Rotational speed [rpm]**
4. **Torque [Nm]**
5. **Tool wear [min]**

---

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Setup Environment
Clone the repository, create a virtual environment, and install dependencies:
```powershell
# Clone the repository
git clone https://github.com/Mansi091/Predictive-Maintainance.git
cd Predictive-Maintainance

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Model Training
To train the neural network model and save the weights/scalers:
```powershell
python train.py
```

### 4. Running the REST API
Start the FastAPI server:
```powershell
python -m uvicorn app:app --reload
```
* Access the interactive API Swagger documentation at: **`http://127.0.0.1:8000/docs`**

### 5. Running the QA Integration Test Suite
To verify the system end-to-end (starts background API, tests healthy/failure payloads, runs batch CSV uploads, and shuts down):
```powershell
python test_api.py
```
