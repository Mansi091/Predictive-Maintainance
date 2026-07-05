# Predictive Maintenance System & Explainable AI (XAI) Engine

This repository contains an end-to-end Machine Learning and REST API system designed to monitor industrial machine telemetry, predict imminent mechanical failures, and provide transparent explanations for predictions using Explainable AI (SHAP).

## 🚀 Key Highlights

* **Advanced Tabular Model:** An XGBoost Classifier trained on stratified 5-fold cross-validation, outperforming standard dense neural networks.
* **Domain-Driven Feature Engineering:** Includes physical derived features: `Power = Torque * Rotational Speed` and `Temp Difference = Process Temp - Air Temp` to guide decision boundaries.
* **Precision Optimization:** Uses precision-recall threshold tuning (`threshold = 0.8615`) to reduce false alarms by ~54% compared to default settings.
* **Explainable AI (XAI):** Employs **SHAP TreeExplainer** on the backend for near-instantaneous (under 1ms) feature contribution analysis.
* **Production-Grade API & Dashboard:** Includes a FastAPI backend serving a lightweight, interactive HTML/CSS dark-mode telemetry diagnostic dashboard.

---

## 📊 Performance Metrics

* **Precision (False Alarms):** **82.4%** (Only ~1 in 5 predictions is a false positive)
* **Recall (Detect Rate):** **74.6%** (Successfully captures three-quarters of actual failures)
* **F1-Score:** **78.3%**
* **PR-AUC:** **83.3%**
* **Avg API Latency:** **<5ms** (SHAP explanation latency reduced by 98% using TreeExplainer)
* **Dataset Size:** **10,000** machine telemetry records

### 📈 Model Evaluation Visualizations

| Precision-Recall Curve & Threshold Tuning | Confusion Matrix |
| :---: | :---: |
| ![Precision-Recall Curve](static/eda/pr_curve.png) | ![Confusion Matrix](static/eda/confusion_matrix.png) |

---

## 🛠️ Project Architecture

```
                 +---------------------------+
                 |       rui-dataset.csv     |
                 +-------------+-------------+
                               |
                               v
                 +---------------------------+
                 |  train_advanced.py (XGB)  |
                 +-------+-----------+-------+
                         |           |
                         v           v
             +---------------+   +-----------+
             |   model.pkl   |   | eda plots |
             +-------+-------+   +-----------+
                     |
                     v
                 +---------------------------+
                 |       app.py (FastAPI)    | <--- Client / Dashboard Request
                 +-----------+-----------+---+
                             |
                             v
              +-----------------------------+
              | - Failure Probability       |
              | - SHAP Tree Contribution    |
              | - Troubleshooting Guidance  |
              +-----------------------------+
```

---

## 📋 Telemetry & Engineered Features

The system monitors 5 core sensor parameters:
1. **Air temperature [K]**
2. **Process temperature [K]**
3. **Rotational speed [rpm]**
4. **Torque [Nm]**
5. **Tool wear [min]**

Additionally, 2 physics-derived features are engineered:
6. **Power_Nm_RPM** (`Torque * Rotational Speed` - workload index)
7. **Temp_Difference_K** (`Process temperature - Air temperature` - thermal efficiency)

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

### 3. Exploratory Data Analysis & Feature Engineering
Generate EDA trends and correlation heatmap plots:
```powershell
python eda_analyzer.py
```
Plots will be saved inside the `static/eda` folder.

### 4. Advanced Model Training
To train the XGBoost and Random Forest classifiers, tune decision thresholds, and output validation metrics:
```powershell
python train_advanced.py
```

### 5. Running the REST API & Dashboard
Start the FastAPI server:
```powershell
python -m uvicorn app:app --reload
```
* Access the interactive API Swagger documentation at: **`http://127.0.0.1:8000/docs`**
* Access the interactive Dashboard at: **`http://127.0.0.1:8000/`**

### 6. Running the QA Integration Test Suite
To verify the system end-to-end (starts background API, tests healthy/failure payloads, runs batch CSV uploads, and shuts down):
```powershell
python test_api.py
```

