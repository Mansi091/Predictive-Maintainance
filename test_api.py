import os
import pandas as pd
import requests
import time
import subprocess

API_URL = "http://127.0.0.1:8000"
DATASET_PATH = "rui-dataset.csv"

def run_tests():
    print("Starting API Integration Tests ")
    df = pd.read_csv(DATASET_PATH)
    
    normal_sample = df[df['Machine failure'] == 0].iloc[0]
    failure_sample = df[df['Machine failure'] == 1].iloc[0]
    
    payload_normal = {
        "air_temp": float(normal_sample['Air temperature [K]']),
        "process_temp": float(normal_sample['Process temperature [K]']),
        "rotational_speed": float(normal_sample['Rotational speed [rpm]']),
        "torque": float(normal_sample['Torque [Nm]']),
        "tool_wear": float(normal_sample['Tool wear [min]'])
    }
    response = requests.post(f"{API_URL}/predict", json=payload_normal)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    payload_failure = {
        "air_temp": float(failure_sample['Air temperature [K]']),
        "process_temp": float(failure_sample['Process temperature [K]']),
        "rotational_speed": float(failure_sample['Rotational speed [rpm]']),
        "torque": float(failure_sample['Torque [Nm]']),
        "tool_wear": float(failure_sample['Tool wear [min]'])
    }
    response = requests.post(f"{API_URL}/predict", json=payload_failure)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    test_batch_path = "test_batch.csv"
    df.head(20).to_csv(test_batch_path, index=False)
    
    with open(test_batch_path, 'rb') as f:
        files = {'file': (test_batch_path, f, 'text/csv')}
        response = requests.post(f"{API_URL}/predict-batch", files=files)
        
    print(f"Status: {response.status_code}")
    batch_res = response.json()
    print(f"Total: {batch_res['total_records']}")
    print(f"Failures: {batch_res['failures_detected']}")
    for pred in batch_res['predictions'][:3]:
        print(f"  Air Temp={pred['Air temperature [K]']}K, Speed={pred['Rotational speed [rpm]']}rpm, Prob={pred['Failure_Probability']:.4f}, Prediction={pred['Failure_Prediction']}")
        
    if os.path.exists(test_batch_path):
        os.remove(test_batch_path)

if __name__ == "__main__":
    try:
        r = requests.get(API_URL)
        if r.status_code == 200:
            run_tests()
    except requests.exceptions.ConnectionError:
        proc = subprocess.Popen(
            [".venv/Scripts/python", "-m", "uvicorn", "app:app", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(10)
        try:
            run_tests()
        finally:
            proc.terminate()
            proc.wait()
