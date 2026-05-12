import requests
import json
import pandas as pd
import time

def test_health():
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url)
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error connecting to API: {e}")

def test_predict():
    url = "http://localhost:8000/predict"
    
    # Create sample dummy data (24 points as per LOOKBACK)
    data = []
    base_time = time.time()
    for i in range(30): # More than 24 to allow for indicators
        data.append({
            "timestamp": str(pd.to_datetime(base_time - (30-i)*3600, unit='s')),
            "close": 150.0 + i*0.1,
            "volume": 1000000 + i*1000,
            "rsi": 50.0 + (i%10),
            "macd": 0.5 + (i/100),
            "signal_line": 0.4 + (i/100),
            "atr": 1.5,
            "sentiment": 0.1 * (i%5),
            "hour": (i % 24),
            "day_of_week": (i // 24) % 7
        })
    
    try:
        response = requests.post(url, json=data)
        print(f"Prediction Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Prediction Result: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error connecting to API: {e}")

if __name__ == "__main__":
    print("Ensure the FastAPI app is running (uvicorn app:app --reload)")
    test_health()
    test_predict()
