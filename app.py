from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import mlflow.keras
import os
from typing import List

# =========================
# 1. SETTINGS & LOADING
# =========================

app = FastAPI(
    title="Market Prediction API",
    description="Real-time market movement prediction using Sequential Deep Learning",
    version="1.0.0"
)

MODEL_NAME = "SimpleRNN" # Based on the best performing model from latest runs
LOOKBACK = 24
FEATURES = [
    'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 
    'ATR', 'sentiment', 'Hour', 'DayOfWeek', 'Volatility'
]

# Load Scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Load Model from MLflow
def load_best_model():
    try:
        # We try to load the latest run from the experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Market_Prediction_V3")
        if not experiment:
             experiment = client.get_experiment_by_name("Market_Prediction_V2")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.directional_accuracy DESC"],
            max_results=1
        )
        
        if not runs:
            return None
        
        best_run_id = runs[0].info.run_id
        model_uri = f"runs:/{best_run_id}/model_simplernn" # Or generic model path
        model = mlflow.keras.load_model(model_uri)
        print(f"Loaded model from run: {best_run_id}")
        return model
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None

model = load_best_model()

# =========================
# 2. SCHEMAS
# =========================

class MarketData(BaseModel):
    timestamp: str
    close: float
    volume: float
    rsi: float
    macd: float
    signal_line: float
    atr: float
    sentiment: float
    hour: int
    day_of_week: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_used: str
    status: str

# =========================
# 3. PREPROCESSING UTILS
# =========================

def preprocess_input(data_list: List[MarketData]):
    if len(data_list) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} historical data points for prediction.")
    
    # Convert to DataFrame
    df = pd.DataFrame([d.dict() for d in data_list])
    
    # Calculate Volatility (rolling std of log returns as per train.py)
    df['log_ret'] = np.log((df['close'] + 1e-6) / (df['close'].shift(1) + 1e-6))
    df['Volatility'] = df['log_ret'].rolling(window=10).std()
    
    # We also need log returns for Close and Volume as features
    df['Close_log'] = df['log_ret']
    df['Volume_log'] = np.log((df['volume'] + 1e-6) / (df['volume'].shift(1) + 1e-6))
    
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < LOOKBACK:
         raise ValueError("Not enough data after calculating indicators/lags.")

    # Map to FEATURES list used in training
    # Note: FEATURES in train.py were ['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 'ATR', 'sentiment', 'Hour', 'DayOfWeek', 'Volatility']
    # But Close and Volume were logged in-place.
    
    feature_data = df.tail(LOOKBACK).copy()
    
    # Prepare the actual input array
    # Re-aligning with train.py: for col in ['Close', 'Volume']: df_proc[col] = np.log(...)
    input_features = pd.DataFrame()
    input_features['Close'] = feature_data['Close_log']
    input_features['Volume'] = feature_data['Volume_log']
    input_features['RSI'] = feature_data['rsi']
    input_features['MACD'] = feature_data['macd']
    input_features['Signal_Line'] = feature_data['signal_line']
    input_features['ATR'] = feature_data['atr']
    input_features['sentiment'] = feature_data['sentiment']
    input_features['Hour'] = feature_data['hour']
    input_features['DayOfWeek'] = feature_data['day_of_week']
    input_features['Volatility'] = feature_data['Volatility']
    
    # Scale
    scaled_data = scaler.transform(input_features)
    
    # Reshape for RNN (1, LOOKBACK, num_features)
    return np.array([scaled_data])

# =========================
# 4. ENDPOINTS
# =========================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: List[MarketData]):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded")
    
    try:
        X = preprocess_input(data)
        prob = float(model.predict(X)[0][0])
        pred = 1 if prob > 0.5 else 0
        
        return PredictionResponse(
            prediction=pred,
            probability=prob,
            model_used=MODEL_NAME,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
