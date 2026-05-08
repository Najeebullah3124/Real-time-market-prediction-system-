import pandas as pd
import numpy as np
import os
import pickle
import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. SETTINGS
# =========================
DATA_PATH = "dataset/final_dataset.csv"
LOOKBACK = 60  # Use last 60 hours to predict next 1
FEATURES = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'sentiment']
TARGET = 'Close'

# =========================
# 2. PREPROCESSING
# =========================
def prepare_data(df):
    print("Preprocessing data...")
    # Select features
    data = df[FEATURES].values
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler for future predictions
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i])
        y.append(scaled_data[i, 0]) # Index 0 is 'Close'
        
    return np.array(X), np.array(y), scaler

# =========================
# 3. MODEL BUILDERS
# =========================
def build_rnn(input_shape):
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# =========================
# 4. TRAINING LOOP
# =========================
def train_and_log(model_name, model_builder, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining {model_name}...")
        model = model_builder((X_train.shape[1], X_train.shape[2]))
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("lookback", LOOKBACK)
        mlflow.log_param("features", FEATURES)
        
        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Log metrics
        val_loss = history.history['val_loss'][-1]
        mlflow.log_metric("mse", val_loss)
        print(f"{model_name} Validation MSE: {val_loss:.6f}")
        
        # Save model
        mlflow.keras.log_model(model, f"model_{model_name.lower()}")

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run build_dataset.py first.")
        exit()
        
    df = pd.read_csv(DATA_PATH)
    X, y, scaler = prepare_data(df)
    
    # Split data (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Setup MLflow
    mlflow.set_experiment("Market_Prediction_V2")
    
    # Run experiments
    models_to_train = [
        ("SimpleRNN", build_rnn),
        ("LSTM", build_lstm),
        ("GRU", build_gru)
    ]
    
    for name, builder in models_to_train:
        train_and_log(name, builder, X_train, y_train, X_test, y_test)
        
    print("\nSUCCESS: All models trained and logged to MLflow.")
    print("Run 'mlflow ui' in your terminal to see the comparison.")
