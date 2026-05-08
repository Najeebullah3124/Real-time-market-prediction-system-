import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN, Dense, Dropout
import os

# Settings
DATA_PATH = "dataset/final_dataset.csv"
LOOKBACK = 60
FEATURES = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'sentiment']

def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == "SimpleRNN":
        model.add(SimpleRNN(50, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50, activation='relu'))
    elif model_type == "LSTM":
        model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
    elif model_type == "GRU":
        model.add(GRU(50, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50, activation='relu'))
    
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_plots():
    if not os.path.exists(DATA_PATH):
        print("Dataset not found.")
        return

    # 1. Load and Preprocess
    df = pd.read_csv(DATA_PATH)
    data = df[FEATURES].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # 2. Train and Predict for each model
    model_types = ["SimpleRNN", "LSTM", "GRU"]
    results = {}

    for mt in model_types:
        print(f"Generating predictions for {mt}...")
        model = build_model(mt, (X.shape[1], X.shape[2]))
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        predictions = model.predict(X_test)
        
        # Inverse Scaling
        dummy = np.zeros((len(predictions), len(FEATURES)))
        dummy[:, 0] = predictions.flatten()
        results[mt] = scaler.inverse_transform(dummy)[:, 0]

    # Inverse Scaling for Actual
    dummy = np.zeros((len(y_test), len(FEATURES)))
    dummy[:, 0] = y_test.flatten()
    inv_actual = scaler.inverse_transform(dummy)[:, 0]

    # 3. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    colors = ['orange', 'green', 'red']

    for i, mt in enumerate(model_types):
        axes[i].plot(inv_actual[-100:], label="Actual Price", color='blue', linewidth=2)
        axes[i].plot(results[mt][-100:], label=f"Predicted ({mt})", color=colors[i], linestyle='--', linewidth=2)
        axes[i].set_title(f"Model: {mt}", fontsize=14)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel("Price (USD)")

    plt.xlabel("Time (Last 100 Hours)", fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.abspath("all_models_plot.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    generate_plots()
