import pandas as pd
import numpy as np
import os
import pickle
import mlflow
import mlflow.keras

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    SimpleRNN,
    Dropout,
    Bidirectional
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# =========================
# 1. SETTINGS
# =========================

DATA_PATH = "dataset/final_dataset.csv"

LOOKBACK = 24

FEATURES = [
    'Close',
    'Volume',
    'RSI',
    'MACD',
    'Signal_Line',
    'ATR',
    'sentiment',
    'Hour',
    'DayOfWeek'
]

# =========================
# 2. PREPROCESSING
# =========================

def prepare_data(df):

    print("Preprocessing data...")

    df_proc = df.copy()

    # -------------------------
    # TARGET LABEL
    # Predict significant 6-hour movement
    # -------------------------

    future_return = (
        df_proc['Close'].shift(-6) - df_proc['Close']
    ) / df_proc['Close']

    df_proc['Target'] = (future_return > 0.003).astype(int)

    # -------------------------
    # STATIONARY FEATURES
    # -------------------------

    for col in ['Close', 'Volume']:
        df_proc[col] = np.log(
            (df_proc[col] + 1e-6) /
            (df_proc[col].shift(1) + 1e-6)
        )

    # Additional volatility feature
    df_proc['Volatility'] = (
        df_proc['Close']
        .rolling(window=10)
        .std()
    )

    df_proc = df_proc.dropna().reset_index(drop=True)

    FEATURES_EXTENDED = FEATURES + ['Volatility']

    # -------------------------
    # TRAIN TEST SPLIT FIRST
    # -------------------------

    split_idx = int(len(df_proc) * 0.8)

    train_df = df_proc.iloc[:split_idx]
    test_df = df_proc.iloc[split_idx:]

    # -------------------------
    # FIT SCALER ONLY ON TRAIN
    # -------------------------

    scaler = RobustScaler()

    scaler.fit(train_df[FEATURES_EXTENDED])

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # -------------------------
    # TRANSFORM
    # -------------------------

    train_scaled = scaler.transform(
        train_df[FEATURES_EXTENDED]
    )

    test_scaled = scaler.transform(
        test_df[FEATURES_EXTENDED]
    )

    # -------------------------
    # CREATE SEQUENCES
    # -------------------------

    X_train, y_train = [], []
    X_test, y_test = [], []

    # Train sequences
    for i in range(LOOKBACK, len(train_scaled)):

        X_train.append(
            train_scaled[i - LOOKBACK:i]
        )

        y_train.append(
            train_df.iloc[i]['Target']
        )

    # Test sequences
    for i in range(LOOKBACK, len(test_scaled)):

        X_test.append(
            test_scaled[i - LOOKBACK:i]
        )

        y_test.append(
            test_df.iloc[i]['Target']
        )

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test)
    )

# =========================
# 3. MODEL BUILDERS
# =========================

def build_rnn(input_shape):

    model = Sequential([

        Bidirectional(
            SimpleRNN(
                32,
                activation='tanh',
                return_sequences=True,
                kernel_regularizer=l2(0.001)
            ),
            input_shape=input_shape
        ),

        Dropout(0.4),

        Bidirectional(
            SimpleRNN(
                16,
                activation='tanh',
                kernel_regularizer=l2(0.001)
            )
        ),

        Dropout(0.4),

        Dense(16, activation='relu'),

        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# -------------------------

def build_lstm(input_shape):

    model = Sequential([

        Bidirectional(
            LSTM(
                64,
                activation='tanh',
                return_sequences=True,
                kernel_regularizer=l2(0.001)
            ),
            input_shape=input_shape
        ),

        Dropout(0.4),

        Bidirectional(
            LSTM(
                32,
                activation='tanh',
                kernel_regularizer=l2(0.001)
            )
        ),

        Dropout(0.4),

        Dense(16, activation='relu'),

        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# -------------------------

def build_gru(input_shape):

    model = Sequential([

        Bidirectional(
            GRU(
                64,
                activation='tanh',
                return_sequences=True,
                kernel_regularizer=l2(0.001)
            ),
            input_shape=input_shape
        ),

        Dropout(0.4),

        Bidirectional(
            GRU(
                32,
                activation='tanh',
                kernel_regularizer=l2(0.001)
            )
        ),

        Dropout(0.4),

        Dense(16, activation='relu'),

        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================
# 4. TRAINING LOOP
# =========================

def train_and_log(
    model_name,
    model_builder,
    X_train,
    y_train,
    X_test,
    y_test
):

    with mlflow.start_run(run_name=model_name):

        print(f"\nTraining {model_name}...")

        model = model_builder(
            (X_train.shape[1], X_train.shape[2])
        )

        # -------------------------
        # LOG PARAMETERS
        # -------------------------

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("lookback", LOOKBACK)
        mlflow.log_param("features", FEATURES)

        # -------------------------
        # CALLBACKS
        # -------------------------

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # -------------------------
        # CLASS WEIGHTS
        # -------------------------

        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        dict_weights = dict(enumerate(weights))

        # -------------------------
        # TRAIN
        # -------------------------

        history = model.fit(

            X_train,
            y_train,

            epochs=50,

            batch_size=64,

            validation_data=(X_test, y_test),

            callbacks=[
                early_stop,
                reduce_lr
            ],

            class_weight=dict_weights,

            shuffle=False,

            verbose=1
        )

        # -------------------------
        # PREDICTIONS
        # -------------------------

        predictions = (
            model.predict(X_test, verbose=0) > 0.5
        ).astype(int)

        # -------------------------
        # METRICS
        # -------------------------

        acc = accuracy_score(y_test, predictions)

        f1 = f1_score(y_test, predictions)

        mlflow.log_metric(
            "directional_accuracy",
            acc
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        print(
            f"{model_name} -> "
            f"Accuracy: {acc:.2%}, "
            f"F1: {f1:.4f}"
        )

        # -------------------------
        # SAVE MODEL
        # -------------------------

        mlflow.keras.log_model(
            model,
            f"model_{model_name.lower()}"
        )

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    if not os.path.exists(DATA_PATH):

        print(
            f"Error: {DATA_PATH} not found."
        )

        exit()

    # -------------------------
    # LOAD DATA
    # -------------------------

    df = pd.read_csv(DATA_PATH)

    # -------------------------
    # PREPARE DATA
    # -------------------------

    X_train, y_train, X_test, y_test = prepare_data(df)

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # -------------------------
    # MLFLOW
    # -------------------------

    mlflow.set_experiment(
        "Market_Prediction_V3"
    )

    # -------------------------
    # MODELS
    # -------------------------

    models_to_train = [

        ("GRU", build_gru),

        ("LSTM", build_lstm),

        ("SimpleRNN", build_rnn)
    ]

    # -------------------------
    # TRAIN
    # -------------------------

    for name, builder in models_to_train:

        try:

            train_and_log(
                name,
                builder,
                X_train,
                y_train,
                X_test,
                y_test
            )

        except Exception as e:

            print(
                f"Error training {name}: {e}"
            )

    print("\nSUCCESS: Training Complete")
    print("Run: mlflow ui")