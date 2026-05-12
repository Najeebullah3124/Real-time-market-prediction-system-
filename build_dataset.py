import yfinance as yf
import requests
import pandas as pd
import numpy as np
from transformers import pipeline

# =========================
# 1. SENTIMENT MODEL SETUP
# =========================
print("Loading sentiment model...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# =========================
# SETTINGS
# =========================
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "INTC",
    "GOOG", "BABA", "V", "MA", "DIS", "ADBE", "CRM", "ORCL", "CMCSA", "PEP", 
    "KO", "NKE", "XOM", "CVX", "JPM", "BAC", "WMT", "COST", "T", "VZ"
]
api_key = "f8c1ec12f9794e959cb331f42ca3e010"

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    print("Calculating advanced indicators (MACD, Bollinger Bands)...")
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['20STD'] * 2)
    
    # Moving Averages (Already have 20, 50, let's keep them)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    # ROC (Rate of Change)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Bollinger %B
    df['Bollinger_B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    
    # ATR (Average True Range) approximation
    df['TR'] = df['High'] - df['Low']
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Drop TR as it's just for ATR
    df.drop('TR', axis=1, inplace=True)
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # ADD LAGGED FEATURES (Increased to 10 for more historical context)
    for i in range(1, 11):
        df[f'Lag_{i}'] = df['Close'].pct_change().shift(i)
    
    return df

# =========================
# 2-5. DATA COLLECTION LOOP
# =========================
all_data = []

for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    
    # Stock Data
    stock = yf.Ticker(ticker).history(period="2y", interval="1h")
    stock = stock.reset_index()
    if 'Datetime' in stock.columns:
        stock.rename(columns={"Datetime": "Timestamp"}, inplace=True)
    else:
        stock.rename(columns={"Date": "Timestamp"}, inplace=True)
    stock["Ticker"] = ticker
    stock["Date_Only"] = pd.to_datetime(stock["Timestamp"]).dt.date
    stock["Hour"] = pd.to_datetime(stock["Timestamp"]).dt.hour
    stock["DayOfWeek"] = pd.to_datetime(stock["Timestamp"]).dt.dayofweek
    
    # News Data for this ticker
    print(f"Fetching news for {ticker}...")
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    articles = requests.get(url).json().get("articles", [])
    
    news_data = []
    for article in articles[:100]: # Increased news limit for better sentiment coverage
        if article.get("title") and article.get("publishedAt"):
            date = pd.to_datetime(article["publishedAt"]).date()
            sentiment = classifier(article["title"])[0]
            score = 1 if sentiment["label"] == "POSITIVE" else -1
            news_data.append([date, score])
    
    news_df = pd.DataFrame(news_data, columns=["Date_Only", "sentiment"])
    if not news_df.empty:
        news_df = news_df.groupby("Date_Only").mean().reset_index()
    else:
        news_df = pd.DataFrame(columns=["Date_Only", "sentiment"])

    # Merge
    merged = stock.merge(news_df, on='Date_Only', how='left').fillna(0)
    merged = add_indicators(merged)
    all_data.append(merged)

# Combine all tickers
final_df = pd.concat(all_data, ignore_index=True)
final_df = final_df.dropna().reset_index(drop=True)
final_df.drop(columns=["Date_Only"], inplace=True)

# =========================
# 6. SAVE FINAL DATASET
# =========================
final_df.to_csv("dataset/final_dataset.csv", index=False)
print(f"\nSUCCESS: Dataset created with {len(final_df)} rows and {len(tickers)} tickers.")
print(f"Columns: {list(final_df.columns)}")
print(final_df.head())