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
ticker = "AAPL"
api_key = "f8c1ec12f9794e959cb331f42ca3e010"

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================
# 2. STOCK DATA (HOURLY)
# =========================
print(f"Fetching hourly stock data for {ticker}...")
# Hourly data is limited to last 730 days (2 years) by yfinance
stock = yf.Ticker(ticker).history(period="2y", interval="1h")
stock = stock.reset_index()

# Rename column if necessary (yfinance usually returns 'Datetime' for hourly)
if 'Datetime' in stock.columns:
    stock.rename(columns={"Datetime": "Timestamp"}, inplace=True)
else:
    stock.rename(columns={"Date": "Timestamp"}, inplace=True)

# Add Ticker and Date helper column for merging
stock["Ticker"] = ticker
stock["Date_Only"] = pd.to_datetime(stock["Timestamp"]).dt.date

# =========================
# 3. TECHNICAL INDICATORS
# =========================
print("Calculating technical indicators...")
stock['SMA_20'] = stock['Close'].rolling(window=20).mean()
stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
stock['RSI'] = calculate_rsi(stock['Close'])

# =========================
# 4. NEWS DATA (DAILY)
# =========================
print(f"Fetching recent news for {ticker}...")
url = (
    "https://newsapi.org/v2/everything?"
    f"q={ticker} stock OR finance&"
    "language=en&"
    "sortBy=publishedAt&"
    f"apiKey={api_key}"
)

try:
    response = requests.get(url)
    articles = response.json().get("articles", [])
    news_data = []

    for article in articles:
        if article.get("title") and article.get("publishedAt"):
            date = pd.to_datetime(article["publishedAt"]).date()
            sentiment = classifier(article["title"])[0]
            score = 1 if sentiment["label"] == "POSITIVE" else -1
            news_data.append([date, score])

    news_df = pd.DataFrame(news_data, columns=["Date_Only", "sentiment"])
    if not news_df.empty:
        news_df = news_df.groupby("Date_Only").mean().reset_index()
except Exception as e:
    print(f"News fetch failed: {e}")
    news_df = pd.DataFrame(columns=["Date_Only", "sentiment"])

# =========================
# 5. MERGE & CLEANUP
# =========================
print("Merging data...")
merged = stock.merge(news_df, on="Date_Only", how="left")
merged["sentiment"] = merged["sentiment"].fillna(0)

# Drop rows with NaN indicators (first 50 rows usually)
merged.dropna(subset=['SMA_50', 'RSI'], inplace=True)

# Drop helper column
merged.drop(columns=["Date_Only"], inplace=True)

# =========================
# 6. SAVE FINAL DATASET
# =========================
merged.to_csv("final_dataset.csv", index=False)

print(f"\nSUCCESS: Dataset created with {len(merged)} rows.")
print(f"Columns: {list(merged.columns)}")
print(merged.head())