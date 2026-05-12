import pandas as pd
import numpy as np

def check_balance():
    df = pd.read_csv("dataset/final_dataset.csv")
    # Simulate the preprocessing to get y
    # ... Simplified
    y = (df['Close'].pct_change() > 0).astype(int).dropna()
    print(f"Total samples: {len(y)}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")

if __name__ == "__main__":
    check_balance()
