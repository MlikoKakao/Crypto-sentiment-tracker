import pandas as pd
from src.utils.helpers import load_csv, save_csv

def merge_sentiment_and_price(sentiment_file, price_file, output_file):
    # Load raw data
    sentiment_df = load_csv(sentiment_file)
    price_df = load_csv(price_file)

    # Convert timestamp columns safely (handles ISO strings with +00:00)
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], format="ISO8601", errors="coerce")
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], format="ISO8601", errors="coerce")

    # Drop any rows with invalid timestamps
    sentiment_df = sentiment_df.dropna(subset=["timestamp"])
    price_df = price_df.dropna(subset=["timestamp"])

    # Remove timezone info
    sentiment_df["timestamp"] = sentiment_df["timestamp"].dt.tz_localize(None)
    price_df["timestamp"] = price_df["timestamp"].dt.tz_localize(None)

    # Sort both DataFrames by timestamp BEFORE merge_asof
    sentiment_df = sentiment_df.sort_values("timestamp")
    price_df = price_df.sort_values("timestamp")

    # Merge based on nearest timestamp
    merged = pd.merge_asof(
        sentiment_df,
        price_df,
        on="timestamp",
        direction="nearest"
    )

    # Save to file
    save_csv(merged, output_file)
    print("✅ Merged data saved:", output_file)
    print(merged.head())