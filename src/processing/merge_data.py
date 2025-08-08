import pandas as pd
from src.utils.helpers import load_csv, save_csv
from src.utils.cache import load_cached_csv, cache_csv

def merge_sentiment_and_price(sentiment_file, price_file, output_file, cache_settings):
    merged_cached = load_cached_csv(    
        cache_settings, parse_dates=["timestamp"], freshness_minutes=30
    )
    if merged_cached is not None:
        save_csv(merged_cached, output_file)
        print("Loaded merged data from cache:", output_file)
        print(merged_cached.head())
        return

    

    sentiment_df = load_csv(sentiment_file)
    price_df = load_csv(price_file)


    # Convert timestamp columns safely (handles ISO strings with +00:00)
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], errors="coerce")
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], errors="coerce")

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
    cache_csv(merged, cache_settings)
    save_csv(merged, output_file)
    print("✅ Merged data saved:", output_file)
    print(merged.head())