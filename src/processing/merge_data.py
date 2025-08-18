import pandas as pd
from src.utils.helpers import load_csv, save_csv, _to_naive
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


    # Convert timestamp columns safely
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    price_df["timestamp"]     = pd.to_datetime(price_df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)

    sentiment_df = sentiment_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    price_df     = price_df.dropna(subset=["timestamp","price"]).sort_values("timestamp")

    # dynamic tolerance
    span = sentiment_df["timestamp"].max() - sentiment_df["timestamp"].min()
    tol  = pd.Timedelta("30min") if span <= pd.Timedelta("2D") else (pd.Timedelta("2H") if span <= pd.Timedelta("14D") else pd.Timedelta("1D"))

    merged = pd.merge_asof(
        sentiment_df,
        price_df[["timestamp","price"]],
        on="timestamp",
        direction="backward",
        tolerance=tol,
    )

    # Save to file
    cache_csv(merged, cache_settings)
    save_csv(merged, output_file)
    print("✅ Merged data saved:", output_file)
    print(merged.head())