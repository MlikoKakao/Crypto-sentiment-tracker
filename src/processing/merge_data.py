import pandas as pd
from src.utils.helpers import load_csv, save_csv, _to_naive
from src.utils.cache import load_cached_csv, cache_csv
from src.processing.indicators import add_indicators
from config.settings import DEMO_MODE, get_demo_data_path

def merge_sentiment_and_price(sentiment_file, price_file, output_file, cache_settings):
    if DEMO_MODE:
        return pd.read_csv(get_demo_data_path("combined_sentiment.csv"), parse_dates=["timestamp"])
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

    for df in (sentiment_df, price_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc = True, errors = "coerce").dt.tz_convert(None)

    sentiment_df = sentiment_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    price_df     = price_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # dynamic tolerance
    span = sentiment_df["timestamp"].max() - sentiment_df["timestamp"].min()
    tol  = pd.Timedelta("30min") if span <= pd.Timedelta("2D") else (pd.Timedelta("12H") if span <= pd.Timedelta("14D") else pd.Timedelta("1D"))

    merged = pd.merge_asof(
        sentiment_df,
        price_df[["timestamp","price"]],
        on="timestamp",
        direction="backward",
        tolerance=tol,
    )
    merged = add_indicators(
        merged,
        price_col="price",
        sma_windows=(20,50),
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )



    # Save to file
    cache_csv(merged, cache_settings)
    save_csv(merged, output_file)
    print("✅ Merged data saved:", output_file)
    print(merged.head())
    return merged