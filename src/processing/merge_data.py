import pandas as pd

def merge_sentiment_and_price_df(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:

    if sentiment_df.empty or price_df.empty:
        return pd.DataFrame(columns=["timestamp", "price", "text", "sentiment", "source"])

    sentiment_df = sentiment_df.copy()
    price_df = price_df.copy()

    for df in (sentiment_df, price_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc = True, errors = "coerce").dt.tz_convert(None)

    sentiment_df = sentiment_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    price_df     = price_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if sentiment_df.empty or price_df.empty:
        return pd.DataFrame(columns=["timestamp", "price", "text", "sentiment", "source"])

    # dynamic tolerance
    span = sentiment_df["timestamp"].max() - sentiment_df["timestamp"].min()
    tol  = pd.Timedelta("30min") if span <= pd.Timedelta("2D") else (pd.Timedelta("12H") if span <= pd.Timedelta("14D") else pd.Timedelta("1D"))

    merged = pd.merge_asof(
        price_df[["timestamp","price"]],
        sentiment_df,
        on="timestamp",
        direction="backward",
        tolerance=tol,
    )
    return merged