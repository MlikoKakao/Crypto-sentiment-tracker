import pandas as pd



def build_signal_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()
    df["positive_sentiment"] = df["sentiment"] > 0.2
    df["negative_sentiment"] = df["sentiment"] < -0.2
    
    rolling_mean = df["sentiment"].rolling(20).mean()
    rolling_std = df["sentiment"].rolling(20).std()

    df["sentiment_spike"] = (
        (df["sentiment"] - rolling_mean).abs() >= 2 * rolling_std
        )
    
    df["sma_bullish_cross"] = (
        (df["sma_20"].shift(1) <= df["sma_50"].shift(1))
        & (df["sma_20"] > df["sma_50"])
    )
    
    df["price_change"] = df["price"].pct_change()
    df["sentiment_change"] = df["sentiment"].diff()

    df["bearish_divergence"] = (
        (df["price_change"] > 0)
        & (df["sentiment_change"] < 0)
    )

    df["bullish_divergence"] = (
        (df["price_change"] < 0)
         & (df["sentiment_change"] > 0)
    )
    
    
    
    
    return df