from __future__ import annotations
import pandas as pd
from typing import Tuple


def add_indicators(
    df: pd.DataFrame,
    price_col: str = "price",
    sma_windows: Tuple[int, int] = (20, 50),
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    if df.empty or price_col not in df.columns:
        return df

    df = df.sort_values("timestamp").copy()
    p = df[price_col].astype(float)

    for w in sma_windows:
        df[f"sma_{w}"] = p.rolling(window=w, min_periods=w).mean()

    delta = p.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df[f"rsi_{rsi_period}"] = 100 - (100 / (1 + rs))

    ema_fast = p.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = p.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()
    hist = macd - signal

    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    return df

