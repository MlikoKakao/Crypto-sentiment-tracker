import pandas as pd
from src.domain.market.dto import IndicatorConfig


def calculate_sma(df: pd.DataFrame, state: IndicatorConfig) -> pd.DataFrame:
    result = df[["timestamp"]].copy()
    p = df["price"].astype(float)

    for w in (state.sma_fast, state.sma_slow):
        result[f"sma_{w}"] = p.rolling(window=w, min_periods=w).mean()

    return result


def calculate_rsi(df: pd.DataFrame, state: IndicatorConfig) -> pd.DataFrame:
    result = df[["timestamp"]].copy()
    p = df["price"].astype(float)

    signal_name = f"rsi_{state.rsi_period}"

    delta = p.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(state.rsi_period, min_periods=state.rsi_period).mean()

    avg_loss = loss.rolling(state.rsi_period, min_periods=state.rsi_period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result[signal_name] = 100 - (100 / (1 + rs))

    return result


def calculate_macd(df: pd.DataFrame, state: IndicatorConfig) -> pd.DataFrame:
    result = df[["timestamp"]].copy()
    p = df["price"].astype(float)

    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    ema_fast = p.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = p.ewm(span=macd_slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    result["macd"] = macd
    result["macd_signal"] = signal
    result["macd_hist"] = macd - signal

    return result
