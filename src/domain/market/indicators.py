from __future__ import annotations
import pandas as pd
from src.presentation.sidebar import IndicatorConfig
from src.infra.storage.db.signal_repository import (
    save_signal_df,
    load_signal_df,
    has_signal_coverage,
)


def add_indicators_to_df(
    df: pd.DataFrame,
    state: IndicatorConfig,
) -> pd.DataFrame:
    price_col: str = "price"
    sma_windows = (state.sma_fast, state.sma_slow)
    rsi_period = state.rsi_period
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    if df.empty or price_col not in df.columns:
        return df

    df = df.sort_values("timestamp").copy()
    p = df[price_col].astype(float)

    if state.use_sma:
        for w in sma_windows:
            signal_name = f"sma_{w}"
            cached = load_signal_df(state, signal_name)

            if has_signal_coverage(state, cached):
                df = df.merge(cached, on="timestamp", how="left")
            else:
                df[signal_name] = p.rolling(window=w, min_periods=w).mean()
                save_signal_df(df, signal_name, state.coin)

    if state.use_rsi:
        signal_name = f"rsi_{rsi_period}"
        cached = load_signal_df(state, signal_name)

        if has_signal_coverage(state, cached):
            df = df.merge(cached, on="timestamp", how="left")
        else:
            delta = p.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(rsi_period, min_periods=rsi_period).mean()
            avg_loss = loss.rolling(rsi_period, min_periods=rsi_period).mean()
            rs = avg_gain / avg_loss.replace(0, pd.NA)

            df[signal_name] = 100 - (100 / (1 + rs))
            save_signal_df(df, signal_name, state.coin)

    if state.use_macd:
        macd_signals = ("macd", "macd_signal", "macd_hist")
        cached_signals = {
            signal_name: load_signal_df(state, signal_name)
            for signal_name in macd_signals
        }

        if all(has_signal_coverage(state, cached) for cached in cached_signals.values()):
            for cached in cached_signals.values():
                df = df.merge(cached, on="timestamp", how="left")
        else:
            ema_fast = p.ewm(span=macd_fast, adjust=False).mean()
            ema_slow = p.ewm(span=macd_slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=macd_signal, adjust=False).mean()
            hist = macd - signal

            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = hist

            save_signal_df(df, "macd", state.coin)
            save_signal_df(df, "macd_signal", state.coin)
            save_signal_df(df, "macd_hist", state.coin)

    return df
