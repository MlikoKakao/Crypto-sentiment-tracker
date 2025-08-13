import numpy as np
import pandas as pd

def compute_indicators(df: pd.DataFrame, ema_window: int = 20, med_window_bars: int = 96) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("timestamp")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")
    out = out.dropna(subset=["price","sentiment"])

    out["ema20"] = out["price"].ewm(span=ema_window, adjust=False).mean()
    out["sent_smooth"] = out["sentiment"].ewm(span=med_window_bars//2, adjust=False).mean()
    out["sent_med"] = out["sent_smooth"].rolling(med_window_bars, min_periods=med_window_bars//2).median()
    return out

def build_signal(df: pd.DataFrame) -> pd.Series:
    cond_sent = df["sent_smooth"] > df["sent_med"]
    cond_ema = df["price"] > df["ema20"]
    sig = (cond_sent & cond_ema).astype(int)

    return sig.shift(1).fillna(0).astype(int)

def backtest_long_only(df: pd.DataFrame,
                       cost_bps: float = 5.0,
                       slippage_bps: float = 5.0
                       ) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out["price"]).diff()
    pos = build_signal(out)
    out["pos"] = pos

    trades = pos.diff().abs().fillna(0)

    cost = (cost_bps+slippage_bps) / 10000.0
    out["ret_net"] = pos * out["ret"] - trades * cost

    out["eq_strategy"] = out["ret_net"].cumsum().pipe(np.exp)
    out["eq_hodl"] = out["ret"].fillna(0).cumsum().pipe(np.exp)
    peak = out["eq_strategy"].cummax()
    out["dd"] = out["eq_strategy"]/peak -1.0
    return out

def summarize(df_bt: pd.DataFrame, bars_per_year: int)-> dict:
    ret_series = df_bt["ret_net"].dropna()
    total_years = len(ret_series) / bars_per_year if bars_per_year else np.nan
    cagr = (df_bt["eq_strategy"].iloc[-1])**(1/max(total_years, 1e-9)) - 1 if len(df_bt)>0 else np.nan
    vol_ann = ret_series.std() * np.sqrt(bars_per_year) if bars_per_year else np.nan
    sharpe = (ret_series.mean()*bars_per_year) / vol_ann if vol_ann and vol_ann > 0 else np.nan
    max_dd = df_bt["dd"].min() if "dd" in df_bt else np.nan
    hit = (ret_series > 0).mean()

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "HitRate": hit}

def run_backtest(df_merged: pd.DataFrame,
                 cost_bps: float,
                 slippage_bps: float,
                 resample: str = "5min"):
    df5 = (df_merged.set_index("timestamp")
           .resample(resample).ffill()
           .reset_index())
    df_ind = compute_indicators(df5, ema_window=20, med_window_bars=96)
    bt = backtest_long_only(df_ind, cost_bps=cost_bps, slippage_bps=slippage_bps)

    dt = df5["timestamp"].diff().median()

    if pd.isna(dt) or getattr(dt, "total_seconds", lambda: 0)() <= 0:
        bars_per_year = 52560
    else:
        bars_per_year = int(round((365*24*3600) / dt.total_seconds()))
    
    stats = summarize(bt, bars_per_year=bars_per_year)
    return bt,stats