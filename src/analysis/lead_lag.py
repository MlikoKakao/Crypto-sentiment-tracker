import numpy as np
import pandas as pd
from src.utils.cache import load_cached_csv, cache_csv
from src.utils.helpers import file_sha1

def compute_lead_lag(merged_df: pd.DataFrame,
                     lag_seconds:list[int],
                     metric: str = "pearson") -> pd.DataFrame:
    df = merged_df[["timestamp", "price", "sentiment"]].dropna().sort_values("timestamp")
    df = df.set_index("timestamp")

    out = []
    for lag in lag_seconds:
        s = df["sentiment"].copy()
        p = df["price"].copy()
        #Shift sentiment in time(lag, either pos or neg)
        s_lag = s.copy()
        s_lag.index = s_lag.index + pd.to_timedelta(lag, unit="s")
        aligned = (
            pd.concat([p, s_lag], axis=1, join="inner")
            .rename(columns={"sentiment": "sentiment_lag"})
            .dropna()
        )
        if len(aligned) < 5:
            out.append((lag, np.nan, np.nan, len(aligned)))
            continue
        if metric == "pearson":
            r = aligned["price"].corr(aligned["sentiment_lag"])
            out.append((lag, r, np.nan, len(aligned)))
        else:
            out.append((lag, np.nan,np.nan,len(aligned)))
    return pd.DataFrame(out, columns=["lag_seconds","r","p_value","n"])

def load_or_build_features(settings: dict, merged_path: str) -> pd.DataFrame:
    df_cached = load_cached_csv(settings, parse_dates=None, freshness_minutes=None)
    if df_cached is not None: #Check if cached
        return df_cached
    
    #Build if not
    merged_df = pd.read_csv(merged_path, parse_dates=["timestamp"])
    lag_seconds = list(range(settings["lag_min_s"], settings["lag_max_s"]+1,settings["lag_step_s"]))
    feats = compute_lead_lag(merged_df, lag_seconds, metric=settings["metric"])
    
    #Cache
    cache_csv(feats,settings)
    return feats