import numpy as np
import pandas as pd
from src.utils.cache import load_cached_csv, cache_csv
from src.utils.helpers import file_sha1
from scipy.stats import pearsonr

def compute_lead_lag(merged_df: pd.DataFrame,
                     lag_seconds:list[int],
                     metric: str = "pearson",
                     resample: str = "5min",
                     min_points: int = 50) -> pd.DataFrame:
    df = (merged_df[["timestamp", "price", "sentiment"]].dropna().sort_values("timestamp"))
    if df.empty:
        return pd.DataFrame(columns=["lag_seconds","r","p_value","n"])
    
    g = (df.set_index("timestamp")
         .resample(resample).mean()
         .interpolate("time"))

    s = g["sentiment"]
    p = g["price"]
    step = pd.to_timedelta(resample).total_seconds()

    out = []
    for lag in lag_seconds:
        k = int(round(lag/step))
        p_shift = p.shift(-k)
        valid = pd.concat([s, p_shift], axis = 1).dropna()
        n = len(valid)
        if n < min_points:
            out.append([lag, np.nan, np.nan, n])
            continue
        if metric == "spearman":
            r = float(valid.corr(method="spearman").iloc[0,1])
            pval = np.nan
        else:
            r = float(valid.corr().iloc[0,1])
            try:
                _,pval = pearsonr(valid.iloc[:,0].values, valid.iloc[:,1].values)
            except Exception:
                pval = np.nan
        out.append([lag,r,float(pval),int(n)])
        return pd.DataFrame(out,columns=["lag_seconds","r","p_value","n"])



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