import pandas as pd 
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def apply_loess(df: pd.DataFrame, x_col: str, y_col: str, frac: float = 0.1) -> pd.DataFrame:
    if df.empty:
        return df
    
    x_vals = df[x_col]
    if np.issubdtype(x_vals.dtype,np.datetime64):
        x_vals = x_vals.astype(np.int64)

    loess_smoothed = lowess(df[y_col], x_vals, frac=frac, return_sorted=False)
    df[f"{y_col}_loess"] = loess_smoothed
    return df