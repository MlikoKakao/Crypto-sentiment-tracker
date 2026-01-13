import numpy as np
import pandas as pd
import os
from src.utils.cache import load_cached_csv, cache_csv
from scipy.stats import pearsonr
from typing import Any

def _to_float(x: Any) -> float:
    """Safely convert pandas/numpy scalars and builtins to Python float.

    Raises TypeError for complex / non-convertible values so the type checker
    sees we've excluded complex before calling float(...).

    Come back to this later, this is a mess.
    """
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, complex):
        raise TypeError("complex cannot be converted to float")
    if isinstance(x, (int, float, bool)):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cannot convert {type(x)!r} to float") from exc

def compute_lead_lag(
    merged_df: pd.DataFrame,
    lag_seconds: list[int],
    metric: str = "pearson",
    resample: str = "5min",
    min_points: int = 50,
) -> pd.DataFrame:
    df = (
        merged_df[["timestamp", "price", "sentiment"]]
        .dropna()
        .sort_values(by=["timestamp"])
    )  # type: ignore
    if df.empty:
        return pd.DataFrame(columns=["lag_seconds", "r", "p_value", "n"])

    df_time_interpolated = df.set_index("timestamp").resample(resample).mean().interpolate("time")

    s = df_time_interpolated["sentiment"]
    p = df_time_interpolated["price"]
    step = pd.to_timedelta(resample).total_seconds()

    out = []
    for lag_seconds_value in lag_seconds:
        lag_steps = int(round(lag_seconds_value / step))
        shifted_price = p.shift(-lag_steps)
        valid = pd.concat([s, shifted_price], axis=1).dropna()
        n = len(valid)
        if n < min_points:
            out.append([lag_seconds_value, np.nan, np.nan, n])
            continue
        if metric.lower() == "spearman":
            r_val = valid.corr(method="spearman").iloc[0, 1]
            r = _to_float(r_val)
            pval = np.nan
        else:
            r_val = valid.corr().iloc[0, 1]
            r = _to_float(r_val)
            try:
                pearson_res = pearsonr(
                    valid.iloc[:, 0].to_numpy(), valid.iloc[:, 1].to_numpy()
                )
                pval = float(pearson_res[1])
            except Exception:
                pval = float("nan")

            # Ensure pval is a Python float for downstream consumption and typing
            try:
                if not isinstance(pval, float):
                    pval = _to_float(pval)
                else:
                    pval = float(pval)
            except TypeError:
                pval = float("nan")
        out.append([lag_seconds_value, r, pval, int(n)])
    return pd.DataFrame(out, columns=["lag_seconds", "r", "p_value", "n"])


def load_or_build_lead_lag_features(settings: dict[str, Any], merged_path: str) -> pd.DataFrame:
    if "depends_on" not in settings:
        from src.utils.helpers import file_sha1

        settings = dict(settings)
        settings["depends_on"] = file_sha1(merged_path)

    df_cached = load_cached_csv(settings, parse_dates=None, freshness_minutes=None)
    if df_cached is not None:  # Check if cached
        return df_cached

    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"merged_path not found {merged_path}")

    # Build if not
    merged_df = pd.read_csv(merged_path, parse_dates=["timestamp"])
    lag_seconds = list(
        range(settings["lag_min_s"], settings["lag_max_s"] + 1, settings["lag_step_s"])
    )
    feats = compute_lead_lag(merged_df, lag_seconds, metric=settings["metric"])

    # Cache
    cache_csv(feats, settings)
    return feats
