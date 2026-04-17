from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr  # type: ignore
from src.shared.helpers import normalize_timestamp_column



LEAD_LAG_COLUMNS = ["lag_seconds", "r", "p_value", "n"]

def compute_lead_lag(
    merged_df: pd.DataFrame,
    lag_hours: int = 24,
    lag_step_min: int = 30,
    metric: str = "pearson",
    resample: str = "5min",
    min_points: int = 50,
) -> pd.DataFrame:
    price_sentiment_df = (
        merged_df[["timestamp", "price", "sentiment"]]
        .dropna()
        .sort_values(by=["timestamp"])
    )
    price_sentiment_df = normalize_timestamp_column(
        price_sentiment_df,
        drop_invalid=True,
    )
    if price_sentiment_df.empty:
        return pd.DataFrame(columns=LEAD_LAG_COLUMNS)

    lag_seconds = list(
        range(
            -lag_hours * 3600,
            lag_hours * 3600 + 1,
            lag_step_min * 60,
        )
    )

    resampled_df = (
        price_sentiment_df
        .set_index("timestamp")
        .resample(resample)
        .mean()
        .interpolate("time")
    )

    sentiment_series = resampled_df["sentiment"]
    price_series = resampled_df["price"]
    seconds_per_row = pd.to_timedelta(resample).total_seconds()

    rows = []
    for current_lag_seconds in lag_seconds:
        lag_rows = int(round(current_lag_seconds / seconds_per_row))
        shifted_price_series = price_series.shift(-lag_rows)

        valid_pairs = pd.concat([sentiment_series, shifted_price_series], axis=1).dropna()
        valid_pair_count = len(valid_pairs)

        if valid_pair_count < min_points:
            rows.append([current_lag_seconds, np.nan, np.nan, valid_pair_count])
            continue

        valid_sentiment = valid_pairs.iloc[:, 0]
        valid_price = valid_pairs.iloc[:, 1]

        if metric.lower() == "spearman":
            correlation = float(valid_sentiment.corr(valid_price, method="spearman"))
            p_value = np.nan
        else:
            correlation = float(valid_sentiment.corr(valid_price))
            try:
                _, p_value = cast(
                    tuple[float,float],
                    pearsonr(
                        valid_sentiment.to_numpy(),
                        valid_price.to_numpy(),
                    ),
                )
                p_value = float(p_value)
            except Exception:
                p_value = float("nan")

        rows.append([current_lag_seconds, correlation, p_value, valid_pair_count])

    return pd.DataFrame(rows, columns=LEAD_LAG_COLUMNS)
