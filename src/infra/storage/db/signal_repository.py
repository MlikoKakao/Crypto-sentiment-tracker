from datetime import timedelta
from typing import Any, cast

import pandas as pd
from sqlalchemy import text

from src.domain.market.dto import IndicatorConfig
from src.infra.storage.db.connection import get_engine
from src.shared.dataframe_schema import require_columns
from src.shared.helpers import normalize_timestamp_column


def save_signal_df(
    signal_df: pd.DataFrame,
    signal: str,
    coin: str = "btc",
) -> None:
    require_columns(signal_df, {"timestamp", signal}, "signal_df")

    df = signal_df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["coin"] = coin.upper()
    df["signal_name"] = signal
    df["value"] = df[signal]
    df = df.dropna(subset=["value"])

    rows = cast(
        list[dict[str, Any]],
        df[["coin", "timestamp", "signal_name", "value"]].to_dict(orient="records"),
    )

    if not rows:
        return

    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO signals (
                    coin,
                    timestamp,
                    signal_name,
                    value
                )
                VALUES (
                    :coin,
                    :timestamp,
                    :signal_name,
                    :value
                )
                ON CONFLICT (coin, timestamp, signal_name)
                DO UPDATE SET value = EXCLUDED.value
                """
            ),
            rows,
        )


def load_signal_df(state: IndicatorConfig, signal: str) -> pd.DataFrame:
    engine = get_engine()

    query = text(
        """
        SELECT timestamp, value
        FROM signals
        WHERE coin = :coin
          AND signal_name = :signal_name
          AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY timestamp ASC
        """
    )

    with engine.connect() as conn:
        params: dict[str, Any] = {
            "coin": state.coin.upper(),
            "signal_name": signal,
            "start_date": state.start_date,
            "end_date": state.end_date,
        }
        df = pd.read_sql_query(
            query,
            conn,
            params=params,
        )

    if df.empty:
        return pd.DataFrame(columns=["timestamp", signal])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"value": signal})

    return df[["timestamp", signal]]


def has_signal_coverage(state: IndicatorConfig, signal_df: pd.DataFrame) -> bool:
    if signal_df.empty:
        return False

    tolerance = timedelta(hours=1)
    min_time = signal_df["timestamp"].min()
    max_time = signal_df["timestamp"].max()

    start_date = pd.to_datetime(state.start_date, utc=True).tz_convert(None)
    end_date = pd.to_datetime(state.end_date, utc=True).tz_convert(None)

    starts_near = min_time <= start_date + tolerance
    ends_near = max_time >= end_date - tolerance

    return starts_near and ends_near
