import pandas as pd
from typing import Any, cast
from datetime import timedelta
from sqlalchemy import text

from src.infra.storage.db.connection import get_engine
from src.shared.dataframe_schema import REQUIRED_PRICE_COLUMNS, require_columns
from src.shared.helpers import normalize_timestamp_column
from src.app.dto import AnalysisConfig


def save_price_df(prices_df: pd.DataFrame, coin: str = "btc") -> None:
    require_columns(prices_df, REQUIRED_PRICE_COLUMNS, "prices_df")
    df = prices_df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["coin"] = coin.upper()

    rows = cast(
        list[dict[str, Any]],
        df[["coin", "timestamp", "price"]].to_dict(orient="records"),
    )

    if not rows:
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO prices (
                    coin,
                    timestamp,
                    price
                )
                VALUES (
                    :coin,
                    :timestamp,
                    :price
                )
                ON CONFLICT DO NOTHING
                    """
            ),
            rows,
        )


def load_price_df(config: AnalysisConfig) -> pd.DataFrame:
    engine = get_engine()
    query = text(
        """
        SELECT *
        FROM prices
        WHERE coin = :coin
            AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY timestamp DESC
        """
    )

    with engine.begin() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "coin": config.coin.upper(),
                "start_date": config.start_date,
                "end_date": config.end_date,
            },
        )

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def has_price_coverage(config: AnalysisConfig, price_df: pd.DataFrame) -> bool:
    if price_df.empty:
        return False

    tolerance = timedelta(hours=1)
    min_time = price_df["timestamp"].min()
    max_time = price_df["timestamp"].max()

    start_date = pd.to_datetime(config.start_date, utc=True).tz_convert(None)
    end_date = pd.to_datetime(config.end_date, utc=True).tz_convert(None)

    starts_near = min_time <= start_date + tolerance
    ends_near = max_time >= end_date - tolerance

    return starts_near and ends_near
