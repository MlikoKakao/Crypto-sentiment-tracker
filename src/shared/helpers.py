import pandas as pd
from datetime import datetime
from typing import cast

from src.app.dto import Coin

def normalize_timestamp_column(
    df: pd.DataFrame,
    column: str = "timestamp",
    drop_invalid: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Missing timestamp column: {column}")

    df[column] = pd.to_datetime(
        df[column],
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None)

    if drop_invalid:
        df = df.dropna(subset=[column])

    return df


def is_date_correct(start_date: datetime, end_date: datetime) -> bool:
    return end_date > start_date


# Text cleanup for sentiment analysis
def clean_text(text: str) -> str:
    return str(text).lower().strip()


SUPPORTED_COINS = {"BTC", "ETH", "XMR"}


def normalize_coin(coin: str) -> Coin:
    normalized = coin.upper()

    if normalized not in SUPPORTED_COINS:
        raise ValueError(f"Unsupported coin: {coin}")

    return cast(Coin, normalized)
