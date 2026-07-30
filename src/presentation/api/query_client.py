import os

import pandas as pd
import requests

from src.presentation.api.contracts.requests import (
    PostQuery,
    PriceQuery,
    SentimentQuery,
    SignalQuery,
)

QueryParams = dict[str, str] | list[tuple[str, str | int]]


def _get_table(path: str, params: QueryParams) -> pd.DataFrame:
    base_url = os.getenv("QUERY_API_URL", "http://localhost:8080").rstrip("/")
    response = requests.get(
        f"{base_url}/{path}",
        params=params,
        timeout=10,
    )
    response.raise_for_status()

    df = pd.DataFrame(response.json())
    df.rename(
        columns={
            "sourceId": "source_id",
            "contentHash": "content_hash",
            "signalName": "signal_name",
        },
        inplace=True,
    )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_prices(query: PriceQuery) -> pd.DataFrame:
    return _get_table("prices", query.to_params())


def get_posts(query: PostQuery) -> pd.DataFrame:
    return _get_table("posts", query.to_params())


def get_sentiment(query: SentimentQuery) -> pd.DataFrame:
    return _get_table("sentiment", query.to_params())


def get_signals(query: SignalQuery) -> pd.DataFrame:
    return _get_table("signals", query.to_params())
