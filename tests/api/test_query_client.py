from datetime import datetime, timezone
from unittest.mock import Mock

import pandas as pd

from src.presentation.api import query_client
from src.presentation.api.contracts.requests import SentimentQuery, SignalQuery


def test_get_sentiment_serializes_repeated_sources(monkeypatch) -> None:
    response = Mock()
    response.json.return_value = [
        {
            "sourceId": "post-1",
            "contentHash": "hash-1",
            "timestamp": "2026-07-30T00:00:00Z",
            "sentiment": 0.5,
        }
    ]
    monkeypatch.setattr(query_client.requests, "get", Mock(return_value=response))

    query = SentimentQuery(
        coin="BTC",
        start_date=datetime(2026, 7, 29, tzinfo=timezone.utc),
        end_date=datetime(2026, 7, 30, tzinfo=timezone.utc),
        sources=("reddit", "news"),
        analyzer="vader",
        limit=100,
    )

    result = query_client.get_sentiment(query)

    request_params = query_client.requests.get.call_args.kwargs["params"]
    assert ("source", "reddit") in request_params
    assert ("source", "news") in request_params
    assert list(result.columns) == [
        "source_id",
        "content_hash",
        "timestamp",
        "sentiment",
    ]
    assert isinstance(result.loc[0, "timestamp"], pd.Timestamp)
    response.raise_for_status.assert_called_once()


def test_get_signals_uses_dotnet_parameter_names(monkeypatch) -> None:
    response = Mock()
    response.json.return_value = []
    monkeypatch.setattr(query_client.requests, "get", Mock(return_value=response))

    query = SignalQuery(
        coin="ETH",
        start_date=datetime(2026, 7, 29, tzinfo=timezone.utc),
        end_date=datetime(2026, 7, 30, tzinfo=timezone.utc),
        signal_names=("sma_20", "rsi"),
        num_signals=25,
    )

    query_client.get_signals(query)

    request_params = query_client.requests.get.call_args.kwargs["params"]
    assert ("signalName", "sma_20") in request_params
    assert ("signalName", "rsi") in request_params
    assert ("numSignals", 25) in request_params
