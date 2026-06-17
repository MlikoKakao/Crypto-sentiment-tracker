from datetime import datetime

import pandas as pd
from pydantic import ValidationError
import pytest
from fastapi.encoders import jsonable_encoder

from src.app.dto import AnalysisConfig
from src.domain.sentiment import registry
from src.presentation.api.helpers.validate import DateRangeParams
from src.presentation.api.routes import sentiment
from src.presentation.api.schemas.sentiment import SentimentResponse


def test_sentiment_endpoint_success_returns_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_sentiment_df(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
        assert config.coin == "BTC"
        assert analyzer == "vader"
        return pd.DataFrame(
            {
                "coin": ["BTC"],
                "source": ["reddit"],
                "content_hash": ["abc123"],
                "timestamp": [pd.Timestamp("2026-05-30 00:00:00")],
                "text": ["test row"],
                "url": [None],
                "analyzer": [analyzer],
                "sentiment": [0.5],
            }
        )

    monkeypatch.setattr(sentiment, "load_sentiment_df", fake_load_sentiment_df)

    result = sentiment.get_sentiment(
        params=DateRangeParams(
            coin="BTC", start_date=datetime(2026, 5, 30), end_date=datetime(2026, 6, 1)
        ),
        sources=["reddit"],
        analyzer="vader",
    )

    assert jsonable_encoder(result) == [
        {
            "coin": "BTC",
            "source": "reddit",
            "source_id": None,
            "content_hash": "abc123",
            "timestamp": "2026-05-30T00:00:00",
            "text": "test row",
            "url": None,
            "analyzer": "vader",
            "sentiment": 0.5,
        }
    ]


def test_sentiment_endpoint_invalid_coin_returns_400() -> None:
    with pytest.raises(ValidationError):
        sentiment.get_sentiment(
            params=DateRangeParams(
                coin="DOGE",
                start_date=datetime(2026, 5, 30),
                end_date=datetime(2026, 6, 1),
            ),
            sources=["reddit"],
        )


def test_sentiment_endpoint_analyzer_all_combines_analyzers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_sentiment_df(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coin": ["BTC"],
                "timestamp": [pd.Timestamp("2026-05-30 00:00:00")],
                "text": [f"{analyzer} row"],
                "source": ["reddit"],
                "analyzer": [analyzer],
                "sentiment": [0.5],
                "content_hash": ["abc123"],
                "url": [None],
            }
        )

    monkeypatch.setattr(registry, "ALL_ANALYZER_NAMES", ("vader", "textblob"))
    monkeypatch.setattr(sentiment, "load_sentiment_df", fake_load_sentiment_df)

    result: list[SentimentResponse] = sentiment.get_sentiment(
        params=DateRangeParams(
            coin="BTC", start_date=datetime(2026, 5, 30), end_date=datetime(2026, 6, 1)
        ),
        sources=["reddit"],
        analyzer="all",
    )

    assert [row.analyzer for row in result] == ["vader", "textblob"]
