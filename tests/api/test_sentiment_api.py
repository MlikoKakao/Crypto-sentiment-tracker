from datetime import datetime

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from src.app.dto import AnalysisConfig
from src.domain.sentiment import registry
from src.presentation.api.routes import sentiment


def test_sentiment_endpoint_success_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_sentiment_df(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
        assert config.coin == "BTC"
        assert analyzer == "vader"
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-05-30 00:00:00")],
                "text": ["BTC looks strong"],
                "source": ["reddit"],
                "analyzer": ["vader"],
                "sentiment": [0.8],
            }
        )

    monkeypatch.setattr(sentiment, "load_sentiment_df", fake_load_sentiment_df)

    result = sentiment.get_sentiment(
        coin="BTC",
        start_date=datetime(2026, 5, 30),
        end_date=datetime(2026, 6, 1),
        sources=["reddit"],
        analyzer="vader",
    )

    assert jsonable_encoder(result) == [
        {
            "timestamp": "2026-05-30T00:00:00",
            "text": "BTC looks strong",
            "source": "reddit",
            "analyzer": "vader",
            "sentiment": 0.8,
        }
    ]


def test_sentiment_endpoint_invalid_coin_returns_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        sentiment.get_sentiment(
            coin="DOGE",
            start_date=datetime(2026, 5, 30),
            end_date=datetime(2026, 6, 1),
            sources=["reddit"],
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unsupported coin: DOGE"


def test_sentiment_endpoint_analyzer_all_combines_analyzers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_sentiment_df(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-05-30 00:00:00")],
                "text": [f"{analyzer} row"],
                "source": ["reddit"],
                "analyzer": [analyzer],
                "sentiment": [0.5],
            }
        )

    monkeypatch.setattr(registry, "ALL_ANALYZER_NAMES", ("vader", "textblob"))
    monkeypatch.setattr(sentiment, "load_sentiment_df", fake_load_sentiment_df)

    result = sentiment.get_sentiment(
        coin="BTC",
        start_date=datetime(2026, 5, 30),
        end_date=datetime(2026, 6, 1),
        sources=["reddit"],
        analyzer="all",
    )

    assert [row["analyzer"] for row in result] == ["vader", "textblob"]
