import pandas as pd
import pytest
from typing import Any

from src.domain.sentiment.service import add_sentiment_to_df
from src.domain.sentiment import service


def test_add_sentiment_to_df_adds_sentiment_and_analyzer() -> None:
    df = pd.DataFrame(
        {
            "coin": ["btc"] * 2,
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="30min"),
            "text": ["BITCOIN IS GOING UP", "bitcoin is falling"],
        }
    )

    result = add_sentiment_to_df(df, "vader")

    assert "sentiment" in result.columns
    assert "analyzer" in result.columns
    assert len(result) == 2
    assert result["analyzer"].tolist() == ["vader", "vader"]


def test_add_sentiment_to_df_rejects_unknown_analyzer() -> None:
    df = pd.DataFrame({"text": ["hello"]})

    with pytest.raises(ValueError):
        add_sentiment_to_df(df, "missing-analyzer")

class FakeBatchAnalyzer:
    def analyze_many(self, texts: list[str]) -> list[float]:
        return [0.5] * len(texts)

    def __call__(self, text: str | None) -> float:
        raise AssertionError("single analyzer path was used")


def test_add_sentiment_uses_batch_analyzer(monkeypatch: Any):
    monkeypatch.setitem(service.ANALYZERS, "fake-batch", FakeBatchAnalyzer())

    df = pd.DataFrame({"text": ["a", "b", "c"]})

    result = service.add_sentiment_to_df(df, "fake-batch")

    assert result["sentiment"].tolist() == [0.5, 0.5, 0.5]