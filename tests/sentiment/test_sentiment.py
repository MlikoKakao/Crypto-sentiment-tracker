import pandas as pd
import pytest

from src.domain.sentiment.service import add_sentiment_to_df


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
