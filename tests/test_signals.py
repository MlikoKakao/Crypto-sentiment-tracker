import pandas as pd

from src.domain.signals.engine import build_signal_df


def test_build_signal_df_adds_signal_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=7, freq="30min"),
            "price": [100, 101, 99, 100, 99, 95, 100],
            "sentiment": [0.1, -0.1, 0.2, 1, -1, -0.5, -0.6],
        }
    )

    result = build_signal_df(df)

    assert "positive_sentiment" in result.columns
    assert "negative_sentiment" in result.columns
    assert "bearish_divergence" in result.columns
    assert "bullish_divergence" in result.columns

    assert result.loc[3, "positive_sentiment"]
    assert result.loc[4, "negative_sentiment"]
    assert result.loc[5, "bullish_divergence"]
    assert result.loc[6, "bearish_divergence"]


def test_build_signal_df_detects_sma_bullish_cross() -> None:
    df = pd.DataFrame(
        {
            "price": [100, 101],
            "sentiment": [0.1, 0.2],
            "sma_20": [9, 11],
            "sma_50": [10, 10],
        }
    )

    result = build_signal_df(df)

    assert result["sma_bullish_cross"].iloc[1]


def test_build_signal_df_handles_empty_dataframe_with_required_columns() -> None:
    df = pd.DataFrame(columns=["timestamp", "price", "sentiment"])

    result = build_signal_df(df)

    assert result.empty
    assert "positive_sentiment" in result.columns
    assert "negative_sentiment" in result.columns
