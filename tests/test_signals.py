import pandas as pd

from src.domain.signals.engine import build_signal_df


def test_build_signal_df_adds_signal_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="30min"),
            "price": [100, 101, 99],
            "sentiment": [0.1, -0.1, 0.2],
        }
    )

    result = build_signal_df(df)

    assert "positive_sentiment" in result.columns
    assert "negative_sentiment" in result.columns
    assert "bearish_divergence" in result.columns
    assert "bullish_divergence" in result.columns


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

    assert result["sma_bullish_cross"].iloc[1] == True
