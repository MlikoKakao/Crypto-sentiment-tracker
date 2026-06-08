import pandas as pd

from src.domain.market.merge import merge_sentiment_and_price_df


def test_merge_sentiment_and_price_df_uses_previous_sentiment_within_tolerance():
    price_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-01 10:00:00",
                    "2026-06-01 11:00:00",
                ]
            ),
            "price": [100, 110],
        }
    )
    sentiment_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-01 10:15:00",
                    "2026-06-01 10:45:00",
                    "2026-06-01 11:20:00",
                ]
            ),
            "sentiment": [0.5, -0.1, 0.8],
        }
    )

    result = merge_sentiment_and_price_df(price_df, sentiment_df)

    assert result.loc[0, "price"] == 100
    assert pd.isna(result.loc[0, "sentiment"])
    assert result.loc[1, "price"] == 110
    assert result.loc[1, "sentiment"] == -0.1


def test_merge_sentiment_and_price_df_returns_expected_columns():
    price_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01 10:00:00"]),
            "price": [100],
        }
    )
    sentiment_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01 09:45:00"]),
            "sentiment": [0.5],
        }
    )

    result = merge_sentiment_and_price_df(price_df, sentiment_df)

    assert len(result) == 1
    assert "timestamp" in result.columns
    assert "price" in result.columns
    assert "sentiment" in result.columns


def test_merge_sentiment_and_price_df_empty_input_returns_empty_schema():
    price_df = pd.DataFrame(columns=["timestamp", "price"])
    sentiment_df = pd.DataFrame(columns=["timestamp", "sentiment"])

    result = merge_sentiment_and_price_df(price_df, sentiment_df)

    assert result.empty
    assert list(result.columns) == ["timestamp", "price", "text", "sentiment", "source"]
