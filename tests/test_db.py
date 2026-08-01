import pandas as pd

from src.app.dto import AnalysisConfig
from src.domain.market.dto import IndicatorConfig
from src.infra.storage.db.content_repository import (
    has_content_coverage,
    load_content_df,
    save_content_df,
)
from src.infra.storage.db.price_repository import load_price_df, save_price_df
from src.infra.storage.db.sentiment_repository import (
    has_sentiment_coverage,
    load_sentiment_df,
    save_sentiment_df,
)
from src.infra.storage.db.signal_repository import load_signal_df, save_signal_df


def test_price_repository_saves_and_loads_rows(
    analysis_config: AnalysisConfig,
) -> None:
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1h"),
            "price": [100.0, 101.5],
        }
    )
    save_price_df(prices, "btc")

    result = load_price_df(analysis_config)

    assert len(result) == 2
    assert result["price"].tolist() == [101.5, 100.0]


def test_content_repository_saves_and_loads_rows(
    analysis_config: AnalysisConfig,
) -> None:
    content = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1h"),
            "text": ["BTC is strong", "BTC is quiet"],
            "source": ["reddit", "reddit"],
            "source_id": ["post-1", "post-2"],
            "url": ["https://example.com/1", "https://example.com/2"],
        }
    )

    save_content_df(content, "btc")

    result = load_content_df(analysis_config, "reddit")

    assert len(result) == 2
    assert set(result["source_id"]) == {"post-1", "post-2"}
    assert "content_hash" in result.columns
    assert result["timestamp"].dt.tz is None
    assert has_content_coverage(analysis_config, result) is False


def test_sentiment_repository_saves_and_loads_rows(
    analysis_config: AnalysisConfig,
) -> None:
    content = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "text": ["BTC is strong"],
            "source": ["reddit"],
            "source_id": ["post-1"],
        }
    )
    save_content_df(content, "btc")

    saved_content = load_content_df(
        analysis_config,
        "reddit",
    )
    sentiment = saved_content.assign(analyzer="vader", sentiment=0.7)

    save_sentiment_df(sentiment, "btc")
    result = load_sentiment_df(analysis_config, "vader")

    assert len(result) == 1
    assert result.loc[0, "text"] == "BTC is strong"
    assert result.loc[0, "sentiment"] == 0.7
    assert result["timestamp"].dt.tz is None
    assert has_sentiment_coverage(analysis_config, result) is False


def test_signal_repository_saves_and_loads_rows(
    indicator_config: IndicatorConfig,
) -> None:
    signals = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1h"),
            "sma_20": [100.0, 101.5],
        }
    )

    save_signal_df(signals, "sma_20", "btc")

    result = load_signal_df(indicator_config, "sma_20")

    assert len(result) == 2
    assert result["sma_20"].tolist() == [100.0, 101.5]
