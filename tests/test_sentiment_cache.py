from dataclasses import replace

import pandas as pd
import pytest

from src.app.dto import AnalysisConfig
from src.app.use_cases import sentiment_cache


def test_get_or_create_single_sentiment_uses_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    cached = pd.DataFrame(
        {
            "source": ["reddit"],
            "content_hash": ["hash-1"],
            "sentiment": [0.9],
            "analyzer": ["vader"],
        }
    )

    def load_cached_sentiment(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
        return cached

    def cache_hit(config: AnalysisConfig, df: pd.DataFrame) -> bool:
        return True

    monkeypatch.setattr(sentiment_cache, "load_sentiment_df", load_cached_sentiment)
    monkeypatch.setattr(sentiment_cache, "has_sentiment_coverage", cache_hit)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("sentiment should not be recomputed on cache hit")

    monkeypatch.setattr(sentiment_cache, "add_sentiment_to_df", fail_if_called)
    monkeypatch.setattr(sentiment_cache, "save_sentiment_df", fail_if_called)

    result = sentiment_cache.get_or_create_sentiment_df(
        analysis_config,
        pd.DataFrame({"text": ["BTC"]}),
    )

    assert result.equals(cached)


def test_get_or_create_single_sentiment_recomputes_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    posts = pd.DataFrame(
        {
            "source": ["reddit"],
            "content_hash": ["hash-1"],
            "text": ["BTC is strong"],
        }
    )
    computed = posts.assign(sentiment=0.7, analyzer="vader")
    saved: list[pd.DataFrame] = []

    def load_empty_sentiment(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
        return pd.DataFrame()

    def cache_miss(config: AnalysisConfig, df: pd.DataFrame) -> bool:
        return False

    def compute_sentiment(df: pd.DataFrame, analyzer_name: str) -> pd.DataFrame:
        return computed

    def save_sentiment(df: pd.DataFrame, coin: str) -> None:
        saved.append(df)

    monkeypatch.setattr(sentiment_cache, "load_sentiment_df", load_empty_sentiment)
    monkeypatch.setattr(sentiment_cache, "has_sentiment_coverage", cache_miss)
    monkeypatch.setattr(sentiment_cache, "add_sentiment_to_df", compute_sentiment)
    monkeypatch.setattr(sentiment_cache, "save_sentiment_df", save_sentiment)

    result = sentiment_cache.get_or_create_sentiment_df(analysis_config, posts)

    assert result.equals(computed)
    assert len(saved) == 1


def test_get_or_create_multiple_sentiment_averages_all_analyzers(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    posts = pd.DataFrame(
        {
            "source": ["reddit"],
            "content_hash": ["hash-1"],
            "text": ["BTC is strong"],
        }
    )

    def fake_add_sentiment_to_df(df: pd.DataFrame, analyzer_name: str) -> pd.DataFrame:
        score = 1.0 if analyzer_name == "vader" else -1.0
        return df.assign(sentiment=score, analyzer=analyzer_name)

    def load_empty_sentiment(config: AnalysisConfig, analyzer: str) -> pd.DataFrame:
        return pd.DataFrame()

    def cache_miss(config: AnalysisConfig, df: pd.DataFrame) -> bool:
        return False

    def save_sentiment(df: pd.DataFrame, coin: str) -> None:
        return None

    monkeypatch.setattr(sentiment_cache, "ALL_ANALYZER_NAMES", ("vader", "textblob"))
    monkeypatch.setattr(sentiment_cache, "load_sentiment_df", load_empty_sentiment)
    monkeypatch.setattr(sentiment_cache, "has_sentiment_coverage", cache_miss)
    monkeypatch.setattr(sentiment_cache, "add_sentiment_to_df", fake_add_sentiment_to_df)
    monkeypatch.setattr(sentiment_cache, "save_sentiment_df", save_sentiment)

    result = sentiment_cache.get_or_create_sentiment_df(
        replace(analysis_config, analyzer="all"),
        posts,
    )

    assert result.loc[0, "sentiment"] == 0.0
    assert result.loc[0, "analyzer"] == "all"
