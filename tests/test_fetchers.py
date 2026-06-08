from dataclasses import replace

import pandas as pd
import pytest

from src.app.dto import AnalysisConfig
from src.infra.fetchers import service
from src.infra.fetchers import reddit
from src.infra.fetchers import youtube
from src.shared.helpers import normalize_coin


def test_fetch_posts_combines_mocked_fetchers(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    def fake_reddit_fetcher(config: AnalysisConfig) -> pd.DataFrame:
        return pd.DataFrame({"source": ["reddit"], "text": ["BTC reddit"]})

    def fake_news_fetcher(config: AnalysisConfig) -> pd.DataFrame:
        return pd.DataFrame({"source": ["news"], "text": ["BTC news"]})

    monkeypatch.setattr(
        service,
        "FETCHERS",
        {
            "reddit": fake_reddit_fetcher,
            "news": fake_news_fetcher,
        },
    )

    config = replace(analysis_config, sources=("reddit", "news"))
    result = service.fetch_posts(config)

    assert result["source"].tolist() == ["reddit", "news"]
    assert result["text"].tolist() == ["BTC reddit", "BTC news"]


def test_fetch_posts_returns_empty_schema_when_no_fetchers_match(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    monkeypatch.setattr(service, "FETCHERS", {})

    result = service.fetch_posts(analysis_config)

    assert result.empty
    assert list(result.columns) == [
        "timestamp",
        "text",
        "sentiment",
        "source",
        "source_id",
        "url",
    ]


def test_reddit_client_without_api_key_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)

    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID"):
        reddit.get_reddit_client()


def test_youtube_fetch_without_api_key_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    def empty_cached_content(config: AnalysisConfig, source: str) -> pd.DataFrame:
        return pd.DataFrame()

    def cache_miss(config: AnalysisConfig, df: pd.DataFrame) -> bool:
        return False

    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(youtube, "load_content_df", empty_cached_content)
    monkeypatch.setattr(youtube, "has_content_coverage", cache_miss)

    with pytest.raises(RuntimeError, match="YOUTUBE_API_KEY"):
        youtube.fetch_youtube_posts(replace(analysis_config, sources=("youtube",)))


def test_unsupported_coin_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported coin: DOGE"):
        normalize_coin("DOGE")
