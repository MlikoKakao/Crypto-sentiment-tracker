from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.app.dto import AnalysisConfig
from src.infra.fetchers import service
from src.infra.fetchers import reddit
from src.infra.fetchers import youtube
from src.infra.fetchers import price
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
        "get_fetchers",
        lambda: {
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
    monkeypatch.setattr(service, "get_fetchers", lambda: {})  # type: ignore
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


def test_reddit_fetch_returns_newly_fetched_posts_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    submission = SimpleNamespace(
        created_utc=datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc).timestamp(),
        title="Bitcoin update",
        selftext="BTC adoption is growing",
        id="post-1",
        stickied=False,
        url="https://example.com/post-1",
        permalink="/r/bitcoin/post-1",
        score=10,
        upvote_ratio=0.9,
        num_comments=3,
        subreddit=SimpleNamespace(display_name="bitcoin"),
    )
    reddit_client = SimpleNamespace(
        subreddit=lambda name: SimpleNamespace(new=lambda limit: [submission])
    )

    monkeypatch.setattr(reddit, "load_content_df", lambda config, source: pd.DataFrame())
    monkeypatch.setattr(reddit, "has_content_coverage", lambda config, df: False)
    monkeypatch.setattr(reddit, "get_reddit_client", lambda: reddit_client)

    result = reddit.fetch_reddit_posts(analysis_config)

    assert result["source_id"].tolist() == ["post-1"]
    assert result["source"].tolist() == ["reddit"]


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


def test_xmr_price_fallback_returns_fetched_prices(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    monkeypatch.setattr(price, "load_price_df", lambda config: pd.DataFrame())
    monkeypatch.setattr(price, "has_price_coverage", lambda config, df: False)
    monkeypatch.setattr(
        price.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"prices": [[1_700_000_000_000, 150.0]]},
        ),
    )

    result = price.get_price_history(replace(analysis_config, coin="XMR"))

    assert result["price"].tolist() == [150.0]
