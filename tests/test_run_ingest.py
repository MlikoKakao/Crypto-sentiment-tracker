import pandas as pd
import pytest

from src.app.dto import AnalysisConfig
from src.app.use_cases import run_ingest


def test_fetch_and_save_posts_returns_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    posts = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "text": ["BTC is strong"],
            "source": ["reddit"],
            "source_id": ["post-1"],
            "url": ["https://example.com/post-1"],
        }
    )

    def fetch_posts(config: AnalysisConfig) -> pd.DataFrame:
        return posts

    monkeypatch.setattr("src.infra.fetchers.service.fetch_posts", fetch_posts)

    result = run_ingest.fetch_and_save_posts(analysis_config)

    assert "content_hash" in result.columns
    assert result.loc[0, "content_hash"]


def test_analyze_and_save_sentiment_skips_empty_posts(
    monkeypatch: pytest.MonkeyPatch,
    analysis_config: AnalysisConfig,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("empty posts should not be scored or saved")

    monkeypatch.setattr(
        "src.domain.sentiment.service.add_sentiment_to_df",
        fail_if_called,
    )
    monkeypatch.setattr(
        "src.infra.storage.db.sentiment_repository.save_sentiment_df",
        fail_if_called,
    )

    result = run_ingest.analyze_and_save_sentiment(analysis_config, pd.DataFrame())

    assert result.empty
