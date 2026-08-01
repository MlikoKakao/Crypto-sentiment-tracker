import pandas as pd
import pytest

from src.app.jobs import backfill_sentiment
from src.infra.storage.db.content_repository import (
    count_content_missing_sentiment,
    load_content_missing_sentiment,
    save_content_df,
)
from src.infra.storage.db.sentiment_repository import save_sentiment_df


def test_missing_sentiment_queries_skip_existing_rows() -> None:
    content = save_content_df(
        pd.DataFrame(
            {
                "source": ["reddit", "reddit"],
                "source_id": ["one", "two"],
                "timestamp": [
                    pd.Timestamp("2026-01-01"),
                    pd.Timestamp("2026-01-02"),
                ],
                "text": ["first", "second"],
                "url": ["https://example.com/one", "https://example.com/two"],
            }
        ),
        "BTC",
    )
    existing = content.iloc[[0]].assign(
        analyzer="textblob",
        sentiment=0.5,
    )
    save_sentiment_df(existing, "BTC")

    assert count_content_missing_sentiment("BTC", "textblob") == 1
    missing = load_content_missing_sentiment("BTC", "textblob", limit=10)
    assert missing["text"].tolist() == ["second"]


def test_backfill_dry_run_never_scores_or_saves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backfill_sentiment, "count_content_missing_sentiment", lambda *_: 3)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run must not score or save")

    monkeypatch.setattr(backfill_sentiment, "add_sentiment_to_df", fail)
    monkeypatch.setattr(backfill_sentiment, "save_sentiment_df", fail)

    saved = backfill_sentiment.backfill_sentiment(
        coins=("BTC",),
        analyzers=("textblob",),
        batch_size=2,
        execute=False,
    )

    assert saved == 0


def test_backfill_saves_batches_and_stops_when_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        pd.DataFrame({"text": ["one", "two"]}),
        pd.DataFrame({"text": ["three"]}),
        pd.DataFrame(),
    ]
    saved_sizes: list[int] = []

    monkeypatch.setattr(backfill_sentiment, "count_content_missing_sentiment", lambda *_: 3)
    monkeypatch.setattr(
        backfill_sentiment,
        "load_content_missing_sentiment",
        lambda *_: batches.pop(0),
    )
    monkeypatch.setattr(
        backfill_sentiment,
        "add_sentiment_to_df",
        lambda df, analyzer: df.assign(analyzer=analyzer, sentiment=0.1),
    )
    monkeypatch.setattr(
        backfill_sentiment,
        "save_sentiment_df",
        lambda df, coin: saved_sizes.append(len(df)),
    )

    saved = backfill_sentiment.backfill_sentiment(
        coins=("BTC",),
        analyzers=("textblob",),
        batch_size=2,
        execute=True,
    )

    assert saved == 3
    assert saved_sizes == [2, 1]
