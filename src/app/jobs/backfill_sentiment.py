import argparse
from collections.abc import Sequence

from src.domain.sentiment.registry import ALL_ANALYZER_NAMES
from src.domain.sentiment.service import add_sentiment_to_df
from src.infra.storage.db.content_repository import (
    count_content_missing_sentiment,
    load_content_missing_sentiment,
)
from src.infra.storage.db.sentiment_repository import save_sentiment_df

COINS = ("BTC", "ETH", "XMR")
DEFAULT_ANALYZERS = tuple(
    analyzer for analyzer in ALL_ANALYZER_NAMES if analyzer != "vader"
)


def backfill_sentiment(
    coins: Sequence[str],
    analyzers: Sequence[str],
    batch_size: int,
    execute: bool,
    max_batches: int | None = None,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1")

    missing_total = 0
    for analyzer in analyzers:
        for coin in coins:
            missing = count_content_missing_sentiment(coin, analyzer)
            missing_total += missing
            print(f"{coin} {analyzer}: {missing} missing rows")

    if not execute:
        print(f"Dry run only. {missing_total} rows would be analyzed.")
        return 0

    saved_total = 0
    for analyzer in analyzers:
        for coin in coins:
            batches = 0
            while max_batches is None or batches < max_batches:
                content_df = load_content_missing_sentiment(
                    coin,
                    analyzer,
                    batch_size,
                )
                if content_df.empty:
                    break

                sentiment_df = add_sentiment_to_df(content_df, analyzer)
                save_sentiment_df(sentiment_df, coin)
                batches += 1
                saved_total += len(sentiment_df)
                print(
                    f"Saved {len(sentiment_df)} {coin} {analyzer} rows "
                    f"(total saved: {saved_total})"
                )

    print(f"Backfill finished. Saved {saved_total} rows.")
    return saved_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stored content that is missing sentiment rows."
    )
    parser.add_argument(
        "--coin",
        action="append",
        choices=COINS,
        dest="coins",
        help="Coin to process. Repeat for multiple coins. Defaults to all.",
    )
    parser.add_argument(
        "--analyzer",
        action="append",
        choices=DEFAULT_ANALYZERS,
        dest="analyzers",
        help="Analyzer to process. Repeat for multiple analyzers. Defaults to all missing.",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Maximum batches per coin/analyzer pair.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write results. Without this flag, only missing counts are shown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backfill_sentiment(
        coins=args.coins or COINS,
        analyzers=args.analyzers or DEFAULT_ANALYZERS,
        batch_size=args.batch_size,
        execute=args.execute,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
