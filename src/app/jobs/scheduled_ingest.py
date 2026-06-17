from dataclasses import replace

from src.app.defaults import default_config
from src.app.use_cases.run_ingest import run_ingest


def run_scheduled_ingest() -> None:
    COINS = ["BTC", "ETH", "XMR"]

    for coin in COINS:
        config = replace(
            default_config(),
            coin=coin,
            analyzer="all",
            num_posts=100,
        )
        run_ingest(config)

        result = run_ingest(config)

        print(
            f"Ingested {result.posts_ingested} posts, for coin {coin}, "
            f"{result.sentiment_rows} sentiment rows"
        )


if __name__ == "__main__":
    run_scheduled_ingest()
