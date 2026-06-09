import logging
import pandas as pd

from src.app.defaults import DEFAULT_CONFIG
from src.app.dto import AnalysisConfig, AnalysisIssue, AnalysisResult
from src.infra.fetchers.coinbase_price import get_coinbase_price_history
from src.infra.fetchers.service import fetch_posts
from src.domain.market.merge import merge_sentiment_and_price_df
from src.shared.helpers import save_csv
from src.infra.storage.db.schema import init_db
from src.app.use_cases.sentiment_cache import get_or_create_sentiment_df

logger = logging.getLogger(__name__)


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    init_db()

    issues: list[AnalysisIssue] = []

    logger.info("Fetching posts for %s from %s", config.coin, config.sources)
    try:
        posts_df = fetch_posts(config)
    except Exception as e:
        issues.append(AnalysisIssue(stage="posts", message=str(e)))
        posts_df = pd.DataFrame()
    logger.info("Fetched %s posts", len(posts_df))

    logger.info(
        "Analyzing sentiment for %s from %s posts with %s",
        config.coin,
        config.sources,
        config.analyzer,
    )
    try:
        sentiment_df = get_or_create_sentiment_df(config, posts_df)
    except Exception as e:
        issues.append(AnalysisIssue(stage="sentiment", message=str(e)))
        sentiment_df = pd.DataFrame()
    logger.info("Analyzed %s posts", len(sentiment_df))

    logger.info(
        "Fetching price points for %s from %s to %s",
        config.coin,
        config.end_date,
        config.start_date,
    )
    try:
        price_df = get_coinbase_price_history(config)
    except Exception as e:
        issues.append(AnalysisIssue(stage="price", message=str(e)))
        price_df = pd.DataFrame()
    logger.info("Fetched %s price points", len(price_df))

    logger.info("Merging price and posts for %s from %s", config.coin, config.sources)
    merged_df = merge_sentiment_and_price_df(price_df, sentiment_df)
    logger.info("Merged %s posts", len(merged_df))
    if merged_df.empty:
        issues.append(AnalysisIssue(stage="merge", message="Merged output is empty"))
    elif len(merged_df.dropna(subset=["price", "sentiment"])) < 5:
        issues.append(
            AnalysisIssue(
                stage="merge",
                message="Merged output has fewer than 5 usable price/sentiment rows",
            )
        )

    status = "ok" if not issues else "partial"

    return AnalysisResult(
        posts_df=posts_df,
        price_df=price_df,
        merged_df=merged_df,
        status=status,
        issues=tuple(issues),
    )


if __name__ == "__main__":
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Test running analysis with default config...")
    result = run_analysis(DEFAULT_CONFIG)
    save_csv(result.posts_df, "data/tests/run_analysis_posts.csv")
    save_csv(result.price_df, "data/tests/run_analysis_prices.csv")
    save_csv(result.merged_df, "data/tests/run_analysis_merged.csv")
