from src.app.defaults import DEFAULT_CONFIG
from src.app.dto import AnalysisConfig, AnalysisResult
from src.infra.fetchers.coinbase_price import get_coinbase_price_history
from src.infra.fetchers.service import fetch_posts
from src.domain.sentiment.service import add_sentiment_to_df
from src.domain.market.merge import merge_sentiment_and_price_df
from src.shared.helpers import save_csv
from src.infra.storage.db.schema import init_db

def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    init_db()
    posts_df = fetch_posts(config)
    sentiment_df = add_sentiment_to_df(posts_df, analyzer_name=config.analyzer)
    price_df = get_coinbase_price_history(config)
    merged_df = merge_sentiment_and_price_df(price_df, sentiment_df)

    return AnalysisResult(posts_df=posts_df, price_df=price_df, merged_df=merged_df)

if __name__ == "__main__":
    import logging
    from src.infra.storage.logging_config import configure_logging
    logger = logging.getLogger(__name__)
    configure_logging()

    logger.info("Test running analysis with default config...")
    result = run_analysis(DEFAULT_CONFIG)
    save_csv(result.posts_df, "data/tests/run_analysis_posts.csv")
    save_csv(result.price_df, "data/tests/run_analysis_prices.csv")
    save_csv(result.merged_df, "data/tests/run_analysis_merged.csv")
