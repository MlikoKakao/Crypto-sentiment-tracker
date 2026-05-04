import logging
import pandas as pd
from src.app.dto import AnalysisResult


def run_demo() -> AnalysisResult:
    logger = logging.getLogger(__name__)
    logger.info("Fetching demo csv...")

    posts_df = pd.read_csv(("data/demo/demo_posts.csv"))
    price_df = pd.read_csv("data/demo/demo_prices.csv")
    merged_df = pd.read_csv("data/demo/demo_merged.csv")

    return AnalysisResult(posts_df=posts_df, price_df=price_df, merged_df=merged_df)