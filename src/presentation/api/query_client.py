import os

import pandas as pd
import requests

from src.presentation.api.contracts.requests import (
    PostQuery,
    PriceQuery,
    SentimentQuery,
    SignalQuery,
)
from src.app.dto import AnalysisConfig, AnalysisIssue, AnalysisResult
from src.domain.market.merge import merge_sentiment_and_price_df

QueryParams = dict[str, str] | list[tuple[str, str | int]]

def run_analysis_with_api(config: AnalysisConfig) -> AnalysisResult:
    issues: list[AnalysisIssue] = []
    
    price_query = PriceQuery(
                       coin=config.coin,
                       start_date=config.start_date,
                       end_date=config.end_date)
    
    try:
        price_df = get_prices(price_query)
    except requests.RequestException as error:
        issues.append(AnalysisIssue(stage="price", message=str(error)))
        price_df = pd.DataFrame()
    
    posts_query = PostQuery(
        coin = config.coin,
        start_date=config.start_date,
        end_date=config.end_date,
        sources=config.sources,
        num_posts=config.num_posts
    )
    
    try:
        posts_df = get_posts(posts_query)
    except requests.RequestException as error:
        issues.append(AnalysisIssue(stage="posts", message=str(error)))
        posts_df = pd.DataFrame()
        
    sentiment_query = SentimentQuery(
        coin=config.coin,
        start_date=config.start_date,
        end_date=config.end_date,
        sources=config.sources,
        analyzer=config.analyzer,
        limit=config.num_posts
    )
    
    try:
        sentiment_df = get_sentiment(sentiment_query)
    except requests.RequestException as error:
        issues.append(AnalysisIssue(stage="sentiment", message=str(error)))
        sentiment_df = pd.DataFrame()
    
    merged_df = merge_sentiment_and_price_df(price_df, sentiment_df)
    
    return AnalysisResult(
            posts_df=posts_df,
            price_df=price_df,
            merged_df=merged_df,
            status="ok" if not issues else "partial",
            issues=tuple(issues),
        )

def _get_table(path: str, params: QueryParams) -> pd.DataFrame:
    base_url = os.getenv("QUERY_API_URL", "http://localhost:8080").rstrip("/")
    response = requests.get(
        f"{base_url}/{path}",
        params=params,
        timeout=10,
    )
    response.raise_for_status()

    df = pd.DataFrame(response.json())
    df.rename(
        columns={
            "sourceId": "source_id",
            "contentHash": "content_hash",
            "signalName": "signal_name",
        },
        inplace=True,
    )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_prices(query: PriceQuery) -> pd.DataFrame:
    return _get_table("prices", query.to_params())


def get_posts(query: PostQuery) -> pd.DataFrame:
    return _get_table("posts", query.to_params())


def get_sentiment(query: SentimentQuery) -> pd.DataFrame:
    return _get_table("sentiment", query.to_params())


def get_signals(query: SignalQuery) -> pd.DataFrame:
    return _get_table("signals", query.to_params())
