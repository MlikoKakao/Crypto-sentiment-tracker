from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from typing import Literal

Coin = Literal["BTC", "ETH", "XMR"]
Analyzer = Literal["vader", "textblob", "twitter-roberta", "finbert", "all"]
Source = Literal["reddit", "youtube", "news"]

Status = Literal["ok", "partial", "failed"]


@dataclass(frozen=True)
class AnalysisConfig:
    coin: Coin
    start_date: datetime
    end_date: datetime
    analyzer: Analyzer
    sources: tuple[Source, ...]
    num_posts: int
    subreddits: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisResult:
    posts_df: pd.DataFrame
    price_df: pd.DataFrame
    merged_df: pd.DataFrame
    status: Status = "ok"
    issues: tuple[AnalysisIssue, ...] = ()


@dataclass(frozen=True)
class AnalysisIssue:
    stage: str
    message: str
