from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
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

    def _to_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_date", self._to_utc(self.start_date))
        object.__setattr__(self, "end_date", self._to_utc(self.end_date))


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
