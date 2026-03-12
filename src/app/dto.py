from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from typing import Literal

Analyzer = Literal["vader", "textblob", "roberta", "finberta"]
Source = Literal["reddit", "twitter", "news"]


@dataclass(frozen=True)
class AnalysisConfig:
    coin: str
    start_date: datetime
    end_date: datetime
    analyzer: tuple[Analyzer, ...]
    sources: tuple[Source, ...]
    num_posts: int
    subreddits: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisResult:
    posts_df: pd.DataFrame
    price_df: pd.DataFrame
    merged_df: pd.DataFrame
