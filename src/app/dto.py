from dataclasses import dataclass
from datetime import datetime
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
