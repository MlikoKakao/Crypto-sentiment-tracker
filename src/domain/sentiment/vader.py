from __future__ import annotations

from functools import lru_cache
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore


@lru_cache(maxsize=1)
def get_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def vader_analyze(text: Optional[str]) -> float:
    s = "" if text is None else str(text)
    analyzer = get_vader()
    return float(analyzer.polarity_scores(s)["compound"])
