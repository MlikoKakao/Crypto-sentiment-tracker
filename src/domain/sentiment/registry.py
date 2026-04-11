from typing import Callable

from src.domain.sentiment.vader import vader_analyze
from src.domain.sentiment.textblob import textblob_analyze
from src.domain.sentiment.roberta import roberta_analyze
from src.domain.sentiment.finbert import finbert_analyze

AnalyzerFunc = Callable[[str | None], float]

ANALYZERS: dict[str, AnalyzerFunc] = {
    "vader": vader_analyze,
    "textblob": textblob_analyze,
    "twitter-roberta": roberta_analyze,
    "finbert": finbert_analyze,
}

ALL_ANALYZER_NAMES = ("vader", "textblob", "twitter-roberta", "finbert")

