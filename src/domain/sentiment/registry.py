from typing import Callable, Protocol

from src.domain.sentiment.vader import vader_analyze
from src.domain.sentiment.textblob import textblob_analyze
from src.domain.sentiment.roberta import RobertaAnalyzer
from src.domain.sentiment.finbert import finbert_analyze


class BatchAnalyzer(Protocol):
    def __call__(self, text: str | None) -> float: ...
    def analyze_many(self, texts: list[str | None]) -> list[float]: ...
    
Analyzer = Callable[[str | None], float] | BatchAnalyzer


ANALYZERS: dict[str, Analyzer] = {
    "vader": vader_analyze,
    "textblob": textblob_analyze,
    "twitter-roberta": RobertaAnalyzer(),
    "finbert": finbert_analyze,
}

ALL_ANALYZER_NAMES = ("vader", "textblob", "twitter-roberta", "finbert")
