from __future__ import annotations
from typing import Optional
import nltk  # type: ignore

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore[import]
except ImportError:
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]


_vader = None


def vader_analyze(text: Optional[str]) -> float:
    global _vader

    if _vader is None:
        if SentimentIntensityAnalyzer is None:
            raise RuntimeError("nltk SentimentIntensityAnalyzer not available")

        try:
            _vader = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _vader = SentimentIntensityAnalyzer()

    s = "" if text is None else str(text)
    return float(_vader.polarity_scores(s)["compound"])
