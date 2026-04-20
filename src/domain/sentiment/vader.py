from __future__ import annotations
from typing import Optional
import nltk # type: ignore

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore[import]
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore  
    
_vader = None
    
def vader_analyze(text: Optional[str]) -> float:
    global _vader
    if _vader is None:
        if SentimentIntensityAnalyzer is None:
            raise RuntimeError("nltk SentimentIntensityAnalyzer not available")
        try:
            _vader = SentimentIntensityAnalyzer()
        except LookupError:
            if nltk is not None:
                try:
                    nltk.download("vader_lexicon", quiet=True)
                except Exception:
                    pass
            _vader = SentimentIntensityAnalyzer()
    s = "" if text is None else str(text)
    return _vader.polarity_scores(s)["compound"]