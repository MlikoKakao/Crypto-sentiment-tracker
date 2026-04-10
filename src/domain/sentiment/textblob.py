from __future__ import annotations
from typing import Optional
try:
    from textblob import TextBlob  # type: ignore[import]
except Exception:
    TextBlob = None  # type: ignore


def textblob_analyze(text: Optional[str]) -> float:
    if TextBlob is None:
        raise RuntimeError("textblob not available")
    return float(getattr(TextBlob(str(text)).sentiment, "polarity", 0.0))