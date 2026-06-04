from __future__ import annotations
from typing import Optional

try:
    from textblob import TextBlob  # type: ignore[import]
except ImportError:
    TextBlob = None  # type: ignore


def textblob_analyze(text: Optional[str]) -> float:
    if TextBlob is None:
        raise RuntimeError(
            "textblob is not installed. Install it with: pip install textblob"
        )

    if text is None:
        text = ""

    return float(TextBlob(text).sentiment.polarity)
