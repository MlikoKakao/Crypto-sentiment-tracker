from typing import Protocol

class _Sentiment(Protocol):
    polarity: float
    subjectivity: float

class vaderSentiment:
    raw: str