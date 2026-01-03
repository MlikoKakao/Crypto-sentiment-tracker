from typing import Protocol

class _Sentiment(Protocol):
    polarity: float
    subjectivity: float

class TextBlob:
    raw: str
    def __init__(self, text: str) -> None: ...
    @property
    def sentiment(self) -> _Sentiment: ...