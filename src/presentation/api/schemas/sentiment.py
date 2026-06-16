from datetime import datetime
from pydantic import BaseModel

from src.app.dto import Source


class SentimentResponse(BaseModel):
    coin: str
    source: Source
    source_id: str | None = None
    timestamp: datetime
    text: str
    url: str | None = None
    content_hash: str
    sentiment: float
