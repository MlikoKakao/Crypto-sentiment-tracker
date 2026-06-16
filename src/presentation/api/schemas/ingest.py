from pydantic import BaseModel

from src.app.dto import Source, Analyzer
from src.presentation.api.helpers.validate import DateRangeParams


class IngestRequest(BaseModel):
    params: DateRangeParams
    num_posts: int
    analyzer: Analyzer
    sources: list[Source]


class IngestResponse(BaseModel):
    status: str
    coin: str
    sources: list[Source]
    price_points: int
    posts_ingested: int
    sentiment_rows: int
