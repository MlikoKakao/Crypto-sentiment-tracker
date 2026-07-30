from dataclasses import dataclass
from datetime import datetime

from src.app.dto import Coin, Source


@dataclass(frozen=True)
class PostQuery:
    coin: Coin
    start_date: datetime
    end_date: datetime
    sources: tuple[Source, ...]
    num_posts: int

    def to_params(self) -> list[tuple[str, str | int]]:
        params: list[tuple[str, str | int]] = [
            ("coin", self.coin),
            ("start_date", self.start_date.isoformat()),
            ("end_date", self.end_date.isoformat()),
            ("numPosts", self.num_posts),
        ]
        params.extend(("source", source) for source in self.sources)
        return params
