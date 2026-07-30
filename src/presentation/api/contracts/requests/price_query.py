from dataclasses import dataclass
from datetime import datetime

from src.app.dto import Coin


@dataclass(frozen=True)
class PriceQuery:
    coin: Coin
    start_date: datetime
    end_date: datetime

    def to_params(self) -> dict[str, str]:
        return {
            "coin": self.coin,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }
