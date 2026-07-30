from dataclasses import dataclass
from datetime import datetime

from src.app.dto import Coin


@dataclass(frozen=True)
class SignalQuery:
    coin: Coin
    start_date: datetime
    end_date: datetime
    signal_names: tuple[str, ...]
    num_signals: int

    def to_params(self) -> list[tuple[str, str | int]]:
        params: list[tuple[str, str | int]] = [
            ("coin", self.coin),
            ("start_date", self.start_date.isoformat()),
            ("end_date", self.end_date.isoformat()),
            ("numSignals", self.num_signals),
        ]
        params.extend(("signalName", name) for name in self.signal_names)
        return params
