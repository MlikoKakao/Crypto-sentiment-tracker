from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class IndicatorConfig:
    coin: str
    use_sma: bool
    use_rsi: bool
    use_macd: bool
    sma_fast: int
    sma_slow: int
    rsi_period: int
    start_date: datetime
    end_date: datetime
