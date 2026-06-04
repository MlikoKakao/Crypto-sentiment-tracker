from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class IndicatorConfig:
    coin: str
    start_date: datetime
    end_date: datetime
    use_sma: bool = False
    use_rsi: bool = False
    use_macd: bool = False
    sma_fast: int = 20
    sma_slow: int = 50
    rsi_period: int = 14
