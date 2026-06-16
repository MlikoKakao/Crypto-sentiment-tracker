from datetime import datetime
from pydantic import BaseModel


class SignalResponse(BaseModel):
    coin: str
    timestamp: datetime
    signal: str
    value: float
