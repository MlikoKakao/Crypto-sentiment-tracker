from pydantic import BaseModel
from datetime import datetime


class PricePoint(BaseModel):
    coin: str
    timestamp: datetime
    price: float
