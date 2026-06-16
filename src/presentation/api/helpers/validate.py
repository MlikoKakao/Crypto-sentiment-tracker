from datetime import datetime
from pydantic import BaseModel, model_validator

from src.shared.helpers import is_date_correct, normalize_coin


class DateRangeParams(BaseModel):
    coin: str
    start_date: datetime
    end_date: datetime

    @model_validator(mode="after")
    def validate_date_range(self):
        if not is_date_correct(self.start_date, self.end_date):
            raise ValueError("end_date must be after start_date")
        self.coin = normalize_coin(self.coin)
        return self
