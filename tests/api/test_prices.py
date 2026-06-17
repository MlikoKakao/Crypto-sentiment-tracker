from datetime import datetime

import pandas as pd
from pydantic import ValidationError
import pytest
from fastapi.encoders import jsonable_encoder

from src.app.dto import AnalysisConfig
from src.presentation.api.routes import market
from src.presentation.api.helpers.validate import DateRangeParams


def test_prices_endpoint_success_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_price_df(config: AnalysisConfig) -> pd.DataFrame:
        assert config.coin == "BTC"
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-01 00:00:00")],
                "price": [100.0],
            }
        )

    monkeypatch.setattr(market, "load_price_df", fake_load_price_df)

    result = market.get_prices(
        params=DateRangeParams(
            coin="BTC",
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 2),
        )
    )

    assert jsonable_encoder(result) == [
        {"timestamp": "2026-01-01T00:00:00", "price": 100.0}
    ]


def test_prices_endpoint_bad_date_returns_validation_error() -> None:
    with pytest.raises(ValidationError):
        DateRangeParams(
            coin="BTC",
            start_date=datetime(2026, 1, 10),
            end_date=datetime(2026, 1, 1),
        )
