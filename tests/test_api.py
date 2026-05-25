from fastapi import HTTPException
import pytest
from datetime import datetime

from src.presentation.api.main import IngestRequest, get_prices, health_check


def test_health_endpoint_returns_ok() -> None:
    assert health_check() == {"status": "ok"}


def test_prices_rejects_end_date_before_start_date() -> None:
    start_datetime = datetime(2026, 3, 1, 9, 30)
    end_datetime = datetime(2026, 4, 1, 9, 30)

    
    request = IngestRequest(
        coin="BTC",
        num_posts=10,
        start_date=start_datetime,
        end_date=end_datetime,
        analyzer="vader",
        sources=("reddit",),
    )

    with pytest.raises(HTTPException):
        get_prices(request)
