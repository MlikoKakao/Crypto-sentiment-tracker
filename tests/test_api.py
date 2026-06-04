from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from datetime import datetime

from src.presentation.api.main import get_prices, health_check, app

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    assert health_check() == {"status": "ok"}


def test_prices_rejects_end_date_before_start_date() -> None:
    start_datetime = datetime(2026, 4, 1, 9, 30)
    end_datetime = datetime(2026, 3, 1, 9, 30)

    with pytest.raises(HTTPException):
        get_prices(coin="btc", start_date=start_datetime, end_date=end_datetime)


def test_prices_endpoint_bad_date_returns_400():
    response = client.get(
        "/prices",
        params={
            "coin": "BTC",
            "start_date": "2024-01-10T00:00:00",
            "end_date": "2024-01-01T00:00:00",
        },
    )

    assert response.status_code == 400
