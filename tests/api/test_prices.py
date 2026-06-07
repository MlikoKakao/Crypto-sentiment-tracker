from fastapi.testclient import TestClient


def test_prices_endpoint_bad_date_returns_400(client: TestClient) -> None:
    response = client.get(
        "/market/prices",
        params={
            "coin": "BTC",
            "start_date": "2026-01-10T00:00:00",
            "end_date": "2026-01-01T00:00:00",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "end_date must be after start_date"
