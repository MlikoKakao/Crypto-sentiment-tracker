from src.presentation.api.routes.health import health_check


def test_health_endpoint_returns_ok() -> None:
    assert health_check() == {"status": "ok"}
