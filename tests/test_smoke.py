import pandas as pd

from src.domain.market.filtering import contains_coin
from src.domain.analysis.lead_lag import compute_lead_lag

def test_import_core_modules() -> None:
    import src.domain.market.filtering
    import src.domain.analysis.lead_lag
    import src.domain.market.indicators

def test_contains_coin() -> None:
    assert contains_coin("Bitcoin is rising", "BTC") is True
    assert contains_coin("ethereum is rising", "BTC") is False
    assert contains_coin("look at bTc", "BTC") is True
    
def test_lead_lag() -> None:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="30min"),
        "price": [100, 101, 102, 103, 104],
        "sentiment": [0.1, 0.2, 0.3, 0.4, 0.5],
    })

    result = compute_lead_lag(df, lag_hours=1, lag_step_min=30, min_points=2)

    assert list(result.columns) == ["lag_seconds", "r", "p_value", "n"]
    assert len(result) == 5
