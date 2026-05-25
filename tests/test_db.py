from datetime import datetime

import pandas as pd

from src.app.dto import AnalysisConfig
from src.infra.storage.db.price_repository import load_price_df, save_price_df
from src.infra.storage.db.schema import init_db


def test_price_repository_saves_and_loads_rows(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.infra.storage.db.connection.DB_PATH", db_path)
    init_db()

    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1h"),
            "price": [100.0, 101.5],
        }
    )
    save_price_df(prices, "btc")

    config = AnalysisConfig(
        coin="BTC",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 1, 1),
        analyzer="vader",
        sources=("reddit",),
        num_posts=10,
        subreddits=("bitcoin",),
    )
    result = load_price_df(config)

    assert len(result) == 2
    assert result["price"].tolist() == [100.0, 101.5]
