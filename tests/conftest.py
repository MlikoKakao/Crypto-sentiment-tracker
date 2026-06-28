from datetime import datetime

import pytest
from alembic import command
from alembic.config import Config

from src.app.dto import AnalysisConfig
from src.domain.market.dto import IndicatorConfig
from src.infra.storage.db.connection import get_engine

TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/crypto_test"


@pytest.fixture(autouse=True)
def postgres_test_db(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    get_engine.cache_clear()
    command.upgrade(Config("alembic.ini"), "head")

    engine = get_engine()

    yield

    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE sentiment CASCADE")
        conn.exec_driver_sql("TRUNCATE TABLE content_items CASCADE")
        conn.exec_driver_sql("TRUNCATE TABLE prices CASCADE")
        conn.exec_driver_sql("TRUNCATE TABLE signals CASCADE")


@pytest.fixture
def analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        coin="BTC",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 1, 1),
        analyzer="vader",
        sources=("reddit",),
        num_posts=10,
        subreddits=("bitcoin",),
    )


@pytest.fixture
def indicator_config() -> IndicatorConfig:
    return IndicatorConfig(
        coin="BTC",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 1, 1),
    )
