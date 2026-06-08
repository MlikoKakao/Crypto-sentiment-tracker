from datetime import datetime
from pathlib import Path

import pytest

from src.app.dto import AnalysisConfig
from src.domain.market.dto import IndicatorConfig
from src.infra.storage.db.schema import init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


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
