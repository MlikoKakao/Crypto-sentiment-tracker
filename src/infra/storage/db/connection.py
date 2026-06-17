from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.app.settings import get_database_url


@lru_cache
def get_engine() -> Engine:
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )
