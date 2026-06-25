from pathlib import Path

from sqlalchemy import text

from src.infra.storage.db.connection import get_engine


def init_db() -> None:
    migration_path = Path(__file__).parent / "migrations" / "001_initial_schema.sql"
    sql = migration_path.read_text()

    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(sql))