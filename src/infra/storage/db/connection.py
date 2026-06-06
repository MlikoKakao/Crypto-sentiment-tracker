import sqlite3
from pathlib import Path
from src.app.settings import DATABASE_PATH


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    db_path = Path(db_path or DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
