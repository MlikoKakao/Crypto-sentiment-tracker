import sqlite3
from pathlib import Path

DB_PATH = Path("data/app.db")

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)