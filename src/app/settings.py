import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_database_path() -> Path:
    return Path(os.getenv("DATABASE_PATH", "data/app.db"))
