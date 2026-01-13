import pandas as pd
import os
import time
import logging
from datetime import datetime
import hashlib
from typing import Any
from pathlib import Path
from config.settings import DEMO_MODE
logger = logging.getLogger(__name__)


#CSV HANDLING
def load_csv(filepath: Path | str, parse_dates: Any = None):
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise ValueError(f"File not found: {filepath}")
    logger.info(f"Loaded CSV from: {filepath}")
    return pd.read_csv(filepath,parse_dates=parse_dates)

def save_csv(df: pd.DataFrame, filepath: Path | str):
    if DEMO_MODE:
        return
    df.to_csv(filepath, index=False)
    logger.debug(f"Saved CSV to: {filepath} ({len(df)} rows)")

def file_sha1(p:str | os.PathLike[Any]) -> str: #Returns content fingerprint of a file
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        raise IsADirectoryError(p)

    # Fast path (Py 3.11+)
    from typing import Any as _Any
    file_digest: _Any = None
    try:
        from hashlib import file_digest
    except ImportError:
        file_digest = None

    if file_digest is not None:
        with p.open("rb") as f:
            return file_digest(f, hashlib.sha1).hexdigest()

    # Fallback for <=3.10
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() #Automatically reads 1MiB chunks > feeds them to hasher, returns 40-char hex digest

#DEPRECATED CACHING
def is_file_fresh(path: Path | str,freshness_minutes: int = 10):
    if not os.path.exists(path):
        return False
    modified_time = os.path.getmtime(path)
    age_minutes = (time.time() - modified_time) / 60
    return age_minutes < freshness_minutes

#TEXT CLEANUP
def clean_text(text: str) -> str:
    return str(text).lower().strip()

#FILTER DF BY TIME
def filter_date_range(df: pd.DataFrame, start_date: datetime, end_date: datetime, date_column: str = "timestamp") -> pd.DataFrame:
    return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

#Convert currency names to symbols(BTC,ETH,..)
def map_to_cryptopanic_symbol(symbol: str) -> str:
    symbol_map = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "monero": "XMR"
    }
    return symbol_map.get(symbol.lower(),symbol.upper())