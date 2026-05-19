import pandas as pd
from datetime import datetime
import os
import logging
from typing import Any
from pathlib import Path
from src.presentation.config.settings import DEMO_MODE

logger = logging.getLogger(__name__)


# CSV HANDLING
def load_csv(filepath: Path | str, parse_dates: Any = None):
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise ValueError(f"File not found: {filepath}")
    logger.info(f"Loaded CSV from: {filepath}")
    return pd.read_csv(filepath, parse_dates=parse_dates)


def save_csv(df: pd.DataFrame, filepath: Path | str):
    if DEMO_MODE:
        return
    df.to_csv(filepath, index=False)
    logger.debug(f"Saved CSV to: {filepath} ({len(df)} rows)")


def normalize_timestamp_column(
    df: pd.DataFrame,
    column: str = "timestamp",
    drop_invalid: bool = False,
) -> pd.DataFrame:
    df[column] = pd.to_datetime(
        df[column],
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None)
    if drop_invalid:
        df = df.dropna(subset=[column])
    return df


def is_date_correct(start_date: datetime, end_date: datetime) -> bool:
    if end_date <= start_date:
        return False
    else:
        return True


# Text cleanup for sentiment analysis
def clean_text(text: str) -> str:
    return str(text).lower().strip()
