import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


#CSV HANDLING
def load_csv(filepath, parse_dates=None):
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise ValueError(f"File not found: {filepath}")
    logger.info(f"Loaded CSV from: {filepath}")
    return pd.read_csv(filepath,parse_dates=parse_dates)

def save_csv(df, filepath):
    df.to_csv(filepath, index=False)
    logger.debug(f"Saved CSV to: {filepath} ({len(df)} rows)")

#CACHING
def is_file_fresh(path,freshness_minutes=10):
    if not os.path.exists(path):
        return False
    modified_time = os.path.getmtime(path)
    age_minutes = (time.time() - modified_time) / 60
    return age_minutes < freshness_minutes


#TEXT CLEANUP
def clean_text(text):
    return str(text).lower().strip()

#FILTER DF BY TIME
def filter_date_range(df, start_date, end_date, date_column = "timestamp"):
    return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

#Convert currency names to symbols(BTC,ETH,..)
def map_to_cryptopanic_symbol(symbol):
    symbol_map = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "monero": "XMR"
    }
    return symbol_map.get(symbol.lower(),symbol.upper())
