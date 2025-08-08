import json
from config.settings import MAPPING_FILE, CACHE_DIR
import os
import hashlib
import pandas as pd
import time
from pathlib import Path







def load_mapping() -> dict:
    if Path(MAPPING_FILE).exists():
        return json.loads(Path(MAPPING_FILE).read_text() or "{}")
    return {}

def save_mapping(mp: dict) -> None:
    Path(MAPPING_FILE).write_text(json.dumps(mp,indent=2))

def hash_settings(settings: dict) -> str:
    key = "|".join(f"{k}={settings[k]}" for k in sorted(settings))
    return hashlib.md5(key.encode()).hexdigest()

def get_cached_path(settings: dict) -> Path:
    h = hash_settings(settings)
    return Path(CACHE_DIR) / f"{h}.csv"

def load_cached_csv(settings:dict, parse_dates=None, freshness_minutes=None):
    path = get_cached_path(settings)
    
    if not path.exists():
        return None
    if freshness_minutes is not None:
        age_min = (time.time() - path.stat().st_mtime) / 60
        if age_min > freshness_minutes:
            return None
    return pd.read_csv(path, parse_dates=parse_dates)

def cache_csv(df: pd.DataFrame, settings:dict) -> Path:
    path = get_cached_path(settings)
    df.to_csv(path, index=False)
    mp = load_mapping()
    mp[hash_settings(settings)] = {
        "path": str(path),
        "rows": len(df),
        "updated": int(time.time()),
        "settings": settings
    }
    save_mapping(mp)
    return path