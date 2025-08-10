import json
import os
import hashlib
from hashlib import file_digest
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.cache_schema import canonicalize_settings
from config.settings import MAPPING_FILE, CACHE_DIR

#Do those paths exist? If not creates
CACHE_DIR = Path(CACHE_DIR)
MAPPING_FILE = Path(MAPPING_FILE)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_FILE.parent.mkdir(parents=True,exist_ok=True)

#Makes all formatting centralized instead of always having to convert
def _normalize(obj: Any) -> Any:
    import datetime as dt
    try:
        import numpy as np
    except ImportError:
        np = None
    
    if isinstance(obj,dict):
        return {k: _normalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, set):
        return [_normalize(x) for x in sorted(obj)]
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    return obj

#Returns dictionary as json
def _settings_blob(settings: Dict[str, Any]) -> str:
    norm = _normalize(settings)
    return json.dumps(norm, separators=(",",":"), ensure_ascii=False)

def hash_settings(settings:dict) -> str:
    canon = canonicalize_settings(settings) #1. Checks if settings have all required keys
    blob = _settings_blob(canon) #2. Changes dict into json 
    return hashlib.md5(blob.encode("utf-8")).hexdigest() #3. Returns hashed json

#Checks if settings already exist
def load_mapping() -> dict:
    if not MAPPING_FILE.exists(): #Checks if specific settings are in cache_index
        return {}  #if not returns nothing
    try:
        txt = MAPPING_FILE.read_text(encoding="utf-8")
        return json.loads(txt) if txt.strip() else {} #if yes returns it, strip to handle empty files, otherwise would crash)
    except json.JSONDecodeError:
        return {}

#Takes in dictionary of settings puts it in MAPPING_FILE(cache_index)
def save_mapping(mp: dict) -> None:
    tmp = MAPPING_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(mp, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MAPPING_FILE) #puts it in MAPPING_FILE(cache_index)

#Checks last modified time of file
def _is_fresh(path: Path, freshness_minutes: Optional[int]) -> bool:
    if freshness_minutes is None:
        return True
    age_min = (time.time() - path.stat().st_mtime) / 60
    return age_min <= freshness_minutes #returns minutes since last edit

#Returns Path to sent settings, it builds the filename from settings hash
def get_cached_path(settings: dict) -> Path:
    return Path(CACHE_DIR) / f"{hash_settings(settings)}.csv"

#1. Takes in settings
def load_cached_csv(settings:dict, parse_dates=None, freshness_minutes: Optional[int] = None):
    settings = canonicalize_settings(settings) #2. Check for required keys
    path = get_cached_path(settings) #3. Get path to settings
    
    if not path.exists() or not _is_fresh(path, freshness_minutes):
        return None #4. If doesnt exist or is too old > says theres nothing good cached
    return pd.read_csv(path, parse_dates=parse_dates) #5. If is return cached

#Takes in DataFrame and settings, returns filesystem path of CSV it just wrote
def cache_csv(df: pd.DataFrame, settings:dict) -> Path:
    settings = canonicalize_settings(settings) #Check if all keys
    path = get_cached_path(settings) #Get path to settings
    df.to_csv(path, index=False) #Converts df to CSV

    mp = load_mapping() #Checks if settings exist
    key = path.stem #.stem is filename without .csv, used for json mapping key
    mp[key] = {
        "path": str(path),
        "rows": int(len(df)),
        "updated": int(time.time()),
        "settings": _normalize(settings)
    }
    save_mapping(mp) #saves properties of csv and it's settings
    return path


