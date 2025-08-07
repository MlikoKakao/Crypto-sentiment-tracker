import json
from config.settings import MAPPING_FILE, CACHE_DIR
import os
import hashlib
import pandas as pd
import time






def hash_settings(settings: dict) -> str:
    key = "|".join(f"{k}={v}" for k,v in sorted(settings.items()))
    return hashlib.md5(key.encode()).hexdigest()

def load_mapping():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r") as f:
            return json.load(f)
    return {}

def save_mapping(mapping):
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

def cache_csv(df, settings:dict):
    mapping = load_mapping()
    cache_hash = hash_settings(settings)
    filename = f"{cache_hash}.csv"
    path = os.path.join(CACHE_DIR, filename)

    df.to_csv(path, index=False)
    mapping[cache_hash] = settings
    save_mapping(mapping)

    return path

def load_cached_csv(settings:dict, freshness_minutes=10):
    mapping = load_mapping()
    cache_hash = hash_settings(settings)

    if cache_hash in mapping:
        path = os.path.join(CACHE_DIR, f"{cache_hash}.csv")
        if os.path.exists(path):
            modified_time = os.path.getmtime(path)
            age_minutes = (time.time() - modified_time) / 60
            if age_minutes < freshness_minutes:
                return pd.read_csv(path)
    return None