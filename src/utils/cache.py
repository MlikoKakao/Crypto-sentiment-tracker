import json
import os
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional
import stat
import shutil
from contextlib import contextmanager
import pandas as pd
import streamlit as st
import logging
logger = logging.getLogger(__name__)

from config.cache_schema import canonicalize_settings as _canonicalize_settings
from config.settings import MAPPING_FILE, CACHE_DIR, DEMO_MODE

# Do those paths exist? If not create Path objects (keep original config names untouched)
CACHE_DIR_PATH = Path(CACHE_DIR)
MAPPING_FILE_PATH = Path(MAPPING_FILE)
CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
MAPPING_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
# To avoid multiple writes to json > throws windows error
LOCK_FILE = MAPPING_FILE_PATH.with_suffix(".lock")

@contextmanager
def _file_lock(lock_path: Path, timeout: float = 5.0, poll: float = 0.05):
    start = time.time()
    # use sentinel -1 to avoid Optional[int] type confusion in static checker
    fd: int = -1
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            # write pid for easier debugging of stale locks
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            except Exception:
                pass
            break # if acquired
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Could not acquire lock: {lock_path}")
            time.sleep(poll)
    try:
        yield
    finally:
        try:
            if fd >= 0:
                os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass

#Makes all formatting centralized instead of always having to convert
def _normalize(obj: Any) -> Any:
    import datetime as dt
    np: Any = None
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

def hash_settings(settings: Dict[str, Any]) -> str:
    canon = _canonicalize_settings(settings) #1. Checks if settings have all required keys
    blob = _settings_blob(canon) #2. Changes dict into json 
    return hashlib.md5(blob.encode("utf-8")).hexdigest() #3. Returns hashed json

#Checks if settings already exist
def load_mapping() -> Dict[str, Any]:
    if not MAPPING_FILE_PATH.exists():
        return {}
    try:
        txt = MAPPING_FILE_PATH.read_text(encoding="utf-8")
        return json.loads(txt) if txt.strip() else {}
    except json.JSONDecodeError:
        return {}

#Takes in dictionary of settings puts it in MAPPING_FILE(cache_index)
def save_mapping(mp: Dict[str, Any]) -> None:
    tmp = MAPPING_FILE_PATH.with_name(f"{MAPPING_FILE_PATH.stem}.{os.getpid()}.{int(time.time()*1000)}.tmp")
    tmp.write_text(json.dumps(mp, indent=2, ensure_ascii=False), encoding="utf-8")
    with _file_lock(LOCK_FILE):
        for _ in range(10):
            try:
                tmp.replace(MAPPING_FILE_PATH)
                break
            except PermissionError:
                time.sleep(0.05)
        else:
            MAPPING_FILE_PATH.write_text(json.dumps(mp, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

#Checks last modified time of file
def _is_fresh(path: Path, freshness_minutes: Optional[int]) -> bool:
    if freshness_minutes is None:
        return True
    age_min = (time.time() - path.stat().st_mtime) / 60
    return age_min <= freshness_minutes #returns T/F if correct

#Returns Path to sent settings, it builds the filename from settings hash
def get_cached_path(settings: Dict[str, Any]) -> Path:
    return CACHE_DIR_PATH / f"{hash_settings(settings)}.csv"

#1. Takes in settings
def load_cached_csv(settings: Dict[str, Any], parse_dates: Optional[Any] = None, freshness_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    if DEMO_MODE:
        return None
    settings = _canonicalize_settings(settings)
    path = get_cached_path(settings)

    if not path.exists():
        logger.info("CACHE MISS %s -> %s", settings, path.name)
        return None
    if not _is_fresh(path, freshness_minutes):
        logger.info("CACHE STALE")
        return None
    logger.info("CACHE HIT %s", path.name)
    return pd.read_csv(path, parse_dates=parse_dates)

#Takes in DataFrame and settings, returns filesystem path of CSV it just wrote
def cache_csv(df: pd.DataFrame, settings: Dict[str, Any]) -> Optional[Path]:
    if DEMO_MODE:
        return None
    settings = _canonicalize_settings(settings)
    path = get_cached_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    mp = load_mapping()
    key = path.stem
    mp[key] = {
        "path": str(path),
        "rows": int(len(df)),
        "updated": int(time.time()),
        "settings": _normalize(settings)
    }
    save_mapping(mp)
    return path

def _remove_readonly_and_retry(func: Any, path: str, exc_info: Any) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def clear_cache_dir() -> Dict[str, int]:
    bytes_freed = 0
    files_count = 0
    with _file_lock(LOCK_FILE):
        if CACHE_DIR_PATH.exists():
            for p in CACHE_DIR_PATH.rglob("*"):
                if p.is_file():
                    try:
                        bytes_freed += p.stat().st_size
                        files_count += 1
                    except Exception:
                        pass
            # pyright/stubs may disagree about rmtree's kwarg name; ignore call-arg typing here
            shutil.rmtree(CACHE_DIR_PATH, onerror=_remove_readonly_and_retry)  # type: ignore[call-arg]

        CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
        MAPPING_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MAPPING_FILE_PATH.write_text("{}", encoding="utf-8")
    try:
        # some Streamlit versions / environments may not expose cache_data
        st.cache_data.clear()
    except Exception:
        logger.debug("streamlit cache clear skipped or unavailable")
    return {"files_removed": files_count, "bytes_freed": bytes_freed}

def day_str(ts: Any) -> str:
    t = pd.Timestamp(ts)
    # tz_convert only on tz-aware timestamps
    if getattr(t, "tz", None) is not None:
        try:
            t = t.tz_convert(None)
        except Exception:
            pass
    return t.normalize().strftime("%Y-%m-%d")