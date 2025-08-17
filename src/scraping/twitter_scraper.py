from __future__ import annotations
import os, time, math
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd

from src.utils.cache import get_cached_path, load_cached_csv


_COIN_TERMS = {
    "btc": ["bitcoin", "btc", "$btc", "#bitcoin", "#btc"],
    "eth": ["ethereum", "eth", "$eth", "#ethereum", "#eth"],
    "xmr": ["monero", "xmr", "$xmr", "#monero", "#xmr"]
}

def _infer_terms(coin: str) -> List[str]:
    key = (coin or "").lower()
    return _COIN_TERMS.get(key, [coin, f"${coin.upper()}", f"#{coin.lower()}"])

#def build_x_query(coin: str, extra: Optional[str] = None)