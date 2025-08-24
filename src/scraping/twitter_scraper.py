from __future__ import annotations
import os, time, math
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd
from apify_client import ApifyClient
from dotenv import load_dotenv
from src.utils.cache import get_cached_path, load_cached_csv
from config.settings import DEMO_MODE, get_demo_data_path

load_dotenv()

_ALIAS = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "monero": "xmr"
}

_COIN_TERMS = {
    "btc": ["bitcoin", "btc", "$btc", "#bitcoin", "#btc"],
    "eth": ["ethereum", "eth", "$eth", "#ethereum", "#eth"],
    "xmr": ["monero", "xmr", "$xmr", "#monero", "#xmr"]
}

actor_id = "xtdata/twitter-x-scraper"

def _to_date(val) -> Optional[str]:
    if val is None or val == "":
        return None
    ts = pd.to_datetime(val, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")

def fetch_twitter_posts(coin: str,
                        limit: int = 500,
                        lang: str = "en",
                        sort: str = "Latest",
                        start: Optional[str] = None,
                        end: Optional[str] = None
                        ) -> pd.DataFrame:
    if DEMO_MODE:
        return pd.read_csv(get_demo_data_path("twitter_posts.csv"), parse_dates=["timestamp"])
    client = ApifyClient(os.getenv("APIFY_API"))
    if not client:
        raise RuntimeError("Set APIFY_API in .env file")
    
    terms: List[str] = _COIN_TERMS.get(coin.lower(), [coin])
    or_query = "(" + " OR ".join(sorted(set(terms))) + ")"

    input_data = {
        "searchTerms": [or_query],
        "tweetLanguage": lang,
        "maxItems": limit,
        "sort": sort,
        "start": _to_date(start),
        "end": _to_date(end),
        "includeSearchTerms": False
    }

    run = client.actor(actor_id).call(run_input=input_data)
    items = client.dataset(run["defaultDatasetId"]).list_items(limit=limit).items
    
    rows = []
    for it in items:
        author_obj = it.get("author") or {}
        author = author_obj.get("screen_name") or author_obj.get("username") or ""
        url = it.get("url") or it.get("twitterUrl")
        rows.append({
            "id": it.get("id") or it.get("id_str"),
            "timestamp": it.get("created_at"),
            "text": it.get("full_text") or it.get("text") or "",
            "source": "twitter",
            "url": url,
            "author": author,
            "like_count": it.get("favorite_count", 0),
            "retweet_count": it.get("retweet_count", 0),
            "reply_count": it.get("reply_count", 0),
            "quote_count": it.get("quote_count", 0),
            "lang": it.get("lang"),
            "coin": coin.lower()
        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df = df.sort_values("timestamp").drop_duplicates(subset=["id"])

    cols = ["timestamp", "text", "source", "url", "author",
            "like_count", "retweet_count", "reply_count",
            "quote_count", "lang", "coin", "id"]
    return df[cols]