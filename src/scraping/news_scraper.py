from pytz import utc
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.utils.helpers import save_csv, clean_text
import os
import logging
from config.settings import DEMO_MODE, get_demo_data_path

logger = logging.getLogger(__name__)
load_dotenv()

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
def fetch_news_posts(currency:str = "BTC", limit:int = 100):
    if DEMO_MODE:
        return pd.read_csv(get_demo_data_path("news_posts.csv"), parse_dates=["timestamp"])
    logger.info(f"Attempting to fetch news for {currency}..")
    url = "https://cryptopanic.com/api/v1/posts/"
    params={
        "auth_token": CRYPTOPANIC_API_KEY,
        "currencies": currency,
        "kind": "news",
        "public": "true",
        "filter": "hot"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(f"Cryptopanic API failed: {response.status_code} - {response.reason}")
        raise Exception("Failed to fetch news")
    
    results = response.json().get("results",[])
    posts = []

    for item in results[:limit]:
        title = item.get("title","")
        published_at = item.get("published_at","")
        timestamp = pd.to_datetime(published_at, utc=True).tz_convert(utc)
        url = item.get("url","")
        domain = item.get("domain","")
        posts.append({
            "timestamp": timestamp,
            "title": title,
            "source": domain,
            "url": url
        })
    logger.info(f"Fetched {len(posts)} news posts for {currency}")
    df = pd.DataFrame(posts)
    df["text"] = df["title"].apply(clean_text)
    return df

if __name__ == "__main__":
    now = datetime.now(utc)
    one_week_ago = now - timedelta(days=7)
    df = fetch_news_posts("BTC", limit=50)
    save_csv(df, "data/news_posts.csv")
    print(df.head())