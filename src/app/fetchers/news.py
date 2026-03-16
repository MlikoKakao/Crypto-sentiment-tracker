from pytz import utc
import requests
import pandas as pd
from dotenv import load_dotenv
from src.app.defaults import DEFAULT_CONFIG
from src.utils.helpers import save_csv, clean_text
import os
import logging
from src.infra.storage.paths import get_demo_data_path
from src.app.dto import AnalysisConfig

logger = logging.getLogger(__name__)
load_dotenv()

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")


def fetch_news_posts(config: AnalysisConfig) -> pd.DataFrame:
    logger.info(f"Attempting to fetch news for {config.coin}..")
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "currencies": config.coin,  # Could be wrong - dk if they want BTC or bitcoin
        "kind": "news",
        "public": "true",
        "filter": "hot",
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(
            f"Cryptopanic API failed: {response.status_code} - {response.reason}"
        )
        raise Exception("Failed to fetch news")

    results = response.json().get("results", [])
    posts = []

    for item in results[: config.num_posts]:
        title = item.get("title", "")
        published_at = item.get("published_at", "")
        timestamp = pd.to_datetime(published_at, utc=True).tz_convert(utc)
        url = item.get("url", "")
        domain = item.get("domain", "")
        posts.append(
            {"timestamp": timestamp, "title": title, "source": domain, "url": url}
        )
    logger.info(f"Fetched {len(posts)} news posts for {config.coin}")
    df = pd.DataFrame(posts)
    df["text"] = df["title"].apply(clean_text)
    return df


def demo_news_scrape():
    return pd.read_csv(get_demo_data_path("news_posts.csv"), parse_dates=["timestamp"])


if __name__ == "__main__":
    df = fetch_news_posts(DEFAULT_CONFIG)
    save_csv(df, "data/tests/news_posts.csv")
    print(df.head())
