from pytz import utc
import requests
import pandas as pd
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from src.utils.helpers import save_csv
from config.settings import DEMO_MODE, get_demo_data_path

logger = logging.getLogger(__name__)


def get_price_history(symbol: str = "bitcoin", days: str = "1"):
    if DEMO_MODE:
        return pd.read_csv(
            get_demo_data_path("price_history.csv"), parse_dates=["timestamp"]
        )
    load_dotenv()


def fetch_news_posts(config: AnalysisConfig) -> pd.DataFrame:
    logger.info(f"Attempting to fetch news for {config.coin}..")
    url = f"https://api.coingecko.com/api/v3/coins/{COIN_IDS[config.coin]}/market_chart/range?vs_currency=usd&from={int(config.start_date.timestamp())}&to={int(config.end_date.timestamp())}"

    response = requests.get(url)

    if response.status_code != 200:
        logger.error(
            f"Cryptopanic API failed: {response.status_code} - {response.reason}"
        )
        raise Exception("Failed to fetch news")

    API_COINGECKO_ID = os.getenv("API_COINGECKO_ID")  # type: ignore # for now not in use, but may need it later for higher rate limits
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": str(int(days))}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(f"API request failed: {response.status_code} – {response.text}")
        raise Exception(f"API requred failed: {response.status_code}")

    data = response.json()

    if "prices" not in data:
        logger.error("'prices' key not found in API response!")
        return pd.DataFrame()

    prices = []
    for timestamp, price in data["prices"]:
        dt = datetime.fromtimestamp(timestamp / 1000, utc)
        prices.append({"timestamp": dt, "price": price})

    df = pd.DataFrame(prices)
    return df


if __name__ == "__main__":
    df = get_price_history("bitcoin", days="max")
    save_csv(df, "data/bitcoin_prices.csv")
    logger.info("Saved price history to data/bitcoin_prices.csv")
    print(f"Saved {len(df)} price points for all available days")

