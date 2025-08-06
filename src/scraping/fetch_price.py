from pytz import utc
import requests
import pandas as pd
from datetime import datetime
import sys
import logging
import os
from dotenv import load_dotenv
from src.utils.helpers import save_csv

logger = logging.getLogger(__name__)


def get_price_history(symbol="bitcoin", days="1"):
    load_dotenv()
    API_COINGECKO_ID = os.getenv("API_COINGECKO_ID")
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        logging.error(f"API request failed: {response.status_code} – {response.text}")
        raise Exception(f"API requred failed: {response.status_code}")

    data = response.json()

    if "prices" not in data:
        logging.error("'prices' key not found in API response!")
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
    logging.info("Saved price history to data/bitcoin_prices.csv")
    print(f"Saved {len(df)} price points for all available days")