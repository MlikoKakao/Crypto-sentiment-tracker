from pytz import utc
import requests
import pandas as pd
from datetime import datetime
import logging
from src.utils.helpers import save_csv
from src.app.dto import AnalysisConfig
from src.app.defaults import DEFAULT_CONFIG
from src.domain.market.coins import COIN_IDS

logger = logging.getLogger(__name__)


def get_price_history(config: AnalysisConfig) -> pd.DataFrame:

    logger.info(f"Attempting to fetch news for {config.coin}..")
    if config.coin not in COIN_IDS:
        logger.error(f"Unsupported coin: {config.coin}")
        raise ValueError(f"Unsupported coin: {config.coin}")
    
    url = f"https://api.coingecko.com/api/v3/coins/{COIN_IDS[config.coin]}/market_chart/range?vs_currency=usd&from={int(config.start_date.timestamp())}&to={int(config.end_date.timestamp())}"

    response = requests.get(url)

    if response.status_code != 200:
        logger.error(
            f"Coingecko API failed: {response.status_code} - {response.reason}"
        )
        raise Exception("Failed to fetch news")

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
    df = get_price_history(DEFAULT_CONFIG)
    save_csv(df, "data/tests/bitcoin_prices.csv")
    logger.info("Saved price history to data/tests/bitcoin_prices.csv")
    logger.debug(f"Saved {len(df)} price points for all available days")
