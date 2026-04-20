from pytz import utc
import requests
import pandas as pd
from datetime import datetime
import logging
from src.infra.storage.logging_config import configure_logging
from src.app.dto import AnalysisConfig
from src.app.defaults import DEFAULT_CONFIG
from src.domain.market.coins import COIN_IDS
from src.infra.storage.db.price_repository import save_price_df, load_price_df, has_price_coverage

logger = logging.getLogger(__name__)


def get_price_history(config: AnalysisConfig) -> pd.DataFrame:
    logger.info("Checking cache for price points..")
    df = load_price_df(config)
    if has_price_coverage(config, df):
        return df 


    logger.info(f"Attempting to fetch price for {config.coin}..")
    if config.coin not in COIN_IDS:
        logger.error(f"Unsupported coin: {config.coin}")
        raise ValueError(f"Unsupported coin: {config.coin}")
    
    url = f"https://api.coingecko.com/api/v3/coins/{COIN_IDS[config.coin]}/market_chart/range?vs_currency=usd&from={int(config.start_date.timestamp())}&to={int(config.end_date.timestamp())}"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        logger.error("Coingecko API request timed out")
        raise Exception("Failed to fetch price: request timed out")
    
    
    if response.status_code != 200:
        logger.error(
            f"Coingecko API failed: {response.status_code} - {response.reason}"
        )
        raise Exception("Failed to fetch price")

    data = response.json()

    if "prices" not in data:
        logger.error("'prices' key not found in API response!")
        return pd.DataFrame()

    prices = []
    for timestamp, price in data["prices"]:
        dt = datetime.fromtimestamp(timestamp / 1000, utc)
        prices.append({"timestamp": dt, "price": price})

    df = pd.DataFrame(prices)
    save_price_df(df, config.coin)
    return load_price_df(config)


if __name__ == "__main__":
    configure_logging()
    df = get_price_history(DEFAULT_CONFIG)
    save_price_df(df, DEFAULT_CONFIG.coin)
    logger.info("Saved price history to app.db")
    logger.debug(f"Saved {len(df)} price points for all available days")
