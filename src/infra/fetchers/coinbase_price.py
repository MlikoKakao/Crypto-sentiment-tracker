import logging
import requests
from pytz import utc
import pandas as pd
from src.app.dto import AnalysisConfig
from datetime import datetime, timedelta
from src.app.defaults import default_config
from src.shared.helpers import save_csv
from src.infra.fetchers.price import get_price_history

COINBASE_PRODUCTS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

logger = logging.getLogger(__name__)


def get_coinbase_price_history(config: AnalysisConfig) -> pd.DataFrame:
    logger.info("Checking cache for price points..")

    if config.coin not in COINBASE_PRODUCTS:
        df = get_price_history(config)
        return df

    prices = []
    granularity = 300
    max_candles = 300
    chunk_seconds = granularity * max_candles

    current_start = config.start_date

    while current_start < config.end_date:
        current_end = min(
            current_start + timedelta(seconds=chunk_seconds), config.end_date
        )
        url = f"https://api.exchange.coinbase.com/products/{COINBASE_PRODUCTS[config.coin]}/candles"
        params = {
            "granularity": granularity,
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
        }

        try:
            response = requests.get(url, params=params, timeout=10)
        except requests.exceptions.Timeout:
            logger.error("Price fetch failed, Coinbase took too long to respond")
            raise Exception("Price fetch failed, Coinbase took too long to respond")
            # Eventually fallback to coingecko - for XMR

        if response.status_code != 200:
            logger.error(
                f"Coinbase API failed: {response.status_code} - {response.reason}"
            )
            raise Exception(
                f"Coinbase API failed: {response.status_code} - {response.reason}"
            )

        data = response.json()

        for row in data:
            dt = datetime.fromtimestamp(row[0], utc)
            close = row[4]
            prices.append({"timestamp": dt, "price": close})

        current_start = current_end

    df = pd.DataFrame(prices)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"])
    return df


if __name__ == "__main__":
    df = get_coinbase_price_history(default_config())
    save_csv(df, "data/tests/coinbase_price.csv")
