import logging
import requests
from requests.adapters import HTTPAdapter, Retry
from pytz import utc
import pandas as pd
from src.app.dto import AnalysisConfig
from datetime import datetime, timedelta
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

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    while current_start < config.end_date:
        current_end = min(
            current_start + timedelta(seconds=chunk_seconds), config.end_date
        )
        url = f"https://api.exchange.coinbase.com/products/{COINBASE_PRODUCTS[config.coin]}/candles"
        params: dict[str, str | int] = {
            "granularity": granularity,
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
        }

        try:
            response = session.get(url, params=params, timeout=10)
        except requests.exceptions.Timeout as e:
            logger.error("Price fetch failed, Coinbase took too long to respond")
            raise RuntimeError("Price fetch failed, Coinbase took too long to respond") from e
        except requests.exceptions.RequestException as e:
            logger.error("Price fetch failed while contacting Coinbase: %s", e)
            raise RuntimeError("Price fetch failed while contacting Coinbase") from e

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
