import os
from pathlib import Path

DEFAULT_CURRENCY = "usd"
DEFAULT_DAYS = ["1", "7", "30", "90", "180", "365"]

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
COINGECKO_INTERVALS = {
    "short": "",
    "long": "daily" 
}

DEFAULT_SUBS = ["CryptoCurrency","CryptoCurrencyTrading","CryptoMarkets"]
COIN_SUBS = {
    "bitcoin": ["Bitcoin", "btc", "BitcoinMarkets"],
    "ethereum": ["ethereum", "ethtrader", "eth"],
    "monero": ["xmrtrader", "monero"]
}

def subs_for_coin(coin: str) -> list[str]:
    additional = COIN_SUBS.get(coin,[])
    return list(dict.fromkeys(additional))

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
MAPPING_FILE = CACHE_DIR /  "cache_index.json"

for p in (DATA_DIR, LOG_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)



def get_data_path(coin: str, filetype:str) -> str:
    return os.path.join(DATA_DIR,f"{coin.lower()}_{filetype}.csv")

COINS_UI_LABELS = ["Bitcoin", "Ethereum", "Monero"]
COINS_UI_TO_SYMBOL = {label: label.lower() for label in COINS_UI_LABELS}

ANALYZER_UI_LABELS = ["VADER", "TextBlob", "Twitter-RoBERTa", "finBERT", "All"]
POSTS_KIND = ["All", "Reddit", "News"]
