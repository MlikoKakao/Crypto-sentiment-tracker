import os
from pathlib import Path

DEFAULT_CURRENCY = "usd"
DEFAULT_DAYS = ["1", "7", "30", "90", "180", "365"]

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
COINGECKO_INTERVALS = {
    "short": "",
    "long": "daily" 
}

DEFAULT_SUBS = ["CryptoCurrency","Bitcoin","CryptoMarkets","BitcoinMarkets"]
REDDIT_DEFAULT_QUERY = "bitcoin"

SENTIMENT_NEG_THRESHOLD = -0.05
SENTIMENT_POS_THRESHOLD = 0.05

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
MAPPING_FILE = CACHE_DIR /  "cache_index.json"

for p in (DATA_DIR, LOG_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)



def get_data_path(coin: str, filetype:str) -> str:
    return os.path.join(DATA_DIR,f"{coin.lower()}_{filetype}.csv")

COINS_UI_LABELS = ["Bitcoin","Ethereum"]
COINS_UI_TO_SYMBOL = {label: label.lower() for label in COINS_UI_LABELS}

ANALYZER_UI_LABELS = ["VADER", "TextBlob", "Twitter-RoBERTa", "finBERT", "All"]
POSTS_KIND = ["All", "Reddit", "News"]
