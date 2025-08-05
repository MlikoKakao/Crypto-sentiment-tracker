import os


DEFAULT_CURRENCY = "usd"
DEFAULT_DAYS = ["1", "7", "30", "90", "180", "365"]

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
COINGECKO_INTERVALS = {
    "short": "",
    "long": "daily" 
}

REDDIT_SUBREDDIT = "Cryptocurrency"
REDDIT_DEFAULT_QUERY = "bitcoin"

SENTIMENT_NEG_THRESHOLD = -0.05
SENTIMENT_POS_THRESHOLD = 0.05

DATA_DIR = "data/"
LOG_PATH = "logs/app.log"

def get_data_path(coin: str, filetype:str) -> str:
    return os.path.join(DATA_DIR,f"{coin.lower()}_{filetype}.csv")

COINS_UI_LABELS = ["Bitcoin","Ethereum"]
COINS_UI_TO_SYMBOL = {label: label.lower() for label in COINS_UI_LABELS}

ANALYZER_UI_LABELS = ["VADER", "TextBlob", "Twitter-RoBERTa"]
