import os

DEFAULT_DAYS = ["1", "7", "10", "30", "90", "180", "365"]

COIN_SUBS = {
    "bitcoin": ["Bitcoin", "btc", "BitcoinMarkets"],
    "ethereum": ["ethereum", "ethtrader", "eth"],
    "monero": ["xmrtrader", "monero"],
}


DEMO_MODE = os.getenv("DEMO", "0") == "1"
