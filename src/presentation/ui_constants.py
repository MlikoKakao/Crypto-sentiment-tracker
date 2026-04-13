from src.app.dto import Analyzer, Source

COINS_UI_LABELS = ["Bitcoin", "Ethereum", "Monero"]
COIN_UI_TO_SYMBOL = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Monero": "XMR",
}

ANALYZER_UI_TO_LITERAL: dict[str, Analyzer] = {
    "VADER": "vader",
    "TextBlob": "textblob",
    "Twitter-RoBERTa": "twitter-roberta",
    "finBERT": "finbert",
    "All": "all",
}

SOURCE_UI_TO_LITERAL: dict[str, tuple[Source, ...]] = {
    "All": ("reddit", "youtube", "news"),
    "Reddit": ("reddit",),
    "Youtube": ("youtube",),
    "News": ("news",),
}

COIN_SUBS: dict[str, list[str]] = {
    "BTC": ["Bitcoin", "btc", "BitcoinMarkets"],
    "ETH": ["ethereum", "ethtrader", "eth"],
    "XMR": ["xmrtrader", "monero"],
}
