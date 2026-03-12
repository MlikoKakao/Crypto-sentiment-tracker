import re
from src.domain.market.coins import COIN_TERMS


def contains_coin(text: str, coin: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return any(word in COIN_TERMS[coin] for word in words)
