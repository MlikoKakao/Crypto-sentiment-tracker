from pytz import utc
import requests
import pandas as pd
from datetime import datetime
import sys

def get_price_history(symbol="bitcoin", days="1"):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days
    }
    response = requests.get(url, params=params)
    data = response.json()

    prices = []
    for timestamp, price in data["prices"]:
        dt = datetime.fromtimestamp(timestamp / 1000, utc)
        prices.append({"timestamp": dt, "price": price})

    df = pd.DataFrame(prices)
    return df
    
if __name__ == "__main__":
    days = sys.argv[1] if len(sys.argv) > 1 else "90"
    df = get_price_history("bitcoin", days=days)
    df.to_csv("data/bitcoin_prices.csv", index=False)
    print(f"Saved {len(df)} price points for past {days} days")