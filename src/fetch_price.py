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

    if response.status_code != 200:
        print("❌ API request failed:", response.status_code)
        print(response.text)
        return pd.DataFrame()

    data = response.json()

    if "prices" not in data:
        print("❌ 'prices' key not found in API response!")
        print(data)
        return pd.DataFrame()

    prices = []
    for timestamp, price in data["prices"]:
        dt = datetime.fromtimestamp(timestamp / 1000, utc)
        prices.append({"timestamp": dt, "price": price})

    df = pd.DataFrame(prices)
    return df
    
if __name__ == "__main__":
    df = get_price_history("bitcoin", days="max")
    df.to_csv("data/bitcoin_prices.csv", index=False)
    print(f"Saved {len(df)} price points for all available days")