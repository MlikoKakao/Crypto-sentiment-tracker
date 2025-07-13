import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/bitcoin_prices.csv")

plt.figure(figsize=(10,5))
btc_prices = pd.DataFrame(df['price'],
                          index=pd.date_range("2024-06-25", periods=30, freq="d"
                          ))
plt.bar(df['timestamp'],df['price'],color='green')
plt.title("Bitcoin price")
time = np.arange('2024-06-25', '2025-06-25', dtype='datetime64[D]')
plt.plot(df['timestamp'],df['price'])
plt.xlabel("Time")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()