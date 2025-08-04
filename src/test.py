import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("data/bitcoin_prices.csv", parse_dates=['timestamp'])
df = df.sort_values('timestamp')

most_recent_price = df.iloc[-1] ["price"]
average_past_2_weeks = np.mean(df.iloc[:-14]["price"])
#print(average_past_2_weeks)
EMA = (most_recent_price * 0,2) + (average_past_2_weeks * 0,8)
#   Was, returns types instead of calculcating when most_rect+average_past works? The value is not getting pasted in but reference to data type?
#EMA = most_recent_price+average_past_2_weeks
print(EMA)
bar_colors = ['gray'] + [
    'green' if curr > prev else 'red'
    for prev, curr in zip(df['price'][:-1], df['price'][1:]) 
]

line_color = 'green' if df['price'].iloc[-1] > df['price'].iloc[0] else 'red'


plt.figure(figsize=(10,5))
plt.bar(df['timestamp'],df['price'], color = bar_colors)
plt.title("Bitcoin price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.plot(df['timestamp'],df['price'], color = line_color)
plt.ylim(bottom=df['price'].min()*0.98)
plt.xlim(df['timestamp'].min(),df['timestamp'].max())

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))


plt.tight_layout()
plt.show()