import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#df = pd.read_csv("data/bitcoin_posts.csv")

#sample = df.sample(20, random_state=42)
#sample[["timestamp","text"]].to_csv("data/benchmark_raw.csv",index= False)

from sentiment.analyzer import vader_analyze, textblob_analyze, roberta_analyze
import pandas as pd

df = pd.read_csv("data/benchmark_raw.csv")

# Choose one model to test
df["vader"] = df["text"].apply(vader_analyze)
df["textblob"] = df["text"].apply(textblob_analyze)
df["roberta"] = df["text"].apply(roberta_analyze)

# Optional: convert model output to class label
def convert(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_label"] = df["vader"].apply(convert)
df["textblob_label"] = df["textblob"].apply(convert)
df["roberta_label"] = df["roberta"].apply(convert)

# Save for analysis
df.to_csv("data/benchmark_evaluation.csv", index=False)













""""
df = df.sort_values('timestamp')

most_recent_price = df.iloc[-1] ["price"]
average_past_2_weeks = np.mean(df.iloc[:-14]["price"])
#print(average_past_2_weeks)
EMA = (most_recent_price * 0.2) + (average_past_2_weeks * (1-0.2))
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
"""