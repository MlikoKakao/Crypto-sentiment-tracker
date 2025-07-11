from pytz import utc
import praw
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)



def fetch_reddit_posts(query="bitcoin",limit=100):
    posts = []
    for submission in reddit.subreddit("CryptoCurrency").search(query, limit=limit):
        text = submission.title + " " + submission.selftext
        timestamp = datetime.fromtimestamp(submission.created_utc, utc)
        posts.append({
            "timestamp": timestamp,
            "text": text
        })
    return pd.DataFrame(posts)

if __name__ == "__main__":
    df = fetch_reddit_posts("bitcoin",100)
    df.to_csv("data/bitcoin_posts.csv", index=False)
    print(df.head())