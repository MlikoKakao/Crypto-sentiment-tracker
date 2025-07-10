from pytz import utc
import praw
import requests
import pandas as pd
from datetime import datetime

REDDIT_CLIENT_ID = "bVHgQ5ub15vRqmP1Rh76_g"
REDDIT_CLIENT_SECRET = "nzuccpDYcEcei4vmb3Y2wFPCO_1Cxw"
REDDIT_USER_AGENT = "crypto-tracker by u/MlekoKakao"

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