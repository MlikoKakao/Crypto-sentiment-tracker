from pytz import utc
import praw
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.utils.helpers import save_csv, clean_text
import os
import logging

logger = logging.getLogger(__name__)


load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)




def fetch_reddit_posts(query="bitcoin",limit=1000, start_date=None, end_date=None):
    logging.info(f"Fetching Reddit posts with query='{query}', limit={limit}")
    posts = []
    for submission in reddit.subreddit("CryptoCurrency").search(query, limit=limit, sort="new"):
        
        timestamp = datetime.fromtimestamp(submission.created_utc, utc)
        if start_date and timestamp < start_date:
            continue
        if end_date and timestamp > end_date:
            continue
        text = submission.title + " " + submission.selftext
        posts.append({
            "timestamp": timestamp,
            "text": text,
            "url": f"https://www.reddit.com{submission.permalink}"
        })

    logging.info(f"Fetched {len(posts)} posts for query='{query}'")
    df = pd.DataFrame(posts)
    df["text"] = df["text"].apply(clean_text)
    return df

if __name__ == "__main__":
    now = datetime.now(utc)
    one_week_ago = now - timedelta(days=7)
    df = fetch_reddit_posts("bitcoin", limit=500, start_date=one_week_ago, end_date=now)
    save_csv(df, "data/bitcoin_posts.csv")