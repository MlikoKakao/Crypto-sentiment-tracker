from pytz import utc
import praw
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(
    filename='logs/app.log',
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)


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
            "text": text
        })

    logging.info(f"Fetched {len(posts)} posts for query='{query}'")
    return pd.DataFrame(posts)