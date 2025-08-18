from pytz import utc
import praw
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.utils.helpers import save_csv, clean_text
import os
import logging
import re

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




def fetch_reddit_posts(query="(btc OR bitcoin)"
                       ,limit=1000,
                       start_date=None,
                       end_date=None,
                       subreddits=("CryptoCurrency","Bitcoin","CryptoMarkets","BitcoinMarkets")):
    
    terms = [t.strip(" ()") for t in re.split(" OR ", query, flags=re.I)]
    kw = re.compile("|".join(map(re.escape, terms)), flags = re.I)

    logger.info(f"Fetching Reddit posts with query='{query}', limit={limit}, subs={subreddits}")
    posts, seen = [], set()
    
    for sub in sorted(set(subreddits)):
        for submission in reddit.subreddit(sub).new(limit=None):
            ts = datetime.fromtimestamp(submission.created_utc, tz=utc)
            
            if end_date and ts > end_date:
                continue
            if start_date and ts < start_date:
                break

            #Filter by keywords locally
            text = f"{submission.title or ''} {submission.selftext or ''}"
            if not kw.search(text):
                continue 

            sid = submission.id
            if sid in seen:
                continue
            seen.add(sid)

            posts.append({
                "timestamp": ts,
                "text": text,
                "url": getattr(submission,"url",None) or f"https://www.redd.it/{submission.permalink}",
                "score": submission.score,
                "num_comments": submission.num_comments,
                "id": sid,
                "source": "reddit",
                "subreddit": submission.subreddit.display_name
            })

            if len(posts) >= limit:
                break
        if len(posts)>= limit:
            break

    logger.info(f"Fetched {len(posts)} posts for query='{query}' from {subreddits}")
    df = pd.DataFrame(posts)
    if not df.empty:
        df["text"] = df["text"].apply(clean_text)
    return df

if __name__ == "__main__":
    now = datetime.now(utc)
    one_week_ago = now - timedelta(days=7)
    df = fetch_reddit_posts("bitcoin", limit=500, start_date=one_week_ago, end_date=now)
    save_csv(df, "data/bitcoin_posts.csv")