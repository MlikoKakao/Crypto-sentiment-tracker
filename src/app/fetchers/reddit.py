from pytz import utc
from praw import Reddit  # type: ignore no stub file
from praw.models import Submission, Subreddit
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.app.dto import AnalysisConfig
from src.utils.helpers import save_csv, clean_text
import os
import logging
from typing import Optional
from config.settings import get_demo_data_path
from src.domain.market.coins import COIN_TERMS
from src.app.defaults import DEMO_CONFIG

logger = logging.getLogger(__name__)


load_dotenv()


def get_reddit_client() -> Reddit:
    return Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )


def fetch_reddit_posts(config: AnalysisConfig) -> pd.DataFrame:
    logger.info(
        f"Fetching Reddit posts with query='{config.coin}', limit={config.num_posts}, subs={config.subreddits}"
    )
    posts = list[dict] = []
    seen = set[str] = set()

    reddit = get_reddit_client()
    subs = sorted(set(config.subreddits))
    terms = COIN_TERMS[config.coin]

    for sub in subs:
        for submission in reddit.subreddit(sub).new(limit=config.num_posts):
            time_posted = datetime.fromtimestamp(submission.created_utc, tz=utc)

            if config.end_date and time_posted > config.end_date:
                continue
            if config.start_date and time_posted < config.start_date:
                break

            # Filter by keywords locally
            text = f"{submission.title or ''} {submission.selftext or ''}"
            text_lower = text.lower()

            if not any(term in text_lower for term in terms):
                continue

            post_id = submission.id
            if post_id in seen or submission.stickied:
                continue
            seen.add(post_id)

            posts.append(
                {
                    "timestamp": time_posted,
                    "text": text,
                    "url": getattr(submission, "url", None)
                    or f"https://www.redd.it/{submission.permalink}",
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "id": post_id,
                    "source": "reddit",
                    "subreddit": submission.subreddit.display_name,
                    "is_original": submission.is_original_content,
                }
            )

            if len(posts) >= config.num_posts:
                break
        if len(posts) >= config.num_posts:
            break

    logger.info(
        f"Fetched {len(posts)} posts for query='{config.coin}' from {config.subreddits}"
    )
    df = pd.DataFrame(posts)
    if not df.empty:
        df["text"] = df["text"].apply(clean_text)
    return df


def demo_reddit_scrape():
    return pd.read_csv(
        get_demo_data_path("reddit_posts.csv"), parse_dates=["timestamp"]
    )


if __name__ == "__main__":
    now = datetime.now(utc)
    one_week_ago = now - timedelta(days=7)
    df = fetch_reddit_posts(DEMO_CONFIG)
    save_csv(df, "data/bitcoin_posts.csv")
