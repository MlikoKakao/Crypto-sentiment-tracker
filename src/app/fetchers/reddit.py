from pytz import utc
from praw import Reddit  # type: ignore no stub file
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from src.app.dto import AnalysisConfig
from src.infra.storage.logging_config import configure_logging
from src.utils.helpers import save_csv, clean_text
import os
import logging
from src.app.defaults import DEFAULT_CONFIG
from src.domain.market.filtering import contains_coin

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

    posts: list[dict[str, object]] = []
    seen: set[str] = set()

    reddit = get_reddit_client()

    if len(config.subreddits) == 0:
        logger.error("No subreddits specified in config!")
        raise ValueError("At least one subreddit must be specified in config")
    
    for sub in config.subreddits:
        for submission in reddit.subreddit(sub).new(limit=config.num_posts):
            time_posted = datetime.fromtimestamp(submission.created_utc, tz=utc)

            if time_posted > config.end_date:
                continue
            if time_posted < config.start_date:
                break

            # Filter by keywords locally
            text = f"{submission.title or ''} {submission.selftext or ''}"

            if not contains_coin(text, config.coin):
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

if __name__ == "__main__":
    configure_logging()
    print("Testing Reddit fetch")
    df = fetch_reddit_posts(DEFAULT_CONFIG)
    logger.debug(f"Fetched Reddit posts:{df.head()}")
    save_csv(df, "data/tests/bitcoin_reddit_posts.csv")
