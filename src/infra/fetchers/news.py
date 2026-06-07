import pandas as pd
from src.app.defaults import DEFAULT_CONFIG
import feedparser  # type: ignore[import-untyped]
from src.shared.helpers import save_csv

from src.infra.storage.db.content_repository import (
    save_content_df,
    load_content_df,
    has_content_coverage,
)
import logging
from src.app.dto import AnalysisConfig
from src.domain.market.filtering import contains_coin


logger = logging.getLogger(__name__)


def fetch_news_posts(config: AnalysisConfig) -> pd.DataFrame:
    logger.info("Attempting to fetch cached data..")
    df = load_content_df(config, "news")
    if has_content_coverage(config, df):
        return df

    logger.info("Attempting to fetch news for %s..", config.coin)
    feed_urls = [
        "https://www.coindesk.com/arc/outboundfeeds/rss",
        "https://cointelegraph.com/rss/tag/altcoin",
        "https://cointelegraph.com/rss/tag/bitcoin",
        "https://cointelegraph.com/rss/tag/ethereum",
        "https://cointelegraph.com/rss/tag/blockchain",
        "https://cointelegraph.com/rss/category/top-10-cryptocurrencies",
        "https://www.newsbtc.com/feed/",
        "https://thedefiant.io/feed/",
        "https://cryptopotato.com/feed/",
        "https://cryptoslate.com/feed/",
        "https://cryptonews.com/news/feed/",
        "https://smartliquidity.info/feed/",
        "https://finance.yahoo.com/news/rssindex",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "https://benjaminion.xyz/newineth2/rss_feed.xml",
    ]

    posts = []
    published = ["published", "published_parsed", "updated", "updated_parsed"]

    start_date = pd.to_datetime(config.start_date, utc=True)
    end_date = pd.to_datetime(config.end_date, utc=True)

    for feed_url in feed_urls:
        response = feedparser.parse(feed_url)

        if response.bozo:
            logger.warning(
                "Feed parser warning for %s: %s",
                feed_url,
                response.bozo_exception,
            )

        for entry in response.entries:
            published_at = None

            for publish in published:
                published_at = entry.get(publish, "")
                if published_at:
                    break

            if not published_at:
                continue

            try:
                timestamp = pd.to_datetime(str(published_at), utc=True)
            except (ValueError, TypeError):
                continue

            if timestamp < start_date or timestamp > end_date:
                continue
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if not contains_coin(str(title), config.coin) and not contains_coin(
                str(summary), config.coin
            ):
                continue

            url = entry.get("link", "")
            posts.append(
                {
                    "timestamp": timestamp,
                    "title": title,
                    "summary": summary,
                    "text": f"{title} {summary}",
                    "url": url,
                    "source": "news",
                }
            )
            if len(posts) >= config.num_posts:
                break
        if len(posts) >= config.num_posts:
            break
        logger.debug(f"Number of entries in {feed_url}: {len(response.entries)}")
        logger.debug(f"Number of usable entries so far: {len(posts)}")
    logger.debug(f"Number of posts before dedup: {len(posts)}")

    logger.info(f"Fetched {len(posts)} news posts for {config.coin}")
    df = pd.DataFrame(posts)
    if df.empty:
        return df
    dupes = df.duplicated(subset=["url"])
    dupes[df["url"].isna()] = False
    dupes[df["url"] == ""] = False
    df = df[~dupes]
    logger.debug(f"Size of final df: {len(df)}")
    save_content_df(df, config.coin)
    return load_content_df(config, "news")


if __name__ == "__main__":
    df = fetch_news_posts(DEFAULT_CONFIG)
    save_csv(df, "data/tests/news_posts.csv")
    print(df.head())
