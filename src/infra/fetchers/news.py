import pandas as pd
from src.app.defaults import DEFAULT_CONFIG
import feedparser #type: ignore[import-untyped]
from src.infra.storage.logging_config import configure_logging
from src.shared.helpers import save_csv, clean_text
import logging
from src.app.dto import AnalysisConfig
from src.domain.market.filtering import contains_coin


logger = logging.getLogger(__name__)

def fetch_news_posts(config: AnalysisConfig) -> pd.DataFrame:
    logger.info(f"Attempting to fetch news for {config.coin}..")
    feed_urls = ['https://www.coindesk.com/arc/outboundfeeds/rss', 'https://cointelegraph.com/rss/tag/altcoin',
                 'https://cointelegraph.com/rss/tag/bitcoin', 'https://cointelegraph.com/rss/tag/ethereum', 'https://cointelegraph.com/rss/tag/blockchain',
                 'https://cointelegraph.com/rss/category/top-10-cryptocurrencies', 'https://www.newsbtc.com/feed/',
                 'https://thedefiant.io/feed/', 'https://cryptopotato.com/feed/', 'https://cryptoslate.com/feed/',
                 'https://cryptonews.com/news/feed/', 'https://smartliquidity.info/feed/', 'https://finance.yahoo.com/news/rssindex',
                 'https://www.cnbc.com/id/10000664/device/rss/rss.html', 'https://benjaminion.xyz/newineth2/rss_feed.xml']
    

    posts = []
    published = ['published', 'published_parsed', 'updated', 'updated_parsed']

    for feed_url in feed_urls:
        response = feedparser.parse(feed_url)
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
            except ValueError:
                continue
            if timestamp < config.start_date or timestamp > config.end_date:
                continue
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if not contains_coin(str(title), config.coin) and not contains_coin(str(summary), config.coin):
                continue
            
            url = entry.get("link", "")
            domain = feed_url.split("/")[2]
            posts.append(
                {"timestamp": timestamp, "title": title, "summary": summary, "source": domain, "url": url}
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
    df["text"] = df["title"].apply(clean_text)
    return df




if __name__ == "__main__":
    configure_logging()
    df = fetch_news_posts(DEFAULT_CONFIG)
    save_csv(df, "data/tests/news_posts.csv")
    print(df.head())
