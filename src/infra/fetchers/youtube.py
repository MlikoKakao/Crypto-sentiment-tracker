import os
import pandas as pd
from src.app.dto import AnalysisConfig
import logging

from src.app.defaults import default_config
from src.shared.helpers import clean_text, save_csv
from src.infra.storage.db.content_repository import (
    save_content_df,
    load_content_df,
    has_content_coverage,
)


YOUTUBE_COIN_TERMS = {
    "BTC": ("BTC", "bitcoin"),
    "ETH": ("ETH", "ethereum"),
    "XMR": ("XMR", "monero"),
}
logger = logging.getLogger(__name__)


def fetch_youtube_posts(config: AnalysisConfig) -> pd.DataFrame:
    import googleapiclient.discovery  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore


    df = load_content_df(config, "youtube")
    if has_content_coverage(config, df):
        logger.info(f"Success, fetched {len(df)} youtube posts in DB.")
        return df

    api_service_name = "youtube"
    api_version = "v3"
    youtube_api_key = os.getenv("youtube_api_key")
    if not youtube_api_key or youtube_api_key == "":
        raise RuntimeError("Set youtube_api_key in .env file")

    youtube_limit = min(config.num_posts, 400)
    logger.info(
        f"Fetching YouTube posts with query='{config.coin}', limit={youtube_limit}"
    )

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=youtube_api_key
    )

    posts = []
    seen: set[str] = set()

    for coin in YOUTUBE_COIN_TERMS[config.coin]:
        page_token = None

        while len(posts) < youtube_limit:
            remaining = youtube_limit - len(posts)
            request = youtube.search().list(
                part="id,snippet",
                q=coin,
                maxResults=min(50, remaining),
                type="video",
                order="date",
                publishedAfter=config.start_date.isoformat(),
                publishedBefore=config.end_date.isoformat(),
                pageToken=page_token,
            )
            try:
                response = request.execute()
            except HttpError as e:
                logger.warning("Youtube API request failed: %s", e)
                break

            for item in response.get("items", []):
                video_id = item["id"]["videoId"]

                if video_id in seen:
                    continue
                title = item["snippet"]["title"]
                description = item["snippet"]["description"]

                posts.append(
                    {
                        "source_id": video_id,
                        "timestamp": item["snippet"]["publishedAt"],
                        "text": title + " " + description,
                        "source": "youtube",
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "author": item["snippet"]["channelTitle"],
                        "coin": config.coin.lower(),
                    }
                )
                posts[-1]["text"] = clean_text(posts[-1]["text"])
                seen.add(video_id)

                if len(posts) >= youtube_limit:
                    break

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        if len(posts) >= youtube_limit:
            break

    df = pd.DataFrame(posts)
    if df.empty:
        logger.warning(
            f"No YouTube posts found for coin '{config.coin}' with the given config."
        )
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)

    logger.info(f"Fetched {len(df)} YouTube posts for query='{config.coin}'")

    save_content_df(df, config.coin)
    return load_content_df(config, "youtube")


if __name__ == "__main__":
    df = fetch_youtube_posts(default_config())
    save_csv(df, f"data/tests/{default_config().coin}_youtube.csv")
    logger.info(
        f"Saved YouTube posts to data/tests/{default_config().coin}_youtube.csv"
    )
