import os
from dotenv import load_dotenv
import googleapiclient.discovery #type: ignore
import pandas as pd
from src.app.dto import AnalysisConfig
from src.domain.market.coins import COIN_TERMS
from src.infra.storage.logging_config import configure_logging
import logging
from src.app.defaults import DEFAULT_CONFIG
from src.shared.helpers import clean_text, save_csv
from src.infra.storage.db.youtube_repository import save_youtube_df, load_youtube_df, has_youtube_coverage


logger = logging.getLogger(__name__)
load_dotenv()

def fetch_youtube_posts(config: AnalysisConfig) -> pd.DataFrame:
    df = load_youtube_df(config)
    if has_youtube_coverage(config, df):
        logger.info(f"Success, fetched {len(df)} youtube posts in DB.")
        return df
    
    api_service_name = "youtube"
    api_version = "v3"
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "":
        raise RuntimeError("Set YOUTUBE_API_KEY in .env file")
    logger.info(
        f"Fetching YouTube posts with query='{config.coin}', limit={config.num_posts}")

    youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey = YOUTUBE_API_KEY)


    posts = []
    seen: set[str] = set()

    for coin in COIN_TERMS[config.coin]:
        page_token = None

        while len(posts) < config.num_posts:
            remaining = config.num_posts - len(posts)
            request = youtube.search().list(    
                part="id,snippet",  
                q=coin,   
                maxResults=min(50, remaining),
                type="video",
                order="date",
                publishedAfter=config.start_date.isoformat(),
                publishedBefore=config.end_date.isoformat(),
                pageToken=page_token
            )    

            response = request.execute()

            for item in response.get("items", []):
                videoId = item["id"]["videoId"]

                if videoId in seen:
                    continue
                title = item["snippet"]["title"]
                description = item["snippet"]["description"]

                posts.append({
                    "id": videoId,
                    "timestamp": item["snippet"]["publishedAt"],
                    "text": title + " " + description,
                    "source": "youtube",
                    "url": f"https://www.youtube.com/watch?v={videoId}",
                    "author": item["snippet"]["channelTitle"],
                    "coin": config.coin.lower(),
                })
                posts[-1]["text"] = clean_text(posts[-1]["text"])
                seen.add(videoId)

                if len(posts) >= config.num_posts:
                    break

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        if len(posts) >= config.num_posts:
            break

    
    df = pd.DataFrame(posts)
    if df.empty:
        logger.warning(f"No YouTube posts found for coin '{config.coin}' with the given config.")
        return df 
    df["timestamp"] =pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)

    logger.info(f"Fetched {len(df)} YouTube posts for query='{config.coin}'")

    save_youtube_df(df, config.coin)
    return load_youtube_df(config)

if __name__ == "__main__":
    configure_logging()
    df = fetch_youtube_posts(DEFAULT_CONFIG)
    save_csv(df, f"data/tests/{DEFAULT_CONFIG.coin}_youtube.csv")
    logger.info(f"Saved YouTube posts to data/tests/{DEFAULT_CONFIG.coin}_youtube.csv")
