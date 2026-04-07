import os
from dotenv import load_dotenv
import googleapiclient.discovery
import pandas as pd
from src.app.dto import AnalysisConfig
from src.infra.storage.logging_config import configure_logging
import logging
from src.app.defaults import DEFAULT_CONFIG
from src.utils.helpers import save_csv

logger = logging.getLogger(__name__)


scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
load_dotenv()

def fetch_youtube_posts(config: AnalysisConfig) -> pd.DataFrame:
    api_service_name = "youtube"
    api_version = "v3"
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey = YOUTUBE_API_KEY)
    request = youtube.search().list(    
        part="id,snippet",  
        q=config.coin,   
        maxResults=1,
    )    
    
    response = request.execute()    
    
    print(response)
    df = pd.DataFrame(response["items"])
    return df

if __name__ == "__main__":
    configure_logging()
    df = fetch_youtube_posts(DEFAULT_CONFIG)
    save_csv(df, f"data/tests/{DEFAULT_CONFIG.coin}_youtube.csv")
    logger.info(f"Saved YouTube posts to data/tests/{DEFAULT_CONFIG.coin}_youtube.csv")