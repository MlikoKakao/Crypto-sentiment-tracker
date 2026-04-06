import logging
import pandas as pd
from src.infra.storage.paths import get_demo_data_path


def run_demo() -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("Fetching demo csv...")

    # Make running it be able to live just on merged csv.
    return pd.read_csv(get_demo_data_path("bitcoin_merged.csv"), parse_dates=["timestamp"])