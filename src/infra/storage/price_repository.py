from src.infra.storage.db import get_connection
import pandas as pd
from src.shared.helpers import normalize_timestamp_column
from src.app.dto import AnalysisConfig

def save_price_df(prices_df: pd.DataFrame, coin: str = "btc") -> None:
    df = prices_df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["coin"] = coin.upper()

    rows = df[["coin", "timestamp", "price"]].itertuples(index=False, name=None)
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (coin, timestamp, price)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    conn.close()

# Convert config.dates to format where can compare to SQL results
def load_price_df(config: AnalysisConfig) -> pd.DataFrame:
    start_date = config.start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date = config.end_date.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        df = pd.read_sql_query("""
                               SELECT * FROM prices 
                               WHERE coin = ? AND timestamp >= ? AND timestamp <= ?
                               """, conn, config.coin.upper(), start_date, end_date
        )
    conn.close()
    return df