from src.infra.storage.db.connection import get_connection
import pandas as pd
from src.presentation.sidebar import IndicatorConfig
from src.shared.helpers import normalize_timestamp_column
from datetime import timedelta


def save_signal_df(signal_df: pd.DataFrame, signal: str, coin: str = "btc") -> None:
    df = signal_df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["coin"] = coin.upper()
    df["signal_name"] = signal
    df["value"] = df[signal]        
    df = df.dropna(subset=["value"])        

    rows = df[["coin", "timestamp", "signal_name", "value"]].itertuples(index=False, name=None)
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO signals (coin, timestamp, signal_name, value)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    conn.close()


# Convert config.dates to format where can compare to SQL results
def load_signal_df(state: IndicatorConfig, signal: str) -> pd.DataFrame:
    start_date = state.start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date = state.end_date.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        df = pd.read_sql_query(
            """
                               SELECT * FROM signals
                               WHERE coin = ? AND signal_name = ? AND timestamp BETWEEN ? AND ?
                               """,
            conn,
            params=(state.coin.upper(), signal, start_date, end_date),
        )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "value"]].rename(columns={"value": signal})
    return df


def has_signal_coverage(state: IndicatorConfig, signal_df: pd.DataFrame) -> bool:
    if signal_df.empty:
        return False

    tolerance = timedelta(hours=1)
    min_time = signal_df["timestamp"].min()
    max_time = signal_df["timestamp"].max()

    start_date = pd.to_datetime(state.start_date, utc=True).tz_convert(None)
    end_date = pd.to_datetime(state.end_date, utc=True).tz_convert(None)

    starts_near = min_time <= start_date + tolerance
    ends_near = max_time >= end_date - tolerance

    return starts_near and ends_near
