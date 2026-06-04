import pandas as pd

REQUIRED_PRICE_COLUMNS = {"timestamp", "price"}

REQUIRED_CONTENT_COLUMNS = {
    "timestamp",
    "text",
    "source",
    "url",
}

REQUIRED_SENTIMENT_COLUMNS = {"source", "analyzer", "sentiment"}

REQUIRED_SIGNAL_INPUT_COLUMNS = {"timestamp", "price"}


def require_columns(
    df: pd.DataFrame, required: set[str], name: str = "DataFrame"
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
