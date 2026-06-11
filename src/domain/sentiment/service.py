import pandas as pd

from src.domain.sentiment.registry import ANALYZERS
from src.shared.dataframe_schema import require_columns


def add_sentiment_to_df(df: pd.DataFrame, analyzer_name: str = "vader") -> pd.DataFrame:
    if df.empty:
        return df
    require_columns(df, {"text"}, "sentiment input")

    df = df.copy()
    name = analyzer_name.lower()

    analyzer_func = ANALYZERS.get(name)
    if analyzer_func is None:
        raise ValueError(f"Unknown analyzer: {analyzer_name}")

    df["sentiment"] = df["text"].apply(analyzer_func)
    df["analyzer"] = name
    return df
