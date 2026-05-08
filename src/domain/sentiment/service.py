import pandas as pd

from src.domain.sentiment.registry import ALL_ANALYZER_NAMES, ANALYZERS
from src.infra.storage.db.sentiment_repository import load_sentiment_df, save_sentiment_df, has_sentiment_coverage

def add_sentiment_to_df(df: pd.DataFrame, analyzer_name: str = "vader") -> pd.DataFrame:
    df = df.copy()
    name = analyzer_name.lower()

    if name == "all":
        for analyzer in ALL_ANALYZER_NAMES:
            func = ANALYZERS[analyzer]
            col_name = f"sentiment_{analyzer}"
            df[col_name] = df["text"].apply(func)
            df["analyzer"] = analyzer

        sentiment_cols = [f"sentiment_{analyzer}" for analyzer in ALL_ANALYZER_NAMES]
        df["sentiment"] = df[sentiment_cols].mean(axis=1)

    else:
        analyzer_func = ANALYZERS.get(name)
        if analyzer_func is None:
            raise ValueError(f"Unknown analyzer: {analyzer_name}")

        df["sentiment"] = df["text"].apply(analyzer_func)
        df["analyzer"] = analyzer_name
    return df

