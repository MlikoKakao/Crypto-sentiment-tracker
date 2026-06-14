import pandas as pd

from src.domain.sentiment.registry import ANALYZERS
from src.shared.dataframe_schema import require_columns
    

def add_sentiment_to_df(df: pd.DataFrame, analyzer_name: str = "vader") -> pd.DataFrame:
    if df.empty:
        return df
    require_columns(df, {"text"}, "sentiment input")

    df = df.copy()
    name = analyzer_name.lower()

    analyzer = ANALYZERS.get(name)
    if analyzer is None:
        raise ValueError(f"Unknown analyzer: {analyzer_name}")

    texts = df["text"].fillna("").astype(str).tolist()
    analyze_many = getattr(analyzer, "analyze_many", None)

    if analyze_many is not None:
        df["sentiment"] = analyze_many(texts)
    else:
        df["sentiment"] = [analyzer(text) for text in texts]

    df["analyzer"] = name
    return df
