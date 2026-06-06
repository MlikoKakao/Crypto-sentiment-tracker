import pandas as pd


def format_timestamp_for_api(
    df: pd.DataFrame,
    column: str = "timestamp",
) -> pd.DataFrame:
    df = df.copy()

    if column not in df.columns:
        return df

    df[column] = pd.to_datetime(df[column].dt.strftime("%Y-%m-%d %H:%M:%S"))
    return df
