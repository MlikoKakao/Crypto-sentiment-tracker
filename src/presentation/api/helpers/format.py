from typing import TypeVar
import pandas as pd
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def dataframe_to_response_models(
    df: pd.DataFrame,
    model: type[T],
    timestamp_column: str = "timestamp",
) -> list[T]:
    df = df.copy()

    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(
            df[timestamp_column].dt.strftime("%Y-%m-%d %H:%M:%S")
        )

    return [model.model_validate(row) for row in df.to_dict(orient="records")]
