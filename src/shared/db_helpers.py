import pandas as pd
import hashlib


def make_content_hash(
    source: str,
    source_id: str | None,
    url: str | None,
    text: str,
) -> str:
    if source_id:
        raw = f"{source}|id|{source_id}"
    elif url:
        raw = f"{source}|url|{url}"
    else:
        raw = f"{source}|text|{text.strip()}"

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_content_hash(row: pd.Series) -> str:
    source_id = none_if_missing(row.get("source_id"))
    url = none_if_missing(row.get("url"))

    return make_content_hash(
        source=str(row["source"]),
        source_id=source_id,
        url=url,
        text=str(row["text"]),
    )


def none_if_missing(value: str | None) -> str | None:
    if pd.isna(value):
        return None
    return value


def add_optional_cols_inplace(df: pd.DataFrame) -> None:
    if "source_id" not in df.columns:
        df["source_id"] = None

    if "url" not in df.columns:
        df["url"] = None
