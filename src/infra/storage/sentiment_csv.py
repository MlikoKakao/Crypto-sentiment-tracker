import pandas as pd


def load_sentiment_df(paths_by_source: dict[str, str | None], posts_choice: str) -> pd.DataFrame:
    selected_sources = choice_to_sources(posts_choice)

    frames = []
    for source in selected_sources:
        path = paths_by_source.get(source)
        if not path:
            continue

        df = pd.read_csv(path, parse_dates=["timestamp"])
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])

    all_cols = sorted(set().union(*(frame.columns for frame in frames)))
    frames = [frame.reindex(columns=all_cols) for frame in frames]

    return pd.concat(frames, ignore_index=True)

def choice_to_sources(posts_choice: str) -> tuple[str, ...]:
    choices = {
        "News": ("news",),
        "Reddit": ("reddit",),
        "Youtube": ("youtube",),
        "All": ("news", "reddit", "youtube"),
    }

    return choices.get(posts_choice, ("news", "reddit", "youtube"))
