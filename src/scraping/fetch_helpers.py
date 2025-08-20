import pandas as pd
from src.utils.cache import load_cached_csv, cache_csv


def time_windows(start, end, freq_hours: int):
    cur = start
    delta = pd.Timedelta(hours=freq_hours)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt

def fetching_windows(fetch_func, base_settings: dict, start, end,
                     window_hours: int, per_window_limit: int,
                     cache_key_fields: list, id_col:str):
    frames = []
    for s, e in time_windows(start, end, window_hours):
        settings = {**base_settings,
                    "start_date": s.tz_convert(None).isoformat(timespec="seconds"),
                    "end_date": e.tz_convert(None).isoformat(timespec="seconds"),
                    "num_posts": per_window_limit}
        
        df = load_cached_csv(settings, parse_dates=["timestamp"], freshness_minutes=30)
        if df is None:
            df = fetch_func(start=s, end=e, limit=per_window_limit)
            if df is not None and not df.empty:
                cache_csv(df, settings)
        if df is not None and not df.empty:
           frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "text", "source"]) 
    
    df_all = pd.concat(frames, ignore_index=True)
    if id_col in df_all.columns:
        df_all =  df_all.drop_duplicates(subset=[id_col], keep="last")

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    df_all = df_all.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df_all

def window_params(days: int, num_posts: int):
    posts_per_day = max(10, int(num_posts / max(1, days)))
    if days <= 7:
        window_hours = 12
    elif days <= 30:
        window_hours = 24
    else:
        window_hours = 72
    per_window_limit = max(25, int(posts_per_day * window_hours / 24))
    return window_hours, per_window_limit