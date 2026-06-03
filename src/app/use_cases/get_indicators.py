import pandas as pd
from collections.abc import Callable

from src.domain.market.dto import IndicatorConfig
from src.domain.market.indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
)
from src.infra.storage.db.signal_repository import (
    save_signal_df,
    load_signal_df,
    has_signal_coverage,
)


def add_indicators_with_cache(
    df: pd.DataFrame,
    state: IndicatorConfig,
) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    indicator_jobs = get_enabled_indicator_jobs(state)

    for signal_names, calculate_func in indicator_jobs:
        cached_parts = []

        for signal_name in signal_names:
            cached = load_signal_df(state, signal_name)

            if has_signal_coverage(state, cached):
                cached_parts.append(cached)
            else:
                cached_parts = []
                break

        if cached_parts:
            for cached in cached_parts:
                df = df.merge(cached, on="timestamp", how="left")
        else:
            calculated = calculate_func(df, state)

            df = df.merge(calculated, on="timestamp", how="left")

            for signal_name in signal_names:
                save_signal_df(
                    calculated[["timestamp", signal_name]],
                    signal_name,
                    state.coin,
                )

    return df


def get_enabled_indicator_jobs(
    state: IndicatorConfig,
) -> list[tuple[list[str], Callable[[pd.DataFrame, IndicatorConfig], pd.DataFrame]]]:
    jobs = []

    if state.use_sma:
        jobs.append(
            (
                [f"sma_{state.sma_fast}", f"sma_{state.sma_slow}"],
                calculate_sma,
            )
        )

    if state.use_rsi:
        jobs.append(
            (
                [f"rsi_{state.rsi_period}"],
                calculate_rsi,
            )
        )

    if state.use_macd:
        jobs.append(
            (
                ["macd", "macd_signal", "macd_hist"],
                calculate_macd,
            )
        )

    return jobs
