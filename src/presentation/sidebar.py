from pytz import utc
import streamlit as st
from dataclasses import dataclass
from typing import cast
import pandas as pd
from datetime import datetime
from src.app.defaults import DEFAULT_SUBREDDITS
from src.app.dto import AnalysisConfig, Analyzer, Source, Coin
from src.domain.market.dto import IndicatorConfig
from src.presentation.ui_constants import (
    ANALYZER_UI_TO_LITERAL,
    COIN_UI_TO_SYMBOL,
    COINS_UI_LABELS,
    SOURCE_UI_TO_LITERAL,
)
from src.presentation.translations import TEXT, language_from_browser_locale


@dataclass(frozen=True)
class SidebarState:
    language: str
    selected_coin: Coin
    start_date: datetime
    end_date: datetime
    analyzer: Analyzer
    sources: tuple[Source, ...]
    num_posts: int
    run: bool
    benchmark: bool
    backtest: bool
    cost_bps: float
    slip_bps: float
    use_sma: bool
    use_rsi: bool
    use_macd: bool
    lag_hours: int
    lag_step_min: int
    metric_choice: str
    sma_fast: int
    sma_slow: int
    rsi_period: int


def render_sidebar() -> SidebarState:
    with st.sidebar:
        if "language" not in st.session_state:
            st.session_state.language = language_from_browser_locale(st.context.locale)

        language = st.selectbox(
            label=TEXT["en"]["language"],
            options=["en", "zh-TW"],
            format_func=lambda code: {"en": "English", "zh-TW": "繁體中文"}[code],
            key="language",
        )
        st.header(TEXT[language]["settings"])

        selected_coin_label = st.selectbox(TEXT[language]["choose_coin"], COINS_UI_LABELS)
        assert selected_coin_label is not None
        selected_coin = COIN_UI_TO_SYMBOL[selected_coin_label]

        num_posts = st.slider(
            TEXT[language]["num_posts"],
            min_value=100,
            max_value=5000,
            step=100,
            value=300,
        )

        days = st.selectbox(
            TEXT[language]["price_history"],
            ("1", "7", "10", "30", "90", "180", "365"),
            help=TEXT[language]["price_history_help"],
        )
        assert days is not None

        end_date: datetime = pd.Timestamp.now(tz=utc).to_pydatetime()
        start_date: datetime = (
            pd.Timestamp(end_date) - pd.Timedelta(days=int(days))
        ).to_pydatetime()

        analyzer_label = st.selectbox(
            TEXT[language]["choose_analyzer"],
            list(ANALYZER_UI_TO_LITERAL.keys()),
            help=TEXT[language]["analyzer_help"],
        )
        assert analyzer_label is not None

        analyzer: Analyzer = ANALYZER_UI_TO_LITERAL[analyzer_label]

        source_label = st.selectbox(
            TEXT[language]["choose_sources"], list(SOURCE_UI_TO_LITERAL.keys())
        )
        assert source_label is not None
        sources: tuple[Source, ...] = SOURCE_UI_TO_LITERAL[source_label]

        with st.expander(TEXT[language]["advanced_settings"]):
            backtest = st.checkbox(TEXT[language]["run_backtest"])
            cost_bps = 0.0
            slip_bps = 0.0
            if backtest:
                cost_bps = st.number_input(
                    TEXT[language]["cost"], min_value=0.0, max_value=100.0, value=5.0, step=0.5
                )
                slip_bps = st.number_input(
                    TEXT[language]["slippage"],
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=0.5,
                )

            st.header(TEXT[language]["lead_lag_settings"])
            lag_hours = st.slider(TEXT[language]["lag_window"], 1, 48, 24)
            lag_step_min = st.selectbox(TEXT[language]["lag_step"], [5, 15, 30, 60], index=1)
            metric_choice = st.selectbox(TEXT[language]["correlation_metric"], ["pearson"], index=0)

            st.markdown(f"### {TEXT[language]['indicators']}")
            # default values so variables exist even when checkboxes are unchecked
            sma_fast = 20
            sma_slow = 50
            rsi_period = 14
            use_sma = st.checkbox(
                "SMA (20/50)", value=False, help=TEXT[language]["sma_help"]
            )
            use_rsi = st.checkbox(
                "RSI (14)", value=False, help=TEXT[language]["rsi_help"]
            )
            use_macd = st.checkbox(
                "MACD (12,26,9)",
                value=False,
                help=TEXT[language]["macd_help"],
            )
            if use_sma:
                sma_fast = st.number_input(TEXT[language]["sma_fast"], 5, 200, sma_fast, 1)
                sma_slow = st.number_input(TEXT[language]["sma_slow"], 5, 400, sma_slow, 1)
            if use_rsi:
                rsi_period = st.number_input(TEXT[language]["rsi_period"], 5, 50, rsi_period, 1)

        run = st.button(TEXT[language]["run_analysis"], type="primary")

        st.header(TEXT[language]["utilities"])

        benchmark = st.button(TEXT[language]["run_benchmark"])

    return SidebarState(
        language=language,
        selected_coin=cast(Coin, selected_coin),
        start_date=start_date,
        end_date=end_date,
        analyzer=analyzer,
        sources=sources,
        num_posts=num_posts,
        run=run,
        benchmark=benchmark,
        backtest=backtest,
        cost_bps=cost_bps,
        slip_bps=slip_bps,
        lag_hours=lag_hours,
        lag_step_min=lag_step_min,
        metric_choice=metric_choice,
        use_sma=use_sma,
        use_rsi=use_rsi,
        use_macd=use_macd,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        rsi_period=rsi_period,
    )


def sidebar_state_to_config(state: SidebarState) -> AnalysisConfig:
    return AnalysisConfig(
        coin=state.selected_coin,
        start_date=state.start_date,
        end_date=state.end_date,
        analyzer=state.analyzer,
        sources=state.sources,
        num_posts=state.num_posts,
        subreddits=DEFAULT_SUBREDDITS,
    )


def sidebar_to_indicator(state: SidebarState) -> IndicatorConfig:
    return IndicatorConfig(
        coin=state.selected_coin,
        use_sma=state.use_sma,
        use_rsi=state.use_rsi,
        use_macd=state.use_macd,
        sma_fast=state.sma_fast,
        sma_slow=state.sma_slow,
        rsi_period=state.rsi_period,
        start_date=state.start_date,
        end_date=state.end_date,
    )
