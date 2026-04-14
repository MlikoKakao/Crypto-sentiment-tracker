from pytz import utc
import streamlit as st
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
from src.app.defaults import DEFAULT_SUBREDDITS
from src.app.dto import AnalysisConfig, Analyzer, Source
from src.presentation.ui_constants import (
    ANALYZER_UI_TO_LITERAL,
    COIN_SUBS,
    COIN_UI_TO_SYMBOL,
    COINS_UI_LABELS,
    SOURCE_UI_TO_LITERAL,
)
from src.infra.cache.file_cache import clear_cache_dir

@dataclass(frozen=True)
class SidebarState:
    selected_coin: str
    start_date: datetime
    end_date: datetime
    analyzer: Analyzer
    sources: tuple[Source, ...]
    num_posts: int
    subreddits: tuple[str, ...]
    run: bool
    benchtest: bool
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
        st.header("Settings")
        
        selected_coin_label = st.selectbox("Choose cryptocurrency", COINS_UI_LABELS)
        assert selected_coin_label is not None
        selected_coin = COIN_UI_TO_SYMBOL[selected_coin_label]

        num_posts = st.slider(
            "Number of posts to fetch", min_value=100, max_value=1100, step=100, value=300
        )

        days = st.selectbox(
            "Price history in days",
            ("1", "7", "10", "30", "90", "180", "365"),
            help="Choosing day range longer than 90 days causes to only show price point once per day.",
        )
        assert days is not None

        end_date: datetime = pd.Timestamp.now(tz=utc).to_pydatetime()
        start_date: datetime = (
            pd.Timestamp(end_date) - pd.Timedelta(days=int(days))
        ).to_pydatetime()

        analyzer_label = st.selectbox(
            "Choose sentiment analyzer:",
            list(ANALYZER_UI_TO_LITERAL.keys()),
            help=(
                "VADER - all-rounder, decent speed and analysis.\nText-Blob - fastest, but least accurate.\nTwitter-RoBERTa - slowest(can take up to a minute depending on size), but most accurate, conservative.\nfinBERT - finance-specific, good accuracy, medium speed"
            ),
        )
        assert analyzer_label is not None

        analyzer: Analyzer = ANALYZER_UI_TO_LITERAL[analyzer_label]


        source_label = st.selectbox(
            "Choose which sources to include:", list(SOURCE_UI_TO_LITERAL.keys())
        )
        assert source_label is not None
        sources: tuple[Source, ...] = SOURCE_UI_TO_LITERAL[source_label]


        default_subreddits = tuple(
            dict.fromkeys(DEFAULT_SUBREDDITS + tuple(COIN_SUBS.get(selected_coin, ())))
        )
        subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS

        if "reddit" in sources:
            selected_subreddits = st.multiselect(
                "Subreddits",
                options=default_subreddits,
                default=default_subreddits,
            )

            if selected_subreddits:
                subreddits = tuple(selected_subreddits)
            else:
                subreddits = default_subreddits
                st.warning("No subreddits chosen, defaulted.")


        with st.expander("Advanced settings"):
            backtest = st.checkbox("Run backtest")
            cost_bps = 0.0
            slip_bps = 0.0
            if backtest:
                cost_bps = st.number_input(
                    "Cost (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
                )
                slip_bps = st.number_input(
                    "Slippage (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
                )

            st.header("Lead/Lag settings")
            lag_hours = st.slider("Lag window (±hours)", 1, 48, 24)
            lag_step_min = st.selectbox("Lag step(minutes)", [5, 15, 30, 60], index=1)
            metric_choice = st.selectbox("Correlation metric", ["pearson"], index=0)

            st.markdown("### Indicators")
            # default values so variables exist even when checkboxes are unchecked
            sma_fast = 20
            sma_slow = 50
            rsi_period = 14
            use_sma = st.checkbox("SMA (20/50)", value=False, help="Simple Moving Average")
            use_rsi = st.checkbox("RSI (14)", value=False, help="Relative Strength Index")
            use_macd = st.checkbox(
                "MACD (12,26,9)", value=False, help="Moving Average Convergence Divergence"
            )
            if use_sma:
                sma_fast = st.number_input("SMA fast", 5, 200, sma_fast, 1)
                sma_slow = st.number_input("SMA slow", 5, 400, sma_slow, 1)
            if use_rsi:
                rsi_period = st.number_input("RSI period", 5, 50, rsi_period, 1)

        run = st.button("Run Analysis", type="primary")

        st.header("Utils")

        if st.button("Clear cache"):
            res = clear_cache_dir()
            mb = res["bytes_freed"] / 1000000
            st.success(f"Removed {res['files_removed']} files ({mb:.2f} MB)")
            st.session_state.pop("merged_path", None)


        benchtest = st.button("Run analyzer benchmark")
    
    return SidebarState(
        selected_coin=selected_coin,
        start_date=start_date,
        end_date=end_date,
        analyzer=analyzer,
        sources=sources,
        num_posts=num_posts,
        subreddits=subreddits,
        run=run,
        benchtest=benchtest,
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
        rsi_period=rsi_period
    )
    
def sidebar_state_to_config(state: SidebarState) -> AnalysisConfig:
    return AnalysisConfig(
        coin=state.selected_coin,
        start_date=state.start_date,
        end_date=state.end_date,
        analyzer=state.analyzer,
        sources=state.sources,
        num_posts=state.num_posts,
        subreddits=state.subreddits,
    )
