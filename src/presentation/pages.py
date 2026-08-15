import streamlit as st
import pandas as pd
from typing import Any
from dataclasses import replace

from src.app.dto import AnalysisResult
from src.app.use_cases.get_indicators import add_indicators_with_cache
from src.presentation.charts import (
    plot_sentiment_timeline,
    plot_sentiment_vs_price,
    plot_sentiment_with_price,
    plot_lag_correlation,
    plot_price_with_sma,
    plot_rsi,
    plot_macd,
    plot_drawdown,
    plot_equity,
    plot_signal,
)
from src.domain.backtest.engine import run_backtest
from src.presentation.sidebar import (
    SidebarState,
    render_sidebar,
    sidebar_state_to_config,
    sidebar_to_indicator,
)
from src.domain.analysis.lead_lag import compute_lead_lag
from src.presentation.demo_view import render_demo_page
from src.domain.signals.engine import build_signal_df, SIGNAL_COLUMNS
from src.presentation.translations import TEXT


def render_app(demo_mode: bool = False) -> None:
    if demo_mode:
        render_demo_page()
    else:
        state = render_sidebar()
        render_live_page(state)


def render_live_page(state: SidebarState) -> None:
    text = TEXT[state.language]
    st.title(text["title"])
    st.markdown(text["description"])

    sentiment_tab, engine_tab, finance_tab, backtest_tab = st.tabs(
        [text["tab_sentiment"], text["tab_engine"], text["tab_finance"], text["tab_backtest"]]
    )

    tabs = {
        "sentiment": sentiment_tab,
        "engine": engine_tab,
        "finance": finance_tab,
        "backtest": backtest_tab,
    }

    if not state.run:
        with tabs["sentiment"]:
            st.info(
                text["configure_prompt"]
            )
        with tabs["finance"]:
            st.info(text["finance_prompt"])
        with tabs["backtest"]:
            st.info(text["backtest_prompt"])
        return

    config = sidebar_state_to_config(state)

    with st.spinner(text["running_analysis"]):
        from src.presentation.api.query_client import run_analysis_with_api

        result = run_analysis_with_api(config)
        if result.issues:
            st.write(result.issues)

    render_result_tabs(result, state, tabs)


def render_result_tabs(
    result: AnalysisResult, state: SidebarState, tabs: dict[str, Any]
) -> None:
    with tabs["sentiment"]:
        st.plotly_chart(
            plot_sentiment_with_price(result.merged_df, state.selected_coin, state.language),
            key="live_sentiment_price_chart",
        )

        st.plotly_chart(
            plot_sentiment_timeline(result.merged_df, state.selected_coin, state.language),
            key="live_sentiment_timeline_chart",
        )
        st.plotly_chart(
            plot_sentiment_vs_price(result.merged_df, state.language),
            key="live_sentiment_vs_price_chart",
        )

        lead_lag_df = compute_lead_lag(
            result.merged_df, state.lag_hours, state.lag_step_min, state.metric_choice
        )
        st.plotly_chart(
            plot_lag_correlation(lead_lag_df, language=state.language),
            key="live_lag_correlation_chart",
        )

    with tabs["engine"]:
        indic_state = replace(sidebar_to_indicator(state))
        indicator_df = add_indicators_with_cache(result.merged_df, indic_state)
        signal_df = build_signal_df(indicator_df)
        signal_cols = [col for col in SIGNAL_COLUMNS if col in signal_df.columns]

        event_rows = signal_df[signal_df[signal_cols].any(axis=1)]

        st.dataframe(
            event_rows[["timestamp", "price", "sentiment", *signal_cols]],
            hide_index=True,
        )
        st.plotly_chart(plot_signal(signal_df, state.language), key="live_signal_chart")
    with tabs["finance"]:
        if not (state.use_sma or state.use_macd or state.use_rsi):
            st.info(
                TEXT[state.language]["enable_indicators"]
            )
        indic_state = sidebar_to_indicator(state)
        indicators_df = add_indicators_with_cache(result.price_df, indic_state)
        if state.use_sma:
            st.plotly_chart(
                plot_price_with_sma(
                    indicators_df,
                    state.selected_coin,
                    sma_cols=[f"sma_{state.sma_fast}", f"sma_{state.sma_slow}"],
                    language=state.language,
                ),
                key="live_sma_chart",
            )

        if state.use_macd:
            fig = plot_macd(indicators_df, state.language)
            if fig is not None:
                st.plotly_chart(fig, key="live_macd_chart")
            else:
                st.warning(TEXT[state.language]["macd_unavailable"])

        if state.use_rsi:
            fig = plot_rsi(indicators_df, rsi_col=f"rsi_{state.rsi_period}", language=state.language)
            if fig is not None:
                st.plotly_chart(fig, key="live_rsi_chart")
            else:
                st.warning(TEXT[state.language]["rsi_unavailable"])

    with tabs["backtest"]:
        if not state.backtest:
            st.info(TEXT[state.language]["enable_backtest"])
        else:
            df_bt, stats = run_backtest(
                result.merged_df, state.cost_bps, state.slip_bps
            )
            st.plotly_chart(plot_equity(df_bt, state.language), key="live_equity_chart")
            st.plotly_chart(plot_drawdown(df_bt, state.language), key="live_drawdown_chart")
            st.dataframe(pd.DataFrame([stats]), hide_index=True)
