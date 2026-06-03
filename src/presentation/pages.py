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
from src.app.use_cases.run_analysis import run_analysis
from src.domain.analysis.lead_lag import compute_lead_lag
from src.presentation.demo_view import render_demo_page
from src.presentation.benchmark_view import show_benchmark_data
from src.domain.signals.engine import build_signal_df, SIGNAL_COLUMNS


def render_app(demo_mode: bool = False) -> None:
    if demo_mode:
        render_demo_page()
    else:
        state = render_sidebar()
        render_live_page(state)


def render_live_page(state: SidebarState) -> None:
    st.title("Crypto sentiment tracker")
    st.markdown(
        "Visualization of public sentiment based on keywords and further comparison to actual price of cryptocurrencies"
    )

    sentiment_tab, engine_tab, finance_tab, backtest_tab, benchmark_tab = st.tabs(
        ["Sentiment", "Engine", "Finance", "Backtest", "Benchmark"]
    )

    tabs = {
        "sentiment": sentiment_tab,
        "engine": engine_tab,
        "finance": finance_tab,
        "backtest": backtest_tab,
        "benchmark": benchmark_tab,
    }

    if not state.run:
        with tabs["sentiment"]:
            st.info(
                "Configure the settings in the sidebar and click 'Run Analysis' to see results."
            )
        with tabs["finance"]:
            st.info("Run analysis to see finance results.")
        with tabs["backtest"]:
            st.info("Run analysis to see backtest results.")
        with tabs["benchmark"]:
            if state.benchtest:
                show_benchmark_data()
            else:
                st.info("Run model benchmarks from sidebar.")
        return

    config = sidebar_state_to_config(state)

    with st.spinner("Running analysis..."):
        result = run_analysis(config)

    render_result_tabs(result, state, tabs)


def render_result_tabs(
    result: AnalysisResult, state: SidebarState, tabs: dict[str, Any]
) -> None:
    with tabs["sentiment"]:
        st.plotly_chart(
            plot_sentiment_with_price(result.merged_df, state.selected_coin)
        )
        st.plotly_chart(plot_sentiment_timeline(result.merged_df, state.selected_coin))
        st.plotly_chart(plot_sentiment_vs_price(result.merged_df))
        lead_lag_df = compute_lead_lag(
            result.merged_df, state.lag_hours, state.lag_step_min, state.metric_choice
        )
        st.plotly_chart(plot_lag_correlation(lead_lag_df))

    with tabs["engine"]:
        indic_state = replace(sidebar_to_indicator(state), use_sma=True)
        indicator_df = add_indicators_to_df(result.merged_df, indic_state)
        signal_df = build_signal_df(indicator_df)
        signal_cols = [col for col in SIGNAL_COLUMNS if col in signal_df.columns]

        event_rows = signal_df[signal_df[signal_cols].any(axis=1)]

        st.dataframe(
            event_rows[["timestamp", "price", "sentiment", *signal_cols]],
            hide_index=True,
        )
        st.plotly_chart(plot_signal(signal_df))
    with tabs["finance"]:
        if not (state.use_sma or state.use_macd or state.use_rsi):
            st.info(
                "Enable any financial indicators in Advanced settings to see results."
            )
        indic_state = sidebar_to_indicator(state)
        indicators_df = add_indicators_with_cache(result.price_df, indic_state)
        if state.use_sma:
            st.plotly_chart(
                plot_price_with_sma(
                    indicators_df,
                    state.selected_coin,
                    sma_cols=[f"sma_{state.sma_fast}", f"sma_{state.sma_slow}"],
                )
            )

        if state.use_macd:
            fig = plot_macd(indicators_df)
            if fig is not None:
                st.plotly_chart(fig)
            else:
                st.warning("MACD data is not available.")

        if state.use_rsi:
            fig = plot_rsi(indicators_df, rsi_col=f"rsi_{state.rsi_period}")
            if fig is not None:
                st.plotly_chart(fig)
            else:
                st.warning("RSI data is not available.")

    with tabs["backtest"]:
        if not state.backtest:
            st.info("Enable backtest in Advanced settings to see backtest")
        else:
            df_bt, stats = run_backtest(
                result.merged_df, state.cost_bps, state.slip_bps
            )
            st.plotly_chart(plot_equity(df_bt))
            st.plotly_chart(plot_drawdown(df_bt))
            st.dataframe(pd.DataFrame([stats]), hide_index=True)

    with tabs["benchmark"]:
        if not state.benchtest:
            st.info("Run model benchmarks from sidebar.")
        else:
            show_benchmark_data()
