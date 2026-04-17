import streamlit as st
from src.app.dto import AnalysisResult
from src.domain.market.indicators import add_indicators_to_df
from src.presentation.charts import plot_sentiment_timeline, plot_sentiment_vs_price, plot_sentiment_with_price, plot_lag_correlation, plot_price_with_sma, plot_rsi, plot_macd 
from src.presentation.sidebar import SidebarState, render_sidebar, sidebar_state_to_config
from src.app.use_cases.run_analysis import run_analysis
from src.domain.analysis.lead_lag import compute_lead_lag
from src.presentation.demo_view import render_demo_page

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
    if not state.run:
        st.info("Configure the settings in the sidebar and click 'Run Analysis' to see results.")
        return
    
    config = sidebar_state_to_config(state)

    with st.spinner("Running analysis..."):
            result = run_analysis(config)
    st.success("Analysis completed!")

    render_result_tabs(result, state)

def render_result_tabs(result: AnalysisResult, state: SidebarState) -> None:
    sentiment_tab, finance_tab = st.tabs(["Sentiment", "Finance"])
    
    with sentiment_tab:
        st.plotly_chart(plot_sentiment_with_price(result.merged_df, state.selected_coin))
        st.plotly_chart(plot_sentiment_timeline(result.merged_df, state.selected_coin))
        st.plotly_chart(plot_sentiment_vs_price(result.merged_df))
        lead_lag_df = compute_lead_lag(result.merged_df, state.lag_hours, state.lag_step_min, state.metric_choice)
        st.plotly_chart(plot_lag_correlation(lead_lag_df))

    with finance_tab:
        indicators_df = add_indicators_to_df(
             result.price_df,
             price_col="price",
             use_sma=state.use_sma,
             use_rsi=state.use_rsi,
             use_macd=state.use_macd,
             sma_windows=(state.sma_fast, state.sma_slow),
             rsi_period=state.rsi_period,

        )
        if state.use_sma:
            st.plotly_chart(plot_price_with_sma(indicators_df, state.selected_coin, sma_cols=[f"sma_{state.sma_fast}", f"sma_{state.sma_slow}"]))
        if state.use_macd:
            st.plotly_chart(plot_macd(indicators_df))
        if state.use_rsi:
            st.plotly_chart(plot_rsi(indicators_df))
