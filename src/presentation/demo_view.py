import streamlit as st
from src.presentation.charts import (
    plot_sentiment_timeline,
    plot_sentiment_vs_price,
    plot_sentiment_with_price,
    plot_lag_correlation,
    plot_price_with_sma,
    plot_rsi,
    plot_macd,
)
from src.domain.analysis.lead_lag import compute_lead_lag
import pandas as pd
from src.presentation.sidebar import render_sidebar
from src.shared.helpers import normalize_timestamp_column
from src.presentation.translations import TEXT


def render_demo_page() -> None:
    state = render_sidebar()
    text = TEXT[state.language]
    st.title(text["demo_title"])
    st.markdown(text["description"])
    render_demo_result_tabs(state.language)


def render_demo_result_tabs(language: str) -> None:
    text = TEXT[language]
    sentiment_tab, finance_tab = st.tabs([text["tab_sentiment"], text["tab_finance"]])
    demo_merged_df = load_demo_merged_df()
    with sentiment_tab:
        st.plotly_chart(
            plot_sentiment_with_price(demo_merged_df, "BTC", language),
            key="demo_sentiment_price_chart",
        )
        st.plotly_chart(
            plot_sentiment_timeline(demo_merged_df, "BTC", language),
            key="demo_sentiment_timeline_chart",
        )
        st.plotly_chart(
            plot_sentiment_vs_price(demo_merged_df, language),
            key="demo_sentiment_vs_price_chart",
        )
        lead_lag_df = compute_lead_lag(demo_merged_df)
        st.plotly_chart(
            plot_lag_correlation(lead_lag_df, language=language),
            key="demo_lag_correlation_chart",
        )

    with finance_tab:
        st.plotly_chart(
            plot_price_with_sma(demo_merged_df, "BTC", sma_cols=[f"sma_20", f"sma_50"], language=language),
            key="demo_sma_chart",
        )
        st.plotly_chart(plot_macd(demo_merged_df, language), key="demo_macd_chart")
        st.plotly_chart(plot_rsi(demo_merged_df, language=language), key="demo_rsi_chart")


def load_demo_merged_df() -> pd.DataFrame:
    demo_merged_df = pd.read_csv("data/demo/bitcoin_merged.csv")
    return normalize_timestamp_column(demo_merged_df, drop_invalid=True)
