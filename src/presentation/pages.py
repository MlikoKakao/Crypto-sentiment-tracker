import streamlit as st
from src.app.dto import AnalysisResult
from src.presentation.sidebar import SidebarState, render_sidebar, sidebar_state_to_config
from src.app.use_cases.run_analysis import run_analysis

def render_app(demo_mode: bool = False) -> None:
    if demo_mode:
         render_demo_page()
    else:
        state = render_sidebar()
        render_live_page(state)

def render_demo_page() -> None:
    st.title("Demo Mode")
    render_sidebar()
    return

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
        st.write(f"Posts: {result.posts_df.shape}")
        st.dataframe(result.posts_df.head())
    with finance_tab:
        st.write(f"Price data: {result.price_df.shape}")
        st.write(f"Merged data: {result.merged_df.shape}")
        st.dataframe(result.merged_df.head())