import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pytz import utc
from datetime import timedelta

from src.scraping import fetch_reddit_posts, get_price_history
from src.sentiment import add_sentiment_to_file
from src.processing.merge_data import merge_sentiment_and_price
from config.settings import(
    COINS_UI_LABELS, COINS_UI_TO_SYMBOL, DEFAULT_DAYS, ANALYZER_UI_LABELS,  get_data_path
)
from src.plotting.charts import (
    plot_price_time_series,
    plot_sentiment_timeline,
    plot_sentiment_vs_price
)
import os

#Page header
st.set_page_config(page_title="Crypto Sentiment Tracker",layout="wide")

#Page main intro
st.title("Crypto sentiment tracker")
st.markdown("Visualization of Reddit sentiment based on keywords and further comparison to actual price of cryptocurrencies")

#Config
st.sidebar.header("Configuration")
selected_label = st.sidebar.selectbox("Choose cryptocurrency", COINS_UI_LABELS)
selected_coin = COINS_UI_TO_SYMBOL[selected_label]
num_posts = st.sidebar.slider("Number of Reddit posts to fetch", min_value = 100, max_value=1000, step=100, value=300)
days = st.sidebar.selectbox("Price history in days", DEFAULT_DAYS)
analyzer_choice = st.sidebar.selectbox("Choose sentiment analyzer:", ANALYZER_UI_LABELS)


end_date = pd.Timestamp.now(tz=utc)
start_date = end_date - timedelta(days=int(days))

#Load csv(would be better to only read on button press)
df = pd.read_csv("data/merged_sentiment_price.csv", parse_dates=["timestamp"])

#Fetching and merging all data
if st.sidebar.button("Run Analysis"):
    
    with st.spinner("Fetching Reddit posts..."):
        reddit_df = fetch_reddit_posts(selected_coin, num_posts, start_date=start_date, end_date=end_date)
        reddit_path = f"data/{selected_coin}_posts.csv"
        reddit_df.to_csv(reddit_path, index=False)

    with st.spinner("Analyzing sentiment..."):
        sentiment_path = get_data_path(selected_coin, "sentiment")
        add_sentiment_to_file(reddit_path, sentiment_path, analyzer_choice)

    with st.spinner("Fetching price data..."):
        price_df = get_price_history(selected_coin, days)
        price_path = get_data_path(selected_coin, "prices")
        price_df.to_csv(price_path, index=False)

    with st.spinner("Merging sentiment and price data..."):
        merged_path = get_data_path(selected_coin, "merged")
        merge_sentiment_and_price(sentiment_path, price_path, merged_path)

    st.success("Data ready, showing visualization:")
    st.session_state["merged_path"] = merged_path

if "merged_path" in st.session_state and os.path.exists(st.session_state["merged_path"]):
    #Timestamp things
    df = pd.read_csv(st.session_state["merged_path"], parse_dates=["timestamp"])
    min_date = (df["timestamp"].max()-timedelta(days=int(days))).to_pydatetime()
    max_date = df["timestamp"].max().to_pydatetime()
    selected_range = st.slider(
        "Select time range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    #Time filter
    df = df[(df["timestamp"] >= selected_range[0]) & (df["timestamp"] <=selected_range[1])]

    #Price plot
    st.plotly_chart(plot_price_time_series(df, selected_coin), use_container_width=True)
    #Sentiment vs price
    st.plotly_chart(plot_sentiment_vs_price(df), use_container_width=True)
    #Sentiment timeline
    st.plotly_chart(plot_sentiment_timeline(df, selected_coin), use_container_width=True)

    #Average sentiment
    avg_sent = df["sentiment"].mean()
    st.metric(label="Average Sentiment (selected range)", value=f"{avg_sent:.3f}")

else:
    st.info("Run the analysis from the sidebar to see visualization")

