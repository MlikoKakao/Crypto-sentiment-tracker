import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pytz import utc
from datetime import timedelta

from src.utils.helpers import load_csv,save_csv, filter_date_range, map_to_cryptopanic_symbol
from src.scraping import fetch_reddit_posts, get_price_history
from src.scraping.news_scraper import fetch_news_posts
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
days = st.sidebar.selectbox("Price history in days", DEFAULT_DAYS, help="Choosing day range longer than 90 days causes to only show price point once per day.")
analyzer_choice = st.sidebar.selectbox("Choose sentiment analyzer:", ANALYZER_UI_LABELS, help="VADER - all-rounder, decent speed and analysis; Text-Blob - fastest, but least accurate, " \
                                                                                                    "Twitter-RoBERTa - slowest(can take up to 40 seconds), but most accurate, conservative")


end_date = pd.Timestamp.now(tz=utc)
start_date = end_date - timedelta(days=int(days))

#Load csv(would be better to only read on button press)
df = load_csv("data/merged_sentiment_price.csv", parse_dates=["timestamp"])

#Fetching and merging all data
if st.sidebar.button("Run Analysis"):
    
    with st.spinner("Fetching news..."):
        cryptopanic_coin = map_to_cryptopanic_symbol(selected_coin)
        news_df = fetch_news_posts(cryptopanic_coin, num_posts)
        news_df["source"] = "news"
        news_path = f"data/{selected_coin}_news_posts.csv"
        save_csv(news_df, news_path)

    with st.spinner("Fetching Reddit posts..."):
        reddit_df = fetch_reddit_posts(selected_coin, num_posts, start_date=start_date, end_date=end_date)
        reddit_df["source"] = "reddit"
        reddit_path = f"data/{selected_coin}_posts.csv"
        save_csv(reddit_df, reddit_path)

    with st.spinner("Analyzing sentiment..."):
        news_sentiment_path = get_data_path(selected_coin, "news_sentiment")
        add_sentiment_to_file(news_path, news_sentiment_path, analyzer_choice)

        sentiment_path = get_data_path(selected_coin, "sentiment")
        add_sentiment_to_file(reddit_path, sentiment_path, analyzer_choice)

    with st.spinner("Combining sentiment..."):
        news_sent = load_csv(news_sentiment_path)
        reddit_sent = load_csv(sentiment_path)
        combined_df = pd.concat([news_sent, reddit_sent], ignore_index=True)
        combined_df = combined_df.sort_values("timestamp")
        save_csv(combined_df, "data/combined_sentiment.csv")

    with st.spinner("Fetching price data..."):
        price_df = get_price_history(selected_coin, days)
        price_path = get_data_path(selected_coin, "prices")
        save_csv(price_df, price_path)

    with st.spinner("Merging sentiment and price data..."):
        merged_path = get_data_path(selected_coin, "merged")
        merge_sentiment_and_price("data/combined_sentiment.csv", price_path, merged_path)

    st.success("Data ready, showing visualization:")
    st.session_state["merged_path"] = merged_path

if "merged_path" in st.session_state and os.path.exists(st.session_state["merged_path"]):
    #Timestamp things
    df =load_csv(st.session_state["merged_path"], parse_dates=["timestamp"])
    min_date = (df["timestamp"].max()-timedelta(days=int(days))).to_pydatetime()
    max_date = df["timestamp"].max().to_pydatetime()
    selected_range = st.slider(
        "Select time range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    #Time filter
    df = filter_date_range(df, selected_range[0],selected_range[1])

    #Price plot
    st.plotly_chart(plot_price_time_series(df, selected_coin), use_container_width=True)
    #Sentiment vs price
    st.plotly_chart(plot_sentiment_vs_price(df), use_container_width=True)
    #Sentiment timeline
    st.plotly_chart(plot_sentiment_timeline(df, selected_coin), use_container_width=True)

    #Average sentiment
    avg_sent = df["sentiment"].mean()
    st.metric(label=f"Average Sentiment {selected_label}", value=f"{avg_sent:.3f}")

else:
    st.info("Run the analysis from the sidebar to see visualization")

