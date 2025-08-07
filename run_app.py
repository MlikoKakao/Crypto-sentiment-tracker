import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pytz import utc
from datetime import timedelta
import logging

from src.utils.helpers import load_csv,save_csv, filter_date_range, map_to_cryptopanic_symbol, is_file_fresh
from src.scraping import fetch_reddit_posts, get_price_history
from src.scraping.news_scraper import fetch_news_posts
from src.sentiment import add_sentiment_to_file
from src.processing.merge_data import merge_sentiment_and_price
from src.utils.cache import load_cached_csv, cache_csv
from config.settings import(
    COINS_UI_LABELS, COINS_UI_TO_SYMBOL, DEFAULT_DAYS, ANALYZER_UI_LABELS, get_data_path, POSTS_KIND
)
from src.plotting.charts import (
    plot_price_time_series,
    plot_sentiment_timeline,
    plot_sentiment_vs_price,
    plot_sentiment_with_price
)
logger = logging.getLogger(__name__)

import os

#Page header
st.set_page_config(page_title="Crypto Sentiment Tracker",layout="wide")

#Page main intro
st.title("Crypto sentiment tracker")
st.markdown("Visualization of public sentiment based on keywords and further comparison to actual price of cryptocurrencies")

#Config
st.sidebar.header("Configuration")
selected_label = st.sidebar.selectbox("Choose cryptocurrency", COINS_UI_LABELS)
selected_coin = COINS_UI_TO_SYMBOL[selected_label]
num_posts = st.sidebar.slider("Number of posts to fetch", min_value = 100, max_value=1000, step=100, value=300)
days = st.sidebar.selectbox("Price history in days", DEFAULT_DAYS, help="Choosing day range longer than 90 days causes to only show price point once per day.")
analyzer_choice = st.sidebar.selectbox("Choose sentiment analyzer:", ANALYZER_UI_LABELS, help="VADER - all-rounder, decent speed and analysis; Text-Blob - fastest, but least accurate, " \
                                                                                                    "Twitter-RoBERTa - slowest(can take up to a minute depending on size), but most accurate, conservative")
posts_choice = st.sidebar.selectbox("Choose which kind of posts you want to analyze:", POSTS_KIND)


end_date = pd.Timestamp.now(tz=utc)
start_date = end_date - timedelta(days=int(days))



#Fetching and merging all data
if st.sidebar.button("Run Analysis"):
    #Check whether to use
    use_news = False
    use_reddit = False

    #Set sentiment path already for caching
    news_sentiment_path = get_data_path(selected_coin, "news_sentiment")
    reddit_sentiment_path = get_data_path(selected_coin, "sentiment")

    #Set data path for caching
    news_path = f"data/{selected_coin}_news_posts.csv"
    reddit_path = f"data/{selected_coin}_posts.csv"
    
    #News
    if posts_choice in ("All", "News"):
        #news_settings = {
        #    "source": "news",
        #    "coin": selected_coin,
        #    "num_posts": num_posts,
        #    "start_date": str(start_date.date()),
        #    "end_date": str(end_date.date())
        #}

        
        if is_file_fresh(news_path, freshness_minutes=30):
            logger.info("Using cached news data.")
            news_df = load_csv(news_path)
        else:
            try:
        #news_df = load_cached_csv(news_settings, freshness_minutes=30)
        #if news_df is None:
                with st.spinner("Fetching news..."):
                    cryptopanic_coin = map_to_cryptopanic_symbol(selected_coin)
                    news_df = fetch_news_posts(cryptopanic_coin, num_posts)
                    news_df["source"] = "news"
                    save_csv(news_df, news_path)
                    #cache_csv(news_df, news_settings)
        #use_news = True
            except Exception as e:
                logger.warning(f"Couldn't fetch news: {e}")
                logger.warning("OLD NEWS DATA WILL BE USED, MAKE SURE TO CHECK THE DATES")
                st.warning("OLD NEWS DATA WILL BE USED, MAKE SURE TO CHECK THE DATES")
                
                #From news for now just using old data, if wanna get new ones have to 
                #uncomment cache logic, delete is_file_fresh block, else: and try:except
                
        if os.path.exists(news_path):
            use_news = True
    #Reddit
    if posts_choice in ("All", "Reddit"):
        reddit_settings = {
            "source": "reddit",
            "coin": selected_coin,
            "analyzer": analyzer_choice,
            "start_date": start_date.tz_convert(None).isoformat(timespec="seconds"),
            "end_date": end_date.tz_convert(None).isoformat(timespec="seconds"),
            "num_posts": num_posts
        }
        reddit_df = load_cached_csv(reddit_settings, freshness_minutes = 30)
        if reddit_df is None:
                with st.spinner("Fetching Reddit posts..."):
                    reddit_df = fetch_reddit_posts(selected_coin, num_posts, start_date=start_date, end_date=end_date)
                    reddit_df["source"] = "reddit"
                    cache_csv(reddit_df, reddit_settings)
        use_reddit = True

    with st.spinner("Analyzing sentiment..."):
        if use_news:
            #change to cache_setting=news_settings when switching logic
            add_sentiment_to_file(news_path, news_sentiment_path, analyzer_choice, cache_settings=True)
        else:
            logger.warning("News sentiment will not be included")
        if use_reddit:
            add_sentiment_to_file(reddit_path, reddit_sentiment_path, analyzer_choice, cache_settings=reddit_settings)
        else:
            logger.warning("Reddit sentiment will not be included")

    with st.spinner("Combining sentiment..."):
        dfs = []
        if use_news and os.path.exists(news_sentiment_path):
            news_sent = load_csv(news_sentiment_path)
            dfs.append(news_sent)
        if use_reddit and os.path.exists(reddit_sentiment_path):
            reddit_sent = load_csv(reddit_sentiment_path)
            dfs.append(reddit_sent)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values("timestamp")
            save_csv(combined_df, "data/combined_sentiment.csv")
        else:
            logger.error(f"No sentiment data. use_news={use_news}, use_reddit={use_reddit}")
            st.error("No sentiment data could be loaded. Check API limits or local files.")
            st.stop()

    #Price data
    with st.spinner("Fetching price data..."):
        price_df = get_price_history(selected_coin, days)
        price_path = get_data_path(selected_coin, "prices")
        save_csv(price_df, price_path)

    #Merge of price and data
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

    #Sentiment vs price timeline (smoothed)
    st.plotly_chart(plot_sentiment_with_price(df, selected_coin), use_container_width=True)

    #Average sentiment
    avg_sent = df["sentiment"].mean()
    st.metric(label=f"Average Sentiment {selected_label}", value=f"{avg_sent:.3f}")

else:
    st.info("Run the analysis from the sidebar to see visualization")

