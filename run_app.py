import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.scraping import fetch_reddit_posts, get_price_history
from src.sentiment import add_sentiment_to_file
from src.processing.merge_data import merge_sentiment_and_price
import os

#Page header
st.set_page_config(page_title="Crypto Sentiment Tracker",layout="wide")

#Page main intro
st.title("Crypto sentiment tracker")
st.markdown("Visualization of Reddit sentiment based on keywords and further comparison to actual price of cryptocurrencies")

#Config
st.sidebar.header("Configuration")
selected_coin = st.sidebar.selectbox("Choose cryptocurrency", ["Bitcoin","Ethereum"])
num_posts = st.sidebar.slider("Number of Reddit posts to fetch", min_value = 100, max_value=1000, step=100, value=300)
days = st.sidebar.selectbox("Price history in days", ["1","7","30","90","180","365"])

#Loada csv(would be better to only read on button press)
df = pd.read_csv("data/merged_sentiment_price.csv", parse_dates=["timestamp"])

#Fetching and merging all data
if st.sidebar.button("Run Analysis"):
    
    with st.spinner("Fetching Reddit posts..."):
        reddit_df = fetch_reddit_posts(selected_coin, num_posts)
        reddit_path = f"data/{selected_coin}_posts.csv"
        reddit_df.to_csv(reddit_path, index=False)

    with st.spinner("Analyzing sentiment..."):
        sentiment_path = f"data/{selected_coin}_sentiment.csv"
        add_sentiment_to_file(reddit_path, sentiment_path)

    with st.spinner("Fetching price data..."):
        price_df = get_price_history(selected_coin, days)
        price_path = f"data/{selected_coin}_prices.csv"
        price_df.to_csv(price_path, index=False)

    with st.spinner("Merging sentiment and price data..."):
        merged_path = f"data/{selected_coin}_merged.csv"
        merge_sentiment_and_price(sentiment_path, price_path, merged_path)

    st.success("Data ready, showing visualization:")
    st.session_state["merged_path"] = merged_path

if "merged_path" in st.session_state and os.path.exists(st.session_state["merged_path"]):
    #Timestamp things
    df = pd.read_csv(st.session_state["merged_path"], parse_dates=["timestamp"])
    min_date = (df["timestamp"].max()-days).to_pydatetime()
    max_date = df["timestamp"].max().to_pydatetime()
    selected_range = st.slider("Select time range:", min_value=min_date, max_value=max_date, value=(min_date, max_date)) 
    #Time filter
    df = df[(df["timestamp"] >= selected_range[0]) & (df["timestamp"] <=selected_range[1])]

    #Sentiment vs price line
    fig_price = px.line(df,x="timestamp",y="price",title=f"{selected_coin.capitalize()} Price Over Time")
    st.plotly_chart(fig_price, use_container_width=True)

    fig_scatter = px.scatter(df, x="sentiment", y="price", hover_data=["timestamp"], title="Sentiment vs Price")
    st.plotly_chart(fig_scatter, use_container_width=True)

    avg_sent = df["sentiment"].mean()
    st.metric(label="Average Sentiment (selected range)", value=f"{avg_sent:.3f}")

else:
    st.info("Run the analysis from the sidebar to see visualization")

