import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import altair as alt


#Page header
st.set_page_config(page_title="Crypto Sentiment Tracker",layout="wide")
#Load data
df = pd.read_csv("data/merged_sentiment_price.csv", parse_dates=["timestamp"])

#Page main intro
st.title("Crypto sentiment tracker")
st.markdown("Visualization of Reddit sentiment based on keywords and further comparison to actual price of Bitcoin")

#Time range slider
min_date = df["timestamp"].min().to_pydatetime()
max_date = df["timestamp"].max().to_pydatetime()
selected_range = st.slider("Select time range:", min_value=min_date, max_value=max_date, value=(min_date, max_date))

#Filter by time
df = df[(df["timestamp"] >= selected_range[0]) & (df["timestamp"] <=selected_range[1])]

#Smoothen sentiment
smoothen =(
    alt.Chart(df)
    .mark_bar(color="lightgray")
    .encode(
        x = alt.X("timestamp:T", title="Date"),
        y = alt.Y("mean(sentiment):Q",title="Average Sentiment"),
        tooltip=["timestamp:T", "mean(sentiment):Q"]
    )
    .properties(title="Smoothed Sentiment Over Time", width="container")
)

#Line 1: Drawing sentiment
st.altair_chart(smoothen, use_container_width=True)
    

#Line 2: Price over time
fig_price = px.line(df, x="timestamp",y="price", title="Bitcoin Price Over Time")
st.plotly_chart(fig_price, use_container_width=True)

#Line 3: Scatter of sentiment vs price
fig_scatter = px.scatter(df, x="sentiment", y="price", hover_data=["timestamp"], title="Sentiment vs Price")
st.plotly_chart(fig_scatter, use_container_width=True)