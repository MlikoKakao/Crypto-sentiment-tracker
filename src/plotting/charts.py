import plotly.express as px
import pandas as pd

def plot_price_time_series(df: pd.DataFrame, coin:str):
    fig = px.line(
        df,
        x="timestamp",
        y="price",
        title=f"{coin.capitalize()} Price Over Time",
        labels={"timestamp": "Date", "price": "Price (USD)"}
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_sentiment_vs_price(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="sentiment",
        y="price",
        hover_data=["timestamp"],
        title="Sentiment vs Price",
        labels={"sentiment": "Sentiment Score", "price": "Price (USD)"}
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_sentiment_timeline(df: pd.DataFrame, coin: str):
    fig = px.line(
        df,
        x="timestamp",
        y="sentiment",
        title=f"{coin.capitalize()} Sentiment Over Time",
        labels={"timestamp": "Date", "sentiment": "Sentiment Score"},
        markers=True
    )
    fig.update_traces(line=dict(width=2))
    return fig