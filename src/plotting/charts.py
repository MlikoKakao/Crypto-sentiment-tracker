import plotly.express as px
import pandas as pd
from src.processing.smoothing import apply_loess
import plotly.graph_objects as go

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
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    fig = px.scatter(
        df,
        x="sentiment",
        y="price",
        color="source",
        hover_data=["date_str", "source"],
        title="Sentiment vs Price",
        labels={"sentiment": "Sentiment Score", "price": "Price (USD)", "date_str":"Date", "source": "Source of post"}
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_sentiment_timeline(df: pd.DataFrame, coin: str):
    df = apply_loess(df, x_col="timestamp",y_col="sentiment",frac=0.3)
    fig = px.line(
        df,
        x="timestamp",
        y="sentiment",
        title=f"{coin.capitalize()} Sentiment Over Time",
        labels={"timestamp": "Date", "sentiment": "Sentiment Score"},
        markers=True
    )
    fig.update_traces(line=dict(width=2))
    
    #LOESS - smoothed sentiment over time
    fig.add_scatter(
        x=df["timestamp"],
        y=df["sentiment_loess"],
        mode="lines",
        name="Smoothed (LOESS)",
        line=dict(width=3, dash="dot")
    )

    return fig

#Graph showing LOESS/BTC price
def plot_sentiment_with_price(df: pd.DataFrame, coin:str):
    df = apply_loess(df, x_col="timestamp",y_col="sentiment",frac=0.3)
    fig = go.Figure()

    #Price trace
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["price"],
        name=f"{coin.capitalize()} Price",
        yaxis="y2",
        line=dict(color="gray",width=2),
        hoverinfo="x+y"
    ))

    #Sentiment/LOESS
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["sentiment_loess"],
        name="Smoothed Sentiment",
        yaxis="y1",
        line=dict(color="blue",width=3, dash="dot"),
        hoverinfo="x+y"
    ))

    #Layout update to show both
    padding = 0.05
    winner_number = max(-df["sentiment_loess"].min(), df["sentiment_loess"].max())
    sentiment_range = [-winner_number - padding, winner_number + padding]

    fig.update_layout(
        title=f"{coin.capitalize()} Sentiment vs Price Over Time",
        xaxis=dict(title="Date"),
        yaxis=dict(
            title=dict(text="Sentiment Score", font=dict(color="blue")),
            range=sentiment_range,
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title=dict(text="Price (USD)", font=dict(color="gray")),
            overlaying="y",
            side="right",
            tickfont=dict(color="gray")
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40,r=40,t=50,b=40),
        hovermode="x unified"
    )

    return fig
