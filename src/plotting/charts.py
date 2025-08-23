import plotly.express as px
import pandas as pd
from src.processing.smoothing import apply_loess
import plotly.graph_objects as go
import streamlit as st

def plot_price_time_series(df: pd.DataFrame, coin:str):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

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
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

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
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors= "coerce")
    df = df.dropna(subset=["timestamp", "sentiment"])

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
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

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
    padding = 0.1
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

def plot_lag_correlation(feats: pd.DataFrame, unit: str = "min", metric_label: str= "r"):
    if feats is None or feats.empty or not {"lag_seconds", "r"}.issubset(feats.columns):
        st.error("Features DF must include lag_second and r")
        import plotly.graph_objects as go
        return go.Figure()
    
    df = feats.copy()

    if unit == "min":
        df["lag_axis"] = (df["lag_seconds"] / 60).astype(float)
        x_label = "Lag (minutes)"
    elif unit == "hours":
        df["lag_axis"] = (df["lag_seconds"] / 3600).astype(float)
        x_label = "Lag (hours)"
    else:
        df["lag_axis"]= df["lag_seconds"].astype(float)
        x_label = "Lag (seconds)"

    df = df.sort_values("lag_axis")

    fig = px.line(
        df,
        x="lag_axis",
        y="r",
        title="Correlation vs Lag (positive = sentiment leads)",
        labels={"lag_axis": x_label, "r": metric_label},
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(margin=dict(l=20,r=20,t=50,b=20))
    #Zero line
    fig.add_hline(y=0, line_dash = "dot", line_width=1)

    valid = df.dropna(subset=["r"])
    if not valid.empty:
        best_idx = valid["r"].abs().idxmax()
        best_x = float(valid.loc[best_idx, "lag_axis"])
        best_r = float(valid.loc[best_idx, "r"])
        fig.add_vline(x=best_x,line_dash="dash",line_width=1,line_color="gray")
        fig.add_scatter(x=[best_x],y=[best_r],mode="markers",
                        name=f"Best lag: {best_x:g} {unit}", marker=dict(size=9))
    return fig

def plot_equity(df_bt: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bt["timestamp"], y = df_bt["eq_strategy"], name="Strategy", mode="lines"))
    fig.add_trace(go.Scatter(x=df_bt["timestamp"], y = df_bt["eq_hodl"], name="HODL", mode = "lines"))
    fig.update_layout(title="Equity Curve (Strategy vs HODL)", yaxis_title="Growth (x)", xaxis_title="Time")
    return fig

def plot_drawdown(df_bt: pd.DataFrame):
    fig = px.area(df_bt, x="timestamp", y="dd", title="Drawdown (Strategy)")
    fig.update_yaxes(ticksuffix="", tickformat=".0%")
    return fig

def plot_price_with_sma(df, coin, sma_cols):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price"], name="Price", mode="lines"))
    for col in sma_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=col, mode="lines"))
    fig.update_layout(
        title=f"{coin.upper()} Price + SMA",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_dark",
    )
    return fig

def plot_rsi(df, rsi_col="rsi_14"):
    if rsi_col not in df.columns: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[rsi_col], name=rsi_col, mode="lines"))
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, line_width=0)
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="RSI", xaxis_title="Date", yaxis_title="Value", template="plotly_dark")
    return fig

def plot_macd(df):
    if not {"macd", "macd_signal", "macd_hist"}.issubset(df.columns): return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd"], name="MACD", mode="lines"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd_signal"], name="Signal", mode="lines"))
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["macd_hist"], name="Hist", opacity=0.4))
    fig.update_layout(title="MACD", xaxis_title="Date", yaxis_title="Value", barmode="overlay",
                      template="plotly_dark")
    return fig