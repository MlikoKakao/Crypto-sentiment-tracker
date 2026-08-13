import plotly.express as px
import pandas as pd
from src.domain.market.smoothing import apply_loess
import plotly.graph_objects as go
import streamlit as st
from src.shared.helpers import normalize_timestamp_column
from src.domain.signals.engine import SIGNAL_COLUMNS
from statistics import median
from typing import cast, Sequence
from src.presentation.translations import TEXT


# Not needed right now, but keeping just in case.
def plot_price_time_series(df: pd.DataFrame, coin: str, language: str = "en"):
    text = TEXT[language]
    if not df.empty:
        df = df.copy()
        df = normalize_timestamp_column(df, drop_invalid=True)

        fig = px.line(
            df,
            x="timestamp",
            y="price",
            title=text["price_over_time"].format(coin=coin.capitalize()),
            labels={"timestamp": text["date"], "price": text["price_usd"]},
        )

        fig.update_traces(line=dict(width=2))
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig
    else:
        st.warning(text["no_price_data"])
        return go.Figure()


def plot_sentiment_vs_price(df: pd.DataFrame, language: str = "en"):
    text = TEXT[language]
    if df.empty:
        return go.Figure()
    df = df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)
    df.loc[:, "date_str"] = cast(
        pd.Series, df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    )
    fig = px.scatter(
        df,
        x="sentiment",
        y="price",
        color="source",
        hover_data=["date_str", "source"],
        title=text["sentiment_vs_price"],
        labels={
            "sentiment": text["sentiment_score"],
            "price": text["price_usd"],
            "date_str": text["date"],
            "source": text["post_source"],
        },
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_sentiment_timeline(df: pd.DataFrame, coin: str, language: str = "en"):
    text = TEXT[language]
    if df.empty:
        return go.Figure()
    df = df.copy()
    df = normalize_timestamp_column(df)
    df.loc[:, "sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["timestamp", "sentiment"])
    df = apply_loess(df, x_col="timestamp", y_col="sentiment", frac=0.3)
    fig = px.line(
        df,
        x="timestamp",
        y="sentiment",
        title=text["sentiment_over_time"].format(coin=coin.capitalize()),
        labels={"timestamp": text["date"], "sentiment": text["sentiment_score"]},
        markers=True,
    )
    fig.update_traces(line=dict(width=2))

    # LOESS - smoothed sentiment over time
    fig.add_scatter(
        x=df["timestamp"],
        y=df["sentiment_loess"],
        mode="lines",
        line=dict(width=3, dash="dot"),
        showlegend=False,
    )

    return fig


# Graph showing LOESS/BTC price
def plot_sentiment_with_price(df: pd.DataFrame, coin: str, language: str = "en"):
    text = TEXT[language]
    if df.empty:
        return go.Figure()
    df = df.copy()
    df = normalize_timestamp_column(df, drop_invalid=True)

    df = apply_loess(df, x_col="timestamp", y_col="sentiment", frac=0.3)
    fig = go.Figure()

    # Price trace
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["price"],
            name=f"{coin.capitalize()} {text['price']}",
            yaxis="y2",
            line=dict(color="gray", width=2),
            hoverinfo="x+y",
        )
    )

    # Sentiment/LOESS
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["sentiment_loess"],
            name=text["smoothed_sentiment"],
            yaxis="y1",
            line=dict(color="blue", width=3, dash="dot"),
            hoverinfo="x+y",
        )
    )

    # Layout update to show both
    padding = 0.1
    winner_number = max(-df["sentiment_loess"].min(), df["sentiment_loess"].max())
    sentiment_range = [-winner_number - padding, winner_number + padding]

    fig.update_layout(
        title=text["sentiment_vs_price_over_time"].format(coin=coin.upper()),
        xaxis=dict(title=text["date"]),
        yaxis=dict(
            title=dict(text=text["sentiment_score"], font=dict(color="blue")),
            range=sentiment_range,
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title=dict(text=text["price_usd"], font=dict(color="gray")),
            overlaying="y",
            side="right",
            tickfont=dict(color="gray"),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
    )

    return fig


def plot_lag_correlation(
    feats: pd.DataFrame, unit: str = "min", metric_label: str = "r", language: str = "en"
) -> go.Figure:
    text = TEXT[language]
    if feats.empty or not {"lag_seconds", "r"}.issubset(feats.columns):
        st.error("Features DF must include lag_seconds and r")
        return go.Figure()
    df = feats.copy()

    if unit == "min":
        df["lag_axis"] = (df["lag_seconds"] / 60).astype(float)
        x_label = text["lag_minutes"]
    elif unit == "hours":
        df["lag_axis"] = (df["lag_seconds"] / 3600).astype(float)
        x_label = text["lag_hours"]
    else:
        df["lag_axis"] = df["lag_seconds"].astype(float)
        x_label = text["lag_seconds"]

    df = df.sort_values("lag_axis")

    fig = px.line(
        df,
        x="lag_axis",
        y="r",
        title=text["correlation_vs_lag"],
        labels={"lag_axis": x_label, "r": metric_label},
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_width=1)

    df["lag_axis"] = pd.to_numeric(df["lag_axis"], errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce")

    valid = df.dropna(subset=["r", "lag_axis"])
    if not valid.empty:
        best_pos = valid["r"].abs().to_numpy().argmax()
        best_x = float(valid["lag_axis"].to_numpy()[best_pos])
        best_r = float(valid["r"].to_numpy()[best_pos])
        fig.add_vline(x=best_x, line_dash="dash", line_width=1, line_color="gray")
        fig.add_scatter(
            x=[best_x],
            y=[best_r],
            mode="markers",
            name=text["best_lag"].format(value=best_x, unit=unit),
            marker=dict(size=9),
        )
    return fig


def plot_equity(df_bt: pd.DataFrame, language: str = "en"):
    text = TEXT[language]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_bt["timestamp"], y=df_bt["eq_strategy"], name=text["strategy"], mode="lines"
        )
    )
    fig.add_trace(
        go.Scatter(x=df_bt["timestamp"], y=df_bt["eq_hodl"], name="HODL", mode="lines")
    )
    fig.update_layout(
        title=text["equity_curve"], yaxis_title=text["growth"], xaxis_title=text["time"],
    )
    return fig


def plot_drawdown(df_bt: pd.DataFrame, language: str = "en"):
    fig = px.area(df_bt, x="timestamp", y="dd", title=TEXT[language]["drawdown"])
    fig.update_yaxes(ticksuffix="", tickformat=".0%")
    return fig


def plot_price_with_sma(
    df: pd.DataFrame, coin: str, sma_cols: Sequence[str], language: str = "en"
):
    text = TEXT[language]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["price"], name=text["price"], mode="lines")
    )
    for col in sma_cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[col], name=col, mode="lines")
            )
    fig.update_layout(
        title=text["price_sma"].format(coin=coin.upper()),
        xaxis_title=text["date"], yaxis_title=text["price"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_dark",
    )
    return fig


def plot_rsi(df: pd.DataFrame, rsi_col: str = "rsi_14", language: str = "en"):
    if rsi_col not in df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df[rsi_col], name=rsi_col, mode="lines")
    )
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, line_width=0)
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        title="RSI", xaxis_title=TEXT[language]["date"], yaxis_title=TEXT[language]["value"], template="plotly_dark"
    )
    return fig


def plot_macd(df: pd.DataFrame, language: str = "en"):
    if not {"macd", "macd_signal", "macd_hist"}.issubset(df.columns):
        return None
    df = normalize_timestamp_column(df.copy(), drop_invalid=True)
    ts = df["timestamp"]
    timestamps = sorted(pd.Timestamp(value) for value in ts.dropna())
    if len(timestamps) > 1:
        deltas = [
            (current - previous).total_seconds()
            for previous, current in zip(timestamps, timestamps[1:])
        ]
        bar_width = median(deltas) * 1000 * 0.8
    else:
        bar_width = 24 * 60 * 60 * 1000

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=df["macd"], name="MACD", mode="lines"))
    fig.add_trace(go.Scatter(x=ts, y=df["macd_signal"], name="Signal", mode="lines"))
    fig.add_trace(
        go.Bar(x=ts, y=df["macd_hist"], name="Hist", width=bar_width, opacity=0.7)
    )
    fig.add_hline(y=0, line_dash="dot", opacity=0.6)
    fig.update_layout(
        title="MACD",
        xaxis_title=TEXT[language]["date"],
        yaxis_title=TEXT[language]["value"],
        barmode="overlay",
        template="plotly_dark",
    )
    return fig


def plot_signal(df: pd.DataFrame, language: str = "en") -> go.Figure:
    df = normalize_timestamp_column(df.copy(), drop_invalid=True)
    df = df.dropna(subset=["timestamp", "sentiment"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["price"],
            mode="lines",
            name=TEXT[language]["price"],
        )
    )
    signal_cols = [col for col in SIGNAL_COLUMNS if col in df.columns]
    for signal_col in signal_cols:
        signal_rows = df[df[signal_col]]

        fig.add_trace(
            go.Scatter(
                x=signal_rows["timestamp"],
                y=signal_rows["price"],
                mode="markers",
                name=signal_col,
            )
        )

    fig.update_traces(line=dict(width=2), marker=dict(size=15, symbol="arrow"))
    return fig
