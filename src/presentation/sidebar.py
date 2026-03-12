import streamlit as st
import pathlib
import pandas as pd


st.sidebar.header("Settings")


def render_demo_sidebar():
    if DEMO_MODE:
        st.info(
            "Demo mode is ON — using only files from data/demo (no scraping, no cache)."
        )

        # Locking coin for demo
        selected_label = "Bitcoin"
        selected_coin = "bitcoin"

        # Hard-coded demo paths
        merged_path_demo = pathlib.Path("data/demo/bitcoin_merged.csv")
        price_path_demo = pathlib.Path("data/demo/bitcoin_prices.csv")
        combined_path_demo = pathlib.Path("data/demo/bitcoin_combined_sentiment.csv")

        # Load strictly from data/demo
        try:
            merged_df = pd.read_csv(merged_path_demo, parse_dates=["timestamp"])
            price_df = pd.read_csv(price_path_demo, parse_dates=["timestamp"])
            combined_df = pd.read_csv(combined_path_demo, parse_dates=["timestamp"])
        except FileNotFoundError as e:
            st.error(f"Missing demo file: {e.filename}. Ensure it lives in data/demo/.")
            st.stop()
        with st.sidebar:
            # Minimal controls (no scraping)
            st.selectbox("Choose cryptocurrency", COINS_UI_LABELS)
            st.selectbox("Price history in days", DEFAULT_DAYS, index=1)
            posts_choice = st.sidebar.selectbox(
                "Choose which kind of posts you want to analyze:", POSTS_KIND
            )
            st.selectbox(
                "Choose sentiment analyzer:",
                ANALYZER_UI_LABELS,
                help=(
                    "VADER - all-rounder, decent speed and analysis; Text-Blob - fastest, but least accurate, "
                    "Twitter-RoBERTa - slowest(can take up to a minute depending on size), but most accurate, conservative"
                ),
            )

            st.markdown("### Indicators")
            use_sma = st.checkbox("SMA (20/50)", value=True)
            use_rsi = st.checkbox("RSI (14)", value=True)
            use_macd = st.checkbox("MACD (12,26,9)", value=True)
            sma_fast = st.number_input("SMA fast", 5, 200, 20, 1)
            sma_slow = st.number_input("SMA slow", 5, 400, 50, 1)
            rsi_period = st.number_input("RSI period", 5, 50, 14, 1)

            # Backtest controls (demo)
            backtest = st.checkbox("Run backtest", value=False)
            if backtest:
                cost_bps = st.number_input("Cost (bps)", 0.0, 100.0, 5.0, 0.5)
                slip_bps = st.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 0.5)

        # Time range slider (based on merged_df timestamps)
        min_date = (
            merged_df["timestamp"].max() - pd.to_timedelta(int(days), unit="D")
        ).to_pydatetime()
        max_date = merged_df["timestamp"].max().to_pydatetime()
        selected_range = st.slider(
            "Select time range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
        )

        # Filter view
        view = filter_date_range(merged_df, selected_range[0], selected_range[1])

        # Charts
        st.plotly_chart(
            plot_price_time_series(price_df, selected_coin), use_container_width=True
        )
        st.plotly_chart(
            plot_sentiment_timeline(view, selected_coin), use_container_width=True
        )
        st.plotly_chart(
            plot_sentiment_with_price(view, selected_coin), use_container_width=True
        )
        st.plotly_chart(plot_sentiment_vs_price(view), use_container_width=True)
        # Indicators
        if use_sma:
            st.plotly_chart(
                plot_price_with_sma(
                    view,
                    selected_coin,
                    [f"sma_{int(sma_fast)}", f"sma_{int(sma_slow)}"],
                ),
                use_container_width=True,
            )
        if use_rsi:
            fig = plot_rsi(view, rsi_col=f"rsi_{int(rsi_period)}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        if use_macd:
            fig = plot_macd(view)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if "sentiment" in view.columns and not view["sentiment"].empty:
            st.metric(
                label=f"Average Sentiment {selected_label} (demo)",
                value=f"{view['sentiment'].mean():.3f}",
            )

        # Backtest on demo data
        if backtest:
            try:
                bt, stats = run_backtest(
                    view,
                    cost_bps=cost_bps,  # type: ignore
                    slippage_bps=slip_bps,  # type: ignore
                    resample="5min",  # type: ignore
                )
                st.plotly_chart(plot_equity(bt), use_container_width=True)
                st.plotly_chart(plot_drawdown(bt), use_container_width=True)

                with st.expander("What the metrics mean"):
                    st.markdown(
                        """
                    - **CAGR** — Compounded Annual Growth Rate.
                    - **Sharpe** — Risk-adjusted return (higher is better).
                    - **MaxDD** — Maximum drawdown from peak equity.
                    - **Hit Rate** — Share of profitable trades.
                    """.strip()
                    )

                CAGR = stats.get("CAGR")
                Sharpe = stats.get("Sharpe")
                MaxDD = stats.get("MaxDD")
                Hit = stats.get("HitRate")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "CAGR",
                    "—"
                    if (CAGR is None or (isinstance(CAGR, float) and np.isnan(CAGR)))
                    else f"{CAGR:.2%}",
                )
                c2.metric(
                    "Sharpe",
                    "—"
                    if (
                        Sharpe is None
                        or (isinstance(Sharpe, float) and np.isnan(Sharpe))
                    )
                    else f"{Sharpe:.2f}",
                )
                c3.metric(
                    "MaxDD",
                    "—"
                    if (MaxDD is None or (isinstance(MaxDD, float) and np.isnan(MaxDD)))
                    else f"{abs(MaxDD):.2%}",
                )
                c4.metric(
                    "Hit Rate",
                    "—"
                    if (Hit is None or (isinstance(Hit, float) and np.isnan(Hit)))
                    else f"{Hit:.2%}",
                )
            except Exception as e:
                st.warning(f"Backtest unavailable in demo: {e}")
        # Do not proceed to scraping/caching pipeline
        st.stop()


sentiment, finance = st.tabs(["Sentiment", "Finance"])
selected_label = st.sidebar.selectbox("Choose cryptocurrency", COINS_UI_LABELS)
selected_coin = COINS_UI_TO_SYMBOL[selected_label]
num_posts = st.sidebar.slider(
    "Number of posts to fetch", min_value=100, max_value=1100, step=100, value=300
)
days = st.sidebar.selectbox(
    "Price history in days",
    DEFAULT_DAYS,
    help="Choosing day range longer than 90 days causes to only show price point once per day.",
)
analyzer_choice = st.sidebar.selectbox(
    "Choose sentiment analyzer:",
    ANALYZER_UI_LABELS,
    help=(
        "VADER - all-rounder, decent speed and analysis; Text-Blob - fastest, but least accurate, "
        "Twitter-RoBERTa - slowest(can take up to a minute depending on size), but most accurate, conservative"
    ),
)
posts_choice = st.sidebar.selectbox(
    "Choose which kind of posts you want to analyze:", POSTS_KIND, index=1
)
default_subreddits = DEFAULT_SUBS + subs_for_coin(selected_coin)[:1]
if posts_choice in ("All", "Reddit"):
    subreddits = st.sidebar.multiselect(
        "Subreddits",
        DEFAULT_SUBS + subs_for_coin(selected_coin),
        default=default_subreddits,
    )
    if subreddits == []:
        subreddits = default_subreddits
        st.sidebar.warning("No subreddits chosen, defaulted.")

with st.sidebar.expander("Advanced settings"):
    backtest = st.checkbox("Run backtest")
    if backtest:
        cost_bps = st.number_input(
            "Cost (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
        )
        slip_bps = st.number_input(
            "Slippage (bds)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
        )

    st.header("Lead/Lag settings")
    lag_hours = st.slider("Lag window (±hours)", 1, 48, 24)
    lag_step_min = st.selectbox("Lag step(minutes)", [5, 15, 30, 60], index=1)
    metric_choice = st.selectbox("Correlation metric", ["pearson"], index=0)

    st.markdown("### Indicators")
    # default values so variables exist even when checkboxes are unchecked
    sma_fast = 20
    sma_slow = 50
    rsi_period = 14
    use_sma = st.checkbox("SMA (20/50)", value=False, help="Simple Moving Average")
    use_rsi = st.checkbox("RSI (14)", value=False, help="Relative Strength Index")
    use_macd = st.checkbox(
        "MACD (12,26,9)", value=False, help="Moving Average Convergence Divergence"
    )
    if use_sma:
        sma_fast = st.number_input("SMA fast", 5, 200, sma_fast, 1)
        sma_slow = st.number_input("SMA slow", 5, 400, sma_slow, 1)
    if use_rsi:
        rsi_period = st.number_input("RSI period", 5, 50, rsi_period, 1)

run = st.sidebar.button("Run Analysis", type="primary")

st.sidebar.header("Utils")

if st.sidebar.button("Clear cache"):
    res = clear_cache_dir()
    mb = res["bytes_freed"] / 1000000
    st.sidebar.success(f"Removed {res['files_removed']} files ({mb:.2f} MB)")
    st.session_state.pop("merged_path", None)


benchtest = st.sidebar.button("Run analyzer benchmark")
