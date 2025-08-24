import os
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pytz import utc
from datetime import timedelta
import logging
import sys
import hashlib, pathlib

for k, v in st.secrets.items():
    os.environ[str(k)] = str(v)


from src.utils.helpers import (
     load_csv,
     save_csv,
     filter_date_range,
     map_to_cryptopanic_symbol
)
from src.scraping.reddit_scraper import fetch_reddit_posts
from src.scraping.fetch_price import get_price_history
from src.scraping.news_scraper import fetch_news_posts
from src.scraping.twitter_scraper import fetch_twitter_posts
from src.sentiment.analyzer import add_sentiment_to_file, load_sentiment_df
from src.processing.merge_data import merge_sentiment_and_price
from src.utils.cache import load_cached_csv, cache_csv, clear_cache_dir, day_str
from config.settings import(
    COINS_UI_LABELS,
    COINS_UI_TO_SYMBOL,
    DEFAULT_DAYS,
    ANALYZER_UI_LABELS,
    get_data_path,
    POSTS_KIND,
    DEFAULT_SUBS,
    subs_for_coin,
    COIN_SUBS,
    DEMO_MODE,
    get_demo_data_path
)
from src.plotting.charts import (
    plot_price_time_series,
    plot_sentiment_timeline,
    plot_sentiment_vs_price,
    plot_sentiment_with_price,
    plot_lag_correlation,
    plot_equity,
    plot_drawdown,
    plot_rsi,
    plot_macd,
    plot_price_with_sma
)
from src.utils.helpers import file_sha1
from src.analysis.lead_lag import load_or_build_lead_lag_features
from src.backtest.engine import run_backtest
from src.processing.indicators import add_indicators
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)



#Page header
st.set_page_config(page_title="Crypto Sentiment Tracker",layout="wide")

#Page main intro
st.title("Crypto sentiment tracker")
st.markdown("Visualization of public sentiment based on keywords and further comparison to actual price of cryptocurrencies")

#Config
st.sidebar.header("Configuration")

if st.sidebar.button("Clear cache"):
    res = clear_cache_dir()
    mb = res["bytes_freed"]/1e6
    st.sidebar.success(f"Removed {res['files_removed']} files ({mb:.2f} MB)")
    st.session_state.pop("merged_path", None)


benchtest = st.sidebar.button("Run analyzer benchmark")

st.sidebar.divider()

with st.sidebar.form("analysis_form"):
    selected_label = st.selectbox("Choose cryptocurrency", COINS_UI_LABELS)
    selected_coin = COINS_UI_TO_SYMBOL[selected_label]
    num_posts = st.slider("Number of posts to fetch", min_value=100, max_value=1100, step=100, value=300)
    days = st.selectbox("Price history in days", DEFAULT_DAYS,
                          help="Choosing day range longer than 90 days causes to only show price point once per day.")
    analyzer_choice = st.selectbox(
        "Choose sentiment analyzer:", ANALYZER_UI_LABELS,
        help=("VADER - all-rounder, decent speed and analysis; Text-Blob - fastest, but least accurate, "
              "Twitter-RoBERTa - slowest(can take up to a minute depending on size), but most accurate, conservative")
    )
    posts_choice = st.selectbox("Choose which kind of posts you want to analyze:", POSTS_KIND)
    if posts_choice in ("All", "Reddit"):
        subreddits = st.multiselect(
            "Subreddits",
            DEFAULT_SUBS + subs_for_coin(selected_coin),
            default=DEFAULT_SUBS + subs_for_coin(selected_coin)[:1]
        )

    backtest = st.checkbox("Run backtest")
    if backtest:
        cost_bps = st.number_input("Cost (bps)", 0.0, 100.0, 5.0, 0.5)
        slip_bps = st.number_input("Slippage (bds)", 0.0, 100.0, 5.0, 0.5)

    st.header("Lead/Lag settings")
    lag_hours = st.slider("Lag window (±hours)", 1, 48, 24)
    lag_step_min = st.selectbox("Lag step(minutes)", [5, 15, 30, 60], index=1)
    metric_choice = st.selectbox("Correlation metric", ["pearson"], index=0)

    st.markdown("### Indicators")
    use_sma = st.checkbox("SMA (20/50)", value=True, help="Simple Moving Average")
    use_rsi = st.checkbox("RSI (14)", value=True, help="Relative Strength Index")
    use_macd = st.checkbox("MACD (12,26,9)", value=True, help="Moving Average Convergence Divergence")

    sma_fast = st.number_input("SMA fast", 5, 200, 20, 1)
    sma_slow = st.number_input("SMA slow", 5, 400, 50, 1)
    rsi_period = st.number_input("RSI period", 5, 50, 14, 1)
    run_bench = st.checkbox("Also run analyzer benchmark")


    submit = st.form_submit_button("Run Analysis")



#Fetching and merging all data
if submit:

    end_date = pd.Timestamp.now(tz=utc)
    start_date = end_date - timedelta(days=int(days))

    #Check whether to use
    use_news = False
    use_reddit = False
    use_twitter = False

    #Set sentiment path already for caching
    news_sentiment_path = get_data_path(selected_coin, "news_sentiment")
    reddit_sentiment_path = get_data_path(selected_coin, "reddit_sentiment")
    twitter_sentiment_path = get_data_path(selected_coin, "twitter_sentiment")

    #Set data path for simplified caching
    news_path = f"data/{selected_coin}_news_posts.csv"
    reddit_path = f"data/{selected_coin}_reddit_posts.csv"
    twitter_path = f"data/{selected_coin}_twitter_posts.csv"
    
    cryptopanic_coin = map_to_cryptopanic_symbol(selected_coin)

    if DEMO_MODE:
        st.info("Demo mode is ON — using frozen CSVs (no scraping).")
        st.session_state.pop("merged_path", None)

        selected_coin = "bitcoin"
        selected_label = "Bitcoin"

        #Load demo files
        merged_path_demo   = pathlib.Path("data/demo/bitcoin_merged.csv")
        price_path_demo    = pathlib.Path("data/demo/bitcoin_prices.csv")
        combined_path_demo = pathlib.Path("data/demo/bitcoin_combined_sentiment.csv")

        try:
            merged_df   = pd.read_csv(merged_path_demo,   parse_dates=["timestamp"])
            price_df    = pd.read_csv(price_path_demo,    parse_dates=["timestamp"])
            combined_df = pd.read_csv(combined_path_demo, parse_dates=["timestamp"])
        except FileNotFoundError as e:
            st.error(f"Missing demo file: {e.filename}. Put it under data/demo/ or adjust get_demo_data_path().")
            st.stop()

        #Time-range UI for demo
        min_date = (merged_df["timestamp"].max() - timedelta(days=int(days))).to_pydatetime()
        max_date = merged_df["timestamp"].max().to_pydatetime()
        selected_range = st.slider(
            "Select time range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )

        view = filter_date_range(merged_df, selected_range[0], selected_range[1])

        #Plots
        st.plotly_chart(plot_price_time_series(price_df, selected_coin), use_container_width=True)
        st.plotly_chart(plot_sentiment_timeline(view, selected_coin),   use_container_width=True)
        st.plotly_chart(plot_sentiment_with_price(view, selected_coin), use_container_width=True)

        #Lead/lag on demo
        try:
            lead_lag_settings = {
                "dataset": "features",
                "coin": selected_coin,
                "days": int(days),
                "analyzer": analyzer_choice,
                "posts_choice": posts_choice,
                "depends_on": [0],
                "lag_min_s": -lag_hours * 3600,
                "lag_max_s":  lag_hours * 3600,
                "lag_step_s": lag_step_min * 60,
                "metric": metric_choice,
            }
            feats = load_or_build_lead_lag_features(lead_lag_settings, str(merged_path_demo))
            if feats is not None and not feats.empty:
                st.plotly_chart(plot_lag_correlation(feats, unit="min"), use_container_width=True)
            else:
                st.info("Lag features not available for the selected range.")
        except Exception as e:
            st.warning(f"Lead/lag unavailable in demo: {e}")

        #Backtest in demo
        if backtest:
            try:
                bt, stats = run_backtest(view, cost_bps=cost_bps, slippage_bps=slip_bps, resample="5min")
                st.plotly_chart(plot_equity(bt),   use_container_width=True)
                st.plotly_chart(plot_drawdown(bt), use_container_width=True)
                with st.expander("What the metrics mean"):
                    st.markdown("""
                    - **CAGR** — Compounded Annual Growth Rate.
                    - **Sharpe** — Risk-adjusted return (higher is better).
                    - **MaxDD** — Maximum drawdown from peak equity.
                    - **Hit Rate** — Share of profitable trades.
                    """.strip())
                CAGR = stats.get("CAGR"); Sharpe = stats.get("Sharpe")
                MaxDD = stats.get("MaxDD"); Hit = stats.get("HitRate")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CAGR",     "—" if (CAGR   is None or (isinstance(CAGR,   float) and np.isnan(CAGR)))   else f"{CAGR:.2%}")
                c2.metric("Sharpe",   "—" if (Sharpe is None or (isinstance(Sharpe, float) and np.isnan(Sharpe))) else f"{Sharpe:.2f}")
                c3.metric("MaxDD",    "—" if (MaxDD  is None or (isinstance(MaxDD,  float) and np.isnan(MaxDD)))  else f"{abs(MaxDD):.2%}")
                c4.metric("Hit Rate", "—" if (Hit    is None or (isinstance(Hit,    float) and np.isnan(Hit)))    else f"{Hit:.2%}")
            except Exception as e:
                st.warning(f"Backtest unavailable in demo: {e}")

        #Indicators
        sma_cols = [f"sma_{sma_fast}", f"sma_{sma_slow}"]
        if use_sma:
            st.plotly_chart(plot_price_with_sma(view, selected_coin, sma_cols), use_container_width=True)
        if use_rsi:
            fig = plot_rsi(view, rsi_col=f"rsi_{int(rsi_period)}")
            if fig: st.plotly_chart(fig, use_container_width=True)
        if use_macd:
            fig = plot_macd(view)
            if fig: st.plotly_chart(fig, use_container_width=True)

        
        if "sentiment" in view.columns and not view["sentiment"].empty:
            st.metric(label=f"Average Sentiment {selected_label}", value=f"{view['sentiment'].mean():.3f}")

        if run_bench:
            try:
                from src.benchmark.analyzer_eval import run_fixed_benchmark
                from src.benchmark.benchmark_plot import (
                    accuracy_figure,
                    confusion_matrices
                    )
                results, table = run_fixed_benchmark()
                st.session_state["bench_results"] = results
                st.session_state["bench_table"] = table
            except FileNotFoundError:
                st.error("data/benchmark_labeled.csv not found.")
            except Exception as e:
                st.exception(e)


        st.stop()
    #News
    if posts_choice in ("All", "News"):
        #Dont use news for now, API almost used up - add "All" in line above to allow again        

        news_settings = {
            "dataset": "posts_news",
            "source": "news",
            "coin": selected_coin,
            "query": cryptopanic_coin,
            "start_date": start_date.tz_convert(None).isoformat(timespec="seconds"),
            "end_date": end_date.tz_convert(None).isoformat(timespec="seconds"),
            "num_posts": num_posts,
        }
        news_df = load_cached_csv(news_settings, parse_dates=["timestamp"], freshness_minutes=30)
        if news_df is None:
                with st.spinner("Fetching news..."):
                    news_df = fetch_news_posts(cryptopanic_coin, int(num_posts or 200))
                    news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], utc=True)

                    #Enforce selected date window
                    mask = (news_df["timestamp"] >= start_date) & (news_df["timestamp"] <= end_date)
                    news_df = news_df.loc[mask].copy()

                    news_df["source"] = "news"
                    news_df["coin"] = cryptopanic_coin.lower()
                    if "lang" not in news_df.columns:
                        news_df["lang"] = "en"

                    cache_csv(news_df, news_settings)

        #Ensure columns even when loaded from cache
        news_df["source"] = "news"
        if "coin" not in news_df.columns:
            news_df["coin"] = cryptopanic_coin.lower()
        if "lang" not in news_df.columns:
            news_df["lang"] = "en"

        save_csv(news_df, news_path)
        use_news = True
    else:
        use_news = False

    #Reddit
    if posts_choice in ("All", "Reddit"):
        reddit_settings = {
            "dataset": "posts_reddit",
            "source": "reddit",
            "coin": selected_coin,
            "query": f"({selected_coin} OR {cryptopanic_coin})",
            "start_date": start_date.tz_convert(None).isoformat(timespec="seconds"),
            "end_date": end_date.tz_convert(None).isoformat(timespec="seconds"),
            "num_posts": num_posts,
            "tz":"utc",
            "subreddits": subreddits
        }
        reddit_df = load_cached_csv(reddit_settings, parse_dates=["timestamp"],freshness_minutes = 30)
        if reddit_df is None:
                with st.spinner("Fetching Reddit posts..."):
                    reddit_df = fetch_reddit_posts(query=reddit_settings["query"], limit=num_posts, start_date=start_date, end_date=end_date, subreddits=subreddits)
                    reddit_df["source"] = "reddit"
                    reddit_df["timestamp"] = pd.to_datetime(reddit_df["timestamp"], utc=True)

                    mask = (reddit_df["timestamp"] >= start_date) & (reddit_df["timestamp"] <= end_date)
                    reddit_df = reddit_df.loc[mask].copy()

                    cache_csv(reddit_df, reddit_settings)
        reddit_df["source"] = "reddit"
        reddit_df["coin"] = cryptopanic_coin.lower()
        if "lang" not in reddit_df.columns:
            reddit_df["lang"] = "en"
        save_csv(reddit_df, reddit_path)
        use_reddit = True
    #Twitter
    if posts_choice in ("All", "Twitter/X"):
        twitter_settings = {
            "dataset": "posts_twitter",
            "source": "twitter",
            "coin": selected_coin,
            "query": cryptopanic_coin.lower(),
            "start_date": day_str(start_date),
            "end_date": day_str(end_date),
            "num_posts": num_posts,
            "tz":"utc",
        }
        tweets_df = load_cached_csv(twitter_settings, parse_dates=["timestamp"],freshness_minutes = 30)
        if tweets_df is None:
            with st.spinner("Fetching Twitter posts.."):
                tweets_df = fetch_twitter_posts(
                    coin=cryptopanic_coin.lower(),
                    limit=int(num_posts or 200),
                    lang="en",
                    sort="Top",
                    start=start_date,
                    end=end_date,
                )
                cache_csv(tweets_df, twitter_settings)

        tweets_df["source"] = "twitter"
        tweets_df["coin"] = map_to_cryptopanic_symbol(selected_coin).lower()
        if "lang" not in tweets_df.columns:
            tweets_df["lang"] = "en"
        save_csv(tweets_df, twitter_path)
        use_twitter = True
        

    with st.spinner("Analyzing sentiment..."):
        if use_news:
            news_sent_settings={
                "dataset": "sentiment",
                "source": "news",
                "coin": selected_coin,
                "analyzer": analyzer_choice,
                "num_posts": num_posts,
                "input_sha1": file_sha1(news_path)
            }
            add_sentiment_to_file(news_path,
                                  news_sentiment_path,
                                  analyzer_choice,
                                  cache_settings=news_sent_settings,
                                  freshness_minutes=30)
        else:
            logging.warning("News sentiment will not be included")
        if use_reddit:
            reddit_sent_settings={
                "dataset": "sentiment",
                "source": "reddit",
                "coin": selected_coin,
                "analyzer": analyzer_choice,
                "num_posts": num_posts,
                "input_sha1": file_sha1(reddit_path)
                                  }
            add_sentiment_to_file(reddit_path,
                                  reddit_sentiment_path,
                                  analyzer_choice,
                                  cache_settings=reddit_sent_settings,
                                  freshness_minutes=30)
        else:
            logging.warning("Reddit sentiment will not be included")
        if use_twitter:
            twitter_sent_settings = {
                "dataset": "sentiment",
                "source": "twitter",
                "coin": selected_coin,
                "analyzer": analyzer_choice,
                "num_posts": num_posts,
                "input_sha1": file_sha1(twitter_path),
            }
            add_sentiment_to_file(
                twitter_path,
                twitter_sentiment_path,
                analyzer_choice,
                cache_settings=twitter_sent_settings,
                freshness_minutes=30
            )
        else:
            logging.warning("Twitter sentiment will not be included")


    with st.spinner("Combining sentiment..."):
        sentiment_df = load_sentiment_df(
            news_sentiment_path,
            reddit_sentiment_path,
            twitter_sentiment_path,
            posts_choice,
            )
        combined_sentiment_path = get_data_path(selected_coin, "combined_sentiment")
        save_csv(sentiment_df, combined_sentiment_path)
        dfs = []
        
        if posts_choice == "All":
            dfs = [sentiment_df]
        elif use_news and os.path.exists(news_sentiment_path):
            dfs.append(load_csv(news_sentiment_path))
        elif use_reddit and os.path.exists(reddit_sentiment_path):
            dfs.append(load_csv(reddit_sentiment_path))
        elif use_twitter and os.path.exists(twitter_sentiment_path):
            dfs.append(load_csv(twitter_sentiment_path))
        

        if not dfs:
            logging.error(f"No sentiment data. use_news={use_news}, use_reddit={use_reddit}, use_twitter={use_twitter}")
            st.error("No sentiment data could be loaded. Check API limits or local files.")
            st.stop()

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df["timestamp"] = (
            pd.to_datetime(combined_df["timestamp"], errors="coerce", utc=True)
            .dt.tz_localize(None)
        )
        combined_df["source"] = combined_df["source"].astype(str).str.strip().str.lower()
        combined_df = combined_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        logging.info("Combined counts by source: %s", combined_df["source"].value_counts(dropna=False).to_dict())
        combined_sentiment_path = get_data_path(selected_coin, "combined_sentiment")
        save_csv(combined_df, combined_sentiment_path)

    #Price data
    price_settings={
            "dataset": "price",
            "coin": selected_coin,
            "days": int(days),
            "tz": "utc"
    }
    price_df = load_cached_csv(price_settings, parse_dates=["timestamp"], freshness_minutes=30)
    if price_df is None:
        with st.spinner("Fetching price data..."):
            price_df = get_price_history(selected_coin, days)
            cache_csv(price_df, price_settings)
            save_csv(price_df,get_data_path(selected_coin, "prices"))
    save_csv(price_df, get_data_path(selected_coin,"prices"))
    #Merge of price and data
    merged_settings = {
        "dataset": "merged",
        "coin": selected_coin,
        "days": int(days),
        "analyzer": analyzer_choice,
        "posts_choice": posts_choice,
        "depends_on": [
            file_sha1(get_data_path(selected_coin,"prices")),
            file_sha1(combined_sentiment_path)
        ]
    }
    merged_path = get_data_path(selected_coin, "merged")
    merged_df = load_cached_csv(merged_settings, parse_dates=["timestamp"], freshness_minutes=30)
    if merged_df is None:
        with st.spinner("Merging sentiment and price data..."):
            merge_sentiment_and_price(get_data_path(selected_coin, "combined_sentiment"),
                                      get_data_path(selected_coin, "prices"),
                                        merged_path,
                                        cache_settings=merged_settings)
            merged_df = load_csv(merged_path, parse_dates=["timestamp"])
            merged_df = add_indicators(
                            merged_df,
                            price_col="price",
                            sma_windows=(sma_fast, sma_slow),
                            rsi_period=int(rsi_period),
                            macd_fast=12, macd_slow=26, macd_signal=9
                        )
            cache_csv(merged_df, merged_settings)
    else:
        save_csv(merged_df, merged_path)

    lead_lag_settings ={
        "dataset": "features",
        "coin": selected_coin,
        "days": int(days),
        "analyzer": analyzer_choice,
        "posts_choice": posts_choice,
        "depends_on": [file_sha1(merged_path)],
        "lag_min_s": -lag_hours*3600,
        "lag_max_s": lag_hours*3600,
        "lag_step_s": lag_step_min*60,
        "metric": metric_choice
    }

    feats = load_or_build_lead_lag_features(lead_lag_settings, merged_path)

    st.session_state["lead_lag_settings"] = lead_lag_settings
    st.session_state["merged_path"] = merged_path
   
    st.success("Data ready, showing visualization:")

if (not DEMO_MODE) and "merged_path" in st.session_state and os.path.exists(st.session_state["merged_path"]):
    price_settings = {
        "dataset": "price",
        "coin": selected_coin,
        "days": int(days),
        "tz": "utc",
    }
    price_df = load_cached_csv(price_settings, parse_dates=["timestamp"], freshness_minutes=30)


    #Timestamp things
    df = load_csv(st.session_state["merged_path"], parse_dates=["timestamp"])
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

    feats = None
    if "lead_lag_settings" in st.session_state:
        try:
            feats = load_or_build_lead_lag_features(st.session_state["lead_lag_settings"], st.session_state["merged_path"])
        except Exception as e:
            st.warning(f"Could not build lead/lag features: {e}")

    #Price plot
    st.plotly_chart(plot_price_time_series(price_df, selected_coin), use_container_width=True)
   
    #Sentiment timeline
    st.plotly_chart(plot_sentiment_timeline(df, selected_coin), use_container_width=True)

    #Sentiment vs price timeline (smoothed)
    st.plotly_chart(plot_sentiment_with_price(df, selected_coin), use_container_width=True)
    
    if feats is not None and not feats.empty:
        try:
            fig_lag = plot_lag_correlation(feats,  unit="min")
            st.plotly_chart(fig_lag, use_container_width=True)
        except ValueError as e:
            st.warning(f"Lag plot unavailable: {e}")
    else:
        st.info("Lag features not available for the selected range.")

    if backtest:
        bt, stats  = run_backtest(df, cost_bps=cost_bps, slippage_bps=slip_bps, resample="5min")

        st.plotly_chart(plot_equity(bt), use_container_width=True)
        st.plotly_chart(plot_drawdown(bt), use_container_width=True)
        with st.expander("What the metrics mean"):
            st.markdown("""
                - **CAGR** — Compounded Annual Growth Rate.
                - **Sharpe** — Risk-adjusted return (higher is better).
                - **MaxDD** — Maximum drawdown from peak equity.
                - **Hit Rate** — Share of profitable trades.
                """.strip())
            
        CAGR   = stats.get("CAGR")
        Sharpe = stats.get("Sharpe")
        MaxDD  = stats.get("MaxDD")
        Hit    = stats.get("HitRate")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR",    "—" if (CAGR   is None or (isinstance(CAGR,   float) and np.isnan(CAGR)))   else f"{CAGR:.2%}")
        c2.metric("Sharpe",  "—" if (Sharpe is None or (isinstance(Sharpe, float) and np.isnan(Sharpe))) else f"{Sharpe:.2f}")
        c3.metric("MaxDD",   "—" if (MaxDD  is None or (isinstance(MaxDD,  float) and np.isnan(MaxDD)))  else f"{abs(MaxDD):.2%}")
        c4.metric("Hit Rate","—" if (Hit    is None or (isinstance(Hit,    float) and np.isnan(Hit)))    else f"{Hit:.2%}")


    #Sentiment vs price
    st.plotly_chart(plot_sentiment_vs_price(df), use_container_width=True)

    #Average sentiment
    avg_sent = df["sentiment"].mean()
    st.metric(label=f"Average Sentiment {selected_label}", value=f"{avg_sent:.3f}")
    
    sma_cols = [f"sma_{sma_fast}", f"sma_{sma_slow}"]
    view = filter_date_range(df, selected_range[0], selected_range[1])

    if use_sma:
        st.plotly_chart(plot_price_with_sma(view, selected_coin, sma_cols), use_container_width=True)


    if use_rsi:
        fig = plot_rsi(view, rsi_col=f"rsi_{int(rsi_period)}")
        if fig: st.plotly_chart(fig, use_container_width=True)

    if use_macd:
        fig = plot_macd(view)
        if fig: st.plotly_chart(fig, use_container_width=True)
    

   
else:
    st.info("Run the analysis from the sidebar to see visualization")

if benchtest:
    try:
        from src.benchmark.analyzer_eval import run_fixed_benchmark
        from src.benchmark.benchmark_plot import (
            accuracy_figure,
            confusion_matrices
            )
        results, table = run_fixed_benchmark()
        st.session_state["bench_results"] = results
        st.session_state["bench_table"] = table
    except FileNotFoundError:
        st.error("data/benchmark_labeled.csv not found.")
    except Exception as e:
        st.exception(e)

if "bench_results" in st.session_state and "bench_table" in st.session_state:
    table = st.session_state["bench_table"]
    results = st.session_state["bench_results"]

    try:
        from src.benchmark.benchmark_plot import accuracy_figure, confusion_matrices
    except Exception as e:
        st.warning(f"Benchmark plotting not available: {e}")
    else:
        st.dataframe(
            table.style.format({"Accuracy": "{:.3f}", "F1 (macro)": "{:.3f}"}),
            use_container_width=True
        )
        accuracy_figure(table)
        st.markdown("#### Confusion Matrices")
        confusion_matrices(results)


    