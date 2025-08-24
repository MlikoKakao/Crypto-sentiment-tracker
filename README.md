# Crypto Sentiment Tracker

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

Streamlit app that scrapes posts about **Bitcoin / Ethereum**, runs **sentiment analysis**, merges with **price data**, and plots **EMA/RSI/MACD**. Built for **fast demos** and exploratory analysis with CSV caching.
Demo: https://crypto-currency-sentiment-analysis.streamlit.app
<img width="3835" height="1746" alt="image" src="https://github.com/user-attachments/assets/0f1cbe11-9945-487c-aa86-05de0c561725" />


---

## Features

- **One-click pipeline**: fetch → analyze → merge → visualize
- **Sources**: Reddit, X (Twitter), crypto news (CryptoPanic mapping)
- **Analyzers**: VADER, TextBlob, RoBERTa, FinBert
- **Indicators**: EMA, RSI, MACD
- **Sentiment vs Price overlay**: `plot_sentiment_vs_price(df)` in UI
- **Caching**: file-based (CSV) + own-made with settings map and hashing to keep API usage low
- **Modular**: re-use ETL in notebooks or other apps

> Goal = quick signal intuition. Swap in heavier models/sources when needed.

---

## Project Layout

```
.
├─ run_app.py
├─ data/                     # cached CSVs (gitignored)
├─ docs/                     # images/gifs for README
├─ src/
│  ├─ scraping/
│  │  ├─ reddit_scraper.py
│  │  ├─ twitter_scraper.py
│  │  └─ news_scraper.py
│  │  └─ fetch_helpers.py
│  │  └─ fetch_price.py
│  ├─ sentiment/
│  │  └─ analyzer.py         # add_sentiment_to_file(), analyzer selection
│  ├─ processing/
│  │  └─ merge_data.py       # merge_sentiment_and_price()
│  │  └─ indicators.py       # calculating indicators
│  │  └─ smoothing.py        # smoothing for graphs
│  ├─ plotting/
│  │  └─ charts.py           # all functionality of graphs, charts
│  ├─ analysis/
│  │  └─ lead_lag.py         # calculate lead/lag
│  ├─ backtest/
│  │  └─ engine.py           # simulate trading strategy based on sentiment and trend on backtest data
│  ├─ benchmark/
│  │  └─ analyzer_eval.py    # benchmarking sentiment models
│  │  └─ benchmark:plot.py   # plotting benchmarking results
│  └─ utils/
│     ├─ helpers.py          # load_csv(), save_csv(), filter_date_range(), map_to_cryptopanic_symbol()
│     ├─ cache.py            # load_cached_csv(), cache_csv(), clear_cache_dir()
└─ config/
   └─ settings.py            # UI labels, defaults, paths
   └─ cache_schema.py        # Caching and hashing functionality
```

---

## Quickstart

### 1) Install
```bash
git clone https://github.com/<you>/crypto-sentiment-tracker.git
cd crypto-sentiment-tracker

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure secrets
Create `.streamlit/secrets.toml`:
```toml
REDDIT_CLIENT_ID = "xxx"
REDDIT_CLIENT_SECRET = "xxx"
REDDIT_USER_AGENT = "yourapp/0.1 by youruser"
TWITTER_BEARER = "xxx"
CRYPTOPANIC_KEY = "xxx"
```
> No keys? Use cached CSVs in `data/` and skip live fetches.

### 3) Run the app
```bash
streamlit run run_app.py
```

---

## Usage

- Choose **Coin**, **Sources**, **Analyzer**, and **Date Range**.
- Click **Analyze**.
- Plots include:
  - Price + EMA/RSI/MACD
  - **Sentiment vs Price** (calls `plot_sentiment_vs_price(df)`)
- Use the cache to avoid repeated API calls for the same window.

### Data Columns (merged)
| column        | meaning                                            |
|---------------|-----------------------------------------------------|
| `timestamp`   | UTC time (post or price bar)                        |
| `source`      | `reddit` / `twitter` / `news`                       |
| `text`        | post text (for content sources)                     |
| `sentiment`   | polarity score (-1..1)                              |
| `price`       | close price                                         |
| `ema20`       | 20-period EMA                                       |
| `rsi`         | 14-period RSI                                       |
| `macd`        | MACD line                                           |
| `macd_signal` | signal line                                         |
| `macd_hist`   | histogram (macd - signal)                           |
| `sent_smooth` | EWM-smoothed sentiment                               |
| `sent_med`    | rolling median of smoothed sentiment                 |

_Exact columns depend on enabled modules. Plots are defensive to missing ones._

---

## Caching & Storage

- **CSV cache**: under `data/` (gitignored). Big files? use `clear_cache_dir()`.
- **Streamlit cache**: `st.cache_data.clear()` (wire it to a button if desired).
- **Windows tip** (file lock): if `PermissionError` on `data/combined_sentiment.csv`, close Excel/AV scans and check write perms.

---

## Dev Notes

- Indicators are computed in `processing/indicators.py`. Ensure indicator columns exist before plotting MACD/RSI.
- The **Sentiment vs Price** overlay is activated by:
  ```python
  # Sentiment vs price
  st.plotly_chart(plot_sentiment_vs_price(df), use_container_width=True)
  ```
- Keep API calls minimal during demos; prefer cached paths.

---

## Contributing

1. Create a feature branch: `git checkout -b feat/<name>`  
2. Add tests for ETL/metrics when possible  
3. Run `streamlit run run_app.py` and attach screenshots/GIFs  
4. Open a PR

---

## License

MIT — see `LICENSE`.
