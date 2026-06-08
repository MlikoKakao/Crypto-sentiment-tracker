# Crypto Sentiment Tracker

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

Streamlit app that collects crypto discussion around **Bitcoin / Ethereum / Monero**, runs **sentiment analysis**, merges it with **price data**, and visualizes sentiment, lead/lag correlation, indicators, and simple backtests. Built for **fast demos** and exploratory analysis with SQLite-backed caching.
Demo: https://crypto-currency-sentiment-analysis.streamlit.app
<img width="3835" height="1746" alt="image" src="https://github.com/user-attachments/assets/0f1cbe11-9945-487c-aa86-05de0c561725" />

---

## Features

- **One-click pipeline**: fetch -> analyze -> merge -> visualize
- **Sources**: Reddit, YouTube, crypto news, and Coinbase price data
- **Analyzers**: VADER, TextBlob, RoBERTa, FinBert
- **Indicators**: SMA, RSI, MACD
- **Analysis views**: sentiment/price charts, lead-lag correlation, model benchmark view, and optional backtest charts
- **Caching/storage**: SQLite database at `data/app.db`
- **Modular architecture**: presentation, application, domain, and infrastructure layers

> Goal = quick signal intuition. Swap in heavier models/sources when needed.

---

## Project Layout

```
.
├─ run_app.py
├─ ARCHITECTURE.md
├─ DECISIONS.md
├─ requirements.txt
├─ config/
│  └─ settings.py
├─ data/
│  ├─ benchmark/
│  ├─ cache/
│  ├─ demo/
│  ├─ processed/
│  ├─ raw/
│  └─ tests/
├─ docs/
├─ tests/
│  └─ run_smoke_tests.py
├─ stubs/
│  ├─ textblob/
│  └─ vader/
├─ src/
│  ├─ presentation/
│  │  ├─ pages.py              # Streamlit page routing and app rendering
│  │  ├─ sidebar.py            # User controls for coin/source/analyzer/date
│  │  ├─ charts.py             # Plotly chart builders
│  │  ├─ demo_view.py
│  │  ├─ benchmark_view.py
│  │  └─ ui_constants.py
│  ├─ app/
│  │  ├─ dto.py                # AnalysisConfig and AnalysisResult data objects
│  │  ├─ defaults.py
│  │  └─ use_cases/
│  │     ├─ run_analysis.py    # Main fetch -> sentiment -> price -> merge workflow
│  │     └─ run_demo.py        # Loads demo CSVs from data/demo
│  ├─ domain/
│  │  ├─ sentiment/            # VADER, TextBlob, RoBERTa, FinBERT, registry, service
│  │  ├─ market/               # Coins, filtering, indicators, smoothing, merge logic
│  │  ├─ analysis/             # Lead/lag calculations
│  │  └─ backtest/             # Backtest engine
│  ├─ infra/
│  │  ├─ fetchers/             # Reddit, news, YouTube, Twitter, price, Coinbase price
│  │  └─ storage/
│  │     ├─ sentiment_csv.py   # Legacy/helper CSV storage
│  │     └─ db/                # SQLite schema, connection, and source repositories
│  ├─ benchmark/
│  │  ├─ analyzer_eval.py
│  │  └─ benchmark_plot.py
│  └─ shared/
│     └─ helpers.py            # CSV helpers and shared utility functions
```

---

## Quickstart

### 1) Install
```bash
git clone https://github.com/MlikoKakao/crypto-sentiment-tracker.git
cd crypto-sentiment-tracker

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2) Configure secrets
Create `.streamlit/secrets.toml`:
```toml
REDDIT_CLIENT_ID = "xxx"
REDDIT_CLIENT_SECRET = "xxx"
REDDIT_USER_AGENT = "yourapp/0.1 by youruser"
YOUTUBE_API_KEY = "xxx"
```
> Reddit and YouTube need API keys. News and Coinbase price fetching do not currently require keys.

## Environment Setup

Copy the example environment file:

```bash
cp .env.example .env

### 3) Run the app
```bash
streamlit run run_app.py
```

To run the app against bundled demo CSVs instead of live APIs:
```bash
DEMO=1 streamlit run run_app.py
```

---

## Usage

- Choose **Coin**, **Sources**, **Analyzer**, **Number of posts**, and **Price history** in the sidebar.
- Click **Run Analysis**.
- Use **Advanced settings** to enable SMA, RSI, MACD, lead/lag settings, or the backtest.
- Use the **Sentiment**, **Finance**, **Backtest**, and **Benchmark** tabs to inspect the result from different angles.
- Run the analyzer benchmark from the sidebar when you want to compare sentiment models.

### Data Columns (merged)
| column | meaning |
|---------------|-----------------------------------------------------|
| `timestamp` | UTC time (post or price bar) |
| `source` | `reddit` / `youtube` / `news` |
| `text` | post text (for content sources) |
| `sentiment` | polarity score (-1..1) |
| `price` | close price |
| `sma_20`, `sma_50` | optional simple moving averages |
| `rsi_14` | optional 14-period RSI |
| `macd` | MACD line |
| `macd_signal` | signal line |
| `macd_hist` | histogram (macd - signal) |
| `sentiment_loess` | LOESS-smoothed sentiment used by timeline charts |

_Exact columns depend on enabled modules. Plots are defensive to missing ones._

---
## Caching & Storage

- **SQLite cache/storage**: `data/app.db`, initialized by `src/infra/storage/db/schema.py`.
- **Repository modules**: `src/infra/storage/db/*_repository.py` handle cached rows for prices, Reddit, news, and YouTube.
- **Demo CSVs**: `data/demo/` powers `DEMO=1` mode.
- **Streamlit cache**: `st.cache_data.clear()` (wire it to a button if desired).

---

## Architecture Notes

- `run_app.py` loads Streamlit secrets into environment variables, configures the page, and calls `render_app()`.
- `src/presentation/` owns Streamlit UI, sidebar state, tabs, and Plotly charts.
- `src/app/use_cases/` coordinates workflows like live analysis and demo loading.
- `src/domain/` contains the project logic: sentiment analyzers, market transforms, lead/lag analysis, and backtesting.
- `src/infra/` contains external boundaries: API fetchers and SQLite storage.

---

## Contributing

1. Create a feature branch: `git checkout -b feat/<name>` 
2. Add tests for ETL/metrics when possible 
3. Run `streamlit run run_app.py` and attach screenshots/GIFs 
4. Open a PR

---

## Roadmap
- [x] Refactor and clean up application structure
- [x] Replace current X/Twitter scraping API - replaced with YouTube
- [x] Replace CSV storage with a database
- [x] Improve dashboard UI/UX
- [ ] Add anomaly detection and quick insights
- [ ] Deploy the application online

## License

MIT — see `LICENSE`.
