## Overview

`crypto-sentiment-tracker` is a Python project for collecting crypto-related text data, analyzing sentiment, combining it with market data, and presenting results in a dashboard.

---

## Implemented core goals

- Fetch posts/articles about selected crypto assets
- Run sentiment analysis using interchangeable analyzers
- Fetch price data for the same time range
- Merge sentiment + market data into one analysis dataset
- Compute indicators and optional strategy/backtest outputs
- Display results in a Streamlit app
- Cache expensive operations so repeated runs are fast

## Planned core goals

- Change to DB, first SQLite, then more advanced DBs, from CSVs
- Compute abnormalities
- Regular scraping, make the app live

---

## High-level architecture

The project follows a layered structure:

1. **Presentation layer**  
   Streamlit UI, input handling, charts, pages

2. **Application layer**  
   Coordinates workflows like “run analysis” or “run backtest”

3. **Domain layer**  
   Core business logic:
   - sentiment analysis
   - market indicators
   - merge logic
   - backtesting rules/metrics

4. **Infrastructure layer**  
   External system access:
   - Reddit/X/news/price fetchers
   - CSV/(eventually)database storage
   - file cache
   - path/config helpers

---

## Current: Directory structure

Crypto-sentiment-tracker/
├─ run_app.py
├─ README.md
├─ ARCHITECTURE.md
├─ DECISIONS.md
├─ requirements.txt
├─ pyrightconfig.json
├─ mypy.ini
│
├─ config/
│  ├─ settings.py
│  └─ cache_schema.py
│
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ cache/
│  ├─ demo/
│  └─ tests/
│
├─ logs/
│
├─ tests/
│  ├─ smoke/
│  ├─ unit/
│  └─ integration/
│
├─ stubs/
│
└─ src/
   ├─ app/
   │  ├─ dto.py
   │  ├─ defaults.py
   │  └─ use_cases/
   │     ├─ run_analysis.py
   │     ├─ run_demo.py
   │     └─ run_backtest.py
   │
   ├─ domain/
   │  ├─ sentiment/
   │  │  ├─ registry.py
   │  │  ├─ service.py
   │  │  ├─ vader.py
   │  │  ├─ textblob.py
   │  │  ├─ roberta.py
   │  │  └─ finbert.py
   │  │
   │  ├─ market/
   │  │  ├─ coins.py
   │  │  ├─ filtering.py
   │  │  ├─ indicators.py
   │  │  └─ merge.py
   │  │
   │  ├─ backtest/
   │  │  ├─ engine.py
   │  │  └─ metrics.py
   │  │
   │  └─ analysis/
   │     └─ lead_lag.py
   │
   ├─ infra/
   │  ├─ fetchers/
   │  │  ├─ service.py
   │  │  ├─ reddit.py
   │  │  ├─ news.py
   │  │  ├─ youtube.py
   │  │  ├─ twitter.py
   │  │  └─ price.py
   │  │
   │  ├─ storage/
   │  │  ├─ paths.py
   │  │  ├─ sentiment_csv.py
   │  │  └─ logging_config.py
   │  │
   │  └─ cache/
   │     ├─ file_cache.py
   │     └─ keys.py
   │
   ├─ presentation/
   │  ├─ pages.py
   │  ├─ sidebar.py
   │  ├─ charts.py
   │  ├─ metrics.py
   │  ├─ demo_view.py
   │  ├─ benchmark_view.py
   │  └─ ui_constants.py
   │
   └─ shared/
      ├─ text.py
      ├─ csv.py
      └─ time.py
