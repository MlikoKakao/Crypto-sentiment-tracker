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

- Change to DB, for example PostgreSQL, from CSVs
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

## In progress: Directory structure

crypto-sentiment-tracker/
├─ run_app.py
├─ requirements.txt
├─ README.md
├─ ARCHITECTURE.md
├─ AGENTS.md
├─ DECISIONS.md
├─ data/
│  ├─ raw/ #TODO: separate raw and processed
│  ├─ processed/
│  ├─ cache/
│  └─ demo/
├─ tests/ #TODO
├─ src/
│  ├─ app/
│  │  ├─ defaults.py
│  │  ├─ dto.py
│  │  ├─ use_cases/
│  │  │  ├─ run_analysis.py
│  │  │  ├─ run_backtest.py
│  │  │  └─ load_demo_data.py
│  │
│  ├─ domain/
│  │  ├─ sentiment/
│  │  │  ├─ analyzers.py
│  │  │  ├─ registry.py
│  │  │  └─ service.py
│  │  ├─ market/
│  │  │  ├─ indicators.py
│  │  │  └─ merge.py
│  │  └─ backtest/
│  │     ├─ engine.py
│  │     └─ metrics.py
│  │
│  ├─ infra/
│  │  ├─ fetchers/
│  │  │  ├─ reddit.py
│  │  │  ├─ twitter.py
│  │  │  ├─ news.py
│  │  │  └─ price.py
│  │  ├─ cache/
│  │  │  ├─ file_cache.py
│  │  │  └─ keys.py
│  │  ├─ storage/
│  │  │  ├─ csv_io.py
│  │  │  └─ paths.py
│  │  └─ config/
│  │     └─ settings.py
│  │
│  └─ presentation/
│     └─ streamlit/
│        ├─ sidebar.py
│        ├─ charts.py
│        └─ pages.py