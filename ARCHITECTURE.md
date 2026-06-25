# Architecture

## Overview

`crypto-sentiment-tracker` is a layered Python application for collecting crypto-related text, scoring sentiment, merging it with market data, and exposing the results through both Streamlit and FastAPI.

The project currently has:

- Streamlit UI entry point: `run_app.py`
- FastAPI app: `src/presentation/api/main.py`
- Docker services for API, UI, and DB schema initialization
- pytest coverage for domain behavior, storage repositories, API route behavior, fetcher boundaries, and cache behavior

---

## Layers

### Presentation Layer

Location:

```text
src/presentation/
```

Responsibilities:

- Streamlit page rendering
- Sidebar state and user input
- Plotly chart construction
- FastAPI route definitions
- API response formatting

Important files:

```text
src/presentation/pages.py
src/presentation/sidebar.py
src/presentation/charts.py
src/presentation/demo_view.py
src/presentation/benchmark_view.py
src/presentation/api/main.py
src/presentation/api/routes/
```

Current note: the Streamlit UI still imports application use cases directly. The API and UI are separate Docker services, but they are not yet a fully separated frontend/backend system.

---

### Application Layer

Location:

```text
src/app/
```

Responsibilities:

- Define application DTOs and defaults
- Coordinate workflows
- Decide when to use cached sentiment versus recomputing it

Important files:

```text
src/app/dto.py
src/app/defaults.py
src/app/settings.py
src/app/use_cases/run_analysis.py
src/app/use_cases/run_demo.py
src/app/use_cases/get_indicators.py
src/app/use_cases/sentiment_cache.py
```

Typical live-analysis flow:

```text
AnalysisConfig
  -> fetch_posts(config)
  -> get_or_create_sentiment_df(config, posts_df)
  -> get_coinbase_price_history(config)
  -> merge_sentiment_and_price_df(price_df, sentiment_df)
  -> AnalysisResult
```

---

### Domain Layer

Location:

```text
src/domain/
```

Responsibilities:

- Sentiment analyzer wrappers and registry
- Market filtering, indicators, smoothing, and merge logic
- Signal generation
- Lead/lag analysis
- Backtest calculations

Important files:

```text
src/domain/sentiment/
src/domain/market/
src/domain/signals/engine.py
src/domain/analysis/lead_lag.py
src/domain/backtest/engine.py
```

Domain code should stay mostly independent from Docker, Streamlit, FastAPI, and PostgreSQL details.

---

### Infrastructure Layer

Location:

```text
src/infra/
```

Responsibilities:

- External API access
- PostgreSQL schema and repository implementation
- Legacy CSV helper code

Important files:

```text
src/infra/fetchers/service.py
src/infra/fetchers/reddit.py
src/infra/fetchers/news.py
src/infra/fetchers/youtube.py
src/infra/fetchers/coinbase_price.py
src/infra/fetchers/price.py
src/infra/storage/db/schema.py
src/infra/storage/db/connection.py
src/infra/storage/db/content_repository.py
src/infra/storage/db/price_repository.py
src/infra/storage/db/sentiment_repository.py
src/infra/storage/db/signal_repository.py
src/infra/storage/sentiment_csv.py
```

PostgreSQL is the current persistence layer. CSV storage is legacy/transitional and should shrink as demo and runtime data move fully into PostgreSQL.

---

### Shared Utilities

Location:

```text
src/shared/
```

Responsibilities:

- Timestamp normalization
- CSV helper functions
- Coin validation
- DataFrame schema checks
- API/DataFrame formatting helpers
- DB helper utilities such as content hashing

Important files:

```text
src/shared/helpers.py
src/shared/dataframe_schema.py
src/shared/dataframe_utils.py
src/shared/db_helpers.py
```

---

## Current Directory Shape

```text
Crypto-sentiment-tracker/
├─ Dockerfile
├─ docker-compose.yml
├─ pyproject.toml
├─ run_app.py
├─ README.md
├─ ARCHITECTURE.md
├─ DECISIONS.md
├─ LICENSE
├─ mypy.ini
├─ pyrightconfig.json
├─ stubs/
├─ tests/
│  ├─ api/
│  ├─ sentiment/
│  ├─ conftest.py
│  ├─ test_db.py
│  ├─ test_fetchers.py
│  ├─ test_sentiment_cache.py
│  └─ test_signals.py
└─ src/
   ├─ app/
   ├─ benchmark/
   ├─ domain/
   │  ├─ analysis/
   │  ├─ backtest/
   │  ├─ market/
   │  ├─ sentiment/
   │  └─ signals/
   ├─ infra/
   │  ├─ fetchers/
   │  └─ storage/
   │     └─ db/
   ├─ presentation/
   │  ├─ api/
   │  │  └─ routes/
   │  └─ config/
   └─ shared/
```

---

## Storage Model

Current primary storage:

```text
PostgreSQL database
```

Default local path:

```text
data/app.db
```

Docker path:

```text
/usr/src/app/data/app.db
```

Schema creation:

```text
src/infra/storage/db/schema.py
```

Tables:

- `prices`
- `content_items`
- `sentiment`
- `signals`

Repositories:

- `price_repository.py`
- `content_repository.py`
- `sentiment_repository.py`
- `signal_repository.py`

The repositories accept an optional `db_path`, which makes them easy to test against temporary PostgreSQL databases.

---

## API

FastAPI app:

```text
src/presentation/api/main.py
```

Routes:

```text
GET  /health
GET  /market/prices
GET  /market/signals
GET  /posts
GET  /sentiment
POST /ingest
```

API route tests currently call route functions directly and monkeypatch repository dependencies where needed. This avoids real external services and real app DB writes.

---

## Docker

`Dockerfile` installs the app from `pyproject.toml`:

```text
pip install -e .
```

`docker-compose.yml` defines:

- `api`: FastAPI service on container port `8000`, host port `8002`
- `ui`: Streamlit service on container/host port `8501`
- `migrate`: one-off command that calls `init_db()`

Compose uses a named volume:

```text
app_data
```

mounted at:

```text
/usr/src/app/data
```

This gives persistent PostgreSQL storage for Docker runs.

Current limitation: because `/usr/src/app/data` is a named volume, repo CSV files under `data/demo` are not automatically present inside that path. The intended direction is DB-backed demo data instead of runtime demo CSV reads.

---

## Testing Strategy

Test locations:

```text
tests/
tests/api/
tests/sentiment/
```

Current coverage:

- domain merge behavior
- signal generation behavior
- sentiment service behavior
- PostgreSQL repository round-trips
- API route success and validation paths
- fetcher behavior with mocked external boundaries
- missing API key behavior
- unsupported coin behavior
- sentiment cache hit/miss behavior
- `analyzer="all"` behavior

Testing rule of thumb:

- Use real temporary PostgreSQL DBs for repository tests.
- Use monkeypatching for external services, cache path decisions, and API route dependencies.
- Keep network calls out of unit tests.

---
