# Architecture

## Overview

Coin Sentiment is a layered Python application with a .NET query
service. It collects crypto-related content and prices, calculates sentiment
and market signals, stores the results in PostgreSQL, and presents them through
a Streamlit dashboard.

The current infrastructure deliberately separates writes from reads:

- Python owns fetching, sentiment analysis, signal calculation, and database writes.
- FastAPI provides an authenticated endpoint for on-demand ingestion.
- A Python worker runs scheduled batch ingestion.
- .NET and Entity Framework Core provide the read/query API.
- Streamlit reads stored data through the .NET API.
- PostgreSQL is the shared source of truth.

---

## System Map

```text
              External data providers
      ┌──────────┬───────────┬──────────┬──────────┐
      │ Reddit   │ YouTube   │ RSS news │ Coinbase │
      └────┬─────┴─────┬─────┴────┬─────┴────┬─────┘
           └───────────┴───────────┴──────────┘
                              │
                 Python application use cases
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
   FastAPI `POST /ingest`              scheduled ingest worker
            │                                   │
            └─────────────────┬─────────────────┘
                              │ writes
                              ▼
                        PostgreSQL 16
                              │ reads
                              ▼
                       .NET query API
                              │ HTTP/JSON
                              ▼
                      Streamlit dashboard
                              │
           charts, indicators, lead/lag, backtests
```

There are two API processes:

| API | Technology | Responsibility |
|---|---|---|
| Ingest API | Python FastAPI | Validate and trigger writes |
| Query API | ASP.NET Core | Read prices, posts, sentiment, and signals |

This distinction is important: Streamlit does not trigger live fetching when a
user runs an analysis. It queries data already stored in PostgreSQL through the
.NET API.

---

## Runtime Services

`docker-compose.yml` defines the production-shaped service graph.
`docker-compose.dev.yml` adds local builds, source mounts, reload behavior, and
test infrastructure.

### `db`

- Image: PostgreSQL 16
- Persistence: `postgres_data:/var/lib/postgresql/data`
- Health check: `pg_isready`
- Host port: not exposed by the base Compose file; development exposes `5432`

### `migrate`

- Image: Python ingest/API image
- Command: `alembic upgrade head`
- Runs after PostgreSQL becomes healthy
- Must complete successfully before APIs start

### `ingest-api`

- Technology: Python and FastAPI
- Entry point: `src.presentation.api.main:app`
- Container port: `8000`
- Host port: `8002`
- Main write route: authenticated `POST /ingest`
- Database access: SQLAlchemy repositories using `DATABASE_URL`

### `query-api`

- Technology: ASP.NET Core and Entity Framework Core
- Project: `api-dotnet/CryptoTracker.Api`
- Container and host port: `8081`
- Reads the PostgreSQL schema through `CryptoDbContext`
- Health check verifies that the database can be reached

### `ui`

- Technology: Streamlit
- Entry point: `run_app.py`
- Container and host port: `8501`
- Queries `QUERY_API_URL`, which is `http://query-api:8081` in Compose
- Waits for the query API to become healthy

### `ingest-worker`

- Technology: Python
- Entry point: `python -m src.app.jobs.scheduled_ingest`
- Runs ingestion for BTC, ETH, and XMR
- Runs all registered sentiment analyzers
- Calculates indicators/signals after ingestion
- Uses the persistent `crypto-hf-cache` volume for Hugging Face models
- Configured as a one-shot job rather than an always-running scheduler

### `test_db` (development profile)

- Defined in `docker-compose.dev.yml`
- PostgreSQL 16 on host port `5433`
- Uses a temporary in-memory filesystem for database data
- Enabled with the `test` Compose profile

---

## Request and Data Flows

### On-demand ingestion

```text
Client
  -> POST /ingest with admin API key
  -> FastAPI validates IngestRequest
  -> run_ingest(AnalysisConfig)
  -> fetch and save Coinbase prices
  -> fetch and save posts
  -> run selected sentiment analyzer(s)
  -> save sentiment rows
  -> return ingestion counts
```

The FastAPI route is in `src/presentation/api/routes/ingest.py`. Authentication
is enforced by `require_admin_api_key`.

### Scheduled ingestion

```text
scheduled_ingest
  -> build a three-hour config for each supported coin
  -> run_ingest
  -> calculate configured indicators
  -> save generated signals through the application/repository path
  -> report per-coin failures and totals
```

Each coin is handled independently so a failure for one coin does not prevent
the remaining coins from running.

### Dashboard query

```text
Streamlit sidebar
  -> AnalysisConfig
  -> query_client
      -> GET /prices
      -> GET /posts
      -> GET /sentiment
  -> pandas DataFrames
  -> merge sentiment with price
  -> render charts, signals, indicators, and backtests
```

`src/presentation/api/query_client.py` defaults to
`http://localhost:8081` outside Compose.

---

## Python Layers

The Python code uses layered architecture. Dependencies generally point inward:
presentation and infrastructure coordinate around application and domain
behavior.

### Presentation

Location: `src/presentation/`

Responsibilities:

- Render Streamlit pages, sidebar controls, and Plotly charts
- Define FastAPI routes and request/response schemas
- Authenticate ingestion requests
- Convert .NET API responses into pandas DataFrames

Important files:

```text
src/presentation/pages.py
src/presentation/sidebar.py
src/presentation/charts.py
src/presentation/api/main.py
src/presentation/api/query_client.py
src/presentation/api/routes/ingest.py
```

The UI still calls Python domain/application functions for interactive
indicators, lead/lag calculations, signals, and backtests after querying stored
data.

### Application

Location: `src/app/`

Responsibilities:

- Define DTOs and settings
- Coordinate ingestion and analysis workflows
- Manage sentiment cache decisions
- Coordinate indicator calculation and persistence
- Define scheduled jobs

Important files:

```text
src/app/dto.py
src/app/settings.py
src/app/use_cases/run_ingest.py
src/app/use_cases/run_analysis.py
src/app/use_cases/sentiment_cache.py
src/app/use_cases/get_indicators.py
src/app/jobs/scheduled_ingest.py
```

`run_ingest` is the current write workflow used by FastAPI and the worker.
`run_analysis` remains a Python application use case but is not the live
Streamlit query path.

### Domain

Location: `src/domain/`

Responsibilities:

- Wrap sentiment analyzers and maintain their registry
- Merge price and sentiment observations
- Calculate market indicators and smoothing
- Generate signals
- Calculate lead/lag relationships
- Run backtests

Important locations:

```text
src/domain/sentiment/
src/domain/market/
src/domain/signals/engine.py
src/domain/analysis/lead_lag.py
src/domain/backtest/engine.py
```

Domain code is kept independent of Streamlit, FastAPI, Docker, and PostgreSQL
where practical.

### Infrastructure

Location: `src/infra/`

Responsibilities:

- Call Reddit, YouTube, RSS, and Coinbase
- Create SQLAlchemy database connections
- Store and load content, prices, sentiment, and signals

Important locations:

```text
src/infra/fetchers/
src/infra/storage/db/connection.py
src/infra/storage/db/models.py
src/infra/storage/db/content_repository.py
src/infra/storage/db/price_repository.py
src/infra/storage/db/sentiment_repository.py
src/infra/storage/db/signal_repository.py
```

### Shared utilities

Location: `src/shared/`

Responsibilities:

- Normalize timestamps
- Validate DataFrame schemas
- Calculate stable content hashes
- Provide small cross-layer helpers

---

## .NET Query API

Location: `api-dotnet/CryptoTracker.Api/`

The .NET API follows a controller/service/data structure:

```text
HTTP request
  -> Controller
  -> Query validation
  -> Service
  -> CryptoDbContext
  -> PostgreSQL
  -> response contract
```

Main components:

```text
Controllers/       HTTP endpoints
Contracts/         request and response shapes
Services/          query behavior
Data/              Entity Framework Core DbContext
Models/            PostgreSQL table mappings
Validation/        supported-value and date-range validation
```

The API exposes health plus query endpoints for:

- prices
- posts
- sentiment
- signals

The exact URL shape is defined by controller route attributes. Streamlit's
Python query client currently calls `/prices`, `/posts`, `/sentiment`, and
`/signals`.

---

## PostgreSQL Storage

PostgreSQL is the runtime source of truth. It is shared by Python writers and
the .NET reader.

### Schema ownership

- SQLAlchemy models: `src/infra/storage/db/models.py`
- Alembic configuration: `alembic.ini`
- Migrations: `alembic/versions/`
- Python engine: `src/infra/storage/db/connection.py`
- .NET mapping: `api-dotnet/CryptoTracker.Api/Data/CryptoDbContext.cs`

Alembic—not application startup—owns schema creation and upgrades.

### Tables

| Table | Purpose | Key |
|---|---|---|
| `prices` | Coin price observations | coin + timestamp |
| `content_items` | Posts and articles | coin + source + content hash |
| `sentiment` | Analyzer score for content | coin + source + content hash + analyzer |
| `signals` | Calculated signal values | coin + timestamp + signal name |

The sentiment table references content with a foreign key. This lets multiple
analyzers score the same content without duplicating the original text.

### Non-runtime datasets

Files under `data/demo/` and `data/benchmark/` support demo and benchmark
features. They are not substitutes for PostgreSQL runtime persistence.

---

## Configuration

Important environment variables:

| Variable | Consumer | Purpose |
|---|---|---|
| `DATABASE_URL` | Python services | SQLAlchemy PostgreSQL connection |
| `ConnectionStrings__Postgres` | .NET API | Entity Framework connection |
| `QUERY_API_URL` | Streamlit | Base URL of the .NET query API |
| `ADMIN_API_KEY` | FastAPI | Protects the ingestion endpoint |
| `POSTGRES_USER` | Compose/PostgreSQL | Database user |
| `POSTGRES_PASSWORD` | Compose/PostgreSQL | Database password |
| `POSTGRES_DB` | Compose/PostgreSQL | Database name |
| `HF_DEVICE` | sentiment analyzers | Transformer execution device |

Fetcher credentials are supplied through the Reddit and YouTube environment
variables in `.env.example`.

---

## Failure Boundaries

- Compose health checks prevent dependent services from starting too early.
- Alembic must complete before either API starts.
- The .NET health endpoint checks actual PostgreSQL connectivity.
- Individual source fetch failures are skipped so other sources can continue.
- Scheduled ingestion catches failures per coin.
- The query client returns a partial `AnalysisResult` when an HTTP request fails.
- PostgreSQL uniqueness constraints prevent duplicate observations and scores.

---

## Testing Strategy

Python tests live under `tests/` and cover:

- domain merge and signal behavior
- sentiment analyzers and multi-analyzer behavior
- fetcher boundaries
- sentiment caching
- PostgreSQL repositories
- FastAPI validation and routes

.NET integration tests live under
`api-dotnet/CryptoTracker.Api.IntegrationTests/` and cover health, prices,
posts, sentiment, signals, and database-backed endpoint behavior.

Testing rules:

- Keep real external network calls out of unit tests.
- Use the temporary PostgreSQL service for database integration behavior.
- Mock external fetchers and narrow application boundaries.
- Test the .NET HTTP contract separately from Python domain logic.

---

## Deployment Direction

The Compose topology represents the current deployable system:

```text
PostgreSQL + migrations + ingest API + query API + UI + ingest worker
```

CI/CD and Raspberry Pi scheduled ingestion are already represented in the
project configuration. AWS deployment remains the next infrastructure goal.
