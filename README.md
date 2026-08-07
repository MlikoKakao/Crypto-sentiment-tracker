# Crypto Sentiment Tracker

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![.NET](https://img.shields.io/badge/.NET-10-purple.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-ingest_API-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

Crypto Sentiment Tracker collects crypto-related posts and articles, scores their
sentiment, combines the results with market data, and presents the data in a
Streamlit dashboard.

The current infrastructure separates data ingestion from data querying:

- Python fetchers and sentiment analyzers write to PostgreSQL.
- A Python FastAPI service exposes the authenticated ingestion endpoint.
- A .NET API provides read-only query endpoints for the dashboard.
- Streamlit queries the .NET API and performs visualization and interactive analysis.
- A scheduled Python worker can ingest BTC, ETH, and XMR data in batches.

Demo: https://crypto-currency-sentiment-analysis.streamlit.app

<img width="3835" height="1746" alt="Crypto Sentiment Tracker screenshot" src="https://github.com/user-attachments/assets/0f1cbe11-9945-487c-aa86-05de0c561725" />

---

## Features

- Fetches crypto content from Reddit, YouTube, and RSS news feeds
- Fetches market prices from Coinbase
- Scores sentiment with VADER, TextBlob, RoBERTa, or FinBERT
- Supports `analyzer="all"` to run every registered analyzer
- Stores prices, content, sentiment, and signals in PostgreSQL
- Uses Alembic for database schema migrations
- Exposes an authenticated Python ingestion API
- Exposes a .NET query API for prices, posts, sentiment, and signals
- Renders Streamlit charts, lead/lag analysis, technical indicators, benchmarks,
  signals, and backtests
- Includes unit, repository, Python API, and .NET integration tests

---

## Current Infrastructure

```text
Reddit / YouTube / RSS News       Coinbase
              \                    /
               \                  /
                Python ingestion
                 /             \
        FastAPI POST /ingest    scheduled worker
                 \             /
                  PostgreSQL 16
                       |
                  .NET query API
                       |
                Streamlit dashboard
```

The main Docker Compose services are:

| Service | Responsibility | Host port |
|---|---|---:|
| `db` | PostgreSQL 16 persistent storage | not exposed by default |
| `migrate` | Runs `alembic upgrade head` and exits | none |
| `ingest-api` | Python FastAPI write/ingestion API | `8002` |
| `query-api` | .NET read/query API | `8081` |
| `ui` | Streamlit dashboard | `8501` |
| `ingest-worker` | One-shot scheduled ingestion job | none |

`docker-compose.dev.yml` adds source mounts, reload behavior, PostgreSQL host
port `5432`, and a temporary test database on host port `5433`.

---

## Project Layout

```text
.
├─ api-dotnet/
│  ├─ CryptoTracker.Api/                   # .NET query API
│  └─ CryptoTracker.Api.IntegrationTests/
├─ alembic/                                # PostgreSQL migrations
├─ data/
│  ├─ benchmark/
│  └─ demo/
├─ src/
│  ├─ app/                                 # use cases, configuration, jobs
│  ├─ benchmark/
│  ├─ domain/                              # analysis and calculation rules
│  ├─ infra/                               # fetchers and DB repositories
│  ├─ presentation/                        # Streamlit and FastAPI
│  └─ shared/
├─ tests/                                  # Python tests
├─ Dockerfile.api
├─ Dockerfile.ingest
├─ Dockerfile.ui
├─ docker-compose.yml
├─ docker-compose.dev.yml
├─ pyproject.toml
└─ run_app.py
```

`pyproject.toml` defines Python dependency groups. The lock files are split by
container role: API, UI, ingestion, and tests.

---

## Environment Setup

Copy the public template:

```bash
cp .env.example .env
```

The Compose stack requires:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=crypto
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/crypto
ADMIN_API_KEY=
```

Optional integration settings include:

```env
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
YOUTUBE_API_KEY=
DEMO=0
HF_DEVICE=-1
```

Reddit and YouTube require API credentials. RSS news and Coinbase price
fetching do not currently require credentials.

Inside Compose, `DATABASE_URL` is overridden to use the `db` service hostname.
The UI also receives:

```env
QUERY_API_URL=http://query-api:8081
```

For Streamlit Cloud, `run_app.py` can load values from
`.streamlit/secrets.toml`.

---

## Run with Docker Compose

Build and start the long-running services:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Compose waits for PostgreSQL, runs the migration service, then starts the APIs
and UI according to their health checks.

Open:

- Streamlit UI: http://localhost:8501
- .NET query API health: http://localhost:8081/health
- Python ingest API health: http://localhost:8002/health

Trigger an ingestion request through the Python API using the admin API key and
the request contract defined in `src/presentation/api/schemas/ingest.py`.

Run the one-shot ingestion worker:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm ingest-worker
```

Stop the stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

PostgreSQL data persists in the `postgres_data` named volume. Downloaded
Hugging Face models persist in the `crypto-hf-cache` volume.

---

## Run Components Locally

Create a Python environment and install the dependency groups needed for local
development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,ui,fetchers,sentiment,benchmark,dev]"
```

Start PostgreSQL first, set `DATABASE_URL`, and apply migrations:

```bash
alembic upgrade head
```

Run the Python ingestion API:

```bash
uvicorn src.presentation.api.main:app --reload --port 8000
```

Run the .NET query API from its project directory with a valid
`ConnectionStrings__Postgres` setting:

```bash
dotnet run --project api-dotnet/CryptoTracker.Api
```

Set `QUERY_API_URL` to the URL printed by .NET, then run Streamlit:

```bash
streamlit run run_app.py
```

---

## Storage

PostgreSQL is the runtime database. SQLAlchemy models live in
`src/infra/storage/db/models.py`, while Alembic migrations in
`alembic/versions/` create and evolve the schema.

The main tables are:

- `prices`
- `content_items`
- `sentiment`
- `signals`

Python repositories write ingestion results and support Python-side use cases.
The .NET API reads the same tables through Entity Framework Core.

CSV files under `data/demo/` and `data/benchmark/` are bundled datasets for
demo and benchmark flows; they are not the production persistence layer.

---

## Tests

Run Python tests:

```bash
pytest
```

The development Compose file provides a temporary PostgreSQL test service:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile test up test_db
```

Run the .NET integration tests with:

```bash
dotnet test api-dotnet/CryptoTracker.Api.IntegrationTests
```

Tests cover domain behavior, fetcher boundaries, sentiment caching, database
repositories, Python API routes, and .NET query endpoints.

---

## Architecture

- `presentation/` owns Streamlit and Python FastAPI interfaces.
- `app/` coordinates ingestion, analysis, caching, and scheduled jobs.
- `domain/` owns sentiment, market, signal, lead/lag, and backtest logic.
- `infra/` owns external fetchers and PostgreSQL repositories.
- `api-dotnet/` owns the read/query API used by Streamlit.

See `ARCHITECTURE.md` for the complete system and request-flow map.

---

## Roadmap

- [x] Refactor into a layered Python structure
- [x] Move runtime storage to PostgreSQL
- [x] Add Alembic migrations
- [x] Separate ingestion and query responsibilities
- [x] Add the .NET query API
- [x] Add scheduled ingestion
- [x] Add CI/CD
- [ ] Deploy the stack to AWS

## License

MIT — see `LICENSE`.
