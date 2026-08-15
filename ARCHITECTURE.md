# Architecture

## Overview

Coin Sentiment uses a write/read split around one PostgreSQL schema. Python
owns ingestion and schema migrations; ASP.NET Core owns read-only API queries;
Streamlit owns presentation.

```text
Content sources + Coinbase
          |
          v
Python ingestion (FastAPI or worker)
          |
          v
     PostgreSQL database
          |
          v
   ASP.NET Core query API
          |
          v
    Streamlit dashboard
```

Local development runs PostgreSQL as the Compose `db` service. Production on
EC2 connects the same services to AWS RDS. RDS configuration belongs to the
deployment environment and is not embedded in the local Compose file.

## Runtime services

| Service | Technology | Responsibility |
|---|---|---|
| `db` | PostgreSQL 16 | Local development database |
| `migrate` | Python + Alembic | Applies schema migrations before application services start |
| `ingest-api` | FastAPI | Authenticated `POST /ingest` write endpoint |
| `ingest-worker` | Python | Runs scheduled ingestion as a one-shot job |
| `query-api` | ASP.NET Core + EF Core | Read-only prices, posts, sentiment, and signals endpoints |
| `ui` | Streamlit | Queries the .NET API and renders analysis |

`docker-compose.dev.yml` adds source mounts, reload for FastAPI, port `5432`
for the local database, and a temporary `test_db` on port `5433`.

## Data flows

### Ingestion

1. The FastAPI endpoint or worker creates an `AnalysisConfig`.
2. Fetchers retrieve Reddit, YouTube, RSS, and Coinbase data.
3. Python repositories persist content and price rows.
4. The configured sentiment analyzer scores new content.
5. Sentiment rows are stored with the analyzer name.

### Dashboard

1. Streamlit collects a query configuration in the sidebar.
2. The UI calls the .NET query API through `QUERY_API_URL`.
3. The query API reads PostgreSQL and returns typed response objects.
4. Streamlit converts responses to DataFrames and calculates indicators,
   signals, lead/lag results, and backtests for presentation.

The UI never fetches source data or loads transformer models.

## Python layers

- `src/presentation/` contains Streamlit, FastAPI routes, request validation,
  and API clients.
- `src/app/` coordinates ingestion, sentiment caching, indicators, and jobs.
- `src/domain/` contains analyzer adapters, market calculations, signals,
  lead/lag analysis, and backtesting.
- `src/infra/` contains external fetchers and PostgreSQL repositories.
- `src/shared/` contains DataFrame validation and normalization helpers.

## Database ownership

The shared PostgreSQL schema contains:

- `prices`
- `content_items`
- `sentiment`
- `signals`

SQLAlchemy models live in `src/infra/storage/db/models.py`. Alembic migrations
in `alembic/versions/` are the schema-change mechanism. The .NET mapping is in
`api-dotnet/CryptoTracker.Api/Data/CryptoDbContext.cs`.

## Configuration

| Variable | Consumer | Purpose |
|---|---|---|
| `DATABASE_URL` | Python migration and ingestion services | PostgreSQL SQLAlchemy connection |
| `ConnectionStrings__Postgres` | .NET query API | PostgreSQL EF Core connection |
| `QUERY_API_URL` | Streamlit | .NET query API base URL |
| `ADMIN_API_KEY` | FastAPI | Protects `POST /ingest` |
| `HF_DEVICE` | Transformer analyzers | Execution device selection |
| Reddit and YouTube credentials | Fetchers | Source access |

Local Compose constructs service connection values from `POSTGRES_USER`,
`POSTGRES_PASSWORD`, and `POSTGRES_DB`. EC2 uses RDS values configured in its
deployment environment.

## Testing and deployment

Python tests use the temporary `test_db` service on port `5433`. .NET
integration tests use the local Compose `db` service on port `5432`. GitHub
Actions runs both suites, builds images, and deploys the configured targets.

The externally visible query behavior is specified in
[API_CONTRACT.md](API_CONTRACT.md).
