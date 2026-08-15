# Coin Sentiment

Coin Sentiment collects cryptocurrency-related content, scores its sentiment,
stores the results with market prices, and presents the data in a Streamlit
dashboard.

The project separates ingestion from querying:

- Python fetchers and sentiment analyzers write to PostgreSQL.
- FastAPI exposes an authenticated ingestion endpoint.
- ASP.NET Core exposes read-only query endpoints.
- Streamlit reads from the query API and renders charts and analysis.

## Features

- Reddit, YouTube, and RSS news ingestion
- Coinbase market-price ingestion
- VADER, TextBlob, Twitter-RoBERTa, and FinBERT sentiment analyzers
- PostgreSQL storage with Alembic migrations
- Technical indicators, signals, lead/lag analysis, and backtesting
- Python and .NET integration tests

## Runtime topology

```text
Reddit / YouTube / RSS News       Coinbase
              \                    /
               Python ingestion
              /                \
     FastAPI POST /ingest    scheduled worker
              \                /
                   PostgreSQL
                        |
                 .NET query API
                        |
               Streamlit dashboard
```

For local development, Docker Compose runs PostgreSQL in the `db` service.
The EC2 deployment uses AWS RDS instead; its connection settings are managed
in that deployment, not replaced by the local Compose configuration.

| Service | Responsibility | Host port |
|---|---|---:|
| `db` | Local PostgreSQL storage | `5432` in the development override |
| `migrate` | Runs `alembic upgrade head` and exits | — |
| `ingest-api` | Python write/ingestion API | `8002` |
| `query-api` | .NET read/query API | `8081` |
| `ui` | Streamlit dashboard | `8501` |
| `ingest-worker` | One-shot ingestion job | — |

## Project layout

```text
.
├─ api-dotnet/       # ASP.NET Core query API and integration tests
├─ alembic/          # PostgreSQL schema migrations
├─ src/
│  ├─ app/           # use cases, configuration, and jobs
│  ├─ domain/        # analysis and calculation rules
│  ├─ infra/         # fetchers and PostgreSQL repositories
│  ├─ presentation/  # Streamlit and FastAPI interfaces
│  └─ shared/
├─ tests/            # Python tests
├─ Dockerfile.api
├─ Dockerfile.ingest
├─ Dockerfile.ui
├─ docker-compose.yml
└─ docker-compose.dev.yml
```

## Local development

Create a local environment file:

```bash
cp .env.example .env
```

Set the API credentials you use in `.env`. Local Compose provides the database
credentials and overrides `DATABASE_URL` to point at its `db` service.

Start the stack with development mounts and a local PostgreSQL port:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Open:

- UI: http://localhost:8501
- Query API health: http://localhost:8081/health
- Ingestion API health: http://localhost:8002/health

Run a one-shot ingestion job:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm ingest-worker
```

Stop local services with:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

The local database persists in the `postgres_data` Docker volume. Downloaded
transformer models persist in `crypto-hf-cache`.

## AWS deployment

On EC2, PostgreSQL is provided by AWS RDS. Configure the deployment services
with:

- `DATABASE_URL` for the Python migration and ingestion services.
- `ConnectionStrings__Postgres` for the .NET query API.
- `QUERY_API_URL` for the Streamlit UI.
- `ADMIN_API_KEY` for the ingestion API.

Use the RDS endpoint and TLS settings appropriate for your instance. Do not
commit production connection strings or credentials to this repository.

## Running components without Docker

Install development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,ui,fetchers,sentiment,dev]"
```

Set `DATABASE_URL` to a reachable PostgreSQL database, then apply migrations:

```bash
alembic upgrade head
```

Start the Python ingestion API:

```bash
uvicorn src.presentation.api.main:app --reload --port 8000
```

Start the .NET query API with `ConnectionStrings__Postgres` configured:

```bash
dotnet run --project api-dotnet/CryptoTracker.Api
```

Set `QUERY_API_URL` to the query API URL and start the UI:

```bash
streamlit run run_app.py
```

## Storage and API contract

PostgreSQL holds `prices`, `content_items`, `sentiment`, and `signals`.
SQLAlchemy and Alembic own schema writes; Entity Framework Core maps the same
schema for the query API. See [API_CONTRACT.md](API_CONTRACT.md) for the public
read-endpoint contract and [ARCHITECTURE.md](ARCHITECTURE.md) for data flows.

## Tests

Start the temporary Python test database:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile test up -d test_db
```

Run Python tests:

```bash
pytest
```

The .NET integration tests use the local Compose `db` service on port `5432`:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db
dotnet test api-dotnet/CryptoTracker.Api.IntegrationTests
```

## License

MIT — see [LICENSE](LICENSE).
