# Crypto Sentiment Tracker

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-api-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

Crypto Sentiment Tracker collects crypto-related posts/articles, scores their sentiment, combines that with market data, and shows the result in a Streamlit dashboard and FastAPI API.

The current app supports live analysis, SQLite-backed caching/storage, basic technical indicators, lead/lag analysis, and a small backtest view.

Demo: https://crypto-currency-sentiment-analysis.streamlit.app

<img width="3835" height="1746" alt="Crypto Sentiment Tracker screenshot" src="https://github.com/user-attachments/assets/0f1cbe11-9945-487c-aa86-05de0c561725" />

---

## Features

- Fetches crypto content from Reddit, YouTube, and RSS news feeds
- Fetches market prices from Coinbase, with CoinGecko fallback infrastructure
- Scores sentiment with VADER, TextBlob, RoBERTa, or FinBERT
- Supports `analyzer="all"` aggregation across analyzers
- Stores prices, content, sentiment, and signals in SQLite
- Exposes FastAPI endpoints for health, prices, sentiment, posts, signals, and ingest
- Renders a Streamlit UI with charts, lead/lag analysis, indicators, benchmark views, and backtest output
- Includes focused pytest coverage for domain logic, repositories, API routes, fetcher boundaries, and cache behavior

---

## Project Layout

```text
.
├─ Dockerfile
├─ docker-compose.yml
├─ pyproject.toml
├─ run_app.py
├─ README.md
├─ ARCHITECTURE.md
├─ DECISIONS.md
├─ tests/
│  ├─ api/
│  ├─ sentiment/
│  ├─ conftest.py
│  ├─ test_db.py
│  ├─ test_fetchers.py
│  ├─ test_sentiment_cache.py
│  └─ test_signals.py
├─ stubs/
│  ├─ textblob/
│  └─ vader/
└─ src/
   ├─ app/
   │  ├─ dto.py
   │  ├─ defaults.py
   │  └─ use_cases/
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
   │     ├─ sentiment_csv.py
   │     └─ db/
   ├─ presentation/
   │  ├─ api/
   │  │  └─ routes/
   │  ├─ config/
   │  ├─ pages.py
   │  ├─ sidebar.py
   │  └─ charts.py
   └─ shared/
```

`pyproject.toml` is the source of Python dependencies. `requirements.txt` is not used by the Dockerfile or quickstart workflow.

---

## Local Setup

```bash
git clone https://github.com/MlikoKakao/crypto-sentiment-tracker.git
cd crypto-sentiment-tracker

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---

## Environment Setup

Copy the public template:

```bash
cp .env.example .env
```

Then edit `.env` and fill in the API keys you need.

Important variables:

```env
DATABASE_PATH=data/app.db
DEMO=0
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
YOUTUBE_API_KEY=
HF_DEVICE=-1
```

Reddit and YouTube require API keys. News RSS and Coinbase price fetching do not currently require keys.

For Streamlit Cloud, `run_app.py` also supports loading values from `.streamlit/secrets.toml`.

---

## Run Locally

Run the Streamlit UI:

```bash
streamlit run run_app.py
```

Run the FastAPI API:

```bash
uvicorn src.presentation.api.main:app --reload
```

API health check:

```bash
curl http://localhost:8000/health
```

Run tests:

```bash
pytest
```

---

## Docker

The Dockerfile installs the package from `pyproject.toml` and defaults to:

```dockerfile
CMD ["python", "run_app.py"]
```

`docker-compose.yml` defines three services:

- `api`: FastAPI service on host port `8002`
- `ui`: Streamlit service on host port `8501`
- `migrate`: one-off SQLite schema initialization command

Prepare the Docker-managed SQLite volume:

```bash
docker compose run --rm migrate
```

Start API and UI:

```bash
docker compose up --build
```

Start only the UI and its dependency:

```bash
docker compose up ui
```

Open:

```text
http://localhost:8501
```

API health check from the host:

```bash
curl http://localhost:8002/health
```

Stop services:

```bash
docker compose down
```

The Compose setup uses a named volume:

```text
app_data -> /usr/src/app/data
```

That persists the SQLite database across container restarts. Because it is a named volume, bundled CSV demo files from the repo are not automatically available inside `/usr/src/app/data`; the project is moving toward DB-backed demo data.

---

## Storage

Current persistent storage is SQLite:

- schema: `src/infra/storage/db/schema.py`
- connection: `src/infra/storage/db/connection.py`
- repositories:
  - `content_repository.py`
  - `price_repository.py`
  - `sentiment_repository.py`
  - `signal_repository.py`

Default local DB path:

```text
data/app.db
```

Docker DB path:

```text
/usr/src/app/data/app.db
```

Legacy/demo CSV code still exists in a few places, especially demo and benchmark flows. The intended direction is to move demo/runtime data fully into SQLite.

---

## Tests

Run all tests:

```bash
pytest
```

Current test coverage includes:

- domain merge and signal behavior
- sentiment service behavior
- SQLite repository round-trips with temporary DBs
- API route behavior with mocked repository dependencies
- fetcher boundary behavior without real external API calls
- sentiment cache hit/miss behavior

---

## Architecture Notes

- `presentation/` owns Streamlit and FastAPI interfaces.
- `app/use_cases/` coordinates workflows.
- `domain/` owns business logic and calculations.
- `infra/` owns external boundaries: APIs and storage.
- `shared/` contains cross-layer helpers and DataFrame utilities.

See `ARCHITECTURE.md` for the fuller system map.

---

## Roadmap

- [x] Refactor into layered project structure
- [x] Add FastAPI API layer
- [x] Add SQLite repositories for core cached data
- [x] Add Dockerfile and Docker Compose services
- [x] Add healthcheck and migration service
- [x] Replace remaining CSV demo/runtime paths with DB-backed demo data
- [x] Finish code and architecture clean-up
- [x] Move from Sqlite to PostgreSQL
- [x] Improve API to be production(ish)-level
- [x] Docker setup
- [x] Scheduled ingestion through Raspberry Pi
- [x] CI implementation
- [x] CD implementation
- [ ] Kubernetes deployment demo - delayed, other priorities more important
- [ ] Cloud deployment

## License

MIT - see `LICENSE`.
