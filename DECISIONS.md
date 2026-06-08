# Decisions

This file records important project decisions so future changes have context.

---

## 1. Use SQLite First

### Decision

Use SQLite as the first persistent database.

### Reasoning

SQLite is easy to set up, lightweight, local, and does not require running a separate database server. That fits the current stage of the project: learning, prototyping, Dockerizing, and building a working sentiment tracker without extra infrastructure.

The current app stores cached prices, content items, sentiment scores, and signals in SQLite.

### Consequences

- Local development is simple.
- Tests can use temporary SQLite databases with `tmp_path`.
- Docker can persist the database with a single volume.
- The app is not yet designed for high-concurrency writes or multi-user production workloads.

### Revisit When

Move toward PostgreSQL when the project needs:

- multiple users writing at once
- stronger production deployment guarantees
- richer migration tooling
- hosted database infrastructure
- more complex querying/reporting

---

## 2. Use Streamlit + FastAPI

### Decision

Use Streamlit for the UI and FastAPI for API endpoints.

### Reasoning

Both tools are fast to set up and match the current workflow. Streamlit is good for quickly building dashboards and interactive analysis views. FastAPI is good for exposing structured endpoints such as health checks, prices, sentiment, posts, signals, and ingest.

The current workflow does not demand a heavier frontend framework or a more complex service architecture.

### Consequences

- The project can move quickly.
- The UI remains easy to change.
- The API gives a clean path toward separating backend behavior.
- There is some duplication/overlap because the Streamlit UI still imports application use cases directly instead of calling the FastAPI service over HTTP.

### Revisit When

Consider a stronger API/UI split when:

- the UI should be deployed separately from the API
- multiple frontends need to use the same backend
- authentication or user accounts become important
- Streamlit becomes limiting for the desired UX

---

## 3. Replace Twitter With YouTube

### Decision

Replace Twitter/X scraping with YouTube as a supported content source.

### Reasoning

Twitter/X scraping became unreliable. Scrapers break often, access rules change, and unofficial scraping adds maintenance risk.

The YouTube API is more stable and has a usable official client. It still requires an API key, but it is a cleaner integration point than depending on brittle Twitter/X scraping.

### Consequences

- Source collection is more reliable.
- The project uses an official API instead of unstable scraping.
- YouTube content may have different sentiment characteristics than Twitter/X posts.
- YouTube API quota limits and API keys must be managed.

### Revisit When

Reconsider Twitter/X only if there is a stable, legal, maintainable API path that fits the project.

---

## 4. Use `content_hash` Instead Of Only `source_id`

### Decision

Use `content_hash` as the primary identity for stored content items, while still keeping `source_id` when a source provides it.

### Reasoning

Not all sources provide a stable ID for each post or article. Reddit and YouTube usually have IDs, but RSS/news entries may not. Some sources may only provide a URL or text.

`content_hash` gives the app one consistent identity field across all sources.

Current hash priority:

```text
source + source_id
source + url
source + text
```

### Consequences

- Deduplication works across sources with different metadata quality.
- Sentiment rows can reference content consistently.
- The schema does not depend on every scraper having the same fields.
- If only text is available, edited or slightly changed text can produce a new hash.

### Revisit When

Revisit hashing if sources start providing better stable IDs, or if deduplication needs to detect near-duplicate content instead of exact ID/URL/text matches.

---

## 5. Cache Indicators In The App Layer, Not The Domain Layer

### Decision

Keep indicator calculations in the domain layer, but keep indicator cache orchestration in the application layer.

### Reasoning

The domain layer should answer:

```text
Given this price DataFrame and this indicator config, what are the indicator values?
```

The app layer should answer:

```text
Do we already have these values cached?
If yes, load them.
If no, calculate them and save them.
```

That is why `src/domain/market/indicators.py` contains calculation logic, while `src/app/use_cases/get_indicators.py` coordinates cache checks, loading, calculation, and saving.

### Consequences

- Domain indicator functions stay easier to test as pure calculations.
- Storage concerns stay outside the domain layer.
- The app layer owns workflow decisions.
- The app layer is more complex because it coordinates both domain logic and repositories.

### Revisit When

Revisit if caching becomes generic enough to extract into a reusable cache service, or if indicator calculation needs a more formal job/task system.

---

## 6. Make Heavy ML Optional

### Decision

Keep heavy ML dependencies optional.

### Reasoning

Transformer-based models like RoBERTa and FinBERT can improve sentiment quality, but they are large, slow, and expensive compared with lightweight analyzers.

VADER is much smaller and already gives decent baseline performance for quick sentiment exploration. TextBlob is also lightweight enough for normal installation.

Therefore `torch` and `transformers` live in the optional `ml` dependency group instead of the base install.

### Consequences

- Normal installs and Docker builds stay smaller.
- Basic analysis works without downloading large models.
- Heavy analyzers can fail clearly if optional dependencies are not installed.
- Benchmarking or high-quality ML scoring requires an explicit opt-in install.

### Usage

Base install:

```bash
pip install -e .
```

Install with ML analyzers:

```bash
pip install -e ".[ml]"
```

Development install:

```bash
pip install -e ".[dev]"
```

Development plus ML:

```bash
pip install -e ".[dev,ml]"
```

### Revisit When

Revisit if the app becomes primarily ML-focused, or if deployment moves to an environment where model size and startup time are no longer a problem.
