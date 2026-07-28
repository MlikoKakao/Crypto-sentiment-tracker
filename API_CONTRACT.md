# API Contract

This document defines the HTTP behavior that API clients can rely on. Dates use
ISO-8601 timestamps and date-range boundaries are inclusive.

The .NET API owns all GET data endpoints.
FastAPI owns POST /ingest.
Alembic owns schema migrations.
Both .NET and FastAPI use the same PostgreSQL database.

## Prices

### `GET /prices`

Returns every stored price point in the requested date range. There is no row
limit; clients control the result size by choosing the date range.

#### Query parameters

| Parameter | Type | Required? | Default | Meaning |
| --- | --- | ---: | ---: | --- |  
| `coin` | string | Yes | — | `BTC`, `ETH`, or `XMR` |
| `start_date` | ISO-8601 timestamp | Yes | — | Earliest price timestamp to include |
| `end_date` | ISO-8601 timestamp | Yes | — | Latest price timestamp to include |

Example:

```http
GET /prices?coin=BTC&start_date=2026-05-30T00:00:00Z&end_date=2026-06-01T00:00:00Z
```

#### Response

The response is a JSON array ordered by `timestamp` descending (newest first).
An empty result is returned as `200 OK` with `[]`.

```json
[
  {
    "coin": "BTC",
    "timestamp": "2026-05-30T12:00:00Z",
    "priceValue": 104250.50
  }
]
```

#### Validation

- Missing required parameters return `400 Bad Request`.
- `end_date` earlier than `start_date` returns `400 Bad Request`.
- Both timestamps at the boundaries are included.

## Sentiment

### `GET /sentiment`

Returns content together with its calculated sentiment.

#### Query parameters

| Parameter | Type | Required? | Default | Meaning |
| --- | --- | ---: | ---: | --- |
| `coin` | string | Yes | — | `BTC`, `ETH`, or `XMR` |
| `start_date` | ISO-8601 timestamp | Yes | — | Earliest content timestamp to include |
| `end_date` | ISO-8601 timestamp | Yes | — | Latest content timestamp to include |
| `source` | repeated string | Yes | — | One or more of `reddit`, `youtube`, or `news` |
| `analyzer` | string | No | `vader` | `vader`, `textblob`, `twitter-roberta`, `finbert`, or `all` |
| `limit` | integer | No | `10` | Maximum number of rows returned |

Example:

```http
GET /sentiment?coin=BTC&start_date=2026-05-30T00:00:00Z&end_date=2026-06-01T00:00:00Z&source=reddit&source=news&analyzer=vader&limit=10
```

#### Response

The response is a JSON array ordered by the content `timestamp` descending.
An empty result is returned as `200 OK` with `[]`.

```json
[
  {
    "coin": "BTC",
    "source": "reddit",
    "source_id": "post-123",
    "timestamp": "2026-05-30T12:00:00Z",
    "text": "Bitcoin is looking strong",
    "url": "https://example.com/post-123",
    "content_hash": "abc123",
    "analyzer": "vader",
    "sentiment": 0.5
  }
]
```

`source_id` and `url` may be `null`. The content timestamp is returned rather
than the time at which sentiment analysis was performed.

#### Ordering and limits

- Results are ordered by content timestamp descending (newest first).
- Filtering happens before the limit is applied.
- `limit` must be from 1 through 1000; invalid values return `400 Bad Request`.
- `analyzer=all` still returns at most `limit` rows in total, not `limit` rows
  per analyzer.

#### Validation

- Missing required parameters return `400 Bad Request`.
- Unknown coins, sources, or analyzers return `400 Bad Request`.
- `end_date` earlier than or equal to `start_date` returns `400 Bad Request`.
- Coin matching is case-insensitive and response coin values are uppercase.
- Both timestamps at the boundaries are included.
