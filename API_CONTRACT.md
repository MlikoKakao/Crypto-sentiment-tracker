# API Contract

This document defines the HTTP behavior that API clients can rely on. Dates use
ISO-8601 timestamps, and date-range boundaries are inclusive.

The .NET API owns all `GET` data endpoints. FastAPI owns `POST /ingest`.
Alembic owns schema migrations. Both APIs use the same PostgreSQL database.

All read endpoints use these date defaults:

- `end_date`: current UTC time
- `start_date`: seven days before the resolved `end_date`

## Prices

### `GET /prices`

Returns stored price points ordered by `timestamp` descending.

#### Query parameters

| Parameter | Type | Required? | Default |
| --- | --- | ---: | --- |
| `coin` | string | No | `BTC` |
| `start_date` | ISO-8601 timestamp | No | Seven days before `end_date` |
| `end_date` | ISO-8601 timestamp | No | Current UTC time |

Supported coins are `BTC`, `ETH`, and `XMR`. Coin matching is
case-insensitive.

Example:

```http
GET /prices?coin=BTC&start_date=2026-05-30T00:00:00Z&end_date=2026-06-01T00:00:00Z
```

#### Response

```json
[
  {
    "coin": "BTC",
    "timestamp": "2026-05-30T12:00:00Z",
    "price": 104250.50
  }
]
```

An empty result returns `200 OK` with `[]`.

#### Validation

- Unknown coins return `400 Bad Request`.
- `end_date` earlier than `start_date` returns `400 Bad Request`.
- Both timestamps at the boundaries are included.

## Posts

### `GET /posts`

Returns stored content ordered by `timestamp` descending.

#### Query parameters

| Parameter | Type | Required? | Default |
| --- | --- | ---: | --- |
| `coin` | string | No | `BTC` |
| `start_date` | ISO-8601 timestamp | No | Seven days before `end_date` |
| `end_date` | ISO-8601 timestamp | No | Current UTC time |
| `source` | repeated string | No | All sources |
| `numPosts` | integer | No | `100` |

Supported sources are `reddit`, `youtube`, and `news`. Repeat `source` to
request more than one source.

Example:

```http
GET /posts?coin=BTC&source=reddit&source=news&numPosts=25
```

#### Response

```json
[
  {
    "coin": "BTC",
    "source": "reddit",
    "sourceId": "post-123",
    "timestamp": "2026-05-30T12:00:00Z",
    "text": "Bitcoin is looking strong",
    "url": "https://example.com/post-123",
    "contentHash": "abc123"
  }
]
```

`sourceId` and `url` may be `null`. An empty result returns `200 OK` with
`[]`. `numPosts` is constrained to the range 1 through 1000.

#### Validation

- Unknown coins or sources return `400 Bad Request`.
- `end_date` earlier than `start_date` returns `400 Bad Request`.
- Both timestamps at the boundaries are included.

## Sentiment

### `GET /sentiment`

Returns stored content together with its calculated sentiment, ordered by the
content timestamp descending.

#### Query parameters

| Parameter | Type | Required? | Default |
| --- | --- | ---: | --- |
| `coin` | string | No | `BTC` |
| `start_date` | ISO-8601 timestamp | No | Seven days before `end_date` |
| `end_date` | ISO-8601 timestamp | No | Current UTC time |
| `source` | repeated string | No | All sources |
| `analyzer` | string | No | `vader` |
| `limit` | integer | No | `10` |

Supported analyzers are `vader`, `textblob`, `twitter-roberta`, `finbert`, and
`all`. The value `all` returns rows from every supported analyzer.

Example:

```http
GET /sentiment?coin=BTC&source=reddit&source=news&analyzer=vader&limit=10
```

#### Response

```json
[
  {
    "coin": "BTC",
    "source": "reddit",
    "sourceId": "post-123",
    "timestamp": "2026-05-30T12:00:00Z",
    "text": "Bitcoin is looking strong",
    "url": "https://example.com/post-123",
    "contentHash": "abc123",
    "analyzer": "vader",
    "sentiment": 0.5
  }
]
```

`sourceId` and `url` may be `null`. Date filtering and ordering use the
content timestamp, not the time sentiment analysis was performed. Filtering
happens before the limit is applied. `analyzer=all` still returns at most
`limit` rows in total. An empty result returns `200 OK` with `[]`.

#### Validation

- Unknown coins, sources, or analyzers return `400 Bad Request`.
- `limit` must be from 1 through 1000; invalid values return
  `400 Bad Request`.
- `end_date` earlier than `start_date` returns `400 Bad Request`.
- Both timestamps at the boundaries are included.

## Signals

### `GET /signals`

Returns stored signal values ordered by `timestamp` descending.

#### Query parameters

| Parameter | Type | Required? | Default |
| --- | --- | ---: | --- |
| `coin` | string | No | `BTC` |
| `start_date` | ISO-8601 timestamp | No | Seven days before `end_date` |
| `end_date` | ISO-8601 timestamp | No | Current UTC time |
| `signalName` | repeated string | No | `sma_20` and `sma_50` |
| `numSignals` | integer | No | `100` |

Repeat `signalName` to request more than one signal.

Example:

```http
GET /signals?coin=BTC&signalName=sma_20&signalName=rsi&numSignals=25
```

#### Response

```json
[
  {
    "coin": "BTC",
    "timestamp": "2026-05-30T12:00:00Z",
    "signalName": "sma_20",
    "value": 104000.25
  }
]
```

An empty result returns `200 OK` with `[]`. `numSignals` is constrained to the
range 1 through 1000.

#### Validation

- Unknown coins return `400 Bad Request`.
- `end_date` earlier than `start_date` returns `400 Bad Request`.
- Both timestamps at the boundaries are included.
