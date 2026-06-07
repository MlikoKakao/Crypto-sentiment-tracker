from pathlib import Path
from contextlib import closing
from src.infra.storage.db.connection import get_connection


def init_db(db_path: Path | str | None = None) -> None:
    with closing(get_connection(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prices (
                coin TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                PRIMARY KEY (coin, timestamp)
            );

            CREATE TABLE IF NOT EXISTS content_items (
                coin TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                url TEXT,
                content_hash TEXT NOT NULL,

                PRIMARY KEY (coin, source, content_hash)
            );

            CREATE TABLE IF NOT EXISTS sentiment (
                coin TEXT NOT NULL,
                source TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                analyzer TEXT NOT NULL,
                sentiment REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (coin, source, content_hash, analyzer),

                FOREIGN KEY (coin, source, content_hash)
                    REFERENCES content_items (coin, source, content_hash)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS signals (
                coin TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (coin, timestamp, signal_name)
            );

            CREATE INDEX IF NOT EXISTS idx_content_coin_source_timestamp
                ON content_items (coin, source, timestamp);

            CREATE INDEX IF NOT EXISTS idx_content_coin_source_id
                ON content_items (coin, source, source_id);

            CREATE INDEX IF NOT EXISTS idx_sentiment_coin_analyzer_source
                ON sentiment (coin, analyzer, source, content_hash);

            CREATE INDEX IF NOT EXISTS idx_signals_coin_signal_timestamp
                ON signals (coin, signal_name, timestamp);
            """
        )

        conn.commit()
