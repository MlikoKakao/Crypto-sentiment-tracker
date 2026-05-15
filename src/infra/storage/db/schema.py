from src.infra.storage.db.connection import get_connection


def init_db():
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prices 
                (coin TEXT NOT NULL, 
                timestamp TEXT NOT NULL, 
                price REAL NOT NULL, 
                PRIMARY KEY (coin, timestamp)
                );
            
            CREATE TABLE IF NOT EXISTS content_items (
                coin TEXT NOT NULL,
                source TEXT NOT NULL, -- reddit/news/youtube
                source_id TEXT NOT NULL, -- reddit id, youtube id, or news url
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                url TEXT NOT NULL,
                PRIMARY KEY (coin, source, source_id)
            );

            CREATE TABLE IF NOT EXISTS sentiment (
                coin TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                analyzer TEXT NOT NULL,
                sentiment REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (coin, source, source_id, analyzer),
                FOREIGN KEY (coin, source, source_id)
                    REFERENCES content_items(coin, source, source_id)
            );
            CREATE TABLE IF NOT EXISTS signals (
                coin TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY(coin, timestamp, signal_name)
            );
            """
        )
        conn.commit()
    conn.close()
