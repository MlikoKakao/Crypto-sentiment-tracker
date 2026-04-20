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

            CREATE TABLE IF NOT EXISTS news (
                coin TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                source TEXT,
                url TEXT NOT NULL,
                text TEXT,
                PRIMARY KEY (coin, url)
                );
            CREATE TABLE IF NOT EXISTS reddit (
                coin TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                url TEXT NOT NULL,
                score INT,
                num_comments INT,
                upvote_ratio REAL,
                id TEXT NOT NULL,
                source TEXT,
                subreddit TEXT,
                PRIMARY KEY (coin, id)
            );
            CREATE TABLE IF NOT EXISTS youtube (
                coin TEXT NOT NULL,
                id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT,
                url TEXT NOT NULL,
                author TEXT,
                PRIMARY KEY (coin, id)
            );
            """
            )
        conn.commit()
    conn.close()