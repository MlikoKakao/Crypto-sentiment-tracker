from src.infra.storage.db.connection import get_connection

def init_db():
    with get_connection() as conn:
        conn.execute(
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
                PRIMARY KEY (url)
                );
            """
            )
        conn.commit()
    conn.close()