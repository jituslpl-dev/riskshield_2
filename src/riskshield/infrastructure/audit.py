import json
import sqlite3
import threading
from pathlib import Path


class AuditStore:
    """SQLite adapter: retain decisions without raw transaction or feature data."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        self.connection.commit()

    def get(self, key, fingerprint):
        row = self.connection.execute(
            "SELECT fingerprint,response FROM decisions WHERE key=?", (key,)
        ).fetchone()
        if row:
            if row[0] != fingerprint:
                raise ValueError("Idempotency key reused with different input.")
            return json.loads(row[1])

    def save(self, key, fingerprint, response):
        self.connection.execute(
            "INSERT INTO decisions(key,fingerprint,response) VALUES(?,?,?)",
            (key, fingerprint, json.dumps(response)),
        )
        self.connection.commit()

    def close(self):
        self.connection.close()
