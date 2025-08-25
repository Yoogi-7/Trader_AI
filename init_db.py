# init_db.py
import os
import sqlite3
import yaml
from core.schema import ensure_base_schema, migrate_signals_schema

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        ensure_base_schema(conn)
        migrate_signals_schema(conn)
        conn.commit()
        print(f"DB schema ready at: {db_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
