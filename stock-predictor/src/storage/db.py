from __future__ import annotations
import argparse, os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from src.config import settings

def get_engine() -> Engine:
    return create_engine(settings.DATABASE_URL, future=True)

def init_db():
    engine = get_engine()
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sql")
    schema_file = "create_tables_sqlite.sql" if engine.dialect.name == "sqlite" else "create_tables_postgres.sql"
    sql_path = os.path.join(base, schema_file)
    with open(sql_path, "r", encoding="utf-8") as f:
        ddl = f.read()
    # execute statements one by one for cross-dialect safety
    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Initialize DB schema")
    args = parser.parse_args()
    if args.init:
        init_db()
        print("DB initialized.")
