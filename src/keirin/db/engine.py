"""SQLite engine and schema initialization."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_engine(db_path: str | Path) -> Engine:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _conn_record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


# Columns added after the initial schema. CREATE TABLE IF NOT EXISTS doesn't
# alter existing tables, so pre-existing DBs are migrated with ALTER TABLE.
_MIGRATIONS: dict[str, dict[str, str]] = {
    "entries": {
        "nige_cnt": "INTEGER",
        "makuri_cnt": "INTEGER",
        "sashi_cnt": "INTEGER",
        "mark_cnt": "INTEGER",
        "s_count": "INTEGER",
        "b_count": "INTEGER",
        "line_raw": "TEXT",
    },
}


def init_db(engine: Engine) -> None:
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for stmt in _split_sql(schema):
            if stmt.strip():
                conn.exec_driver_sql(stmt)
        for table, columns in _MIGRATIONS.items():
            existing = {
                row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")
            }
            for col, coltype in columns.items():
                if col not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")


def _split_sql(sql: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        if line.strip().startswith("--"):
            continue
        buf.append(line)
        if line.rstrip().endswith(";"):
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out
