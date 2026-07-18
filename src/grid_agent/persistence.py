"""Session persistence — PostgreSQL-backed, with an in-memory stand-in.

What is durable and what is not
-------------------------------
A session's *durable* state is exactly what must survive a server restart:

- the committed table,
- the undo stack (so undo still works after a restart),
- the chat history (so clarification context survives).

The **pending preview is deliberately not persisted**: it is a transient
proposal derived from the committed table, and resurrecting a stale
preview after a crash/restart would be worse than asking the user to
re-issue the instruction.

Storage model
-------------
One row per session in a single `sessions` table, the whole snapshot as a
JSONB document plus a monotonically increasing `version` (written on every
save; useful for audits and a hook for optimistic locking if the app ever
runs multiple replicas — see DECISIONS.md).

DataFrames are serialised with pandas' `orient="table"` JSON, which
embeds the schema so dtypes (bool/int/float/str) survive the round-trip
byte-for-byte — a `records` dump would silently degrade them.

`SessionRepository` is a Protocol: the API depends on the interface only.
`PostgresSessionRepository` is production; `InMemorySessionRepository`
keeps the offline test suite (and DB-less dev) fully functional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from typing import Protocol

import pandas as pd

from .config import DATABASE_URL


@dataclass
class SessionSnapshot:
    """Durable state of one session, ready for JSON storage."""
    table: dict                  # committed df, pandas "table" orient
    undo: list[dict]             # undo stack, oldest first, same encoding
    history: list[dict[str, str]]  # chat history ({role, text} turns)


# --- DataFrame <-> JSON (dtype-faithful) -----------------------------------

def df_to_doc(df: pd.DataFrame) -> dict:
    """Encode a DataFrame as a schema-carrying JSON document."""
    return json.loads(df.to_json(orient="table", index=False))


def doc_to_df(doc: dict) -> pd.DataFrame:
    """Decode `df_to_doc` output back into an identically-typed DataFrame."""
    return pd.read_json(StringIO(json.dumps(doc)), orient="table")


# --- repository interface ---------------------------------------------------

class SessionRepository(Protocol):
    """Anything that can load/save session snapshots by id."""

    def load(self, session_id: str) -> SessionSnapshot | None: ...
    def save(self, session_id: str, snapshot: SessionSnapshot) -> None: ...
    def close(self) -> None: ...


class InMemorySessionRepository:
    """Dict-backed repository for tests and DB-less development. Snapshots
    are stored as JSON strings so the (de)serialisation path is identical
    to the Postgres one — tests exercise the real round-trip."""

    def __init__(self) -> None:
        self._rows: dict[str, str] = {}

    def load(self, session_id: str) -> SessionSnapshot | None:
        raw = self._rows.get(session_id)
        return SessionSnapshot(**json.loads(raw)) if raw else None

    def save(self, session_id: str, snapshot: SessionSnapshot) -> None:
        self._rows[session_id] = json.dumps(snapshot.__dict__)

    def close(self) -> None:
        self._rows.clear()


class PostgresSessionRepository:
    """Sessions in a PostgreSQL table, accessed through a small pool.

    Concurrency stance: the API layer serialises access *per session* with
    an in-process lock, so at most one writer per session exists inside
    one server process. The pool (max 4 connections) covers concurrent
    requests across *different* sessions.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            state       JSONB        NOT NULL,
            version     INTEGER      NOT NULL DEFAULT 1,
            updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
        )
    """

    def __init__(self, dsn: str = DATABASE_URL) -> None:
        if not dsn:
            raise ValueError("DATABASE_URL is not configured.")
        # Imported lazily: the offline test suite must not require psycopg
        # to be importable, let alone a running server.
        from psycopg_pool import ConnectionPool
        self._pool = ConnectionPool(dsn, min_size=1, max_size=4, open=True)
        with self._pool.connection() as conn:
            conn.execute(self._SCHEMA)

    def load(self, session_id: str) -> SessionSnapshot | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT state FROM sessions WHERE session_id = %s",
                (session_id,)).fetchone()
        return SessionSnapshot(**row[0]) if row else None

    def save(self, session_id: str, snapshot: SessionSnapshot) -> None:
        from psycopg.types.json import Jsonb
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, state)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE
                SET state = EXCLUDED.state,
                    version = sessions.version + 1,
                    updated_at = now()
                """,
                (session_id, Jsonb(snapshot.__dict__)))

    def close(self) -> None:
        self._pool.close()


def make_repository(dsn: str = DATABASE_URL) -> SessionRepository:
    """Default wiring: Postgres when DATABASE_URL is set, memory otherwise."""
    return PostgresSessionRepository(dsn) if dsn else InMemorySessionRepository()
