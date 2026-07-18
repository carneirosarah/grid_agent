"""PostgreSQL repository — live integration test.

Auto-skipped unless a database is reachable. Run it with the compose db:

    docker compose up -d db
    DATABASE_URL=postgresql://grid:grid@localhost:5433/grid_agent \
        pytest tests/test_persistence_pg.py -v
"""

import os
import uuid

import pandas as pd
import pytest

from grid_agent.persistence import PostgresSessionRepository
from grid_agent.schemas import Condition, Plan, UpdateWhere
from grid_agent.sessions import restore_session, snapshot_session
from grid_agent.state import TableSession
from grid_agent.trace import NullTracer

DSN = os.getenv("DATABASE_URL", "")

pytestmark = pytest.mark.skipif(not DSN, reason="DATABASE_URL not set")


def test_postgres_roundtrip_and_versioning(small_df):
    repo = PostgresSessionRepository(DSN)
    session_id = f"test-{uuid.uuid4().hex}"       # unique per run
    try:
        assert repo.load(session_id) is None      # unknown id -> None

        session = TableSession(df=small_df.copy())
        session.propose(Plan(operations=[UpdateWhere(
            where=[Condition(column="category", op="eq", value="Electronics")],
            column="price", action="multiply", value=2.0)]))
        session.accept()
        session.history.append({"role": "user", "text": "double electronics"})

        # Save twice: the upsert path must bump version, not fail.
        repo.save(session_id, snapshot_session(session))
        repo.save(session_id, snapshot_session(session))

        restored = restore_session(repo.load(session_id), NullTracer())
        pd.testing.assert_frame_equal(restored.df, session.df)
        assert restored.history == session.history
        assert restored.can_undo
        restored.undo()
        pd.testing.assert_frame_equal(restored.df, small_df)
    finally:
        repo.close()
