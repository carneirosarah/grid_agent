"""Per-user sessions: snapshot round-trips, store behaviour, persistence.

Everything runs against `InMemorySessionRepository`, which stores JSON
strings — the exact (de)serialisation path Postgres uses — so these tests
cover the real round-trip without a database. The Postgres repository
itself is exercised by `test_persistence_pg.py` (skipped without a DB).
"""

import threading

import pandas as pd

from grid_agent.persistence import InMemorySessionRepository
from grid_agent.schemas import Condition, Plan, Sort, SortKey, UpdateWhere
from grid_agent.sessions import SessionStore, restore_session, snapshot_session
from grid_agent.state import TableSession
from grid_agent.trace import NullTracer

PLAN = Plan(operations=[
    UpdateWhere(where=[Condition(column="category", op="eq", value="Electronics")],
                column="price", action="multiply", value=2.0),
    Sort(keys=[SortKey(column="price", ascending=False)])],
    summary="Double electronics prices, sort desc")


def make_store(small_df, repository=None, **kwargs) -> SessionStore:
    return SessionStore(df_factory=lambda: small_df.copy(),
                        repository=repository or InMemorySessionRepository(),
                        tracer=NullTracer(), **kwargs)


# --- snapshot / restore -----------------------------------------------------

def test_snapshot_roundtrip_preserves_dtypes_and_order(small_df):
    session = TableSession(df=small_df.copy())
    session.propose(PLAN)
    session.accept()                        # gives us an undo snapshot too
    session.history.append({"role": "user", "text": "double electronics"})

    restored = restore_session(snapshot_session(session), NullTracer())

    pd.testing.assert_frame_equal(restored.df, session.df)   # dtypes included
    assert restored.history == session.history
    assert len(restored.undo_stack) == 1
    restored.undo()                          # undo works after a round-trip
    pd.testing.assert_frame_equal(
        restored.df.reset_index(drop=True), small_df)


def test_pending_preview_is_not_persisted(small_df):
    session = TableSession(df=small_df.copy())
    session.propose(PLAN)                    # staged but never accepted
    restored = restore_session(snapshot_session(session), NullTracer())
    assert restored.pending is None
    pd.testing.assert_frame_equal(restored.df, small_df)


# --- store behaviour --------------------------------------------------------

def test_store_isolates_sessions(small_df):
    store = make_store(small_df)
    a, b = store.entry("alice"), store.entry("bob")
    a.session.edit_cell("SKU-0001", "price", "999")
    assert b.session.df.loc[0, "price"] == 25.0
    assert store.entry("alice") is a         # cached, not rebuilt


def test_store_survives_restart_with_undo_intact(small_df):
    """Simulated server restart: a new store over the same repository must
    restore the committed table AND the undo stack."""
    repo = InMemorySessionRepository()
    store1 = make_store(small_df, repository=repo)
    entry = store1.entry("alice")
    entry.session.propose(PLAN)
    entry.session.accept()
    store1.persist("alice")

    store2 = make_store(small_df, repository=repo)    # "restart"
    revived = store2.entry("alice")
    assert revived.session.df.loc[0, "price"] == 300.0   # sorted desc
    assert revived.session.can_undo
    revived.session.undo()
    assert revived.session.df.loc[0, "price"] == 25.0


def test_lru_eviction_reloads_from_repository(small_df):
    repo = InMemorySessionRepository()
    store = make_store(small_df, repository=repo, cache_limit=1)
    alice = store.entry("alice")
    alice.session.edit_cell("SKU-0001", "price", "111")
    store.persist("alice")

    store.entry("bob")                        # evicts alice (limit 1)
    revived = store.entry("alice")            # reloaded from repository
    assert revived is not alice
    assert revived.session.df.loc[0, "price"] == 111.0


def test_new_sessions_are_persisted_at_birth(small_df):
    repo = InMemorySessionRepository()
    make_store(small_df, repository=repo).entry("alice")
    assert repo.load("alice") is not None


def test_concurrent_first_contact_creates_one_session(small_df):
    """Many threads racing on the same brand-new id must all get the same
    entry (the creation lock prevents rival sessions)."""
    store = make_store(small_df)
    results: list = []
    barrier = threading.Barrier(8)

    def hit():
        barrier.wait()
        results.append(store.entry("same-id"))

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(e) for e in results}) == 1
