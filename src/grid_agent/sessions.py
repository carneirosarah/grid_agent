"""Per-user sessions: identity, caching, locking, write-through persistence.

Replaces the original single global `TableSession`. Each user (identified
by an opaque cookie value) gets their own isolated session, and the store
is the only way to reach one:

    entry = store.entry(session_id)
    with entry.lock:                 # serialise all work on this session
        ... mutate entry.session ...
        store.persist(session_id)    # write-through to the repository

Concurrency treatment
---------------------
- **Per-session `RLock`.** FastAPI runs sync endpoints on a thread pool,
  so two requests from the same user (double-click on Accept, two open
  tabs) can genuinely race. Every endpoint takes the session's lock for
  the duration of the request: the second Accept finds `pending` empty
  and gets a clean 409 instead of double-committing; a chat turn can't
  interleave with an undo. Sessions of *different* users never contend —
  each has its own lock.
- **One creation lock** guards the id -> entry map itself, so two first
  requests with the same new cookie cannot create two rival sessions.
- **Write-through persistence.** Durable state is saved to the repository
  after every committed mutation (the caller decides when — after accept,
  undo, edits, chat turns). The in-memory entry is therefore always a
  cache over the repository, which makes eviction safe.
- **LRU eviction.** At most `SESSION_CACHE_LIMIT` sessions stay resident;
  the least-recently-used entry is dropped (its state is already in the
  repository, so a later request simply reloads it).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .config import SESSION_CACHE_LIMIT
from .persistence import SessionRepository, SessionSnapshot, df_to_doc, doc_to_df
from .state import TableSession
from .trace import NullTracer, Tracer


# --- TableSession <-> SessionSnapshot ---------------------------------------

def snapshot_session(session: TableSession) -> SessionSnapshot:
    """Extract the durable state (committed table, undo stack, history).
    The pending preview is intentionally excluded — see persistence.py."""
    return SessionSnapshot(
        table=df_to_doc(session.df),
        undo=[df_to_doc(df) for df in session.undo_stack],
        history=list(session.history),
    )


def restore_session(snapshot: SessionSnapshot, tracer: Tracer) -> TableSession:
    """Rebuild a live session from its snapshot (pending starts empty)."""
    return TableSession(
        df=doc_to_df(snapshot.table),
        tracer=tracer,
        undo_stack=[doc_to_df(doc) for doc in snapshot.undo],
        history=list(snapshot.history),
    )


# --- the store ---------------------------------------------------------------

@dataclass
class SessionEntry:
    """One resident session plus the lock that serialises access to it."""
    session: TableSession
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionStore:
    """id -> SessionEntry, backed by a SessionRepository."""

    def __init__(self,
                 df_factory: Callable[[], pd.DataFrame],
                 repository: SessionRepository,
                 tracer: Tracer | None = None,
                 cache_limit: int = SESSION_CACHE_LIMIT) -> None:
        # `df_factory` builds the starting table for brand-new sessions
        # (production: read the generated CSV; tests: a small fixture).
        self._df_factory = df_factory
        self._repository = repository
        self._tracer = tracer or NullTracer()
        self._cache_limit = cache_limit
        self._entries: OrderedDict[str, SessionEntry] = OrderedDict()
        self._creation_lock = threading.Lock()

    def entry(self, session_id: str) -> SessionEntry:
        """Return the entry for `session_id`, loading it from the
        repository or creating it fresh. Thread-safe; LRU-maintained."""
        with self._creation_lock:
            existing = self._entries.get(session_id)
            if existing is not None:
                self._entries.move_to_end(session_id)     # mark recently used
                return existing

            tracer = self._tracer.bind(session_id=session_id)
            snapshot = self._repository.load(session_id)
            if snapshot is not None:
                session = restore_session(snapshot, tracer)
                tracer.log("session_restored",
                           undo_steps=len(session.undo_stack))
            else:
                session = TableSession(df=self._df_factory(), tracer=tracer)
                tracer.log("session_created")
                # Persist immediately so the id is durable from birth.
                self._repository.save(session_id, snapshot_session(session))

            entry = SessionEntry(session=session)
            self._entries[session_id] = entry
            self._evict_if_needed()
            return entry

    def persist(self, session_id: str) -> None:
        """Write the session's durable state through to the repository.
        Call after every committed mutation, while holding the entry lock."""
        entry = self._entries.get(session_id)
        if entry is not None:
            self._repository.save(session_id, snapshot_session(entry.session))

    def _evict_if_needed(self) -> None:
        # Oldest-first eviction; safe because persistence is write-through.
        while len(self._entries) > self._cache_limit:
            self._entries.popitem(last=False)

    def close(self) -> None:
        self._repository.close()
