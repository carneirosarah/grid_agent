"""Trace persistence — every meaningful event is appended to a .jsonl file.

One JSON object per line, so the file is greppable, streamable and safe to
append to (no global structure to corrupt). Events cover the full life of
an instruction: user message, LLM request/response, validation verdicts,
previews, accept/reject/undo, manual edits, and errors.

Read a trace back with:  `python -m json.tool --json-lines traces/trace.jsonl`
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .config import TRACE_PATH


class Tracer:
    """Append-only JSONL event log.

    A lock guards writes: FastAPI may serve concurrent requests and JSONL
    integrity depends on each line being written atomically.
    """

    def __init__(self, path: Path = TRACE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, **payload: object) -> None:
        """Append one event. Non-JSON-native values (numpy scalars, models)
        are stringified via `default=str` — a trace must never crash the app."""
        record = {"ts": time.time(), "event": event, **payload}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class NullTracer(Tracer):
    """No-op tracer for unit tests that don't care about tracing."""

    def __init__(self) -> None:  # deliberately skip parent __init__ (no file)
        self._lock = threading.Lock()

    def log(self, event: str, **payload: object) -> None:
        pass
