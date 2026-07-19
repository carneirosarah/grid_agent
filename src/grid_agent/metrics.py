"""Structured-output quality metrics.

**Structured Output Validity Rate** — the percentage of raw LLM responses
that Pydantic accepted as a `WireReply`. This measures the *first* of the
three walls (schema-constrained decoding): how often the model's output is
at least syntactically usable, regardless of whether it later survives
semantic validation against the table.

Counting rules (decided here, applied in llm.py):

- **Accepted** — the response became a `WireReply`, whether via the SDK's
  `.parsed` field or the raw-text fallback parse.
- **Rejected** — a response arrived but could not be parsed into a
  `WireReply` (the `PlannerError("unparseable")` path).
- **Not counted** — transport failures (network, auth, quota). No response
  ever existed, so there is nothing for Pydantic to judge; including them
  would let an outage masquerade as a model-quality regression.

The counter is per-process and resets on restart — it answers "how is the
model behaving right now". The durable, per-call record lives in the
trace: `llm_reply` events are accepted responses, `planner_error` events
with an "unparseable" message are rejected ones.
"""

from __future__ import annotations

import threading


class ValidityCounter:
    """Thread-safe accept/reject tally.

    A lock is required because the planner (which does the recording) is
    shared across all sessions, and FastAPI serves requests from a thread
    pool — two users' chat turns can finish simultaneously.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted = 0
        self._rejected = 0

    def record(self, accepted: bool) -> None:
        """Tally one LLM response as accepted or rejected by Pydantic."""
        with self._lock:
            if accepted:
                self._accepted += 1
            else:
                self._rejected += 1

    def snapshot(self) -> dict:
        """Current counts plus the derived rate, as a JSON-ready dict.
        `validity_pct` is None (JSON null) until the first response —
        0/0 is "no data yet", not 0% validity."""
        with self._lock:
            accepted, rejected = self._accepted, self._rejected
        total = accepted + rejected
        pct = round(100 * accepted / total, 2) if total else None
        return {"llm_responses": total, "accepted": accepted,
                "rejected": rejected, "validity_pct": pct}
