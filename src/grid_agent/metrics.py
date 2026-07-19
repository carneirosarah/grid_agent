"""LLM-quality metrics, one counter per wall (see README "three walls").

Both metrics share one shape — a pass/fail tally with a derived rate —
so one counter class serves both. What differs is *where* each verdict is
rendered, and each is counted at exactly that place:

**Structured Output Validity Rate** — % of raw LLM responses that
Pydantic accepted as a `WireReply`. Judged in `GeminiPlanner.plan`
(llm.py), the only spot a raw response meets the wire schema.

- passed  — the response became a `WireReply`, whether via the SDK's
  `.parsed` field or the raw-text fallback parse.
- failed  — a response arrived but could not be parsed.
- not counted — transport failures (network, auth, quota). No response
  ever existed, so there was nothing to judge; counting them would let
  an outage masquerade as a model-quality regression.

**Semantic Validation Pass Rate** — % of structural plans that
`validate_plan` accepted against the live table (real columns, coercible
values, numeric-only rules, protected `sku`…). Judged in the graph's
`validate` node (graph.py), the only spot a plan meets the table.

- passed  — `validate_plan` returned a resolved plan.
- failed  — it returned errors (which then feed the repair loop).
- not counted — clarify-intent replies (no plan to validate) and
  replies that failed *structural* conversion in `wire_to_plan` (they
  never reached the semantic validator; the trace's
  `validation_failed` events still record them).

Each repair-loop attempt is judged separately, matching the per-attempt
granularity of `model_call` trace events — a turn that fails once and is
repaired counts one fail and one pass.

Counters are per-process and reset on restart — they answer "how is the
model behaving right now". The durable per-call records live in the
trace: `llm_reply` / unparseable `planner_error` for the first metric,
`plan_validated` / `validation_failed` for the second.
"""

from __future__ import annotations

import threading


class PassRateCounter:
    """Thread-safe pass/fail tally.

    A lock is required because both recording sites are reached from
    FastAPI's thread pool (the planner is shared across all sessions, and
    two users' chat turns can finish simultaneously).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._passed = 0
        self._failed = 0

    def record(self, passed: bool) -> None:
        """Tally one verdict."""
        with self._lock:
            if passed:
                self._passed += 1
            else:
                self._failed += 1

    def snapshot(self) -> dict:
        """Current counts plus the derived rate, as a JSON-ready dict.
        `pass_pct` is None (JSON null) until the first verdict — 0/0 is
        "no data yet", not 0%."""
        with self._lock:
            passed, failed = self._passed, self._failed
        total = passed + failed
        pct = round(100 * passed / total, 2) if total else None
        return {"total": total, "passed": passed, "failed": failed,
                "pass_pct": pct}
