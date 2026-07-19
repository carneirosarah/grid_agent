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

**E2E Success Rate** — % of whole agent turns that delivered a staged
preview. Judged at the end of `run_turn` (graph.py), the only place a
turn's final outcome exists. Unlike the two per-wall metrics this is a
*user-facing* system metric, so infrastructure failures count: a turn
that died because Gemini was unreachable still failed for the user.

- preview — success: a validated plan was staged for review.
- error   — failure, whatever the cause (repair budget exhausted,
  unparseable reply, planner unreachable).
- clarify — tracked but excluded from the rate: a correct clarifying
  question is neither a delivery nor a failure. Counting it against
  the agent would punish correct behaviour on ambiguous input;
  counting it as success would let a clarify-happy model score 100%
  while never delivering anything. `success_pct` is therefore
  `preview / (preview + error)`.

Counters are per-process and reset on restart — they answer "how is the
model behaving right now". The durable per-call records live in the
trace: `llm_reply` / unparseable `planner_error` for the first metric,
`plan_validated` / `validation_failed` for the second, `turn_finished`
(with its `outcome` field) for the third.
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


class TurnOutcomeCounter:
    """Thread-safe tally of whole-turn outcomes for the E2E Success Rate.

    A separate class (not PassRateCounter) because a turn has *three*
    outcomes, and the clarify count must stay visible: hiding it would
    make the derived rate impossible to sanity-check.
    """

    _OUTCOMES = ("preview", "clarify", "error")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(self._OUTCOMES, 0)

    def record(self, outcome: str) -> None:
        """Tally one finished turn. Unknown outcomes are a programming
        bug (the graph's outcome type is a Literal), so fail loudly."""
        if outcome not in self._OUTCOMES:
            raise ValueError(f"Unknown turn outcome: {outcome!r}")
        with self._lock:
            self._counts[outcome] += 1

    def snapshot(self) -> dict:
        """Outcome counts plus the derived rate. `success_pct` excludes
        clarifications from the denominator (see module docstring) and is
        None until the first decisive (preview/error) turn."""
        with self._lock:
            counts = dict(self._counts)
        decisive = counts["preview"] + counts["error"]
        pct = round(100 * counts["preview"] / decisive, 2) if decisive else None
        return {"turns": sum(counts.values()), **counts, "success_pct": pct}
