"""Counter unit tests (see metrics.py for the counting rules)."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from grid_agent.metrics import PassRateCounter, TurnOutcomeCounter


def test_empty_counter_has_null_rate_not_zero():
    """0/0 must read as "no data yet", never as 0% validity."""
    assert PassRateCounter().snapshot() == {
        "total": 0, "passed": 0, "failed": 0,
        "pass_pct": None}


def test_rate_arithmetic_and_rounding():
    counter = PassRateCounter()
    counter.record(True)
    counter.record(True)
    counter.record(False)
    assert counter.snapshot() == {
        "total": 3, "passed": 2, "failed": 1,
        "pass_pct": 66.67}          # round(100 * 2/3, 2)


def test_all_rejected_is_zero_percent():
    counter = PassRateCounter()
    counter.record(False)
    assert counter.snapshot()["pass_pct"] == 0.0


def test_concurrent_records_are_not_lost():
    """The planner is shared across FastAPI's worker threads, so parallel
    records must all land."""
    counter = PassRateCounter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: counter.record(i % 2 == 0), range(200)))
    snap = counter.snapshot()
    assert snap["total"] == 200
    assert snap["passed"] == snap["failed"] == 100


# --- TurnOutcomeCounter (E2E Success Rate) ----------------------------------

def test_empty_outcome_counter_has_null_success_rate():
    assert TurnOutcomeCounter().snapshot() == {
        "turns": 0, "preview": 0, "clarify": 0, "error": 0,
        "success_pct": None}


def test_success_rate_excludes_clarifications():
    """3 previews, 1 error, 2 clarifies -> 75%: clarifying is neither a
    delivery nor a failure, so it must not move the rate."""
    counter = TurnOutcomeCounter()
    for outcome in ["preview", "preview", "preview", "error",
                    "clarify", "clarify"]:
        counter.record(outcome)
    assert counter.snapshot() == {
        "turns": 6, "preview": 3, "clarify": 2, "error": 1,
        "success_pct": 75.0}


def test_only_clarifications_is_still_null_rate():
    """A session of pure clarifications has no decisive turns yet."""
    counter = TurnOutcomeCounter()
    counter.record("clarify")
    assert counter.snapshot()["success_pct"] is None


def test_unknown_outcome_fails_loudly():
    with pytest.raises(ValueError, match="Unknown turn outcome"):
        TurnOutcomeCounter().record("shrug")
