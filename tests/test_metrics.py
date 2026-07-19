"""ValidityCounter unit tests (see metrics.py for the counting rules)."""

from concurrent.futures import ThreadPoolExecutor

from grid_agent.metrics import ValidityCounter


def test_empty_counter_has_null_rate_not_zero():
    """0/0 must read as "no data yet", never as 0% validity."""
    assert ValidityCounter().snapshot() == {
        "llm_responses": 0, "accepted": 0, "rejected": 0,
        "validity_pct": None}


def test_rate_arithmetic_and_rounding():
    counter = ValidityCounter()
    counter.record(True)
    counter.record(True)
    counter.record(False)
    assert counter.snapshot() == {
        "llm_responses": 3, "accepted": 2, "rejected": 1,
        "validity_pct": 66.67}          # round(100 * 2/3, 2)


def test_all_rejected_is_zero_percent():
    counter = ValidityCounter()
    counter.record(False)
    assert counter.snapshot()["validity_pct"] == 0.0


def test_concurrent_records_are_not_lost():
    """The planner is shared across FastAPI's worker threads, so parallel
    records must all land."""
    counter = ValidityCounter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: counter.record(i % 2 == 0), range(200)))
    snap = counter.snapshot()
    assert snap["llm_responses"] == 200
    assert snap["accepted"] == snap["rejected"] == 100
