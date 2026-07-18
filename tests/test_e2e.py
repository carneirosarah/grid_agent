"""Step 9 — End-to-end tests.

The full pipeline — chat message -> LangGraph -> validation/repair ->
preview -> accept/undo — exercised offline with a `FakePlanner` that
replays scripted WireReplies and records what it was asked. This proves
the *system's* behaviour independent of any live model, including the two
scenarios the assignment cares most about: ambiguity -> clarifying
question, and confidently-wrong output -> repair or refuse.
"""

import numpy as np
import pandas as pd
import pytest

from grid_agent.config import MAX_REPAIR_ATTEMPTS
from grid_agent.graph import run_turn
from grid_agent.schemas import (
    WireCondition,
    WireOperation,
    WireReply,
    WireSortKey,
)
from grid_agent.state import TableSession
from grid_agent.trace import NullTracer, Tracer


class FakePlanner:
    """Replays scripted replies and records every request it receives."""

    def __init__(self, *replies: WireReply):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def plan(self, table_context, history):
        from grid_agent.llm import ModelCall

        self.calls.append({"context": table_context,
                           "history": [dict(m) for m in history]})
        call = ModelCall(
            model="fake-planner", system_prompt_ref="fake@00000000",
            table_context_sha1="0" * 12, history_turns=len(history),
            last_message_preview=history[-1]["text"][:200] if history else "",
            input_tokens=10, output_tokens=5, total_tokens=15,
            cost_usd=0.0, latency_ms=0.1)
        return self.replies.pop(0), call


# --- scripted replies ------------------------------------------------------

GOOD_PLAN = WireReply(
    intent="plan",
    operations=[
        WireOperation(kind="update_where",
                      where=[WireCondition(column="category", op="eq",
                                           value="Electronics")],
                      target_column="price", action="multiply", value="1.10"),
        WireOperation(kind="sort",
                      sort_keys=[WireSortKey(column="price", ascending=False)]),
    ],
    plan_summary="Increase electronics prices by 10% and sort by price desc.")

# Confidently wrong: column 'prise' does not exist, category misspelled.
BAD_PLAN = WireReply(
    intent="plan",
    operations=[
        WireOperation(kind="update_where",
                      where=[WireCondition(column="category", op="eq",
                                           value="Eletronics")],
                      target_column="prise", action="multiply", value="1.10"),
    ],
    plan_summary="Increase eletronics prises.")

CLARIFY = WireReply(intent="clarify",
                    clarifying_question="Increase prices by what percentage?")


@pytest.fixture()
def session(small_df) -> TableSession:
    return TableSession(df=small_df.copy())


# --- happy path ------------------------------------------------------------

def test_multi_step_instruction_to_preview_accept_undo(session):
    planner = FakePlanner(GOOD_PLAN)
    result = run_turn(session, planner, NullTracer(),
                      "Increase electronics prices by 10%, then sort descending")

    # 1. A preview is staged, nothing committed.
    assert result.outcome == "preview"
    assert result.pending is not None
    assert session.df.loc[0, "price"] == 25.0
    # 2. The diff shows exactly the electronics price change (25 -> 27.5).
    assert [(c.sku, c.column, c.new) for c in result.pending.changes] == \
        [("SKU-0001", "price", 27.5)]
    assert result.pending.order_changed
    # 3. Accept commits; the table is now sorted with updated price.
    session.accept()
    assert session.df.loc[session.df["sku"] == "SKU-0001", "price"].iloc[0] == 27.5
    assert session.df.loc[0, "sku"] == "SKU-0002"      # 300.0 on top
    # 4. Undo restores the original table byte for byte.
    session.undo()
    assert session.df.loc[0, "price"] == 25.0
    assert list(session.df["sku"]) == [f"SKU-{i:04d}" for i in range(1, 7)]


def test_planner_receives_live_table_context(session):
    planner = FakePlanner(GOOD_PLAN)
    run_turn(session, planner, NullTracer(), "raise electronics 10%")
    context = planner.calls[0]["context"]
    assert "6 rows" in context and "Electronics" in context


# --- ambiguity -> clarifying question --------------------------------------

def test_ambiguous_instruction_yields_question_then_plan(session):
    planner = FakePlanner(CLARIFY, GOOD_PLAN)

    first = run_turn(session, planner, NullTracer(), "Increase prices")
    assert first.outcome == "clarify"
    assert "percentage" in first.message
    assert session.pending is None                     # nothing staged

    second = run_turn(session, planner, NullTracer(), "10 percent")
    assert second.outcome == "preview"
    # The follow-up request contains the full conversation, so the model
    # could connect "10 percent" to its own question.
    roles = [m["role"] for m in planner.calls[1]["history"]]
    assert roles == ["user", "model", "user"]
    assert planner.calls[1]["history"][1]["text"] == first.message


# --- confidently wrong output -> repair loop --------------------------------

def test_invalid_plan_is_repaired_via_error_feedback(session):
    planner = FakePlanner(BAD_PLAN, GOOD_PLAN)
    result = run_turn(session, planner, NullTracer(),
                      "increase eletronics prises by 10%")

    assert result.outcome == "preview"                 # recovered
    assert len(planner.calls) == 2
    # The retry request carried the validator's error report...
    feedback = planner.calls[1]["history"][-1]["text"]
    assert "failed validation" in feedback
    assert "prise" in feedback and "price" in feedback  # did-you-mean hint
    # ...but the durable chat history stays clean of repair noise.
    assert all("failed validation" not in m["text"] for m in session.history)


def test_persistently_wrong_planner_gives_up_without_touching_data(session):
    planner = FakePlanner(*[BAD_PLAN] * (MAX_REPAIR_ATTEMPTS + 1))
    before = session.df.copy(deep=True)
    result = run_turn(session, planner, NullTracer(), "do the thing")

    assert result.outcome == "error"
    assert len(planner.calls) == MAX_REPAIR_ATTEMPTS + 1
    assert "couldn't produce a valid plan" in result.message
    assert session.pending is None
    pd.testing.assert_frame_equal(session.df, before)  # data untouched


# --- trace persistence ------------------------------------------------------

def test_full_turn_is_persisted_to_jsonl(tmp_path, small_df):
    import json

    tracer = Tracer(path=tmp_path / "trace.jsonl")
    session = TableSession(df=small_df.copy(), tracer=tracer)
    run_turn(session, FakePlanner(BAD_PLAN, GOOD_PLAN), tracer, "fix prices")
    session.accept()
    session.undo()

    records = [json.loads(line)
               for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    events = [r["event"] for r in records]
    for expected in ["user_message", "model_call", "llm_reply",
                     "validation_failed", "plan_validated", "preview_created",
                     "turn_finished", "change_accepted", "undo"]:
        assert expected in events, f"missing trace event {expected}"

    # Every model call is fully observable: model, prompt reference,
    # token counts, cost and latency (one event per attempt — the repair
    # turn produces a second one).
    model_calls = [r for r in records if r["event"] == "model_call"]
    assert len(model_calls) == 2                    # BAD_PLAN + GOOD_PLAN
    for call in model_calls:
        assert call["model"] == "fake-planner"
        assert call["system_prompt_ref"]
        assert call["input_tokens"] == 10 and call["output_tokens"] == 5
        assert call["cost_usd"] == 0.0
        assert call["latency_ms"] >= 0


# --- real generated dataset -------------------------------------------------

def test_pipeline_on_full_350_row_dataset():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "generate_data",
        Path(__file__).resolve().parents[1] / "scripts" / "generate_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    session = TableSession(df=module.generate())
    result = run_turn(session, FakePlanner(GOOD_PLAN), NullTracer(),
                      "electronics +10%, sort by price desc")
    assert result.outcome == "preview"

    electronics = int((session.df["category"] == "Electronics").sum())
    assert len(result.pending.changes) == electronics  # every one, only them
    session.accept()
    prices = session.df["price"].to_numpy()
    assert (np.diff(prices) <= 1e-9).all()             # sorted descending
