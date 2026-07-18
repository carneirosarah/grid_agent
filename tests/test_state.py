"""Step 6 — State management tests: preview, accept/reject, undo, edits."""

import pandas as pd
import pytest

from grid_agent.schemas import Condition, Plan, Sort, SortKey, UpdateWhere
from grid_agent.state import StateError, TableSession


@pytest.fixture()
def session(small_df) -> TableSession:
    return TableSession(df=small_df.copy())


PLAN = Plan(operations=[
    UpdateWhere(where=[Condition(column="category", op="eq", value="Electronics")],
                column="price", action="multiply", value=2.0)],
    summary="Double electronics prices")


def test_propose_stages_preview_without_committing(session):
    pending = session.propose(PLAN)
    assert len(pending.changes) == 1
    assert pending.changes[0].new == 50.0
    # Committed table untouched until accept.
    assert session.df.loc[0, "price"] == 25.0


def test_accept_commits_and_enables_undo(session):
    session.propose(PLAN)
    session.accept()
    assert session.df.loc[0, "price"] == 50.0
    assert session.pending is None
    assert session.can_undo


def test_reject_discards_preview(session):
    session.propose(PLAN)
    session.reject()
    assert session.pending is None
    assert session.df.loc[0, "price"] == 25.0
    assert not session.can_undo          # nothing was committed


def test_undo_restores_previous_table(session):
    original = session.df.copy(deep=True)
    session.propose(PLAN)
    session.accept()
    session.undo()
    pd.testing.assert_frame_equal(session.df, original)
    assert not session.can_undo


def test_undo_restores_row_order_after_sort(session):
    original_order = list(session.df["sku"])
    session.propose(Plan(operations=[Sort(keys=[SortKey(column="price")])]))
    session.accept()
    assert list(session.df["sku"]) != original_order
    session.undo()
    assert list(session.df["sku"]) == original_order


def test_multiple_undo_steps_pop_in_reverse_order(session):
    session.propose(PLAN)
    session.accept()                     # price -> 50
    session.propose(PLAN)
    session.accept()                     # price -> 100
    assert session.df.loc[0, "price"] == 100.0
    session.undo()
    assert session.df.loc[0, "price"] == 50.0
    session.undo()
    assert session.df.loc[0, "price"] == 25.0


def test_accept_without_pending_raises(session):
    with pytest.raises(StateError):
        session.accept()


def test_undo_without_history_raises(session):
    with pytest.raises(StateError):
        session.undo()


# --- manual edits ----------------------------------------------------------

def test_manual_edit_is_typed_and_undoable(session):
    session.edit_cell("SKU-0003", "stock", "150")   # string in, int stored
    assert session.df.loc[2, "stock"] == 150
    session.undo()
    assert session.df.loc[2, "stock"] == 200


def test_manual_edit_rejects_bad_type(session):
    with pytest.raises(StateError):
        session.edit_cell("SKU-0003", "stock", "lots")


def test_manual_edit_cannot_touch_sku(session):
    with pytest.raises(StateError):
        session.edit_cell("SKU-0003", "sku", "SKU-9999")


def test_manual_edit_invalidates_stale_preview(session):
    session.propose(PLAN)
    session.edit_cell("SKU-0001", "price", "10")
    assert session.pending is None       # preview was computed pre-edit
