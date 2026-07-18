"""Step 4 — Engine tests.

The engine must be deterministic, non-mutating, atomic, and its diff must
be keyed on row identity (sku), not row position.
"""

import pandas as pd

from grid_agent.engine import apply_plan, build_mask, diff_tables
from grid_agent.schemas import Condition, Plan, Sort, SortKey, UpdateWhere


def plan_of(*ops) -> Plan:
    return Plan(operations=list(ops))


# --- Condition matching ----------------------------------------------------

def test_mask_empty_conditions_matches_all(small_df):
    assert build_mask(small_df, []).all()


def test_mask_eq_is_case_insensitive_for_strings(small_df):
    mask = build_mask(small_df, [Condition(column="category", op="eq",
                                           value="electronics")])
    assert list(small_df[mask]["sku"]) == ["SKU-0001"]


def test_mask_numeric_comparison(small_df):
    mask = build_mask(small_df, [Condition(column="price", op="lt", value=30.0)])
    assert list(small_df[mask]["sku"]) == ["SKU-0001", "SKU-0003", "SKU-0006"]


def test_mask_conditions_are_anded(small_df):
    mask = build_mask(small_df, [
        Condition(column="supplier", op="eq", value="PacRim Ltd"),
        Condition(column="price", op="gte", value=30.0),
    ])
    assert list(small_df[mask]["sku"]) == ["SKU-0004"]


def test_mask_contains_and_in(small_df):
    contains = build_mask(small_df, [Condition(column="name", op="contains",
                                               value="acme")])
    assert int(contains.sum()) == 2
    isin = build_mask(small_df, [Condition(column="category", op="in",
                                           value=["kitchen", "Sports"])])
    assert int(isin.sum()) == 2


# --- update_where ----------------------------------------------------------

def test_multiply_rounds_to_cents(small_df):
    plan = plan_of(UpdateWhere(
        where=[Condition(column="category", op="eq", value="Stationery")],
        column="price", action="multiply", value=1.1))
    result = apply_plan(small_df, plan)
    assert result.df.loc[2, "price"] == 5.5
    assert result.op_results[0].rows_matched == 1
    assert result.op_results[0].cells_changed == 1


def test_set_boolean_flag(small_df):
    plan = plan_of(UpdateWhere(
        where=[Condition(column="rating", op="lt", value=3.0)],
        column="flagged", action="set", value=True))
    result = apply_plan(small_df, plan)
    assert list(result.df[result.df["flagged"]]["sku"]) == ["SKU-0003", "SKU-0006"]


def test_input_dataframe_is_never_mutated(small_df):
    original = small_df.copy(deep=True)
    apply_plan(small_df, plan_of(UpdateWhere(column="price", action="multiply",
                                             value=2.0)))
    pd.testing.assert_frame_equal(small_df, original)


def test_noop_set_counts_zero_changed_cells(small_df):
    plan = plan_of(UpdateWhere(column="flagged", action="set", value=False))
    result = apply_plan(small_df, plan)
    assert result.op_results[0].rows_matched == 6
    assert result.op_results[0].cells_changed == 0


# --- sort ------------------------------------------------------------------

def test_sort_descending(small_df):
    result = apply_plan(small_df, plan_of(
        Sort(keys=[SortKey(column="price", ascending=False)])))
    assert list(result.df["price"]) == sorted(small_df["price"], reverse=True)
    # sku travels with its row
    assert result.df.loc[0, "sku"] == "SKU-0002"


def test_multi_key_sort(small_df):
    result = apply_plan(small_df, plan_of(Sort(keys=[
        SortKey(column="supplier", ascending=True),
        SortKey(column="price", ascending=False)])))
    pacrim = result.df[result.df["supplier"] == "PacRim Ltd"]
    assert list(pacrim["price"]) == [40.0, 20.0]


# --- multi-operation plans + diff ------------------------------------------

def test_multi_step_plan_runs_in_order(small_df):
    """'Increase electronics prices 10%, then sort by price descending.'"""
    plan = plan_of(
        UpdateWhere(where=[Condition(column="category", op="eq",
                                     value="Electronics")],
                    column="price", action="multiply", value=1.10),
        Sort(keys=[SortKey(column="price", ascending=False)]),
    )
    result = apply_plan(small_df, plan)
    row = result.df[result.df["sku"] == "SKU-0001"].iloc[0]
    assert row["price"] == 27.5          # 25.0 * 1.1, updated before sorting
    assert result.df.loc[0, "sku"] == "SKU-0002"  # 300.0 still highest


def test_diff_is_keyed_on_sku_not_position(small_df):
    plan = plan_of(
        UpdateWhere(where=[Condition(column="sku", op="eq", value="SKU-0001")],
                    column="stock", action="set", value=99),
        Sort(keys=[SortKey(column="price", ascending=False)]),
    )
    result = apply_plan(small_df, plan)
    changes, order_changed = diff_tables(small_df, result.df)
    # Only ONE cell changed even though every row moved.
    assert len(changes) == 1
    assert (changes[0].sku, changes[0].column) == ("SKU-0001", "stock")
    assert (changes[0].old, changes[0].new) == (10, 99)
    assert order_changed


def test_pure_sort_diff_reports_order_change_only(small_df):
    result = apply_plan(small_df, plan_of(
        Sort(keys=[SortKey(column="rating", ascending=True)])))
    changes, order_changed = diff_tables(small_df, result.df)
    assert changes == []
    assert order_changed


def test_diff_values_are_native_python_types(small_df):
    """Regression: numpy scalars (e.g. numpy.bool) are not JSON-serialisable
    and must never leak into CellChange old/new values."""
    import json

    plan = plan_of(UpdateWhere(
        where=[Condition(column="rating", op="lt", value=2.0)],
        column="flagged", action="set", value=True))
    result = apply_plan(small_df, plan)
    changes, _ = diff_tables(small_df, result.df)
    assert changes and type(changes[0].new) is bool
    json.dumps([c.__dict__ for c in changes])   # must not raise


def test_determinism_same_plan_same_output(small_df):
    plan = plan_of(UpdateWhere(column="price", action="multiply", value=1.07),
                   Sort(keys=[SortKey(column="price")]))
    a = apply_plan(small_df, plan).df
    b = apply_plan(small_df, plan).df
    pd.testing.assert_frame_equal(a, b)
