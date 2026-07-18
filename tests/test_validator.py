"""Step 5 — Semantic validator tests.

The validator turns wire-level string values into typed values bound to
real columns, and produces feedback-quality errors for everything else.
"""

from grid_agent.schemas import Condition, Plan, Sort, SortKey, UpdateWhere
from grid_agent.validator import validate_plan


def one_op_plan(op) -> Plan:
    return Plan(operations=[op])


# --- Happy path: coercion & resolution -------------------------------------

def test_string_values_are_coerced_to_column_dtypes(small_df):
    plan = one_op_plan(UpdateWhere(
        where=[Condition(column="price", op="gt", value="30")],
        column="stock", action="set", value="42"))
    result = validate_plan(small_df, plan)
    assert result.ok
    op = result.plan.operations[0]
    assert op.value == 42 and isinstance(op.value, int)
    assert op.where[0].value == 30.0 and isinstance(op.where[0].value, float)


def test_boolean_coercion_accepts_word_forms(small_df):
    plan = one_op_plan(UpdateWhere(column="flagged", action="set", value="true"))
    result = validate_plan(small_df, plan)
    assert result.ok
    assert result.plan.operations[0].value is True


def test_column_names_resolved_case_insensitively(small_df):
    plan = one_op_plan(Sort(keys=[SortKey(column="PRICE", ascending=False)]))
    result = validate_plan(small_df, plan)
    assert result.ok
    assert result.plan.operations[0].keys[0].column == "price"


# --- Errors the repair loop feeds back to the model -------------------------

def test_unknown_column_gets_did_you_mean(small_df):
    plan = one_op_plan(UpdateWhere(column="prise", action="set", value="1"))
    result = validate_plan(small_df, plan)
    assert not result.ok
    assert "does not exist" in result.errors[0]
    assert "price" in result.errors[0]          # difflib suggestion


def test_misspelled_category_value_lists_real_values(small_df):
    """The assignment's own example typo: 'eletronics'."""
    plan = one_op_plan(UpdateWhere(
        where=[Condition(column="category", op="eq", value="Eletronics")],
        column="price", action="multiply", value="1.1"))
    result = validate_plan(small_df, plan)
    assert not result.ok
    assert "matches no value" in result.errors[0]
    assert "Electronics" in result.errors[0]    # real value offered back


def test_sku_is_protected(small_df):
    plan = one_op_plan(UpdateWhere(column="sku", action="set", value="X"))
    result = validate_plan(small_df, plan)
    assert not result.ok
    assert "cannot be modified" in result.errors[0]


def test_multiply_rejected_on_text_column(small_df):
    plan = one_op_plan(UpdateWhere(column="name", action="multiply", value="2"))
    result = validate_plan(small_df, plan)
    assert not result.ok
    assert "requires a numeric column" in result.errors[0]


def test_ordering_operator_rejected_on_text_column(small_df):
    plan = one_op_plan(UpdateWhere(
        where=[Condition(column="category", op="gt", value="A")],
        column="price", action="set", value="1"))
    result = validate_plan(small_df, plan)
    assert not result.ok


def test_integer_column_rejects_fractional_value(small_df):
    plan = one_op_plan(UpdateWhere(column="stock", action="set", value="10.5"))
    result = validate_plan(small_df, plan)
    assert not result.ok
    assert "whole numbers" in result.errors[0]


def test_all_errors_collected_in_one_pass(small_df):
    plan = Plan(operations=[
        UpdateWhere(column="prise", action="set", value="1"),
        Sort(keys=[SortKey(column="pricee")]),
    ])
    result = validate_plan(small_df, plan)
    assert len(result.errors) == 2


def test_valid_multi_step_plan_passes(small_df):
    plan = Plan(operations=[
        UpdateWhere(where=[Condition(column="category", op="eq",
                                     value="electronics")],
                    column="price", action="multiply", value="1.10"),
        Sort(keys=[SortKey(column="price", ascending=False)]),
    ])
    result = validate_plan(small_df, plan)
    assert result.ok
    assert result.plan.operations[0].value == 1.1
