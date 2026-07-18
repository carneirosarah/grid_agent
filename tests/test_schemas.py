"""Step 3 — Schema tests.

Covers: domain-model validation rules, the wire->domain converter, and the
error-collection behaviour the repair loop depends on.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from grid_agent.schemas import (
    Operation,
    Plan,
    Sort,
    UpdateWhere,
    WireCondition,
    WireOperation,
    WireReply,
    WireSortKey,
    wire_to_plan,
)


# --- Domain models ---------------------------------------------------------

def test_update_where_roundtrip():
    op = UpdateWhere(
        where=[{"column": "category", "op": "eq", "value": "Electronics"}],
        column="price", action="multiply", value=1.1,
    )
    assert op.kind == "update_where"
    # Serialise -> parse must be lossless (plans are persisted in traces).
    assert UpdateWhere.model_validate(op.model_dump()) == op


def test_sort_requires_at_least_one_key():
    with pytest.raises(ValidationError):
        Sort(keys=[])


def test_plan_requires_operations():
    with pytest.raises(ValidationError):
        Plan(operations=[])


def test_operation_union_discriminates_on_kind():
    adapter = TypeAdapter(Operation)
    op = adapter.validate_python(
        {"kind": "sort", "keys": [{"column": "price", "ascending": False}]}
    )
    assert isinstance(op, Sort)


def test_unknown_kind_rejected():
    adapter = TypeAdapter(Operation)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "delete_rows"})


def test_unknown_condition_operator_rejected():
    with pytest.raises(ValidationError):
        UpdateWhere(
            where=[{"column": "price", "op": "between", "value": "1"}],
            column="price", action="set", value="1",
        )


# --- Wire -> domain conversion --------------------------------------------

def _wire_update(**overrides) -> WireOperation:
    base = dict(
        kind="update_where",
        where=[WireCondition(column="category", op="eq", value="Electronics")],
        target_column="price", action="multiply", value="1.10",
    )
    base.update(overrides)
    return WireOperation(**base)


def test_wire_to_plan_happy_path():
    reply = WireReply(
        intent="plan",
        operations=[
            _wire_update(),
            WireOperation(kind="sort",
                          sort_keys=[WireSortKey(column="price", ascending=False)]),
        ],
        plan_summary="Raise electronics prices 10%, sort by price desc.",
    )
    plan, errors = wire_to_plan(reply)
    assert errors == []
    assert plan is not None
    assert [op.kind for op in plan.operations] == ["update_where", "sort"]
    # Values remain strings at this stage — coercion is the validator's job.
    assert plan.operations[0].value == "1.10"


def test_wire_to_plan_collects_all_errors_instead_of_failing_fast():
    reply = WireReply(
        intent="plan",
        operations=[
            _wire_update(target_column=""),           # error 1
            WireOperation(kind="sort", sort_keys=[]), # error 2
        ],
    )
    plan, errors = wire_to_plan(reply)
    assert plan is None
    assert len(errors) == 2  # both reported in one pass for the repair loop


def test_wire_to_plan_in_operator_requires_values_list():
    bad = _wire_update(where=[WireCondition(column="category", op="in")])
    plan, errors = wire_to_plan(WireReply(intent="plan", operations=[bad]))
    assert plan is None
    assert "requires 'values'" in errors[0]


def test_wire_to_plan_empty_plan_is_an_error():
    plan, errors = wire_to_plan(WireReply(intent="plan", operations=[]))
    assert plan is None
    assert errors


def test_wire_reply_clarify_shape():
    reply = WireReply(intent="clarify",
                      clarifying_question="Increase prices by how much?")
    assert reply.operations == []
