"""Step 2 — Operation schemas.

Two layers of models live here, and the distinction is the core of how we
defend against incorrect-but-confident model output:

1. **Wire models** (`Wire*`)  — what Gemini is asked to produce via
   structured output. Deliberately *flat and permissive*: no unions except a
   string enum discriminator, and every value is a plain string. Gemini's
   schema translation handles this shape very reliably, so a syntactically
   broken reply is rare.

2. **Domain models** (`UpdateWhere`, `Sort`, `Plan`) — the strict,
   discriminated-union representation the deterministic engine consumes.
   `wire_to_plan()` converts layer 1 into layer 2 and *collects* structural
   errors instead of raising, so the LangGraph repair loop can feed them
   back to the model verbatim.

Values stay as strings until the **semantic validator** (validator.py)
coerces them against the actual DataFrame dtypes. The engine therefore only
ever sees plans that are both structurally and semantically valid.

Only two operations exist by design (per the assignment):

- ``update_where``: write one value/transform into one column for all rows
  matching a conjunction of conditions (empty conditions = all rows).
- ``sort``: reorder rows by one or more columns.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# Scalar type a condition/update value may take *after* semantic coercion.
Scalar = Union[str, float, int, bool]

# Comparison operators supported by `update_where` conditions.
ConditionOp = Literal["eq", "neq", "gt", "gte", "lt", "lte", "contains", "in"]

# How the new value is produced from the target column.
#   set       -> column = value
#   multiply  -> column = column * value        (numeric columns only)
#   increment -> column = column + value        (numeric columns only)
UpdateAction = Literal["set", "multiply", "increment"]


# ---------------------------------------------------------------------------
# Domain models — strict, engine-facing
# ---------------------------------------------------------------------------

class Condition(BaseModel):
    """One filter clause. Clauses in a list are AND-combined."""
    column: str
    op: ConditionOp
    # `in` takes a list; every other operator takes a single scalar.
    value: Union[Scalar, list[Scalar]]


class UpdateWhere(BaseModel):
    """Write to `column` on every row matching all `where` clauses."""
    kind: Literal["update_where"] = "update_where"
    where: list[Condition] = Field(default_factory=list)
    column: str
    action: UpdateAction
    value: Scalar


class SortKey(BaseModel):
    column: str
    ascending: bool = True


class Sort(BaseModel):
    """Reorder rows by `keys`, applied with priority left to right."""
    kind: Literal["sort"] = "sort"
    keys: list[SortKey] = Field(min_length=1)


# Discriminated union: pydantic picks the right model from `kind`.
Operation = Annotated[Union[UpdateWhere, Sort], Field(discriminator="kind")]


class Plan(BaseModel):
    """An ordered list of operations — executed atomically by the engine."""
    operations: list[Operation] = Field(min_length=1)
    summary: str = ""          # human-readable description for the preview UI


# ---------------------------------------------------------------------------
# Wire models — permissive, Gemini-facing
# ---------------------------------------------------------------------------

class WireCondition(BaseModel):
    """Condition as emitted by the LLM. Values are strings on purpose:
    the semantic validator coerces them to the target column's dtype and
    produces a precise error message when it can't."""
    column: str
    op: ConditionOp
    # For op == "in" fill `values`; otherwise fill `value`.
    value: str = ""
    values: list[str] = Field(default_factory=list)


class WireSortKey(BaseModel):
    column: str
    ascending: bool = True


class WireOperation(BaseModel):
    """Flat superset of both operations; `kind` says which fields matter.
    Flat because Gemini's structured output is far more dependable without
    nested discriminated unions in the response schema."""
    kind: Literal["update_where", "sort"]
    # -- update_where fields --
    where: list[WireCondition] = Field(default_factory=list)
    target_column: str = ""
    action: UpdateAction = "set"
    value: str = ""
    # -- sort fields --
    sort_keys: list[WireSortKey] = Field(default_factory=list)


class WireReply(BaseModel):
    """Top-level structured reply the model must return on every turn.

    intent == "plan"    -> `operations` + `plan_summary` are filled.
    intent == "clarify" -> `clarifying_question` is filled, operations empty.
    """
    reasoning: str = ""
    intent: Literal["plan", "clarify"]
    operations: list[WireOperation] = Field(default_factory=list)
    clarifying_question: str = ""
    plan_summary: str = ""


# ---------------------------------------------------------------------------
# Wire -> domain conversion (structural validation only)
# ---------------------------------------------------------------------------

def wire_to_plan(reply: WireReply) -> tuple[Plan | None, list[str]]:
    """Convert a wire reply into a strict `Plan`.

    Returns ``(plan, errors)``. Errors are collected (not raised) so the
    repair loop can hand the *complete* list back to the model in one shot.
    Only structural rules are checked here; table-aware rules (does the
    column exist? is the value coercible?) belong to the semantic validator.
    """
    errors: list[str] = []
    operations: list[UpdateWhere | Sort] = []

    for i, op in enumerate(reply.operations):
        label = f"operation[{i}] ({op.kind})"

        if op.kind == "update_where":
            if not op.target_column:
                errors.append(f"{label}: target_column is required.")
                continue
            if op.value == "" and op.action != "set":
                errors.append(f"{label}: action '{op.action}' requires a numeric value.")
                continue
            conditions: list[Condition] = []
            cond_ok = True
            for j, wc in enumerate(op.where):
                if wc.op == "in":
                    if not wc.values:
                        errors.append(f"{label}.where[{j}]: op 'in' requires 'values'.")
                        cond_ok = False
                        continue
                    conditions.append(Condition(column=wc.column, op="in",
                                                value=list(wc.values)))
                else:
                    if wc.value == "":
                        errors.append(f"{label}.where[{j}]: 'value' is required for op '{wc.op}'.")
                        cond_ok = False
                        continue
                    conditions.append(Condition(column=wc.column, op=wc.op, value=wc.value))
            if not cond_ok:
                continue
            operations.append(UpdateWhere(where=conditions, column=op.target_column,
                                          action=op.action, value=op.value))

        elif op.kind == "sort":
            if not op.sort_keys:
                errors.append(f"{label}: sort requires at least one sort key.")
                continue
            operations.append(Sort(keys=[SortKey(column=k.column, ascending=k.ascending)
                                         for k in op.sort_keys]))

    if not operations and not errors:
        errors.append("Reply had intent 'plan' but contained no operations.")

    if errors:
        return None, errors
    return Plan(operations=operations, summary=reply.plan_summary), []
