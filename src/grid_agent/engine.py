"""Step 4 — Deterministic operations engine.

The engine is the *only* code that mutates table data, and it contains no
LLM logic at all. Given a DataFrame and a validated `Plan`, it produces a
**new** DataFrame (inputs are never mutated) plus per-operation statistics.

Guarantees:
- Pure & deterministic: same (df, plan) in, same df out. No randomness,
  no I/O, no clock.
- Atomic: operations run sequentially on a working copy; the caller only
  ever sees the final result, so a plan is all-or-nothing.
- The engine trusts the semantic validator: plans reaching here have known
  columns and correctly-typed values. Anything else raises `EngineError`,
  which indicates a programming bug, not bad user/model input.

String comparisons (`eq`, `neq`, `contains`, `in`) are case-insensitive so
"electronics" matches the stored "Electronics" — users type lowercase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schemas import Condition, Plan, Sort, UpdateWhere


class EngineError(RuntimeError):
    """A plan that should have been rejected upstream reached the engine."""


@dataclass
class OpResult:
    """What one operation did — surfaced in the preview and the trace."""
    kind: str
    description: str
    rows_matched: int          # rows the condition selected (sort: all rows)
    cells_changed: int         # cells whose value actually changed


@dataclass
class PlanResult:
    """Outcome of applying a whole plan."""
    df: pd.DataFrame
    op_results: list[OpResult] = field(default_factory=list)

    @property
    def total_cells_changed(self) -> int:
        return sum(r.cells_changed for r in self.op_results)


# ---------------------------------------------------------------------------
# Condition matching
# ---------------------------------------------------------------------------

def build_mask(df: pd.DataFrame, conditions: list[Condition]) -> pd.Series:
    """AND-combine all conditions into a boolean row mask.

    An empty condition list matches every row (that is how "increase ALL
    prices" is expressed: update_where with no `where`).
    """
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        col = df[cond.column]
        if cond.op == "eq":
            mask &= _eq(col, cond.value)
        elif cond.op == "neq":
            mask &= ~_eq(col, cond.value)
        elif cond.op == "gt":
            mask &= col > cond.value
        elif cond.op == "gte":
            mask &= col >= cond.value
        elif cond.op == "lt":
            mask &= col < cond.value
        elif cond.op == "lte":
            mask &= col <= cond.value
        elif cond.op == "contains":
            # Substring match on the string form, case-insensitive.
            mask &= col.astype(str).str.contains(str(cond.value), case=False,
                                                 regex=False, na=False)
        elif cond.op == "in":
            values = cond.value if isinstance(cond.value, list) else [cond.value]
            if _is_string(col):  # string column: compare lowercased
                lowered = {str(v).lower() for v in values}
                mask &= col.astype(str).str.lower().isin(lowered)
            else:
                mask &= col.isin(values)
        else:  # pragma: no cover — schema forbids other operators
            raise EngineError(f"Unsupported operator: {cond.op}")
    return mask


def _is_string(col: pd.Series) -> bool:
    """True for string columns across pandas versions (pandas 3 uses the
    Arrow-backed `str` dtype, older versions use `object`)."""
    return pd.api.types.is_string_dtype(col) or col.dtype == object


def _eq(col: pd.Series, value) -> pd.Series:
    """Equality that is case-insensitive for string columns."""
    if _is_string(col) and isinstance(value, str):
        return col.astype(str).str.lower() == value.lower()
    return col == value


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def apply_update_where(df: pd.DataFrame, op: UpdateWhere) -> tuple[pd.DataFrame, OpResult]:
    """Apply one update_where. Returns (new_df, stats)."""
    mask = build_mask(df, op.where)
    out = df.copy()
    before = out.loc[mask, op.column].copy()

    if op.action == "set":
        out.loc[mask, op.column] = op.value
    elif op.action == "multiply":
        out.loc[mask, op.column] = (out.loc[mask, op.column] * op.value).round(2)
    elif op.action == "increment":
        out.loc[mask, op.column] = out.loc[mask, op.column] + op.value
    else:  # pragma: no cover — schema forbids other actions
        raise EngineError(f"Unsupported action: {op.action}")

    # Count cells that truly changed (setting price=25 where it is already
    # 25 is not a change worth previewing/undoing).
    changed = int((out.loc[mask, op.column] != before).sum())
    desc = (f"{op.action} {op.column}"
            + (f" (matched {int(mask.sum())} rows)" if op.where else " (all rows)"))
    return out, OpResult("update_where", desc, int(mask.sum()), changed)


def apply_sort(df: pd.DataFrame, op: Sort) -> tuple[pd.DataFrame, OpResult]:
    """Apply one sort. Row identity (`sku`) travels with the rows; only the
    display order changes, so cells_changed is 0."""
    out = df.sort_values(
        by=[k.column for k in op.keys],
        ascending=[k.ascending for k in op.keys],
        kind="mergesort",              # stable: equal keys keep prior order
    ).reset_index(drop=True)
    desc = "sort by " + ", ".join(
        f"{k.column} {'asc' if k.ascending else 'desc'}" for k in op.keys)
    return out, OpResult("sort", desc, len(df), 0)


def apply_plan(df: pd.DataFrame, plan: Plan) -> PlanResult:
    """Run every operation in order on a working copy (atomic)."""
    current = df
    results: list[OpResult] = []
    for op in plan.operations:
        if isinstance(op, UpdateWhere):
            current, res = apply_update_where(current, op)
        elif isinstance(op, Sort):
            current, res = apply_sort(current, op)
        else:  # pragma: no cover
            raise EngineError(f"Unknown operation type: {type(op)}")
        results.append(res)
    return PlanResult(df=current, op_results=results)


# ---------------------------------------------------------------------------
# Diffing (used by the preview)
# ---------------------------------------------------------------------------

@dataclass
class CellChange:
    """One cell-level edit in a preview diff: row (by stable `sku`, not
    position — so sorts don't produce false diffs), column, old and new
    value. The frontend highlights cells from this list, and change counts
    in chat replies/traces are derived from it rather than from anything
    the model claims."""
    sku: str
    column: str
    old: object
    new: object


def _native(value: object) -> object:
    """numpy scalar -> plain Python value (JSON-serialisable)."""
    return value.item() if hasattr(value, "item") else value


def diff_tables(before: pd.DataFrame, after: pd.DataFrame) -> tuple[list[CellChange], bool]:
    """Compare two table versions row-identity-wise (keyed on `sku`).

    Returns (cell_changes, order_changed). Sorts alone produce no cell
    changes but flip `order_changed`, so the UI can still explain the diff.
    """
    changes: list[CellChange] = []
    b = before.set_index("sku")
    a = after.set_index("sku")
    for col in b.columns:
        # Align on sku so row order does not create false positives.
        old_vals, new_vals = b[col], a[col].reindex(b.index)
        unequal = old_vals != new_vals
        for sku in b.index[unequal]:
            changes.append(CellChange(sku=str(sku), column=col,
                                      old=_native(old_vals[sku]),
                                      new=_native(new_vals[sku])))
    order_changed = list(before["sku"]) != list(after["sku"])
    return changes, order_changed
