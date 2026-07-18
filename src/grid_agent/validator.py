"""Step 5 — Semantic validator.

Structural validation (schemas.py) proved the plan is *well-formed*. This
module proves it is *meaningful for the actual table*. It is the second of
the three walls between a confidently-wrong LLM and the data:

    wire schema  ->  wire_to_plan (structure)  ->  validate_plan (semantics)

Checks performed against the live DataFrame:
- every referenced column exists (case-insensitively resolved; unknown
  names get did-you-mean suggestions from difflib),
- the protected `sku` identity column is never written,
- `multiply` / `increment` and ordering operators (gt/gte/lt/lte) only
  touch numeric columns,
- `contains` only applies to string columns,
- every wire value (string) is coerced to the column's real dtype — bools
  accept "true"/"yes"/"1", integer columns reject "10.5", etc.,
- equality/membership tests on string columns must reference a value that
  actually occurs in the column ("Eletronics" -> error listing the real
  category names), which turns silent 0-row matches into repairable errors.

On success it returns a **resolved plan**: a deep copy whose values are
correctly typed and whose column names are canonical. Only resolved plans
reach the engine. On failure it returns the full list of errors — the
LangGraph repair loop sends them back to the model in one message.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import pandas as pd

from .schemas import Condition, Plan, Scalar, Sort, UpdateWhere

# Row identity must survive every operation — the diff, undo and frontend
# all key on it — so no plan may write to it.
PROTECTED_COLUMNS = {"sku"}

# How many example values to include in "value not found" errors.
MAX_VALUE_SUGGESTIONS = 8


@dataclass
class ValidationResult:
    plan: Plan | None                       # resolved plan, or None on errors
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, name: str, errors: list[str],
                    context: str) -> str | None:
    """Map a model-provided column name onto a real one (case-insensitive).
    Returns the canonical name, or None after recording a helpful error."""
    lookup = {c.lower(): c for c in df.columns}
    canonical = lookup.get(name.strip().lower())
    if canonical:
        return canonical
    hints = difflib.get_close_matches(name.lower(), list(lookup), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(lookup[h] for h in hints)}?" if hints else ""
    errors.append(f"{context}: column '{name}' does not exist."
                  f" Available: {', '.join(df.columns)}.{hint}")
    return None


def _is_numeric(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def _is_string(col: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(col) or col.dtype == object


_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def _coerce(value: Scalar, col: pd.Series, errors: list[str],
            context: str) -> Scalar | None:
    """Coerce one wire value (usually a string) to the column's dtype.
    Returns the typed value, or None after recording an error."""
    text = str(value).strip()

    if pd.api.types.is_bool_dtype(col):
        if isinstance(value, bool):
            return value
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        errors.append(f"{context}: expected a boolean, got '{value}'.")
        return None

    if pd.api.types.is_integer_dtype(col):
        try:
            as_float = float(text)
        except ValueError:
            errors.append(f"{context}: expected an integer, got '{value}'.")
            return None
        if as_float != int(as_float):
            errors.append(f"{context}: column holds whole numbers, got '{value}'.")
            return None
        return int(as_float)

    if pd.api.types.is_float_dtype(col):
        try:
            return float(text)
        except ValueError:
            errors.append(f"{context}: expected a number, got '{value}'.")
            return None

    return text  # string column: keep as-is


def _check_value_exists(value: str, col: pd.Series, errors: list[str],
                        context: str, substring: bool = False) -> None:
    """For string columns: fail fast when an eq/in/contains value matches
    nothing, instead of silently producing a 0-row update. The error lists
    real values so the repair loop can fix typos like 'Eletronics'."""
    lowered = value.lower()
    uniques = col.dropna().astype(str).unique()
    if substring:
        if any(lowered in u.lower() for u in uniques):
            return
    elif any(lowered == u.lower() for u in uniques):
        return
    sample = ", ".join(sorted(map(str, uniques))[:MAX_VALUE_SUGGESTIONS])
    errors.append(f"{context}: '{value}' matches no value in this column."
                  f" Examples of actual values: {sample}.")


# ---------------------------------------------------------------------------
# Per-operation validation
# ---------------------------------------------------------------------------

def _validate_condition(df: pd.DataFrame, cond: Condition, errors: list[str],
                        context: str) -> Condition | None:
    column = _resolve_column(df, cond.column, errors, context)
    if column is None:
        return None
    col = df[column]

    # Ordering comparisons only make sense on numeric columns here.
    if cond.op in ("gt", "gte", "lt", "lte") and not _is_numeric(col):
        errors.append(f"{context}: operator '{cond.op}' requires a numeric "
                      f"column, but '{column}' is not numeric.")
        return None

    if cond.op == "contains":
        if not _is_string(col):
            errors.append(f"{context}: 'contains' only works on text columns, "
                          f"and '{column}' is not text.")
            return None
        _check_value_exists(str(cond.value), col, errors, context, substring=True)
        return Condition(column=column, op="contains", value=str(cond.value))

    if cond.op == "in":
        raw = cond.value if isinstance(cond.value, list) else [cond.value]
        coerced: list[Scalar] = []
        for v in raw:
            typed = _coerce(v, col, errors, context)
            if typed is None:
                return None
            if _is_string(col):
                _check_value_exists(str(typed), col, errors, context)
            coerced.append(typed)
        return Condition(column=column, op="in", value=coerced)

    # eq / neq / ordering: single scalar, coerced to dtype.
    typed = _coerce(cond.value, col, errors, context)  # type: ignore[arg-type]
    if typed is None:
        return None
    if cond.op == "eq" and _is_string(col):
        _check_value_exists(str(typed), col, errors, context)
    return Condition(column=column, op=cond.op, value=typed)


def _validate_update(df: pd.DataFrame, op: UpdateWhere, index: int,
                     errors: list[str]) -> UpdateWhere | None:
    context = f"operation[{index}] (update_where)"
    column = _resolve_column(df, op.column, errors, f"{context}.target_column")
    if column is None:
        return None
    if column in PROTECTED_COLUMNS:
        errors.append(f"{context}: column '{column}' is the row identifier "
                      "and cannot be modified.")
        return None
    col = df[column]

    if op.action in ("multiply", "increment"):
        if not _is_numeric(col):
            errors.append(f"{context}: action '{op.action}' requires a numeric "
                          f"column, but '{column}' is not numeric.")
            return None
        try:
            value: Scalar = float(str(op.value))
        except ValueError:
            errors.append(f"{context}: action '{op.action}' needs a numeric "
                          f"value, got '{op.value}'.")
            return None
    else:  # set
        maybe = _coerce(op.value, col, errors, f"{context}.value")
        if maybe is None:
            return None
        value = maybe

    conditions: list[Condition] = []
    for j, cond in enumerate(op.where):
        validated = _validate_condition(df, cond, errors, f"{context}.where[{j}]")
        if validated is None:
            return None
        conditions.append(validated)

    return UpdateWhere(where=conditions, column=column, action=op.action,
                       value=value)


def _validate_sort(df: pd.DataFrame, op: Sort, index: int,
                   errors: list[str]) -> Sort | None:
    context = f"operation[{index}] (sort)"
    keys = []
    for key in op.keys:
        column = _resolve_column(df, key.column, errors, context)
        if column is None:
            return None
        keys.append(key.model_copy(update={"column": column}))
    return Sort(keys=keys)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def validate_plan(df: pd.DataFrame, plan: Plan) -> ValidationResult:
    """Validate a structural plan against the live table.

    Collects every error (no fail-fast) so one repair round-trip to the
    model can fix everything at once.
    """
    errors: list[str] = []
    resolved: list[UpdateWhere | Sort] = []

    for i, op in enumerate(plan.operations):
        if isinstance(op, UpdateWhere):
            validated: UpdateWhere | Sort | None = _validate_update(df, op, i, errors)
        else:
            validated = _validate_sort(df, op, i, errors)
        if validated is not None:
            resolved.append(validated)

    if errors:
        return ValidationResult(plan=None, errors=errors)
    return ValidationResult(plan=Plan(operations=resolved, summary=plan.summary))
