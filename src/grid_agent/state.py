"""Step 6 — State management: committed table, pending preview, undo.

`TableSession` owns the three pieces of mutable state and the *only* legal
transitions between them:

    committed df  --propose(plan)-->  pending preview   (nothing committed)
    pending       --accept()------->  committed (snapshot pushed for undo)
    pending       --reject()------->  discarded
    committed     --undo()--------->  previous snapshot restored
    committed     --edit_cell()---->  manual edit (also undoable)

Design decisions worth noting:

- **Previews are never committed.** `propose` runs the engine on the
  committed df and stores the result + diff separately. The user sees the
  diff; only `accept` swaps it in.
- **Undo = snapshot stack.** At 350 rows, a full DataFrame copy is a few
  KB — snapshots are simpler and safer than inverse operations, and they
  restore row order exactly (an inverse of `sort` is not derivable from
  the operation alone). The stack is capped (config.UNDO_STACK_LIMIT).
- **A manual edit or a new proposal invalidates the current pending
  preview**, because the preview was computed against a table that is
  about to change / no longer current. Explicit is better than a stale
  preview silently overwriting fresh edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import DATASET_PATH, UNDO_STACK_LIMIT
from .engine import CellChange, OpResult, apply_plan, diff_tables
from .schemas import Plan
from .trace import NullTracer, Tracer
from .validator import PROTECTED_COLUMNS, coerce_value


class StateError(RuntimeError):
    """An illegal transition was requested (e.g. accept with no pending)."""


@dataclass
class PendingChange:
    """A fully-computed preview waiting for the user's verdict."""
    plan: Plan
    preview_df: pd.DataFrame
    op_results: list[OpResult]
    changes: list[CellChange]
    order_changed: bool

    @property
    def summary(self) -> str:
        return self.plan.summary or "; ".join(r.description for r in self.op_results)


@dataclass
class TableSession:
    """All mutable state for one user's table + conversation."""
    df: pd.DataFrame
    tracer: Tracer = field(default_factory=NullTracer)
    pending: PendingChange | None = None
    undo_stack: list[pd.DataFrame] = field(default_factory=list)
    # Chat history handed to the planner so clarification answers ("the
    # cheap ones" -> follow-up to a question it asked) keep their context.
    history: list[dict[str, str]] = field(default_factory=list)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_csv(cls, path=DATASET_PATH, tracer: Tracer | None = None) -> TableSession:
        df = pd.read_csv(path)
        return cls(df=df, tracer=tracer or NullTracer())

    # -- preview lifecycle --------------------------------------------------

    def propose(self, plan: Plan) -> PendingChange:
        """Run a validated plan against the committed table and stage the
        result as the pending preview. Commits nothing."""
        result = apply_plan(self.df, plan)
        changes, order_changed = diff_tables(self.df, result.df)
        self.pending = PendingChange(
            plan=plan, preview_df=result.df, op_results=result.op_results,
            changes=changes, order_changed=order_changed,
        )
        self.tracer.log("preview_created",
                        summary=self.pending.summary,
                        cells_changed=len(changes),
                        order_changed=order_changed,
                        plan=plan.model_dump())
        return self.pending

    def accept(self) -> None:
        """Commit the pending preview; previous table goes on the undo stack."""
        if self.pending is None:
            raise StateError("There is no pending change to accept.")
        self._push_undo()
        self.df = self.pending.preview_df
        self.tracer.log("change_accepted", summary=self.pending.summary)
        self.pending = None

    def reject(self) -> None:
        """Discard the pending preview; committed table is untouched."""
        if self.pending is None:
            raise StateError("There is no pending change to reject.")
        self.tracer.log("change_rejected", summary=self.pending.summary)
        self.pending = None

    # -- undo ---------------------------------------------------------------

    def undo(self) -> None:
        """Restore the most recent snapshot. Any pending preview is dropped
        (it was computed against the table being replaced)."""
        if not self.undo_stack:
            raise StateError("Nothing to undo.")
        self.pending = None
        self.df = self.undo_stack.pop()
        self.tracer.log("undo", remaining_undo_steps=len(self.undo_stack))

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def _push_undo(self) -> None:
        self.undo_stack.append(self.df.copy(deep=True))
        if len(self.undo_stack) > UNDO_STACK_LIMIT:
            self.undo_stack.pop(0)      # forget the oldest snapshot

    # -- manual cell edits (the table is user-editable too) -----------------

    def edit_cell(self, sku: str, column: str, value: object) -> object:
        """Apply a hand edit from the grid. Type-checked with the same
        coercion rules the agent's plans go through. Returns the typed value.
        """
        if column not in self.df.columns:
            raise StateError(f"Unknown column '{column}'.")
        if column in PROTECTED_COLUMNS:
            raise StateError(f"Column '{column}' is the row identifier and "
                             "cannot be edited.")
        matches = self.df.index[self.df["sku"] == sku]
        if len(matches) == 0:
            raise StateError(f"Unknown row '{sku}'.")

        errors: list[str] = []
        typed = coerce_value(value, self.df[column], errors, f"cell {sku}.{column}")  # type: ignore[arg-type]
        if errors:
            raise StateError(errors[0])

        # A stale preview must not survive a manual edit underneath it.
        self.pending = None
        self._push_undo()
        self.df.loc[matches[0], column] = typed
        self.tracer.log("manual_edit", sku=sku, column=column, value=typed)
        return typed
