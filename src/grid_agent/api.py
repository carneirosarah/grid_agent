"""Step 10 — FastAPI backend.

Thin HTTP layer over `TableSession` + `run_turn`. It contains no business
logic: every rule (validation, preview, undo, protected columns) lives in
the modules below it, so the API is mostly (de)serialisation.

Interactive documentation: every endpoint below carries an OpenAPI summary
and a Markdown docstring, so the running server self-documents at
    http://127.0.0.1:8000/docs      (Swagger UI)
    http://127.0.0.1:8000/redoc     (ReDoc)

`create_app(...)` is a factory taking injectable (session, planner,
tracer) so the HTTP layer is testable offline with a FakePlanner; the
default wiring (real CSV, real Gemini, real trace file) happens only when
arguments are omitted. A single global session serves all requests —
fine for a take-home demo, called out in DECISIONS.md for production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import DATASET_PATH, PROJECT_ROOT
from .graph import run_turn
from .llm import GeminiPlanner, Planner, PlannerError
from .state import PendingChange, StateError, TableSession
from .trace import Tracer

FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"

# Groups shown in the Swagger sidebar, each with its own description.
OPENAPI_TAGS = [
    {"name": "table", "description": "Read the committed table and any "
     "pending (not yet accepted) preview."},
    {"name": "agent", "description": "Natural-language instructions. The "
     "agent replies with a **preview**, a **clarifying question**, or an "
     "**error** — it never commits changes on its own."},
    {"name": "lifecycle", "description": "Human-in-the-loop verdicts on the "
     "pending preview, plus undo of applied changes."},
    {"name": "editing", "description": "Direct (non-agent) cell edits made "
     "in the grid."},
]


# --- request bodies ---------------------------------------------------------

class ChatRequest(BaseModel):
    """One natural-language instruction for the agent."""
    message: str = Field(
        description="Instruction in plain language.",
        examples=["Increase electronics prices by 10%, then sort descending"])


class CellEditRequest(BaseModel):
    """One manual cell edit made directly in the grid."""
    sku: str = Field(description="Row identifier.", examples=["SKU-0042"])
    column: str = Field(description="Column to edit (the `sku` column is "
                        "protected).", examples=["price"])
    value: str = Field(description="New value as text; it is coerced to the "
                       "column's type and rejected if incompatible.",
                       examples=["199.90"])


# --- serialisation helpers --------------------------------------------------

def df_to_rows(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of row dicts (numpy scalars stripped
    by the round-trip through pandas' own JSON encoder)."""
    return json.loads(df.to_json(orient="records"))


def pending_to_json(pending: PendingChange | None) -> dict | None:
    """Serialise the staged preview: the preview rows, the cell-level diff
    (for highlighting), and per-operation stats."""
    if pending is None:
        return None
    return {
        "summary": pending.summary,
        "order_changed": pending.order_changed,
        "preview_rows": df_to_rows(pending.preview_df),
        "changes": [{"sku": c.sku, "column": c.column,
                     "old": c.old, "new": c.new} for c in pending.changes],
        "op_results": [{"kind": r.kind, "description": r.description,
                        "rows_matched": r.rows_matched,
                        "cells_changed": r.cells_changed}
                       for r in pending.op_results],
    }


# --- app factory -------------------------------------------------------------

def create_app(session: TableSession | None = None,
               planner: Planner | None = None,
               tracer: Tracer | None = None) -> FastAPI:
    app = FastAPI(
        title="grid_agent",
        version="0.1.0",
        description=(
            "Two-panel table-editing agent. Natural-language instructions "
            "become **structured operations** (`update_where`, `sort`) that "
            "a deterministic engine applies — the model never writes table "
            "contents directly. Changes are staged as previews, applied "
            "only on explicit accept, and undoable afterwards."),
        openapi_tags=OPENAPI_TAGS,
    )

    tracer = tracer or Tracer()
    session = session or TableSession.from_csv(DATASET_PATH, tracer=tracer)
    # The Gemini client is created lazily on the first chat message so the
    # app (and its non-chat endpoints) work even before a key is configured.
    state = {"planner": planner}

    def get_planner() -> Planner:
        if state["planner"] is None:
            state["planner"] = GeminiPlanner()      # raises if key missing
        return state["planner"]

    def table_payload() -> dict:
        return {
            "columns": list(session.df.columns),
            "rows": df_to_rows(session.df),
            "pending": pending_to_json(session.pending),
            "can_undo": session.can_undo,
        }

    # -- frontend -----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_INDEX)

    # -- table state --------------------------------------------------------
    @app.get("/api/table", tags=["table"],
             summary="Current table state (committed + pending preview)")
    def get_table() -> dict:
        """Return everything the frontend needs to render both panels:

        - **columns** — column names, in display order.
        - **rows** — the *committed* table as a list of row objects.
        - **pending** — `null`, or the staged preview: `preview_rows`,
          the cell-level `changes` diff (`sku`, `column`, `old`, `new`),
          `order_changed`, per-operation `op_results`, and a `summary`.
        - **can_undo** — whether an undo snapshot exists.
        """
        return table_payload()

    # -- chat: one agent turn ------------------------------------------------
    @app.post("/api/chat", tags=["agent"],
              summary="Send an instruction to the agent",
              responses={
                  400: {"description": "Empty message."},
                  503: {"description": "Gemini is unreachable or "
                        "GEMINI_API_KEY is not configured."},
              })
    def chat(body: ChatRequest) -> dict:
        """Run one full agent turn (LangGraph: plan → validate → repair →
        preview) and return:

        - **outcome** — `preview` (a pending change was staged and awaits
          accept/reject), `clarify` (the instruction was ambiguous;
          `message` holds the agent's question), or `error` (no valid plan
          could be produced; nothing was changed).
        - **message** — the agent's user-facing reply.
        - **table** — fresh table payload (same shape as `GET /api/table`).

        The agent **never commits** anything from this endpoint; a
        `preview` outcome still requires `POST /api/pending/accept`.
        """
        text = body.message.strip()
        if not text:
            raise HTTPException(400, "Message must not be empty.")
        try:
            planner_impl = get_planner()
        except PlannerError as exc:                 # missing/invalid key
            raise HTTPException(503, str(exc)) from exc
        result = run_turn(session, planner_impl, tracer, text)
        return {"outcome": result.outcome, "message": result.message,
                "table": table_payload()}

    # -- preview lifecycle ---------------------------------------------------
    @app.post("/api/pending/accept", tags=["lifecycle"],
              summary="Accept the pending preview",
              responses={409: {"description": "No pending change exists."}})
    def accept() -> dict:
        """Commit the staged preview to the table. The previous table
        version is pushed onto the undo stack first, so the change can be
        reverted with `POST /api/undo`. Returns the fresh table payload."""
        try:
            session.accept()
        except StateError as exc:
            raise HTTPException(409, str(exc)) from exc
        return table_payload()

    @app.post("/api/pending/reject", tags=["lifecycle"],
              summary="Reject the pending preview",
              responses={409: {"description": "No pending change exists."}})
    def reject() -> dict:
        """Discard the staged preview. The committed table is untouched and
        nothing is added to the undo stack. Returns the fresh table payload."""
        try:
            session.reject()
        except StateError as exc:
            raise HTTPException(409, str(exc)) from exc
        return table_payload()

    @app.post("/api/undo", tags=["lifecycle"],
              summary="Undo the most recent applied change",
              responses={409: {"description": "The undo stack is empty."}})
    def undo() -> dict:
        """Restore the table to its state before the last applied change
        (accepted plan **or** manual cell edit). Any pending preview is
        dropped, since it was computed against the replaced table. Undo is
        stacked: repeated calls step further back, up to the configured
        snapshot limit. Returns the fresh table payload."""
        try:
            session.undo()
        except StateError as exc:
            raise HTTPException(409, str(exc)) from exc
        return table_payload()

    # -- manual cell edits ---------------------------------------------------
    @app.patch("/api/cell", tags=["editing"],
               summary="Edit one cell directly",
               responses={400: {"description": "Unknown row/column, "
                                "protected column, or value of the wrong "
                                "type."}})
    def edit_cell(body: CellEditRequest) -> dict:
        """Apply a hand edit from the grid, bypassing the agent but **not**
        the rules: the value is coerced with the same type rules as agent
        plans, `sku` stays read-only, and the edit is undoable. A pending
        preview, if any, is invalidated (it described a table that no
        longer exists). Returns the fresh table payload."""
        try:
            session.edit_cell(body.sku, body.column, body.value)
        except StateError as exc:
            raise HTTPException(400, str(exc)) from exc
        return table_payload()

    return app


# `uvicorn grid_agent.api:app` entry point with default (real) wiring.
if not Path(DATASET_PATH).exists():                 # pragma: no cover
    raise RuntimeError(
        f"Dataset not found at {DATASET_PATH}. "
        "Run `python scripts/generate_data.py` first.")
app = create_app()
