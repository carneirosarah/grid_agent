# Session export — grid_agent build

- **Date:** 2026-07-18
- **Model:** Claude Fable 5 (`claude-fable-5`, set via `/model` at session start)
- **Workspace:** `/Users/sarahrlc/Documents/Cactus_Take_Home/grid_agent`
- **Note:** this is a reconstruction of the session written by the assistant
  (the CLI `/export` command was unavailable). User interruptions are quoted
  verbatim; assistant work is summarized with the actual commands and results.

---

## 1. Initial request (user)

The full take-home brief, now preserved in [initial_prompt.md](initial_prompt.md):
build a two-panel app (editable data table + chat), agent edits the table via
**structured operations only** (`update_where`, `sort`), multi-step commands,
preview with accept/reject, undo, clarifying questions on ambiguity. Stack:
Python + LangGraph + Gemini flash free API with structured output. Split into
11 steps, each with code and how to test. Persist trace to `.jsonl`. Suggest
improvements. Prioritize: operation design, separation of planning from
execution, state management, handling confidently-wrong model output.

## 2. Setup

- Inspected the empty repo (`README.md`, `DECISIONS.md` empty; `.git` present).
- Created `.venv` (Python 3.13.5), wrote `requirements.txt`, installed:
  pandas 3.0.3, pydantic 2.13, langgraph 1.2.9, google-genai 2.12, fastapi,
  uvicorn, pytest, httpx. Added `.gitignore`, `.env.example`,
  `src/grid_agent/{__init__,config}.py`, `pytest.ini` (`pythonpath = src`).

## 3. Step 1 — Data generation

`scripts/generate_data.py`: seeded (seed 42) generator, 350 rows × 9 columns
(`sku, name, category, supplier, price, cost, stock, rating, flagged`),
category price bands, cost derived from price with 5–60% margin.
Ran it → `data/inventory.csv` written and verified.

## 4. Steps 2–3 — Operation schemas + tests

`src/grid_agent/schemas.py` with **two schema layers**:
- *Wire models* (flat, all-string values) — what Gemini structured output fills.
- *Domain models* (strict discriminated union) — what the engine consumes.
- `wire_to_plan()` bridges them, **collecting** errors for the repair loop.

`tests/test_schemas.py` — 11 tests, all passed.

## 5. Step 4 — Deterministic engine

`src/grid_agent/engine.py`: pure `apply_plan(df, plan)`, AND-combined
condition masks, case-insensitive string matching, stable sort, atomic
multi-op plans, identity-keyed `diff_tables` (keyed on `sku`, so sorts
produce zero false cell diffs) returning `CellChange` records.

> **User interruption:** *"Explain the use CellChange class"*

Assistant explained before proceeding: `CellChange` is the unit of the
preview diff — one cell edit keyed by stable row identity rather than
position (sorts don't pollute the diff), and the source of truth for
highlighted cells and honest change counts (derived from data, never from
model claims).

Two test failures on first run: pandas 3.0 uses the Arrow-backed `str`
dtype, so `col.dtype == object` never matched. Fixed with an
`_is_string()` helper using `pd.api.types.is_string_dtype`. 15/15 passed.

## 6. Step 5 — Semantic validator

`src/grid_agent/validator.py`: validates structural plans **against the live
table** — column resolution with difflib did-you-mean, protected `sku`,
numeric-only rules for `multiply`/`increment` and ordering operators, dtype
coercion of wire strings (bool word-forms, integer strictness), and a
value-vocabulary check that turns "Eletronics" into an error listing real
categories. Returns a typed *resolved plan* or the full error list.
`tests/test_validator.py` — 12 tests passed (26 cumulative with engine).

## 7. Step 6 — State, preview, undo, trace

- `src/grid_agent/trace.py`: thread-safe append-only JSONL `Tracer` + `NullTracer`.
- `src/grid_agent/state.py`: `TableSession` — committed df, single pending
  `PendingChange`, capped snapshot undo stack (25), chat history, typed
  manual `edit_cell` (undoable, invalidates stale previews).
- `tests/test_state.py` — 12 tests passed.

## 8. Step 7 — Gemini integration

> **User interruption (on first draft):** *"Put the SYSTEM_PROMPT in a .md
> file and explain the plan function"*

Assistant explained `GeminiPlanner.plan()` (contents assembly with fresh
table context each turn, `response_schema=WireReply` +
`response_mime_type="application/json"`, `temperature=0`, normalization to
`WireReply | PlannerError`), moved the prompt to
`src/grid_agent/prompts/system_prompt.md`, and rewrote `llm.py` around the
injectable `Planner` protocol.

> **User interruption (when graph.py was attempted next):** *"Create the
> test script to llm.py"*

Wrote `tests/test_llm.py` first — 12 tests: offline stubs for request
assembly, parse fallbacks, error wrapping, table-context accuracy, plus a
live smoke test. **All 12 passed, including the live Gemini call** (a
`GEMINI_API_KEY` was present), confirming `gemini-3-flash-preview` returns a
valid `update_where + sort` plan for the example instruction.

## 9. Step 8 — LangGraph

`src/grid_agent/graph.py`: `plan → validate → preview` with conditional
edges; validation failures loop back to `plan` carrying the error report
(max 2 repairs), then `give_up` ends the turn loudly; `clarify` exits to the
user. Repair noise is kept out of durable chat history. Accept/reject is
deliberately outside the graph (documented trade-off vs `interrupt()`).

## 10. Step 9 — End-to-end tests

`tests/test_e2e.py` with a scripted `FakePlanner`: multi-step happy path
(preview → accept → undo), live table context, ambiguity → clarifying
question → contextual follow-up, confidently-wrong plan repaired via error
feedback, persistently-wrong planner gives up with data untouched, trace
events asserted from the `.jsonl`, full 350-row dataset run. 7 tests passed.

## 11. Step 10 — FastAPI

> **User interruption (on first draft):** *"Document each endpoint to be
> available in the Swagger Documentation /docs"*

Rewrote `src/grid_agent/api.py` with OpenAPI tags, per-endpoint summaries,
Markdown docstrings, and documented error codes. Endpoints: `GET /api/table`,
`POST /api/chat`, `POST /api/pending/accept|reject`, `POST /api/undo`,
`PATCH /api/cell`, `GET /` (frontend). `create_app()` factory with
injectable session/planner/tracer. `tests/test_api.py` — 7 tests passed
(after adding `tests/__init__.py` for the cross-test import).

## 12. Step 11 — Frontend

`frontend/index.html`: single self-contained page. Left panel — grid with
sticky header, preview rows with changed cells highlighted (old value struck
through), `PREVIEW`/`row order changed` badges, double-click cell editing,
undo button. Right panel — chat with amber clarify / red error bubbles and
an Accept/Reject bar while a preview is pending.

## 13. Live smoke test + one real bug

Started uvicorn on :8000 and drove the real API:

1. *"Flag every product with margin below 15 percent of the price, then sort
   by rating ascending"* → **clarify** (correct: column-to-column margin math
   is not expressible with the two operations; the agent asked instead of
   guessing).
2. *"Increase eletronics prices by 10%, flag products with rating below 2,
   then sort by price descending"* → **HTTP 500**: `numpy.bool` leaked from
   the pandas Series into `CellChange.old/new` and broke JSON serialization.
   **Fixed** in `engine.py` (`_native()` via `.item()`) + regression test.
3. Retry → **preview**: 62 price cells ×1.10, 88 rows flagged, order changed;
   `accept` committed (top price 591.95, undo available); `undo` restored the
   original table; `traces/trace.jsonl` contained every event type
   (`user_message`, `llm_reply`, `plan_validated`, `preview_created`,
   `change_accepted`, `undo`, `clarification_asked`, …).

## 14. Documentation & wrap-up

- The brief (which the user had saved into `DECISIONS.md`) was preserved
  verbatim as `ASSIGNMENT.md`.
- `README.md`: architecture diagram, the "three walls" against
  confidently-wrong output, quickstart, per-step docs with test commands.
- `DECISIONS.md`: design decisions (two schema layers, planning/execution
  separation, snapshot undo, accept outside the graph, etc.) and suggested
  improvements to both the approach (richer operation vocabulary,
  LangGraph-native HITL, planner eval harness) and the code (sessions &
  persistence, scale path, hardening, tooling).

**Final state:** 76/76 tests passing (including the live Gemini test);
server verified end-to-end at `http://127.0.0.1:8000` (Swagger at `/docs`).

## Files created

```
ASSIGNMENT.md  DECISIONS.md  README.md  requirements.txt  pytest.ini
.gitignore  .env.example
scripts/generate_data.py
data/inventory.csv                      (generated, git-ignored)
src/grid_agent/__init__.py  config.py  schemas.py  engine.py  validator.py
src/grid_agent/state.py  trace.py  llm.py  graph.py  api.py
src/grid_agent/prompts/system_prompt.md
frontend/index.html
tests/__init__.py  conftest.py  test_schemas.py  test_engine.py
tests/test_validator.py  test_state.py  test_llm.py  test_e2e.py  test_api.py
traces/trace.jsonl                      (runtime, git-ignored)
```
