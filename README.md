# grid_agent — a table-editing agent with previews and undo

Two-panel application: an **editable data table** (350-row product
inventory) and a **chat interface**. You instruct an agent in natural
language; the agent answers with **structured operations**
(`update_where`, `sort`) that a deterministic engine applies. The model
never writes table contents directly. Every change is staged as a
**preview** you accept or reject, accepted changes are **undoable**, and
every event is persisted to a **`.jsonl` trace**.

Built with Python, **LangGraph**, and the **Gemini** API (structured
output). The original brief is preserved in [initial_prompt.md](docs/initial_prompt.md);
design decisions and suggested improvements are in
[docs/DECISIONS.md](docs/DECISIONS.md).

---

## Quickstart (Docker — app + PostgreSQL)

```bash
cp .env.example .env            # then paste your key from
                                # https://aistudio.google.com/apikey
docker compose up --build
```

Open <http://127.0.0.1:8000> for the app, <http://127.0.0.1:8000/docs>
for the interactive Swagger API documentation. Each browser gets its own
isolated session (a `grid_session` cookie), persisted in PostgreSQL —
sessions, including undo history and chat context, survive restarts
(`docker compose restart app` and reload to see it). Read the trace with
`docker compose exec app cat traces/trace.jsonl`.

## Quickstart (local, no database)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # add your Gemini key

python scripts/generate_data.py # writes data/inventory.csv (350 x 9)
uvicorn grid_agent.api:app --app-dir src --port 8000
```

Without `DATABASE_URL`, sessions live in process memory (still per-user,
just not restart-proof). To use the compose database from a local server:

```bash
docker compose up -d db
DATABASE_URL=postgresql://grid:grid@localhost:5433/grid_agent \
    uvicorn grid_agent.api:app --app-dir src --port 8000
```

Try, in the chat:

> Increase eletronics prices by 10%, flag products with rating below 2,
> then sort by price descending

(The typo is intentional — see "confidently wrong output" below.)

Run the whole test suite (no API key needed; one live smoke test
auto-skips without a key):

```bash
pytest            # 86 tests (the Postgres one skips without DATABASE_URL)
```

## Architecture

```
 chat message
      │
      ▼
┌───────────── LangGraph (src/grid_agent/graph.py) ─────────────┐
│                                                               │
│  plan ──────► validate ──────────────► preview ──► END        │
│ (Gemini,      (wire_to_plan +          (engine on a           │
│  structured    validate_plan,           copy; still           │
│  output)       zero LLM)               uncommitted)           │
│    ▲             │        │                                   │
│    └── repair ◄──┘        └──► clarify ──► END                │
│    (error report,              (question to user)             │
│     max 2 retries)                                            │
└───────────────────────────────────────────────────────────────┘
      │ pending preview
      ▼
 user verdict:  accept ──► commit (undo snapshot pushed)
                reject ──► discard
                undo   ──► restore previous snapshot
```

Three walls stand between a confidently-wrong model and the data:

1. **Schema-constrained decoding** — Gemini's structured output can only
   emit the flat wire schema; free text or invented operation kinds are
   impossible ([llm.py](src/grid_agent/llm.py)).
2. **Structural conversion** — `wire_to_plan` turns the permissive wire
   format into strict discriminated-union models, collecting every error
   ([schemas.py](src/grid_agent/schemas.py)).
3. **Semantic validation** — table-aware checks: real columns
   (did-you-mean suggestions), dtype coercion, protected `sku`,
   numeric-only rules, and value-vocabulary checks that turn a typo like
   *"Eletronics"* into a repairable error listing the real categories
   ([validator.py](src/grid_agent/validator.py)).

Validation failures loop back to the model with the full error report
(max `MAX_REPAIR_ATTEMPTS` retries), then the agent gives up loudly —
it never guesses, and invalid plans never reach the engine.

## Project layout

```
scripts/generate_data.py     Step 1  — dataset generator (seeded)
src/grid_agent/
  config.py                  paths, model name, limits (.env loaded here)
  schemas.py                 Step 2  — wire + domain operation models
  engine.py                  Step 4  — deterministic apply + diff
  validator.py               Step 5  — semantic validation & coercion
  state.py                   Step 6  — session, preview, undo
  sessions.py                per-user session store (locks, LRU, persist)
  persistence.py             PostgreSQL / in-memory session repositories
  trace.py                   .jsonl event log (session-stamped events)
  llm.py                     Step 7  — Gemini structured-output planner
  prompts/system_prompt.md   the agent's behavioural rules
  graph.py                   Step 8  — LangGraph plan→validate→preview
  api.py                     Step 10 — FastAPI (self-documented at /docs)
frontend/index.html          Step 11 — two-panel UI (vanilla JS)
tests/                       Steps 3, 9 + unit tests for every module
data/inventory.csv           generated dataset (git-ignored)
traces/trace.jsonl           persisted trace (git-ignored)
Dockerfile, docker-compose.yml   app container + PostgreSQL
```

## The 11 steps, and how to test each

### 1. Data generation — [scripts/generate_data.py](scripts/generate_data.py)
350 rows × 9 columns of product inventory (`sku`, `name`, `category`,
`supplier`, `price`, `cost`, `stock`, `rating`, `flagged`). Seeded RNG:
every run reproduces the identical file.

```bash
python scripts/generate_data.py           # prints shape + head
```

### 2. Operations — [src/grid_agent/schemas.py](src/grid_agent/schemas.py)
Only `update_where` (conditions AND-combined; actions `set` /
`multiply` / `increment`) and `sort` (multi-key). Two layers: permissive
**wire models** for Gemini, strict **domain models** for the engine, and
`wire_to_plan` bridging them with error collection.

### 3. Schema tests — [tests/test_schemas.py](tests/test_schemas.py)
```bash
pytest tests/test_schemas.py -v
```

### 4. Deterministic engine — [src/grid_agent/engine.py](src/grid_agent/engine.py)
Pure functions: `(df, plan) -> new df + per-op stats`. Never mutates
input, atomic across operations, stable sort, case-insensitive string
matching, and an identity-keyed diff (`diff_tables`) so a sort produces
zero false cell changes.

```bash
pytest tests/test_engine.py -v
```

### 5. Semantic validator — [src/grid_agent/validator.py](src/grid_agent/validator.py)
Validates a structural plan **against the live table** and returns either
a fully-typed resolved plan or a feedback-quality error list.

```bash
pytest tests/test_validator.py -v
```

### 6. State, preview & undo — [src/grid_agent/state.py](src/grid_agent/state.py)
`TableSession` owns the committed table, the single pending preview, a
capped snapshot-based undo stack, and manual cell edits (typed with the
same coercion rules; also undoable).

```bash
pytest tests/test_state.py -v
```

### 7. Gemini integration — [src/grid_agent/llm.py](src/grid_agent/llm.py)
`GeminiPlanner` calls `generate_content` with
`response_schema=WireReply`, `temperature=0`, and a table context rebuilt
every turn (columns, dtypes, ranges, and the full vocabulary of small
text columns). Everything else depends only on the `Planner` protocol.

```bash
pytest tests/test_llm.py -v                    # offline (stubbed SDK)
GEMINI_API_KEY=... pytest tests/test_llm.py -k live   # real API call
```

### 8. LangGraph — [src/grid_agent/graph.py](src/grid_agent/graph.py)
Nodes `plan → validate → preview` with a bounded repair loop and a
`clarify` exit. Planning (LLM) and execution (engine) never share a node;
repair-loop noise is kept out of the durable chat history.

```bash
pytest tests/test_e2e.py -v                    # graph covered end to end
```

### 9. End-to-end tests — [tests/test_e2e.py](tests/test_e2e.py)
A scripted `FakePlanner` drives the full pipeline offline: multi-step
happy path (preview → accept → undo), ambiguity → clarifying question,
confidently-wrong plan → repaired via error feedback, persistently wrong
→ clean give-up with untouched data, plus trace-file assertions and a
run over the full 350-row dataset.

```bash
pytest tests/test_e2e.py -v
```

### 10. FastAPI — [src/grid_agent/api.py](src/grid_agent/api.py)
Thin, logic-free HTTP layer; every endpoint is documented in Swagger
(`/docs`). Factory `create_app()` takes injectable store/planner/tracer
for offline HTTP tests.

```bash
pytest tests/test_api.py -v
uvicorn grid_agent.api:app --app-dir src --port 8000
curl -s localhost:8000/api/table | head -c 300
```

### Per-user sessions, concurrency & persistence
Each user is identified by an opaque `grid_session` cookie and gets an
isolated session ([sessions.py](src/grid_agent/sessions.py)). Every
request holds that session's lock, so races from the same user (double
Accept, two tabs) serialise cleanly — one wins, the other gets a 409 —
while different users never contend. Durable state (committed table,
undo stack, chat history — **not** the transient preview) is written
through to PostgreSQL after each mutation
([persistence.py](src/grid_agent/persistence.py)), dtype-faithfully, so
sessions survive restarts; without `DATABASE_URL` an in-memory
repository keeps the same code path working. Trace events carry the
`session_id`.

```bash
pytest tests/test_sessions.py tests/test_api.py -v     # offline
docker compose up -d db                                # live Postgres
DATABASE_URL=postgresql://grid:grid@localhost:5433/grid_agent \
    pytest tests/test_persistence_pg.py -v
```

### Docker — [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)
Two services: `db` (postgres:16-alpine, healthchecked, named volume) and
`app` (built from the Dockerfile; regenerates the seeded dataset at
start, waits for the db to be healthy). App containers are disposable —
all session state lives in the database.

```bash
docker compose up --build          # app on :8000, Postgres on :5433
docker compose restart app         # sessions survive (persisted in db)
docker compose exec db psql -U grid -d grid_agent -c 'TABLE sessions'
```

### 11. Frontend — [frontend/index.html](frontend/index.html)
Single self-contained page. Left: the grid — preview rows with changed
cells highlighted green (old value struck through), `PREVIEW` / `row
order changed` badges, double-click to edit a cell, undo button. Right:
chat — clarifying questions in amber, errors in red, and an
Accept / Reject bar whenever a preview is pending.

Manual test script: send the quickstart instruction → green highlights
appear and the committed table is unchanged → **Accept** → **↩ Undo** →
original table returns. Then try "increase prices" (ambiguous → the agent
asks a question) and "delete all furniture rows" (unsupported → the agent
says so rather than improvising).

## Trace (`traces/trace.jsonl`)

One JSON object per line: `user_message`, `llm_reply` (raw structured
output, per attempt), `validation_failed` (error lists), `plan_validated`,
`preview_created`, `change_accepted` / `change_rejected`, `undo`,
`manual_edit`, `clarification_asked`, `planner_error`, `turn_finished`.

```bash
python -m json.tool --json-lines traces/trace.jsonl | less
```
