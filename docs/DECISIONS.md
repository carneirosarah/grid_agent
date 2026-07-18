# Design decisions & suggested improvements

The original brief is preserved verbatim in
[initial_prompt.md](initial_prompt.md). This file explains the choices made
where the brief left room, and lists improvements to both the suggested
approach and this implementation.

---

## Key decisions

### 1. Two schema layers: wire vs domain (tool & operation design)

The LLM fills a **flat, permissive wire schema** (string values, no
nested unions) because that is the shape Gemini's structured output
produces most reliably. The engine consumes a **strict domain schema**
(discriminated unions, typed values). `wire_to_plan` bridges them and
*collects* errors instead of failing fast.

Why: constrained decoding eliminates *syntactic* failures, but pushing
our real invariants into the response schema would make the model fail in
opaque ways. Keeping the wire format easy and validating afterwards gives
precise, feedback-able error messages — the raw material of the repair
loop.

### 2. Three walls against confidently-wrong output

1. **Decoding**: `response_schema=WireReply` — free text and invented
   operation kinds are unrepresentable.
2. **Structure**: `wire_to_plan` — missing fields, empty plans.
3. **Semantics**: `validate_plan` runs against the *live DataFrame* —
   unknown columns (with did-you-mean suggestions), dtype coercion of
   every value, numeric-only rules for `multiply`/`increment` and
   ordering operators, protected `sku`, and a value-vocabulary check
   that turns "Eletronics" into an error listing the real categories
   instead of a silent 0-row update.

Failures loop back to the model with the complete error report, at most
`MAX_REPAIR_ATTEMPTS` times, then the agent **gives up loudly** with the
errors shown. It never "does its best" with an invalid plan, and the
engine treats receiving one as a programming bug (`EngineError`), not a
user error. Change counts shown to the user are computed from the actual
before/after diff — never from what the model claims it did.

### 3. Separation of planning from execution

The LLM's entire authority is producing a `WireReply` inside the `plan`
node. Application is a pure function in
[engine.py](../src/grid_agent/engine.py) (no LLM imports, no I/O,
deterministic, input never mutated) — and even that only stages a
preview. The **commit** requires a third, human step. So the pipeline is:
*model proposes → code validates and applies to a copy → human disposes*.

### 4. State management

- **Preview is not a commit.** `TableSession.propose()` stores the result
  and identity-keyed diff separately; `accept()` swaps it in, `reject()`
  discards it.
- **Undo = snapshot stack** (capped at 25). At 350 rows a deep copy costs
  a few KB; snapshots are simpler and strictly safer than inverse
  operations — a `sort` has no derivable inverse without storing the
  prior order anyway. Manual cell edits go through the same stack.
- **Row identity** (`sku`) is the backbone: diffs are keyed on it (a sort
  yields zero false cell changes), the frontend addresses edits with it,
  and the validator refuses to let anything write it.
- **Stale-preview invalidation**: a manual edit or a new proposal drops
  the pending preview, because it was computed against a table that no
  longer exists.

### 5. Per-user sessions, concurrency & persistence (v0.2)

Originally a single global session (an honest take-home simplification);
now replaced by:

- **Identity**: an opaque, httponly `grid_session` cookie names the
  session; first contact mints a UUID.
- **Concurrency**: one `RLock` per session, held for the whole request.
  FastAPI serves sync endpoints from a thread pool, so same-user races
  are real: a double-submitted Accept now yields exactly one commit and
  one clean 409 (`tests/test_api.py::test_concurrent_double_accept_commits_exactly_once`);
  a chat turn cannot interleave with an undo. Different users never share
  a lock, so there is no global bottleneck. A separate creation lock
  prevents two first-requests from minting rival sessions for one id.
- **Persistence**: durable state (committed table, undo stack, chat
  history) is written through to **PostgreSQL** (JSONB, one row per
  session, version counter) after every mutation. DataFrames are encoded
  with pandas' schema-carrying `orient="table"` JSON so dtypes survive
  the round-trip exactly. Sessions — including undo — survive restarts;
  app containers are disposable.
- **The pending preview is deliberately not persisted**: it is a
  transient derivation, and resurrecting a stale preview after a restart
  would be worse than re-asking.
- **In-memory fallback**: without `DATABASE_URL` the same store runs on
  an in-memory repository (identical serialisation path), keeping tests
  and DB-less development first-class.
- **LRU cache**: at most `SESSION_CACHE_LIMIT` sessions stay resident;
  eviction is safe because persistence is write-through.

### 6. Accept/reject outside the LangGraph run

The graph ends at "preview staged"; the verdict arrives as a separate
HTTP call handled by plain session methods. The alternative — LangGraph's
`interrupt()` + checkpointer, resuming the thread on accept — is the
right shape for long-lived multi-agent flows, but with a stateless HTTP
frontend it adds a second persistence layer and thread lifecycle for no
behavioural gain here (session durability is already handled by the
Postgres repository).

### 7. Smaller decisions

- **Repair noise is not conversation.** Validator error reports go into a
  transient history copy for the retry; the durable chat history keeps
  only user-facing turns. Everything is still traced.
- **Table context is rebuilt every turn** (dtypes, ranges, full
  vocabulary of low-cardinality text columns) so the model plans against
  current data and can spell values correctly in the first place.
- **`temperature=0`** — planning should be reproducible.
- **Case-insensitive string matching** in the engine ("electronics"
  matches "Electronics") — users type lowercase.
- **The dataset ships a `flagged` column** so "flag every row where…" is
  expressible within the two allowed operations (no add-column op exists
  by design — the agent correctly *asks* when told to compute a Margin
  column, as the brief's example requires clarification under a
  two-operation vocabulary).
- **One shared `.jsonl` trace, session-stamped.** The trace file stays
  single and append-only (greppable, as the brief asks); every event now
  carries its `session_id` via a bound tracer.
- **Compose ports**: Postgres is published on host port 5433 to avoid
  colliding with a locally installed Postgres; the trace lives in a named
  volume because macOS Docker Desktop requires explicit permission for
  bind mounts under `~/Documents`.

---

## Suggested improvements

### To the suggested approach

1. **The operation vocabulary is the bottleneck.** The brief's own
   example ("Add a Margin column computed from Price and Cost") is not
   expressible with only `update_where` + `sort`; the agent must clarify.
   The highest-value extensions, in order: `add_column` (with a small
   safe expression language over columns), column-to-column comparisons
   in conditions (`price < cost`), `delete_where`, and a read-only
   `aggregate` (count/sum/avg) so questions don't require edits.
2. **Wire values as typed unions.** Values are strings on the wire and
   coerced by the validator. Gemini's structured output has grown more
   reliable with `anyOf`; a typed wire schema would move some errors from
   runtime validation to decode time. Worth measuring before switching —
   the string+coercion path produces better repair messages.
3. **LangGraph-native human-in-the-loop** (`interrupt()` + a
   checkpointer, one thread per session) once reviews become multi-step
   or agents long-lived; the Postgres instance added in v0.2 could host
   the LangGraph checkpointer as well.
4. **An eval harness over the planner.** A table of (instruction →
   expected plan | expected clarification) pairs replayed against the
   live model in CI would catch prompt/model regressions — the system's
   riskiest dependency is prompt drift, not code.
5. **Clarification with options.** Instead of a free-text question, the
   wire schema could carry structured choices ("Which column: price or
   cost?") the UI renders as buttons — fewer round-trips, less ambiguity.

### To this implementation

1. ~~Sessions & persistence~~ — **implemented in v0.2** (cookie-scoped
   sessions, per-session locking, Postgres write-through, docker
   compose). Natural next steps: real authentication instead of a bare
   cookie, a session-expiry sweep (`DELETE FROM sessions WHERE
   updated_at < …`), and optimistic locking on the `version` column if
   the app ever runs multiple replicas (the in-process lock only
   serialises within one process).
2. **Scale path**: pandas copies + full-table JSON are fine at 350 rows.
   At ~100k+: polars/DuckDB for the engine, server-side pagination, undo
   as inverse patches (store the diff, which `diff_tables` already
   computes) instead of snapshots, and a virtualized grid.
3. **Redo + labeled history**: the snapshot stack trivially extends to a
   redo stack; surfacing it as a timeline ("3 changes ago: flagged 88
   rows") would make undo legible.
4. **Streaming & transparency**: stream the reply, and show the validated
   plan JSON in a collapsible chat element so power users can audit what
   will run before accepting.
5. **Hardening**: request size limits and rate limiting on `/api/chat`,
   auth, secure cookies behind TLS, and non-root user + read-only
   filesystem in the container.
6. **Tooling**: move to `pyproject.toml` packaging, add `ruff` + `mypy
   --strict` and a CI workflow running the offline suite (plus a
   Postgres service container for `test_persistence_pg.py`);
   property-based tests (hypothesis) for `build_mask`/`diff_tables`
   invariants.
7. **Frontend**: at current scope vanilla JS is the right weight; if it
   grows (sorting UI, filters, multi-select), moving to a small component
   framework and a virtualized table is the first step.
