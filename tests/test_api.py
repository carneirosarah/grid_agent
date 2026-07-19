"""Step 10 — HTTP layer tests (offline: FakePlanner injected)."""

import pytest
from fastapi.testclient import TestClient

from grid_agent.api import create_app
from grid_agent.persistence import InMemorySessionRepository
from grid_agent.sessions import SessionStore
from grid_agent.trace import NullTracer

from .test_e2e import CLARIFY, GOOD_PLAN, FakePlanner


def make_client(small_df, *replies) -> TestClient:
    store = SessionStore(df_factory=lambda: small_df.copy(),
                         repository=InMemorySessionRepository(),
                         tracer=NullTracer())
    app = create_app(store=store,
                     planner=FakePlanner(*replies),
                     tracer=NullTracer())
    # TestClient persists cookies across requests, so each client behaves
    # like one browser/user with a stable session.
    return TestClient(app)


@pytest.fixture()
def client(small_df):
    return make_client(small_df, GOOD_PLAN, CLARIFY)


def test_get_table_shape(client):
    payload = client.get("/api/table").json()
    assert payload["columns"][0] == "sku"
    assert len(payload["rows"]) == 6
    assert payload["pending"] is None
    assert payload["can_undo"] is False


def test_chat_preview_accept_undo_cycle(client):
    # 1. Instruction -> preview staged, not committed.
    r = client.post("/api/chat", json={"message": "electronics +10%, sort desc"})
    body = r.json()
    assert body["outcome"] == "preview"
    assert body["table"]["pending"]["changes"] == [
        {"sku": "SKU-0001", "column": "price", "old": 25.0, "new": 27.5}]
    assert body["table"]["rows"][0]["price"] == 25.0      # still committed

    # 2. Accept -> committed, undo available.
    table = client.post("/api/pending/accept").json()
    assert table["pending"] is None
    assert table["rows"][0]["sku"] == "SKU-0002"          # sorted desc
    assert table["can_undo"] is True

    # 3. Undo -> original table back.
    table = client.post("/api/undo").json()
    assert table["rows"][0]["price"] == 25.0
    assert table["can_undo"] is False


def test_chat_clarify_stages_nothing(client):
    client.post("/api/chat", json={"message": "raise prices"})   # GOOD_PLAN
    client.post("/api/pending/reject")
    r = client.post("/api/chat", json={"message": "increase prices"})  # CLARIFY
    assert r.json()["outcome"] == "clarify"
    assert r.json()["table"]["pending"] is None


def test_empty_message_is_400(client):
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400


def test_lifecycle_conflicts_are_409(client):
    assert client.post("/api/pending/accept").status_code == 409
    assert client.post("/api/pending/reject").status_code == 409
    assert client.post("/api/undo").status_code == 409


def test_manual_cell_edit_and_validation(client):
    table = client.patch("/api/cell", json={
        "sku": "SKU-0003", "column": "stock", "value": "150"}).json()
    assert table["rows"][2]["stock"] == 150
    assert table["can_undo"] is True

    bad_type = client.patch("/api/cell", json={
        "sku": "SKU-0003", "column": "stock", "value": "many"})
    assert bad_type.status_code == 400

    protected = client.patch("/api/cell", json={
        "sku": "SKU-0003", "column": "sku", "value": "SKU-9999"})
    assert protected.status_code == 400


def test_sessions_are_isolated_per_cookie(small_df):
    """Two browsers (= two cookie jars) must never see each other's data."""
    store = SessionStore(df_factory=lambda: small_df.copy(),
                         repository=InMemorySessionRepository(),
                         tracer=NullTracer())
    app = create_app(store=store, planner=FakePlanner(GOOD_PLAN),
                     tracer=NullTracer())
    alice, bob = TestClient(app), TestClient(app)

    # First contact mints a cookie for each client.
    assert "grid_session" in alice.get("/api/table").cookies
    bob.get("/api/table")

    # Alice edits; Bob's table must be untouched.
    alice.patch("/api/cell", json={"sku": "SKU-0001", "column": "price",
                                   "value": "999"})
    assert alice.get("/api/table").json()["rows"][0]["price"] == 999.0
    assert bob.get("/api/table").json()["rows"][0]["price"] == 25.0
    assert bob.get("/api/table").json()["can_undo"] is False


def test_concurrent_double_accept_commits_exactly_once(small_df):
    """The session lock must make a double-submitted Accept safe: one 200,
    one 409, and the change applied exactly once (not compounded)."""
    from concurrent.futures import ThreadPoolExecutor

    client = make_client(small_df, GOOD_PLAN)
    client.post("/api/chat", json={"message": "electronics +10%"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = sorted(r.status_code for r in pool.map(
            lambda _: client.post("/api/pending/accept"), range(2)))
    assert codes == [200, 409]

    row = next(r for r in client.get("/api/table").json()["rows"]
               if r["sku"] == "SKU-0001")
    assert row["price"] == 27.5          # applied once: 25.0 * 1.1


def test_metrics_endpoint_reports_validity_rate(small_df):
    """GET /api/metrics surfaces the planner's ValidityCounter."""
    from grid_agent.metrics import ValidityCounter

    planner = FakePlanner(GOOD_PLAN)
    planner.validity = ValidityCounter()        # as GeminiPlanner carries
    store = SessionStore(df_factory=lambda: small_df.copy(),
                         repository=InMemorySessionRepository(),
                         tracer=NullTracer())
    client = TestClient(create_app(store=store, planner=planner,
                                   tracer=NullTracer()))

    # No LLM responses yet: null rate, not 0%.
    assert client.get("/api/metrics").json() == {
        "llm_responses": 0, "accepted": 0, "rejected": 0,
        "validity_pct": None}

    planner.validity.record(True)
    planner.validity.record(False)
    assert client.get("/api/metrics").json()["validity_pct"] == 50.0


def test_metrics_endpoint_works_without_a_counter(client):
    """A planner with no counter (or none created yet) yields the empty
    shape — the endpoint must never 500 over observability."""
    assert client.get("/api/metrics").json()["llm_responses"] == 0


def test_openapi_documents_all_endpoints(client):
    """The Swagger doc (/docs) must cover the whole API surface."""
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for route in ["/api/table", "/api/chat", "/api/pending/accept",
                  "/api/pending/reject", "/api/undo", "/api/cell",
                  "/api/metrics"]:
        assert route in paths
    # Spot-check: summaries and rich descriptions made it into the spec.
    chat = paths["/api/chat"]["post"]
    assert chat["summary"] == "Send an instruction to the agent"
    assert "never commits" in chat["description"]
    assert "503" in chat["responses"]
