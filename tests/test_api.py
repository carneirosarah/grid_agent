"""Step 10 — HTTP layer tests (offline: FakePlanner injected)."""

import pytest
from fastapi.testclient import TestClient

from grid_agent.api import create_app
from grid_agent.state import TableSession
from grid_agent.trace import NullTracer

from .test_e2e import CLARIFY, GOOD_PLAN, FakePlanner


@pytest.fixture()
def client(small_df):
    session = TableSession(df=small_df.copy())
    app = create_app(session=session,
                     planner=FakePlanner(GOOD_PLAN, CLARIFY),
                     tracer=NullTracer())
    return TestClient(app)


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


def test_openapi_documents_all_endpoints(client):
    """The Swagger doc (/docs) must cover the whole API surface."""
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for route in ["/api/table", "/api/chat", "/api/pending/accept",
                  "/api/pending/reject", "/api/undo", "/api/cell"]:
        assert route in paths
    # Spot-check: summaries and rich descriptions made it into the spec.
    chat = paths["/api/chat"]["post"]
    assert chat["summary"] == "Send an instruction to the agent"
    assert "never commits" in chat["description"]
    assert "503" in chat["responses"]
