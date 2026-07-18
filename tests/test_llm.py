"""Step 7 — LLM integration tests.

Everything here runs offline: the Gemini client is replaced with stubs.
What we verify is OUR side of the contract:
- the table context given to the model is accurate and current,
- the request is assembled correctly (context first, history in order,
  structured-output config),
- every possible SDK outcome is normalised to `WireReply | PlannerError`.

The final test is a live smoke test, auto-skipped unless GEMINI_API_KEY
is set:  GEMINI_API_KEY=... pytest tests/test_llm.py -k live
"""

import os
from types import SimpleNamespace

import pytest

from grid_agent.llm import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_REF,
    GeminiPlanner,
    PlannerError,
    build_table_context,
)
from grid_agent.schemas import WireReply


# --- system prompt ---------------------------------------------------------

def test_system_prompt_loaded_from_markdown_file():
    assert "update_where" in SYSTEM_PROMPT
    assert "clarify" in SYSTEM_PROMPT


# --- table context ---------------------------------------------------------

def test_context_reports_shape_and_numeric_ranges(small_df):
    context = build_table_context(small_df)
    assert "6 rows" in context
    assert "price: numeric, min=5.0, max=300.0" in context


def test_context_embeds_small_text_vocabulary(small_df):
    """Low-cardinality columns list every value — this is what lets the
    model spell 'Electronics' correctly instead of guessing."""
    context = build_table_context(small_df)
    assert "Apparel" in context and "Sports" in context
    assert "boolean" in context               # flagged column


def test_context_samples_large_text_columns(small_df):
    # Shrink the threshold so `name` (6 uniques) counts as large.
    import grid_agent.llm as llm
    original = llm.VOCABULARY_MAX_UNIQUES
    llm.VOCABULARY_MAX_UNIQUES = 3
    try:
        context = build_table_context(small_df)
        assert "6 distinct values" in context
    finally:
        llm.VOCABULARY_MAX_UNIQUES = original


def test_context_reflects_current_data_not_stale(small_df):
    before = build_table_context(small_df)
    small_df.loc[0, "price"] = 999.0
    after = build_table_context(small_df)
    assert before != after
    assert "max=999.0" in after


# --- GeminiPlanner plumbing (stubbed SDK) ----------------------------------

def make_planner(response=None, error: Exception | None = None) -> GeminiPlanner:
    """Build a planner whose client returns `response` or raises `error`."""
    planner = GeminiPlanner.__new__(GeminiPlanner)   # skip real __init__
    planner._model = "stub-model"

    def generate_content(**kwargs):
        generate_content.last_kwargs = kwargs        # captured for asserts
        if error is not None:
            raise error
        return response

    planner._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content))
    return planner


GOOD_REPLY = WireReply(intent="clarify", clarifying_question="By how much?")

USAGE = SimpleNamespace(prompt_token_count=120, candidates_token_count=30,
                        total_token_count=150)


def test_missing_api_key_raises_clear_error(monkeypatch):
    with pytest.raises(PlannerError, match="GEMINI_API_KEY"):
        GeminiPlanner(api_key="")


def test_parsed_reply_is_returned(small_df):
    planner = make_planner(SimpleNamespace(parsed=GOOD_REPLY, text="ignored",
                                           usage_metadata=USAGE))
    reply, _ = planner.plan(build_table_context(small_df), [])
    assert reply is GOOD_REPLY


def test_falls_back_to_parsing_raw_text(small_df):
    raw = GOOD_REPLY.model_dump_json()
    planner = make_planner(SimpleNamespace(parsed=None, text=raw))
    reply, _ = planner.plan(build_table_context(small_df), [])
    assert reply.intent == "clarify"
    assert reply.clarifying_question == "By how much?"


# --- ModelCall observability record ----------------------------------------

def test_model_call_records_usage_prompt_ref_and_latency(small_df):
    planner = make_planner(SimpleNamespace(parsed=GOOD_REPLY, text="",
                                           usage_metadata=USAGE))
    history = [{"role": "user", "text": "increase electronics prices by 10%"}]
    _, call = planner.plan(build_table_context(small_df), history)

    assert call.model == "stub-model"
    # Prompt is referenced, not copied: versioned system prompt + hashed
    # table context + conversation shape.
    assert call.system_prompt_ref == SYSTEM_PROMPT_REF
    assert len(call.table_context_sha1) == 12
    assert call.history_turns == 1
    assert call.last_message_preview.startswith("increase electronics")
    # Token accounting straight from the API's usage metadata.
    assert (call.input_tokens, call.output_tokens, call.total_tokens) == \
        (120, 30, 150)
    assert call.cost_usd == 0.0            # free tier: prices default to 0
    assert call.latency_ms >= 0


def test_model_call_degrades_gracefully_without_usage(small_df):
    """A response with no usage_metadata must yield None counts/cost,
    never an exception."""
    planner = make_planner(SimpleNamespace(parsed=GOOD_REPLY, text=""))
    _, call = planner.plan(build_table_context(small_df), [])
    assert call.input_tokens is None
    assert call.cost_usd is None
    assert call.latency_ms >= 0


def test_cost_uses_configured_prices(small_df, monkeypatch):
    import grid_agent.llm as llm
    monkeypatch.setattr(llm, "GEMINI_PRICE_INPUT_PER_1M", 0.10)
    monkeypatch.setattr(llm, "GEMINI_PRICE_OUTPUT_PER_1M", 0.40)
    planner = make_planner(SimpleNamespace(parsed=GOOD_REPLY, text="",
                                           usage_metadata=USAGE))
    _, call = planner.plan(build_table_context(small_df), [])
    # 120/1M * 0.10 + 30/1M * 0.40 = 0.000012 + 0.000012
    assert call.cost_usd == pytest.approx(0.000024)


def test_unparseable_reply_raises_planner_error(small_df):
    planner = make_planner(SimpleNamespace(parsed=None, text="I think you should…"))
    with pytest.raises(PlannerError, match="unparseable"):
        planner.plan(build_table_context(small_df), [])


def test_sdk_exception_is_wrapped(small_df):
    planner = make_planner(error=RuntimeError("quota exceeded"))
    with pytest.raises(PlannerError, match="quota exceeded"):
        planner.plan(build_table_context(small_df), [])


def test_request_contains_context_history_and_schema_config(small_df):
    planner = make_planner(SimpleNamespace(parsed=GOOD_REPLY, text=""))
    history = [{"role": "user", "text": "double all prices"},
               {"role": "model", "text": "By how much?"},
               {"role": "user", "text": "sorry — increase by 10%"}]
    planner.plan(build_table_context(small_df), history)

    kwargs = planner._client.models.generate_content.last_kwargs
    contents = kwargs["contents"]
    # Turn 0 is the live table; the conversation follows in order.
    assert "CURRENT TABLE" in contents[0].parts[0].text
    assert [c.role for c in contents[1:]] == ["user", "model", "user"]
    # Structured output is enforced at the decoding level.
    config = kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is WireReply
    assert config.temperature == 0.0
    assert "update_where" in config.system_instruction


# --- live smoke test (requires a real key; skipped otherwise) --------------

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"),
                    reason="GEMINI_API_KEY not set")
def test_live_gemini_returns_valid_wire_reply(small_df):
    planner = GeminiPlanner()
    reply, call = planner.plan(
        build_table_context(small_df),
        [{"role": "user",
          "text": "Increase electronics prices by 10%, then sort by price descending"}])
    assert reply.intent == "plan"
    kinds = [op.kind for op in reply.operations]
    assert "update_where" in kinds and "sort" in kinds
    # The live API reports real usage; the call record must capture it.
    assert call.input_tokens and call.input_tokens > 0
    assert call.output_tokens and call.output_tokens > 0
    assert call.latency_ms > 0
    assert call.cost_usd is not None
