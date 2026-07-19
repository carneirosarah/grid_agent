"""Step 7 — Gemini API integration.

One job: turn (table context + chat history) into a `WireReply`, using
Gemini's *structured output* mode so the model is constrained to our wire
schema at the decoding level — it cannot answer with free text or invent
operation types.

The behavioural rules live in `prompts/system_prompt.md` (kept as Markdown
so they can be reviewed and edited without touching code).

Layering note: everything else in the app depends only on the `Planner`
protocol, never on Gemini directly. Tests inject a `FakePlanner`
(tests/test_e2e.py) and the whole pipeline runs offline; swapping vendors
means rewriting this file only.

Trust boundary: even though decoding is schema-constrained, the reply is
still *content*-untrusted (wrong columns, misspelled values, wrong
factors). That is handled downstream by wire_to_plan + validate_plan, not
here. This module only guarantees "syntactically valid WireReply or
PlannerError" — it never lets a half-parsed reply escape.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_PRICE_INPUT_PER_1M,
    GEMINI_PRICE_OUTPUT_PER_1M,
)
from .metrics import PassRateCounter
from .schemas import WireReply

# Behavioural rules for the model, versioned as a standalone Markdown file.
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system_prompt.md").read_text(
    encoding="utf-8")

# Content-addressed reference to the system prompt: lets a trace reader
# know exactly which prompt version produced a call without storing the
# full text on every event.
SYSTEM_PROMPT_REF = ("prompts/system_prompt.md@"
                     + hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:8])

# Low-cardinality string columns get their full vocabulary embedded in the
# prompt, which removes most misspelled-value errors at the source.
VOCABULARY_MAX_UNIQUES = 25


class PlannerError(RuntimeError):
    """The LLM call failed or returned something that is not a WireReply."""


@dataclass
class ModelCall:
    """Observability record for one LLM invocation, logged verbatim as a
    `model_call` trace event by the caller (which owns the session-bound
    tracer — the planner itself is shared across sessions).

    The prompt is captured by *reference*, not by value: the system prompt
    as file+content-hash, the table context as a hash (it can be rebuilt
    from the table state the surrounding events describe), plus the
    conversation size and a preview of the newest message. Full prompt
    text on every event would bloat the trace without adding information.
    """
    model: str
    system_prompt_ref: str          # e.g. "prompts/system_prompt.md@1a2b3c4d"
    table_context_sha1: str         # hash of the table description sent
    history_turns: int              # conversation turns included
    last_message_preview: str       # first 200 chars of the newest turn
    input_tokens: int | None        # None when the API reported no usage
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None          # tokens x configured prices; None if unknown
    latency_ms: float


def _cost_usd(input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Cost from configured per-1M prices (0 on the free tier)."""
    if input_tokens is None or output_tokens is None:
        return None
    return round(input_tokens / 1e6 * GEMINI_PRICE_INPUT_PER_1M
                 + output_tokens / 1e6 * GEMINI_PRICE_OUTPUT_PER_1M, 6)


class Planner(Protocol):
    """Anything that can turn a conversation into a structured reply."""

    def plan(self, table_context: str,
             history: list[dict[str, str]]) -> tuple[WireReply, ModelCall]:
        """`history` items are {"role": "user"|"model", "text": ...}.
        Returns the structured reply plus the observability record of the
        call that produced it."""
        ...


def build_table_context(df: pd.DataFrame) -> str:
    """Describe the live table to the model: shape, columns with dtypes and
    ranges, and the exact vocabulary of small text columns. Rebuilt every
    turn so the model always plans against current data."""
    lines = [f"The table has {len(df)} rows. Columns:"]
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            desc = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            desc = f"numeric, min={series.min()}, max={series.max()}"
        else:
            uniques = series.dropna().unique()
            if len(uniques) <= VOCABULARY_MAX_UNIQUES:
                desc = "text, values: " + ", ".join(sorted(map(str, uniques)))
            else:
                sample = ", ".join(map(str, uniques[:3]))
                desc = f"text, {len(uniques)} distinct values, e.g. {sample}"
        lines.append(f"- {col}: {desc}")
    return "\n".join(lines)


class GeminiPlanner:
    """Planner backed by the google-genai SDK with structured output."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL,
                 validity: PassRateCounter | None = None):
        if not api_key:
            raise PlannerError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "add your key (https://aistudio.google.com/apikey).")
        # Imported lazily so offline tests never touch the SDK.
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        # Structured Output Validity Rate tally (see metrics.py for the
        # counting rules); exposed by the API at GET /api/metrics.
        self.validity = validity or PassRateCounter()

    def plan(self, table_context: str,
             history: list[dict[str, str]]) -> tuple[WireReply, ModelCall]:
        from google.genai import types

        # 1. Assemble the conversation: the fresh table context rides on
        #    the first user turn, then the chat history follows in order
        #    (including any repair-loop error reports).
        contents = [
            types.Content(role="user", parts=[types.Part(
                text=f"CURRENT TABLE:\n{table_context}")]),
        ]
        for message in history:
            contents.append(types.Content(
                role=message["role"], parts=[types.Part(text=message["text"])]))

        # 2. Call Gemini with decoding constrained to the WireReply schema,
        #    timing the round-trip for the model_call trace event.
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=WireReply,
                    temperature=0.0,     # planning should be reproducible
                ),
            )
        except Exception as exc:                      # network/auth/quota
            raise PlannerError(f"Gemini request failed: {exc}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        # 3. Build the observability record. Token usage comes from the
        #    API's own accounting; read defensively — a missing field must
        #    degrade to None, never break planning.
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        last_text = history[-1]["text"] if history else ""
        call = ModelCall(
            model=self._model,
            system_prompt_ref=SYSTEM_PROMPT_REF,
            table_context_sha1=hashlib.sha1(table_context.encode()).hexdigest()[:12],
            history_turns=len(history),
            last_message_preview=last_text[:200],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=_cost_usd(input_tokens, output_tokens),
            latency_ms=latency_ms,
        )

        # 4. Normalise the result: parsed WireReply out, PlannerError
        #    otherwise — callers never see a half-parsed reply. The SDK
        #    usually fills `.parsed`; fall back to parsing the raw text.
        #    Either way the verdict feeds the Structured Output Validity
        #    Rate — a response reached this point, so Pydantic gets to
        #    judge it (transport failures above are deliberately not
        #    counted: there was no response to judge).
        reply = getattr(response, "parsed", None)
        if isinstance(reply, WireReply):
            self.validity.record(True)
            return reply, call
        try:
            reply = WireReply.model_validate_json(response.text or "")
        except Exception as exc:
            self.validity.record(False)
            raise PlannerError(
                f"Gemini returned an unparseable reply: {response.text!r}") from exc
        self.validity.record(True)
        return reply, call
