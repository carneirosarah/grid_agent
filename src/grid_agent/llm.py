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

from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import GEMINI_API_KEY, GEMINI_MODEL
from .schemas import WireReply

# Behavioural rules for the model, versioned as a standalone Markdown file.
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system_prompt.md").read_text(
    encoding="utf-8")

# Low-cardinality string columns get their full vocabulary embedded in the
# prompt, which removes most misspelled-value errors at the source.
VOCABULARY_MAX_UNIQUES = 25


class PlannerError(RuntimeError):
    """The LLM call failed or returned something that is not a WireReply."""


class Planner(Protocol):
    """Anything that can turn a conversation into a structured reply."""

    def plan(self, table_context: str, history: list[dict[str, str]]) -> WireReply:
        """`history` items are {"role": "user"|"model", "text": ...}."""
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

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        if not api_key:
            raise PlannerError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "add your key (https://aistudio.google.com/apikey).")
        # Imported lazily so offline tests never touch the SDK.
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def plan(self, table_context: str, history: list[dict[str, str]]) -> WireReply:
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

        # 2. Call Gemini with decoding constrained to the WireReply schema.
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

        # 3. Normalise the result: parsed WireReply out, PlannerError
        #    otherwise — callers never see a half-parsed reply. The SDK
        #    usually fills `.parsed`; fall back to parsing the raw text.
        reply = getattr(response, "parsed", None)
        if isinstance(reply, WireReply):
            return reply
        try:
            return WireReply.model_validate_json(response.text or "")
        except Exception as exc:
            raise PlannerError(
                f"Gemini returned an unparseable reply: {response.text!r}") from exc
