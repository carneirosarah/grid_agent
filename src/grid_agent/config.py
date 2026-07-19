"""Central configuration.

Everything tunable lives here so the rest of the codebase never reads
environment variables directly. `.env` is loaded once at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# The root under which data/, frontend/ and traces/ live. Defaults to the
# repo root (two levels up from this file: src/grid_agent/config.py ->
# src -> root), which is correct for an editable install. When the package
# is installed into site-packages — as in the Docker image — that guess
# lands inside the interpreter tree, so the image sets GRID_AGENT_ROOT
# explicitly.
PROJECT_ROOT = Path(os.getenv("GRID_AGENT_ROOT")
                    or Path(__file__).resolve().parents[2])
load_dotenv(PROJECT_ROOT / ".env")

# --- Data ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "inventory.csv"
DATASET_SEED = 42          # deterministic dataset generation
DATASET_ROWS = 350         # requirement: at least 300 rows

# --- Tracing ---------------------------------------------------------------
TRACE_DIR = PROJECT_ROOT / "traces"
TRACE_PATH = TRACE_DIR / "trace.jsonl"

# --- LLM -------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# How many times the planner may retry after the semantic validator rejects
# its plan (feedback loop for "confidently wrong" model output).
MAX_REPAIR_ATTEMPTS = 2

# Per-million-token prices used to compute the cost recorded on each
# `model_call` trace event. The Gemini flash free tier costs nothing, so
# both default to 0 — set these when running on a paid tier.
GEMINI_PRICE_INPUT_PER_1M = float(os.getenv("GEMINI_PRICE_INPUT_PER_1M", "0"))
GEMINI_PRICE_OUTPUT_PER_1M = float(os.getenv("GEMINI_PRICE_OUTPUT_PER_1M", "0"))

# --- State -----------------------------------------------------------------
UNDO_STACK_LIMIT = 25      # snapshots kept for undo

# --- Sessions & persistence ------------------------------------------------
# Postgres DSN, e.g. postgresql://grid:grid@localhost:5432/grid_agent
# Empty -> sessions live in process memory only (dev/tests without a DB).
DATABASE_URL = os.getenv("DATABASE_URL", "")
SESSION_COOKIE = "grid_session"     # cookie carrying the per-user session id
SESSION_CACHE_LIMIT = 100           # live sessions kept in memory (LRU)
