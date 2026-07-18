"""Central configuration.

Everything tunable lives here so the rest of the codebase never reads
environment variables directly. `.env` is loaded once at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from the project root (two levels up from this file:
# src/grid_agent/config.py -> src -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

# --- State -----------------------------------------------------------------
UNDO_STACK_LIMIT = 25      # snapshots kept for undo
