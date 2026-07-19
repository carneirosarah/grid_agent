"""grid_agent — a two-panel app where an LLM agent edits a data table.

The agent never regenerates table contents. It emits *structured operations*
(`update_where`, `sort`) that the deterministic engine applies to a pandas
DataFrame. Pending plans are previewed, and applied plans can be undone.
"""

__version__ = "0.2.0"
