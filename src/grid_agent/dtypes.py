"""Column-dtype predicates shared by the engine and the validator.

Centralised because pandas' notion of "a string column" changed across
major versions (pandas 3 uses the Arrow-backed ``str`` dtype, older
versions used ``object``) — that compatibility knowledge should live in
exactly one place.
"""

from __future__ import annotations

import pandas as pd


def is_string_col(col: pd.Series) -> bool:
    """True for text columns across pandas versions."""
    return pd.api.types.is_string_dtype(col) or col.dtype == object


def is_numeric_col(col: pd.Series) -> bool:
    """True for real numeric columns (bool is numeric to pandas, not to us:
    multiplying a flag column makes no sense)."""
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
