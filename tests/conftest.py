"""Shared fixtures.

`small_df` is a 6-row miniature of the real inventory: same columns, same
dtypes. Unit tests run on this so failures are easy to eyeball; the e2e
tests use the real generated CSV.
"""

import pandas as pd
import pytest


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sku": [f"SKU-{i:04d}" for i in range(1, 7)],
        "name": ["Acme Mouse", "Zenith Desk", "Orbit Pen Set",
                 "Lumen Kettle", "Acme Hoodie", "Vertex Yoga Mat"],
        "category": ["Electronics", "Furniture", "Stationery",
                     "Kitchen", "Apparel", "Sports"],
        "supplier": ["Acme Corp", "Global Traders", "Acme Corp",
                     "PacRim Ltd", "Iberia Goods", "PacRim Ltd"],
        "price": [25.0, 300.0, 5.0, 40.0, 35.0, 20.0],
        "cost": [20.0, 150.0, 4.5, 30.0, 20.0, 10.0],
        "stock": [10, 5, 200, 50, 80, 30],
        "rating": [4.5, 3.0, 2.5, 4.0, 5.0, 1.5],
        "flagged": [False, False, False, False, False, False],
    })
