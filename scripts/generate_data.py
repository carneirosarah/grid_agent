"""Step 1 — Data generation.

Generates a deterministic product-inventory dataset (350 rows x 9 columns)
and writes it to `data/inventory.csv`.

Columns
-------
- sku       : unique row identifier, e.g. "SKU-0001" (stable across sorts,
              used by the diff/undo machinery and the frontend)
- name      : product display name
- category  : one of 6 categories (Electronics, Furniture, ...)
- supplier  : one of 8 suppliers
- price     : retail price (float, 2 decimals)
- cost      : acquisition cost (float, always below price so margins are sane)
- stock     : units on hand (int)
- rating    : average review score 1.0-5.0 (float, 1 decimal)
- flagged   : boolean marker the agent can set via `update_where`
              (e.g. "flag every low-margin row")

Run:
    python scripts/generate_data.py

The generator is seeded (see config.DATASET_SEED) so every run reproduces
the exact same file — important for reproducible tests and traces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a plain script: put `src/` on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_agent.config import DATASET_PATH, DATASET_ROWS, DATASET_SEED

# Vocabulary used to compose realistic-looking product names per category.
CATEGORIES: dict[str, list[str]] = {
    "Electronics": ["Wireless Mouse", "USB-C Hub", "Bluetooth Speaker", "Webcam",
                    "Mechanical Keyboard", "Monitor Stand", "Power Bank"],
    "Furniture": ["Office Chair", "Standing Desk", "Bookshelf", "Filing Cabinet",
                  "Desk Lamp", "Monitor Arm"],
    "Stationery": ["Notebook Pack", "Gel Pen Set", "Sticky Notes", "Stapler",
                   "Whiteboard Markers", "Paper Ream"],
    "Kitchen": ["Coffee Maker", "Electric Kettle", "Mug Set", "Toaster",
                "Water Filter", "Lunch Box"],
    "Apparel": ["Hoodie", "T-Shirt", "Cap", "Fleece Jacket", "Polo Shirt"],
    "Sports": ["Yoga Mat", "Resistance Bands", "Water Bottle", "Jump Rope",
               "Foam Roller"],
}

BRANDS = ["Acme", "Nordika", "Zenith", "Orbit", "Lumen", "Vertex", "Pioneer", "Atlas"]
SUPPLIERS = ["Acme Corp", "Global Traders", "Nordic Supply", "PacRim Ltd",
             "Iberia Goods", "Delta Wholesale", "Summit Partners", "EastBridge"]

# Category-level price bands: (min_price, max_price). Cost is derived from
# price with a margin factor, guaranteeing cost < price for every row.
PRICE_BANDS: dict[str, tuple[float, float]] = {
    "Electronics": (15.0, 250.0),
    "Furniture": (40.0, 600.0),
    "Stationery": (2.0, 30.0),
    "Kitchen": (10.0, 120.0),
    "Apparel": (8.0, 80.0),
    "Sports": (5.0, 90.0),
}


def generate(rows: int = DATASET_ROWS, seed: int = DATASET_SEED) -> pd.DataFrame:
    """Build the inventory DataFrame. Pure function of (rows, seed)."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    category_names = list(CATEGORIES)
    for i in range(rows):
        # 1. Pick a category, then a product + brand within it.
        category = category_names[rng.integers(len(category_names))]
        product = CATEGORIES[category][rng.integers(len(CATEGORIES[category]))]
        brand = BRANDS[rng.integers(len(BRANDS))]

        # 2. Price is uniform inside the category band; cost is price minus a
        #    margin between 5% and 60%, so margins vary enough to query on.
        lo, hi = PRICE_BANDS[category]
        price = round(float(rng.uniform(lo, hi)), 2)
        margin_pct = float(rng.uniform(0.05, 0.60))
        cost = round(price * (1.0 - margin_pct), 2)

        records.append({
            "sku": f"SKU-{i + 1:04d}",
            "name": f"{brand} {product}",
            "category": category,
            "supplier": SUPPLIERS[rng.integers(len(SUPPLIERS))],
            "price": price,
            "cost": cost,
            "stock": int(rng.integers(0, 500)),
            "rating": round(float(rng.uniform(1.0, 5.0)), 1),
            "flagged": False,
        })

    return pd.DataFrame.from_records(records)


def main() -> None:
    df = generate()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns -> {DATASET_PATH}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
