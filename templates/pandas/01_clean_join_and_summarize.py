"""pandas: clean a small order ledger. Beginner/intermediate; 45–60 minutes.

The data below is fictional and intentionally messy. Work cell by cell.
Replace ... and follow the checkpoints; no CSV download is needed.
"""

# %% 0. Supplied data — one row should represent one order
import numpy as np
import pandas as pd

raw_orders = pd.DataFrame(
    {
        "order_id": [101, 102, 103, 104, 104, 105, 106, 107, 108],
        "customer_id": [1, 1, 2, 2, 2, 3, 3, 4, 99],
        "date": [
            "2026-01-03",
            "2026-01-04",
            "2026-01-04",
            "2026-01-05",
            "2026-01-05",
            "not a date",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ],
        "category": [
            " Books ",
            "books",
            " Games",
            "games",
            "games",
            " Books",
            "games",
            " Books",
            "books",
        ],
        "quantity": [2, 1, None, 3, 3, 1, 2, 1, 1],
        "unit_price": ["10", "12", "20", "15", "15", "bad", "25", "8", "9"],
        "status": ["paid"] * 7 + ["cancelled", "paid"],
    }
)
customers = pd.DataFrame(
    {"customer_id": [1, 2, 3, 4], "city": ["Athens", "Rome", "Paris", "Berlin"]}
)
print(raw_orders.head())
print(raw_orders.dtypes)

# %% 1. Work on a copy and normalize strings
# TODO 1: Copy raw_orders to orders. Strip whitespace and lowercase category.
orders = ...
assert orders is not Ellipsis, "Complete TODO 1."
# Write the string-cleaning assignment here.
assert set(orders["category"]) == {"books", "games"}
assert raw_orders.loc[0, "category"] == " Books "

# %% 2. Convert dirty columns
# TODO 2: Convert date to datetime and unit_price to a numeric dtype.
# Use errors="coerce" for both; inspect what happens to invalid input.
# Write your conversions here.
assert pd.api.types.is_datetime64_any_dtype(orders["date"]), "Complete TODO 2."
assert pd.api.types.is_numeric_dtype(orders["unit_price"])
assert orders["date"].isna().sum() == orders["unit_price"].isna().sum() == 1

# %% 3. Deduplicate using a business key
# TODO 3: Inspect duplicated order_id values, then keep the first row per order_id.
# For this dataset the repeated rows are identical. In real data, check conflicts.
deduplicated = ...
assert deduplicated is not Ellipsis, "Complete TODO 3."
assert len(deduplicated) == 8 and deduplicated["order_id"].is_unique

# %% 4. Apply an explicit quality policy
# TODO 4: Keep paid orders with a valid date, quantity, and unit_price.
# Reject incomplete orders here; do not silently impute their revenue.
# Hint: dropna(subset=...) and .loc[mask].copy().
valid_orders = ...
assert valid_orders is not Ellipsis, "Complete TODO 4."
assert set(valid_orders["order_id"]) == {101, 102, 104, 106, 108}

# %% 5. Compute a vectorized column
# TODO 5: Add revenue = quantity * unit_price to valid_orders.
# Write your assignment here.
assert "revenue" in valid_orders, "Complete TODO 5."
np.testing.assert_allclose(valid_orders["revenue"].sum(), 136.0)

# %% 6. Join without multiplying rows
# TODO 6: Left-join customers on customer_id, with validate="many_to_one" and
# indicator=True. Preserve orders whose customer has no lookup entry.
enriched = ...
assert enriched is not Ellipsis, "Complete TODO 6."
assert len(enriched) == len(valid_orders)
assert (enriched["_merge"] == "left_only").sum() == 1
assert enriched.loc[enriched["customer_id"] == 99, "city"].isna().all()
# Optional Data Wrangler: visually inspect the unmatched order. Verify it in code.

# %% 7. Named aggregations
# TODO 7: Group enriched by category. Create a table indexed by category with
# columns total_revenue (sum), order_count (size), and mean_order_value (mean).
category_summary = ...
assert category_summary is not Ellipsis, "Complete TODO 7."
np.testing.assert_allclose(category_summary.loc["books", "total_revenue"], 41)
np.testing.assert_allclose(category_summary.loc["games", "total_revenue"], 95)
assert category_summary["order_count"].sum() == 5

# %% 8. Keep the unknown customer visible
# TODO 8: Fill missing city names with "Unknown", then pivot revenue into a table:
# city on rows, category on columns, aggregation sum, missing cells filled with 0.
# Hint: fillna and pivot_table. Avoid dropping revenue from unmatched customers.
city_revenue = ...
assert city_revenue is not Ellipsis, "Complete TODO 8."
assert "Unknown" in city_revenue.index
np.testing.assert_allclose(city_revenue.to_numpy().sum(), 136.0)

# %% [markdown]
# ## TODO 9: Explain the cleaning decisions
#
# - Which orders were removed, and why? Report the lost row count at each step.
# - Why would an inner join have silently lost a valid order?
# - Why did this exercise reject missing quantities instead of imputing them?
# - How could duplicate customer keys inflate reported revenue?
#
# Optional stretch: deliberately duplicate a customer row and confirm that merge
# validation rejects it. Then produce daily revenue with a datetime index.
#
# VS Code practice: inspect orders in the Jupyter Variables/Data Viewer and use
# F2 to rename an intermediate variable.
# Reference: https://pandas.pydata.org/docs/user_guide/merging.html
