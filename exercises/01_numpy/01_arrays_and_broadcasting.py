"""NumPy: weekly shop sales. Beginner; allow 30–45 minutes.

Run the supplied setup, then complete TODOs in order with Run Cell.
Replace each ... with your code. Checkpoints fail until that task is complete.
Supplied imports are kept for later TODOs. Reset by copying the template again.
"""

# %% 0. Setup — rows are shops, columns are Monday through Sunday
import numpy as np

units = np.array(
    [
        [12, 15, 9, 0, 20, 25, 18],
        [8, 10, 12, 14, 16, 18, 20],
        [30, 28, 26, 24, 22, 20, 18],
        [5, 7, 9, 11, 13, 15, 17],
    ],
    dtype=float,
)
prices = np.array([2.0, 3.0, 1.5, 4.0])
shop_names = np.array(["North", "South", "East", "West"])
print("Units:", units.shape, units.dtype)
print("Prices:", prices.shape)

# %% 1. Slice a two-dimensional array
# TODO 1: Select Saturday and Sunday for all shops, using one slice.
weekend_units = units[:, -2:]
assert weekend_units is not Ellipsis, "Complete TODO 1."
assert weekend_units.shape == (4, 2)
np.testing.assert_array_equal(weekend_units[0], [25, 18])

# %% 2. Understand reduction axes
# TODO 2: Calculate totals per shop and per day, without a Python loop.
# Hint: sum along the dimension that you want to remove.
shop_totals = np.sum(units, axis=1)
daily_totals = np.sum(units, axis=0)
assert shop_totals is not Ellipsis and daily_totals is not Ellipsis
np.testing.assert_array_equal(shop_totals, [99, 98, 168, 77])
assert daily_totals.shape == (7,)
assert daily_totals.sum() == shop_totals.sum()

# %% 3. Boolean indexing
# TODO 3: Build a Boolean mask for shop-days with strictly more than 20 units.
# Then select the matching units. Is the result still two-dimensional?
busy_mask = units > 20
busy_units = units[busy_mask]
assert busy_mask is not Ellipsis and busy_units is not Ellipsis
assert busy_mask.shape == units.shape and busy_mask.dtype == bool
assert busy_units.ndim == 1 and busy_units.size == 6
assert np.all(busy_units > 20)

# %% 4. Broadcasting
# TODO 4: Multiply every shop's daily units by that shop's price.
# Hint: give prices a length-one day axis using reshape or np.newaxis.
revenue = units * prices.T[:, None]
assert revenue is not Ellipsis, "Complete TODO 4."
assert revenue.shape == units.shape
np.testing.assert_allclose(revenue.sum(axis=1), [198, 294, 252, 308])

# %% 5. Rank shops
# TODO 5: Use argsort to find shop indices in descending total-revenue order.
# Use those indices to select the shop names.
ranking = np.argsort(revenue.sum(axis=1))[::-1]
ranked_names = shop_names[ranking]
assert ranking is not Ellipsis and ranked_names is not Ellipsis
assert ranked_names.tolist() == ["West", "South", "East", "North"]

# %% 6. Standardize within each shop
# TODO 6: Calculate each row's mean and population std (ddof=0).
# Keep singleton dimensions so subtraction and division broadcast correctly.
row_means = np.mean(units, axis=1, keepdims=True)
row_stds = np.std(units, axis=1, keepdims=True)  # ddof = 0 by default
standardized = (units - row_means) / row_stds
assert standardized is not Ellipsis, "Complete TODO 6."
assert row_means.shape == row_stds.shape == (4, 1)
np.testing.assert_allclose(standardized.mean(axis=1), 0, atol=1e-12)
np.testing.assert_allclose(standardized.std(axis=1), 1)

# %% 7. Missing data
observations = units.copy()
observations[0, 2] = np.nan
observations[2, 4] = np.nan
# TODO 7: Count missing observations by shop and calculate row means ignoring NaN.
# Hint: isnan and nanmean. Compare mean with nanmean for the first row.
missing_counts = np.sum(np.isnan(observations), axis=1)
observed_means = np.nanmean(observations, axis=1)
assert missing_counts is not Ellipsis and observed_means is not Ellipsis
np.testing.assert_array_equal(missing_counts, [1, 0, 1, 0])
np.testing.assert_allclose(observed_means[0], 15.0)

# %% 8. Views versus copies
# TODO 8: Make an independent copy of the first shop's row and change its first
# value to -1. Demonstrate that units has not changed.
independent_row = units[0].copy()
independent_row[0] = -1
assert independent_row is not Ellipsis, "Complete TODO 8."
# Write your assignment here.
assert independent_row[0] == -1, "Modify your copy before running this check."
assert units[0, 0] == 12, "You changed a view of units; restart the setup."
assert not np.shares_memory(independent_row, units)

# %% [markdown]
# ## TODO 9: Explain your results
#
# - Why does a shape of (4,) fail to broadcast against (4, 7)?
# - Which axis means "shops" before and after a reduction?
# - What happens to standardization if one row has zero variance?
# - What is one benefit of comparing a vectorized result with a small loop?
#
# Optional stretch: write a zero-variance-safe standardization function. Test it
# on a row of identical values and state what you choose to return for that row.
#
# VS Code practice: set a breakpoint on the broadcasting task and inspect shapes.
# Reference: https://numpy.org/doc/stable/user/basics.broadcasting.html
