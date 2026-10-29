"""seaborn + Matplotlib: four views of commute times. 40–60 minutes.

Use fictional, generated data to practise choosing and interpreting charts.
Run cells in order. Fill each TODO and label every chart before exporting it.
"""
# ruff: noqa: F401

# %% 0. Supplied reproducible data and output directory
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
modes = np.tile(["Walk", "Bike", "Bus"], 40)
distance = rng.uniform(1, 8, len(modes))
minutes_per_km = pd.Series(modes).map({"Walk": 12, "Bike": 4, "Bus": 3}).to_numpy()
commutes = pd.DataFrame(
    {
        "mode": modes,
        "distance_km": distance,
        "minutes": np.maximum(1, 5 + distance * minutes_per_km + rng.normal(0, 4, 120)),
    }
)
MODE_ORDER = ["Walk", "Bike", "Bus"]
sns.set_theme(style="whitegrid")

start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = next(
    (p for p in [start, *start.parents] if (p / "skills.md").is_file()), None
)
if PROJECT_ROOT is None:
    raise RuntimeError("Open this project in VS Code and run from inside it.")
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
print(commutes.head())

# %% 1. Set up an explicit layout
# TODO 1: Create a 2-by-2 grid of Axes with plt.subplots, figsize=(12, 8).
# Do not share axes: the panels have different units and meanings.
fig = ...
axes = ...
assert fig is not Ellipsis and axes is not Ellipsis, "Complete TODO 1."
assert axes.shape == (2, 2)

# %% 2. A distribution
# TODO 2: On axes[0, 0], use sns.histplot to plot minutes with 20 bins.
# Pass data, x, and ax explicitly. Add a title and labels with units.
# Write your plotting code here.
assert len(axes[0, 0].patches) > 0, "Complete TODO 2."
# Interpretation: does a pooled histogram hide differences between modes?

# %% 3. Compare groups
# TODO 3: On axes[0, 1], draw a boxplot of mode versus minutes.
# Use MODE_ORDER for the category order; label the chart.
# Write your plotting code here.
assert axes[0, 1].has_data(), "Complete TODO 3."
assert [tick.get_text() for tick in axes[0, 1].get_xticklabels()] == MODE_ORDER
# Interpretation: why does a boxplot not by itself establish which mode is faster
# for the same distance?

# %% 4. Check a potential confounder
# TODO 4: On axes[1, 0], plot distance_km versus minutes, colored by mode.
# Use sns.scatterplot; choose alpha < 1 and keep a readable legend.
# Write your plotting code here.
assert len(axes[1, 0].collections) > 0, "Complete TODO 4."
assert axes[1, 0].get_legend() is not None
# Interpretation: are the trends consistent with the pooled comparison?

# %% 5. Prepare a summary explicitly
# TODO 5: Calculate mean_minutes and n (row count) per mode.
# Return a DataFrame indexed by mode and reorder its rows with MODE_ORDER.
mode_summary = ...
assert mode_summary is not Ellipsis, "Complete TODO 5."
assert mode_summary.index.tolist() == MODE_ORDER
assert mode_summary["n"].tolist() == [40, 40, 40]

# %% 6. Plot an aggregation with Matplotlib
# TODO 6: Use axes[1, 1].bar to plot mode_summary["mean_minutes"].
# Keep the y-axis starting at zero, and label the chart including minutes.
# Write your plotting code here.
assert len(axes[1, 1].patches) == 3, "Complete TODO 6."
assert np.isclose(axes[1, 1].get_ylim()[0], 0)

# %% 7. Export a figure
# TODO 7: Give the figure an overall title, call fig.tight_layout(), and save it
# to FIGURE_DIR / "commute_overview.png" at 150 dpi before calling plt.show().
# Check the figure at full size: titles, tick labels, and legend must be legible.
for ax in axes.flat:
    assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel(), "Label every panel."
# Write layout, export, and display code here.
assert (FIGURE_DIR / "commute_overview.png").is_file(), "Complete TODO 7."

# %% [markdown]
# ## TODO 8: Tell the story
#
# - Give one finding supported by each panel.
# - How does distance affect the comparison between modes?
# - Which panel shows variation that the bar chart hides?
# - Why can't these synthetic observations support a real transport policy?
#
# Optional stretch: annotate each bar with its sample count, then make an ECDF
# comparison. Redraw after restarting the kernel to check reproducibility.
#
# VS Code practice: use Run Cell, then open the exported PNG next to the script.
# Reference: https://seaborn.pydata.org/tutorial/axis_grids.html
