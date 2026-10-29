"""Guided exploratory data analysis worksheet.

How to use this file:
1. Copy it into ``exercises/03_visualization/`` and rename the copy.
2. Change the configuration values below.
3. Replace each ``...`` or write beneath each TODO, one cell at a time.
4. Run cells with Shift+Enter in VS Code.

This is intentionally incomplete. The goal is to type the important pandas and
plotting operations yourself, not only to execute a finished report.
"""

# %% [markdown]
# # Exploratory Data Analysis
#
# **Question:** What do I want to learn from this dataset?
#
# TODO 0: Write one primary analysis question and two supporting questions.

# %% 1. Imports and display settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", context="notebook")

RANDOM_SEED = 42

# %% 2. Configuration
# TODO 1: Point this at a CSV inside data/raw/.
DATA_PATH = Path("data/raw/your_dataset.csv")

# TODO 2: Fill these after inspecting the columns. Use None when not applicable.
TARGET_COLUMN: str | None = None
ID_COLUMNS: list[str] = []
DATE_COLUMNS: list[str] = []

FIGURE_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# %% 3. Load the data
# TODO 3: Read the CSV into a DataFrame named df.
# Hint: start with pd.read_csv(DATA_PATH). Add parse_dates=DATE_COLUMNS later.
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. Update DATA_PATH before continuing."
    )

df: pd.DataFrame = ...  # type: ignore[assignment]

# %% 4. First look
# TODO 4: Display each of the following:
# - the number of rows and columns
# - the first five rows
# - five random rows using RANDOM_SEED
# - df.info()
#
# Question: What does each view reveal that the others do not?

# Optional Data Wrangler checkpoint:
# If a visual profile would help, output df in the cell and choose "Open 'df' in
# Data Wrangler". Record anything that was easier to notice visually, then
# reproduce any important check with pandas below. This does not replace TODO 4.

# %% 5. Column roles and data types
# TODO 5: Create lists for numeric, categorical, boolean, and datetime columns.
# Useful tools: df.select_dtypes(...).columns.tolist()
numeric_columns: list[str] = ...  # type: ignore[assignment]
categorical_columns: list[str] = ...  # type: ignore[assignment]

# TODO: Print the lists and investigate any column whose dtype looks wrong.

# %% 6. Data quality: missing values and duplicates
# TODO 6: Build a table with missing_count and missing_percent for every column.
# Sort it so the columns with the most missing data appear first.
missing_summary: pd.DataFrame = ...  # type: ignore[assignment]

# TODO: Count fully duplicated rows. Then decide whether duplicates are truly
# errors or valid repeated observations.
duplicate_count: int = ...  # type: ignore[assignment]

# %% 7. Numeric summaries
# TODO 7: Produce descriptive statistics for numeric columns and transpose the
# output so that one variable appears per row.
numeric_summary: pd.DataFrame = ...  # type: ignore[assignment]

# TODO: Add range and interquartile range (IQR) columns to numeric_summary.
# Hint: use quantiles at 0.25 and 0.75.

# %% 8. Categorical summaries
# TODO 8: For every categorical column, inspect:
# - number of unique values
# - most common values, including missing values
# - proportion of the most common category
#
# Start with one column manually before deciding whether a loop is useful.

# %% 9. Validate ranges and domain rules
# TODO 9: Write at least three checks based on what the columns mean.
# Examples: impossible ages, negative prices, end dates before start dates, or
# category values outside an expected set.
#
# Use assertions only for rules that must always be true. Otherwise, create a
# Boolean mask and inspect the suspicious rows.

# %% 10. Univariate numeric distributions
# TODO 10: Choose one important numeric variable and draw a histogram with KDE.
PLOT_COLUMN: str | None = None

if PLOT_COLUMN is not None:
    # TODO: Create a figure and call sns.histplot(...).
    # Add a title, x label, and tight layout. Then display it with plt.show().
    ...

# Reflection: Is the distribution symmetric, skewed, multimodal, or truncated?

# %% 11. Categorical distributions
# TODO 11: Choose a categorical variable, calculate its ten most common values,
# and make a horizontal count plot. Decide how missing values should appear.

# %% 12. Potential outliers
# TODO 12: For one numeric column, calculate Q1, Q3, IQR, and the standard
# 1.5-IQR lower and upper fences. Inspect the rows beyond those fences.
OUTLIER_COLUMN: str | None = None

if OUTLIER_COLUMN is not None:
    q1: float = ...  # type: ignore[assignment]
    q3: float = ...  # type: ignore[assignment]
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # TODO: Create outlier_mask and use it to inspect suspicious rows.
    outlier_mask: pd.Series = ...  # type: ignore[assignment]

# Reflection: An outlier is not automatically an error. What could explain it?

# %% 13. Relationships between numeric variables
# TODO 13: Calculate Pearson and Spearman correlation matrices. Compare them.
# Plot a heatmap for a useful subset instead of every numeric column if the
# dataset is wide.
pearson_correlation: pd.DataFrame = ...  # type: ignore[assignment]
spearman_correlation: pd.DataFrame = ...  # type: ignore[assignment]

# TODO: Pick one interesting pair and draw a scatter plot with a fitted trend.
# Useful function: sns.regplot(...)

# %% 14. Target relationships
if TARGET_COLUMN is not None:
    # TODO 14: Choose plots based on the target type:
    # - numeric target vs numeric feature: scatter/regression plot
    # - numeric target vs category: box or violin plot
    # - categorical target vs numeric feature: grouped box/violin plot
    # - categorical target vs category: normalized crosstab/heatmap
    ...

# %% 15. Grouped questions
# TODO 15: Write one groupby aggregation that answers a real question.
# Prefer named aggregations so the output columns are clear.
# Example shape (replace names):
# grouped_summary = df.groupby("group").agg(
#     mean_value=("value", "mean"),
#     median_value=("value", "median"),
#     observations=("value", "size"),
# )
grouped_summary: pd.DataFrame = ...  # type: ignore[assignment]

# %% 16. Optional statsmodels relationship check
# TODO 16: If a linear relationship is relevant, write a formula and fit an OLS
# model. This is exploratory evidence, not proof of causation.
OLS_FORMULA: str | None = None  # Example: "target ~ feature_1 + C(category)"

if OLS_FORMULA is not None:
    ols_model = smf.ols(formula=OLS_FORMULA, data=df).fit()
    print(ols_model.summary())

    # TODO: Check residual plots and state which OLS assumptions may be weak.

# %% 17. Save useful outputs
# TODO 17: Save one final figure and one useful summary table.
# Hints:
# figure.savefig(FIGURE_DIR / "descriptive_name.png", dpi=150, bbox_inches="tight")
# summary.to_csv(TABLE_DIR / "descriptive_name.csv", index=True)

# %% [markdown]
# ## EDA conclusions
#
# TODO 18: Write concise answers for all five prompts.
#
# 1. What are the three most important findings?
# 2. Which data-quality issues require action?
# 3. Which relationships deserve deeper analysis?
# 4. Which transformations or features might be useful later?
# 5. What can this dataset *not* answer?
