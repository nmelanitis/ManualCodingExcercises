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

from scipy import stats

print(Path.cwd())
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", context="notebook")

RANDOM_SEED = 42

# %% 2. Configuration
# TODO 1: Point this at a CSV inside data/raw/.
DATA_PATH = Path("../../data/raw/titanic.csv")
# path for interactive and non-interactive runs is not the same
# path non interactive "data/raw/titanic.csv"
# path interactive "../../data/raw/titanic.csv"

# TODO 2: Fill these after inspecting the columns. Use None when not applicable.
TARGET_COLUMN: str | None = "survived"
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

df: pd.DataFrame = pd.read_csv(DATA_PATH)  # type: ignore[assignment]

# %% 4. First look
# TODO 4: Display each of the following:
# - the number of rows and columns
# - the first five rows
# - five random rows using RANDOM_SEED
# - df.info()
#
# Question: What does each view reveal that the others do not?
df.info()
(r, c) = df.shape
print(f"Number of rows: {r}, Number of columns: {c}")
print(df.head(5))
sample_df = df.sample(5, random_state=RANDOM_SEED)
print(sample_df)

# Optional Data Wrangler checkpoint:
# If a visual profile would help, output df in the cell and choose "Open 'df' in
# Data Wrangler". Record anything that was easier to notice visually, then
# reproduce any important check with pandas below. This does not replace TODO 4.

# %% 5. Column roles and data types
# TODO 5: Create lists for numeric, categorical, boolean, and datetime columns.
# Useful tools: df.select_dtypes(...).columns.tolist()
numeric_columns: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
categorical_columns: list[str] = df.select_dtypes(include="bool").columns.tolist()
datetime_columns: list[str] = df.select_dtypes(include="datetime").columns.tolist()
str_columns: list[str] = df.select_dtypes(include="str").columns.tolist()
# TODO: Print the lists and investigate any column whose dtype looks wrong.
print(f"Numeric columns: {numeric_columns}")
print("Categorical columns:", categorical_columns)
print("Datetime columns:", datetime_columns)
print("String columns:", str_columns)
print(
    "Dont use prints; actually inspect the variables either with dataframe viewers (extensions)"
)
print("or through the jupyter tab next to terminal, debug console etc. Jupyter tab has")
print("a variable explorer/variable workspace that is very useful for this purpose.")

# %% 6. Data quality: missing values and duplicates
# TODO 6: Build a table with missing_count and missing_percent for every column.
# Sort it so the columns with the most missing data appear first.
print("Again this step is done with variable explorer or df viewer extensions")
print("just reporting some values with prints")
missing_summary: pd.DataFrame = df.isna().sum().to_frame(name="missing_count")
missing_summary["missing_percent"] = missing_summary["missing_count"] / len(df) * 100
print("Missing summary:")
print(missing_summary.sort_values("missing_percent", ascending=False))
# TODO: Count fully duplicated rows. Then decide whether duplicates are truly
# errors or valid repeated observations.
duplicate_count: int = df.duplicated().sum()  # type: ignore[assignment]
print("Duplicates are:", duplicate_count)
print("Since we dont have an ID column, we cannot distinguish real duplicates from")
print("valid repeated observations. So we will not drop duplicates for now.")
# %% 7. Numeric summaries
# TODO 7: Produce descriptive statistics for numeric columns and transpose the
# output so that one variable appears per row.
numeric_summary: pd.DataFrame = df[numeric_columns].describe().T
print("Numeric summary:", numeric_summary.round(2))


# %% 8. Categorical summaries
# TODO 8: For every categorical column, inspect:
# - number of unique values
# - most common values, including missing values
# - proportion of the most common category
#
# Start with one column manually before deciding whether a loop is useful.
c_smr: pd.DataFrame = df[categorical_columns].describe().T
c_smr_agg: pd.DataFrame = (
    df[categorical_columns]
    .agg(["nunique", "count", lambda x: x.value_counts().iloc[0] / x.count()])
    .T.rename(columns={"<lambda_0>": "most_common_proportion"})
)
# %% 9. Validate ranges and domain rules
# TODO 9: Write at least three checks based on what the columns mean.
# Examples: impossible ages, negative prices, end dates before start dates, or
# category values outside an expected set.
#
# Use assertions only for rules that must always be true. Otherwise, create a
# Boolean mask and inspect the suspicious rows.
assert ((df["age"] >= 0) | (df["age"].isna())).all(), (
    "Age must be non-negative or missing"
)

# %% 10. Univariate numeric distributions
# TODO 10: Choose one important numeric variable and draw a histogram with KDE.
PLOT_COLUMN: str | None = "fare"

if PLOT_COLUMN is not None:
    # TODO: Create a figure and call sns.histplot(...).
    # Add a title, x label, and tight layout. Then display it with plt.show().
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df[PLOT_COLUMN], kde=True, ax=ax)
    ax.set_title(f"Distribution of {PLOT_COLUMN}")
    plt.xlabel(PLOT_COLUMN)
    plt.tight_layout()
    plt.show()

# Reflection: Is the distribution symmetric, skewed, multimodal, or truncated?

# %% 11. Categorical distributions
# TODO 11: Choose a categorical variable, calculate its ten most common values,
# and make a horizontal count plot. Decide how missing values should appear.

PLOT_CAT: str | None = "embark_town"
n = 10

if PLOT_CAT is not None:
    # This can be implemented with sns.countplot(...)
    topn = df[PLOT_CAT].value_counts().nlargest(n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=topn.values, y=topn.index, ax=ax)

    missing_count = df[PLOT_CAT].isna().sum()
    ax.barh(y=["Missing"], width=missing_count, color="red")
    ax.set_yticks(range(len(topn) + 1))
    ax.set_yticklabels([*topn.index, "Missing"])

    ax.set_title(f"Top {n} most common values of {PLOT_CAT}")
    plt.xlabel("Count")
    plt.ylabel(PLOT_CAT)
    plt.tight_layout()
    plt.show()


# %% 12. Potential outliers
# TODO 12: For one numeric column, calculate Q1, Q3, IQR, and the standard
# 1.5-IQR lower and upper fences. Inspect the rows beyond those fences.
OUTLIER_COLUMN: str | None = "fare"

if OUTLIER_COLUMN is not None:
    values = df[OUTLIER_COLUMN]
    q1: float = values.quantile(0.25)
    q3: float = values.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # TODO: Create outlier_mask and use it to inspect suspicious rows.
    outlier_mask: pd.Series = (values < lower_fence) | (values > upper_fence)
    df_outliers = df.loc[outlier_mask, ["age", "fare", "survived"]]

    # make a nice boxplot too
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=values, ax=ax)
    ax.set_title(f"Boxplot of {OUTLIER_COLUMN} with outliers")
    plt.tight_layout()
    plt.show()
# Reflection: An outlier is not automatically an error. What could explain it?

# %% 13. Relationships between numeric variables
# TODO 13: Calculate Pearson and Spearman correlation matrices. Compare them.
# Plot a heatmap for a useful subset instead of every numeric column if the
# dataset is wide.
pearson_correlation: stats.SignificanceResult = stats.pointbiserialr(
    df["survived"], df["fare"], nan_policy="omit"
)

print(
    f"We got a pearson correlation of {pearson_correlation.correlation} with a p-value of {pearson_correlation.pvalue}"
)
# TODO: Pick one interesting pair and draw a scatter plot with a fitted trend.
# Useful function: sns.regplot(...)
sns.regplot(data=df, x="fare", y="survived", logistic=True, y_jitter=0.03)
plt.show()
print(
    "Observe how nicely the prob of survival rises with fare, however with greater uncertainty in high fare because of fewer data points"
)

# %% 14. Target relationships
TARGET_COLUMN: str | None = "survived"
FEATURE_L: list(str) = ["fare", "age", "sex"]
if TARGET_COLUMN is not None:
    # TODO 14: Choose plots based on the target type:
    # - numeric target vs numeric feature: scatter/regression plot
    # - numeric target vs category: box or violin plot
    # - categorical target vs numeric feature: grouped box/violin plot
    # - categorical target vs category: normalized crosstab/heatmap
    numeric_features = [f for f in FEATURE_L if f in numeric_columns]
    categorical_features = [f for f in FEATURE_L if f in str_columns]
    for feature in numeric_features:
        sns.regplot(data=df, x=feature, y=TARGET_COLUMN, logistic=True, y_jitter=0.03)
        plt.title(f"Relationship between {feature} and {TARGET_COLUMN}")
        plt.show()
        sns.boxplot(data=df, y=feature, x=TARGET_COLUMN)
        plt.title(f"Relationship between {feature} and {TARGET_COLUMN}")
        plt.show()

    for feature in categorical_features:
        crosstab = pd.crosstab(
            df[feature], df[TARGET_COLUMN], margins=True, normalize="index"
        )
        sns.heatmap(
            crosstab,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",  # good for values from 0 to 1
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Proportion"},
        )
        plt.title(f"Relationship between {feature} and {TARGET_COLUMN}")
        plt.show()
# %% 15. Grouped questions
# TODO 15: Write one groupby aggregation that answers a real question.
# Prefer named aggregations so the output columns are clear.
# Example shape (replace names):
# grouped_summary = df.groupby("group").agg(
#     mean_value=("value", "mean"),
#     median_value=("value", "median"),
#     observations=("value", "size"),
# )
grouped_summary: pd.DataFrame = df.groupby("sex").agg(
    mean_survival=("survived", "mean"),
    mean_fare=("fare", "mean"),
    survived_count=("survived", "sum"),
    observations=("survived", "size"),
)  # type: ignore[assignment]

# %% 16. Optional statsmodels relationship check
# TODO 16: If a linear relationship is relevant, write a formula and fit an OLS
# model. This is exploratory evidence, not proof of causation.
OLS_FORMULA: str | None = (
    "survived ~ sex+fare+age"  # Example: "target ~ feature_1 + C(category)"
)

if OLS_FORMULA is not None:
    ols_model = smf.logit(formula=OLS_FORMULA, data=df).fit()
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
