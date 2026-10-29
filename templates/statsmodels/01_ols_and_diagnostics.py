"""statsmodels: interpret a linear model and check assumptions. 60–75 minutes.

These are synthetic learning-session scores on an unbounded numerical scale.
This is an inference/diagnostics exercise, not a held-out prediction benchmark.
"""
# ruff: noqa: F401

# %% 0. Supplied data — noise grows with training hours
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
n = 180
hours = rng.uniform(1, 8, n)
experience = rng.uniform(0, 10, n)
session_format = rng.choice(["solo", "group"], n)
noise = rng.normal(0, 0.5 + 0.9 * hours, n)
sessions = pd.DataFrame(
    {
        "training_hours": hours,
        "prior_experience": experience,
        "session_format": session_format,
        "score": 30
        + 4 * hours
        + 1.5 * experience
        + 3 * (session_format == "group")
        + noise,
    }
)
start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = next(
    (p for p in [start, *start.parents] if (p / "skills.md").is_file()), None
)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from inside the project.")
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# %% 1. Write the model before fitting it
# TODO 1: Define a formula for score using training_hours, prior_experience, and
# a categorical session_format term. Keep an intercept.
# Hint: use C(...) for an explicitly categorical predictor.
formula = ...
assert isinstance(formula, str), "Complete TODO 1."
# Write your expected signs and a sentence about each coefficient here.

# %% 2. Fit and read OLS
# TODO 2: Use smf.ols(..., data=sessions).fit(), then print the summary.
ols_result = ...
assert ols_result is not Ellipsis, "Complete TODO 2."
assert ols_result.nobs == len(sessions)
assert "Intercept" in ols_result.params and len(ols_result.params) == 4
print(ols_result.summary())

# %% 3. Extract interpretable quantities
# TODO 3: Extract the training_hours coefficient and its 95% confidence interval.
# Hint: params and conf_int(). The known generating slope is 4, but sampling
# variation means your estimate will not be exactly 4.
hours_coefficient = ...
hours_interval = ...
assert hours_coefficient is not Ellipsis and hours_interval is not Ellipsis
assert len(hours_interval) == 2
assert hours_interval.iloc[0] < hours_coefficient < hours_interval.iloc[1]
# Write what "holding the other included predictors fixed" means here.

# %% 4. Inspect residuals
# TODO 4: Extract residuals and fitted values. On the first Axes draw residuals
# against fitted values plus a horizontal zero line. On the second, use
# sm.qqplot(residuals, fit=True, line="45", ax=...) for a normal Q-Q comparison.
residuals = ...
fitted_values = ...
assert residuals is not Ellipsis and fitted_values is not Ellipsis
np.testing.assert_allclose(fitted_values + residuals, sessions["score"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# Write both plots, titles, labels, and fig.tight_layout() here.
assert axes[0].has_data() and axes[1].has_data(), "Complete the TODO 4 plots."
# Describe any funnel shape, curvature, or heavy tails before using a test.

# %% 5. Test a specific assumption
# TODO 5: Call het_breuschpagan with residuals and ols_result.model.exog.
# It returns LM statistic, LM p-value, F statistic, and F p-value.
# Store the result, then extract and print the LM p-value.
bp_result = ...
lm_pvalue = ...
assert bp_result is not Ellipsis and lm_pvalue is not Ellipsis, "Complete TODO 5."
assert len(bp_result) == 4 and 0 <= lm_pvalue <= 1
print("Breusch–Pagan LM p-value:", lm_pvalue)
# State the null hypothesis. A large p-value does not prove constant variance.

# %% 6. Compare robust and ordinary standard errors
# TODO 6: Fit the same formula again with .fit(cov_type="HC3").
# Compare params, bse, pvalues, and conf_int with the ordinary result.
robust_result = ...
assert robust_result is not Ellipsis, "Complete TODO 6."
np.testing.assert_allclose(robust_result.params, ols_result.params)
assert not np.allclose(robust_result.bse, ols_result.bse)
# Explain why robust standard errors change uncertainty but not coefficients.
# They do not fix a wrong functional form, omitted variables, or biased sampling.

# %% 7. Distinguish two kinds of interval
new_sessions = pd.DataFrame(
    {
        "training_hours": [2.0, 6.0],
        "prior_experience": [3.0, 3.0],
        "session_format": ["solo", "solo"],
    }
)
# TODO 7: Use ols_result.get_prediction(...).summary_frame(alpha=0.05).
# Compare the mean CI to the observation/prediction interval.
prediction_summary = ...
assert prediction_summary is not Ellipsis, "Complete TODO 7."
assert len(prediction_summary) == 2
mean_width = prediction_summary["mean_ci_upper"] - prediction_summary["mean_ci_lower"]
observation_width = (
    prediction_summary["obs_ci_upper"] - prediction_summary["obs_ci_lower"]
)
assert (observation_width > mean_width).all()
print(prediction_summary)
# These OLS observation intervals assume constant residual variance. Discuss
# why their coverage can be misleading for the deliberately heteroskedastic data.

# %% 8. Save diagnostic evidence
# TODO 8: Label both panels, call fig.tight_layout(), then save the figure to
# FIGURE_DIR / "ols_diagnostics.png" before plt.show().
# Write your export and display code here.
assert (FIGURE_DIR / "ols_diagnostics.png").is_file(), "Complete TODO 8."

# %% [markdown]
# ## TODO 9: Write a short interpretation
#
# - Interpret one coefficient, its units, and its uncertainty.
# - Identify the reference session-format category.
# - Which plots/test suggest an assumption problem, and what does HC3 address?
# - Why does a small p-value not establish a causal training effect in real data?
#
# Optional stretch: add training_hours squared to the formula. Compare estimates
# and residual plots; explain why adding a term can improve fit without being
# justified by the data-generating process.
#
# VS Code practice: hover on get_prediction and inspect its return object's help.
# Reference: https://www.statsmodels.org/stable/regression.html
