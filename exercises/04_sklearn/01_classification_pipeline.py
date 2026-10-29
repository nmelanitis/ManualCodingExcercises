"""scikit-learn: predict late deliveries. Intermediate; 60–75 minutes.

Practise a baseline, mixed-column preprocessing, cross-validation, and a held-out
test. All data is fictional and generated locally. Work through TODOs in order.
"""

# %% 0. Supplied dataset
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 1907
rng = np.random.default_rng(RANDOM_SEED)
N_SAMPLES = 400
distance = rng.uniform(1, 15, N_SAMPLES)
preparation = np.maximum(1, rng.normal(12, 4, N_SAMPLES))
weather = rng.choice(["clear", "rain"], N_SAMPLES, p=[0.7, 0.3])
travel_minutes = (
    10
    + 2.4 * distance
    + preparation
    + 7 * (weather == "rain")
    + rng.normal(0, 5, N_SAMPLES)
)
deliveries = pd.DataFrame(
    {
        "distance_km": distance,
        "preparation_minutes": preparation,
        "weather": weather,
        "late": (travel_minutes > 45).astype(int),
    }
)
# Simulate missing inputs after generating the label.
deliveries.loc[rng.choice(N_SAMPLES, 20, replace=False), "distance_km"] = np.nan
deliveries.loc[rng.choice(N_SAMPLES, 15, replace=False), "preparation_minutes"] = np.nan
NUMERIC_FEATURES = ["distance_km", "preparation_minutes"]
CATEGORICAL_FEATURES = ["weather"]

# %% 1. Separate inputs and label
# TODO 1: Define X from the three input columns and y from late.
# travel_minutes is a post-delivery measurement: do not include it in X.
X = deliveries[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = deliveries["late"]
assert X is not Ellipsis and y is not Ellipsis, "Complete TODO 1."
assert X.shape == (400, 3) and "late" not in X
assert set(y.unique()) == {0, 1}

# %% 2. Split before fitting any preprocessing
# TODO 2: Use an 80/20 train/test split with random_state=RANDOM_SEED, stratify=y.
list_s = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
X_train = list_s[0]
X_test = list_s[1]
y_train = list_s[2]
y_test = list_s[3]
assert X_train is not Ellipsis and X_test is not Ellipsis, "Complete TODO 2."
assert len(X_train) == len(y_train) == 320
assert len(X_test) == len(y_test) == 80
assert set(X_train.index).isdisjoint(X_test.index)

# %% 3. Fit a baseline
# TODO 3: Create DummyClassifier(strategy="most_frequent"), fit only on the
# training data, and predict X_test. Keep these results for the final comparison.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
assert baseline_predictions is not Ellipsis, "Complete TODO 3."
assert baseline_predictions.shape == y_test.shape

# %% 4. Build preprocessing pipelines
# TODO 4: numeric_pipeline: median imputation -> StandardScaler.
# categorical_pipeline: most-frequent imputation -> OneHotEncoder.
# Set handle_unknown="ignore" in the encoder. Do not call fit yet.
numeric_pipeline = Pipeline(
    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
)
categorical_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]
)
assert isinstance(numeric_pipeline, Pipeline), "Complete TODO 4."
assert isinstance(categorical_pipeline, Pipeline)

# %% 5. Combine columns and classifier
# TODO 5: Use ColumnTransformer with the feature lists from setup.
# Build a Pipeline with steps named "preprocess" and "model". Use
# LogisticRegression(max_iter=1000, random_state=RANDOM_SEED) as the model.
preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)
classifier = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ]
)
assert isinstance(preprocessor, ColumnTransformer), "Complete TODO 5."
assert isinstance(classifier, Pipeline)
assert list(classifier.named_steps) == ["preprocess", "model"]

# %% 6. Cross-validation belongs inside the training set
# TODO 6: Create a shuffled 5-fold StratifiedKFold with RANDOM_SEED.
# Pass the entire classifier pipeline to cross_val_score with scoring="f1".
# Use X_train/y_train. Each fold must fit its own preprocessing.
cv = StratifiedKFold(shuffle=True, random_state=RANDOM_SEED)
cv_f1 = cross_val_score(classifier, X_train, y_train, cv=cv, scoring="f1")
assert cv_f1 is not Ellipsis, "Complete TODO 6."
assert cv_f1.shape == (5,) and np.isfinite(cv_f1).all()
assert np.all((cv_f1 >= 0) & (cv_f1 <= 1))
print("Training CV F1 mean/std:", cv_f1.mean(), cv_f1.std())

# %% 7. Fit the final model and evaluate once
# TODO 7: Fit classifier on all training data. Predict labels and positive-class
# probabilities on X_test. For this exercise the positive class is 1.
# Write the fit call here.
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
positive_probabilities = classifier.predict_proba(X_test)[:, 1]
assert predictions is not Ellipsis and positive_probabilities is not Ellipsis
assert predictions.shape == positive_probabilities.shape == (80,)
assert np.all((positive_probabilities >= 0) & (positive_probabilities <= 1))

# %% 8. Metrics answer different questions
# TODO 8: Build a results table indexed by "baseline" and "logistic", with
# accuracy and f1 columns. Use zero_division=0 for F1 if needed.
# Calculate ROC AUC from probabilities, and the confusion matrix from labels.
results = pd.DataFrame(
    data=[
        [
            accuracy_score(y_test, baseline_predictions),
            f1_score(y_test, baseline_predictions, zero_division=0),
        ],
        [
            accuracy_score(y_test, predictions),
            f1_score(y_test, predictions, zero_division=0),
        ],
    ],
    columns=["accuracy", "f1"],
    index=["baseline", "logistic"],
)
auc = roc_auc_score(y_test, positive_probabilities)
confusion = confusion_matrix(y_test, predictions)
assert results is not Ellipsis and auc is not Ellipsis and confusion is not Ellipsis
assert set(results.index) == {"baseline", "logistic"}
assert {"accuracy", "f1"} <= set(results.columns)
assert 0 <= auc <= 1 and confusion.shape == (2, 2)
assert confusion.sum() == len(y_test)
print(results)
print("ROC AUC:", auc, "\nConfusion matrix [true rows, predicted columns]:", confusion)

# %% 9. Inspect what preprocessing produced
# TODO 9: Get output feature names from the fitted "preprocess" step and inspect
# classifier["model"].coef_. Check that the coefficient count matches.
X_processed = classifier.named_steps["preprocess"].transform(X_test)
feature_names = classifier.named_steps["preprocess"].get_feature_names_out()
assert feature_names is not Ellipsis, "Complete TODO 9."
assert len(feature_names) == classifier["model"].coef_.shape[1]

# %% [markdown]
# ## TODO 10: Interpret and audit
#
# - Did the model improve on the baseline? Quote at least two metrics.
# - What do false positives and false negatives mean for these deliveries?
# - Why would fitting the imputer/scaler before cross-validation leak information?
# - Why is accuracy alone insufficient when one class dominates?
#
# Optional stretch: tune model__C with GridSearchCV on training data only.
# Treat the existing test result as already seen; use a fresh held-out dataset
# for an unbiased assessment after making further choices.
#
# VS Code practice: inspect named_steps and transformed feature shapes in a
# debugging session. Ask for a hint about one TODO if you get stuck.
# Reference: https://scikit-learn.org/stable/modules/compose.html
