"""Model definitions, baseline pipelines, hyperparameter search grids, and builders."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .features import build_preprocessor


def get_baseline_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return naive baseline models wrapped in pipelines."""
    return {
        "Dummy (Most Frequent)": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Dummy (Stratified)": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DummyClassifier(strategy="stratified", random_state=random_state),
                ),
            ]
        ),
    }


def get_candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return standard candidate classification models wrapped in leakage-safe pipelines.

    All classifiers include the full preprocessing ColumnTransformer so that
    cross-validation folds fit both transformations and models together.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    KNeighborsClassifier(n_neighbors=7, weights="distance"),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        max_depth=5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def get_hyperparameter_grids() -> dict[str, dict]:
    """Return hyperparameter search spaces for the top candidate models."""
    return {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "classifier__solver": ["lbfgs"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
        "HistGradientBoosting": {
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_iter": [50, 100],
            "classifier__max_leaf_nodes": [15, 31],
        },
    }
