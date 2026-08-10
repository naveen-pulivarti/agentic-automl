"""
The verification layer.

Everything the LLM *proposes* is *verified* here against real data. This module
knows nothing about agents or LLMs — it just builds real scikit-learn pipelines,
runs real cross-validation, and returns real numbers. Those numbers are the
ground truth that every agentic decision is judged against.

Two scales of measurement live here:

* **fast search** — used by the feature/model search loops, which run many
  evaluations. On a large dataset these are the bottleneck, so the data is
  sampled down to `fast_search_row_cap` rows with a *fixed* seed and fewer CV
  folds. The fixed seed matters: every candidate is then scored on exactly the
  same rows, so comparisons between them stay fair.
* **final training** — the chosen pipeline is fitted on the full training data
  and scored once on the untouched test set.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from ..agent.schemas import ModelChoice, ProblemType, TrainingResult
from ..config import CONFIG
from ..utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics: we always express the primary score as "higher is better".
# ---------------------------------------------------------------------------
CLASSIFICATION_PRIMARY = "f1_macro"
REGRESSION_PRIMARY = "r2"

_CLF_SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}
_REG_SCORING = {
    "r2": "r2",
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
}


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------
def build_estimator(choice: ModelChoice, problem: ProblemType,
                    params: dict[str, Any] | None = None):
    """Return an untrained sklearn estimator for the given choice + task."""
    params = params or {}
    rs = CONFIG.ml.random_state
    is_clf = problem == ProblemType.CLASSIFICATION

    # Guard: linear/logistic are task-specific. Remap before lookup so a
    # mismatched proposal degrades gracefully instead of raising.
    if choice == ModelChoice.LOGISTIC_REGRESSION and not is_clf:
        choice = ModelChoice.RIDGE
    if choice in (ModelChoice.LINEAR_REGRESSION, ModelChoice.RIDGE) and is_clf:
        choice = ModelChoice.LOGISTIC_REGRESSION

    table = {
        ModelChoice.LOGISTIC_REGRESSION: lambda: LogisticRegression(
            max_iter=1000, random_state=rs, **params),
        ModelChoice.LINEAR_REGRESSION: lambda: LinearRegression(**params),
        ModelChoice.RIDGE: lambda: Ridge(random_state=rs, **params),
        ModelChoice.RANDOM_FOREST: lambda: (
            RandomForestClassifier(random_state=rs, n_jobs=-1, **params) if is_clf
            else RandomForestRegressor(random_state=rs, n_jobs=-1, **params)),
        ModelChoice.GRADIENT_BOOSTING: lambda: (
            GradientBoostingClassifier(random_state=rs, **params) if is_clf
            else GradientBoostingRegressor(random_state=rs, **params)),
        ModelChoice.HIST_GRADIENT_BOOSTING: lambda: (
            HistGradientBoostingClassifier(random_state=rs, **params) if is_clf
            else HistGradientBoostingRegressor(random_state=rs, **params)),
        ModelChoice.EXTRA_TREES: lambda: (
            ExtraTreesClassifier(random_state=rs, n_jobs=-1, **params) if is_clf
            else ExtraTreesRegressor(random_state=rs, n_jobs=-1, **params)),
        ModelChoice.KNN: lambda: (
            KNeighborsClassifier(**params) if is_clf
            else KNeighborsRegressor(**params)),
        ModelChoice.SVM: lambda: (
            SVC(random_state=rs, **params) if is_clf
            else SVR(**params)),
    }
    return table[choice]()


def valid_models_for(problem: ProblemType, n_rows: int | None = None) -> list[ModelChoice]:
    """Which algorithms make sense for this task type AND this data size.

    Size matters as much as task type. SVM is roughly quadratic in samples and
    KNN's prediction cost grows with the training set, so on large data they
    turn a minutes-long run into an hours-long one for no benefit. Likewise the
    classic GradientBoosting implementation is far slower than the histogram
    based one once there are more than a few thousand rows.

    Filtering here — rather than hoping the LLM knows — is what lets the system
    stay hands-off on a dataset of any size.
    """
    n_rows = n_rows or 0
    big = n_rows > CONFIG.ml.hist_boosting_row_threshold
    very_big = n_rows > 50_000

    if problem == ProblemType.CLASSIFICATION:
        models = [ModelChoice.LOGISTIC_REGRESSION, ModelChoice.RANDOM_FOREST,
                  ModelChoice.EXTRA_TREES]
        models.append(ModelChoice.HIST_GRADIENT_BOOSTING if big
                      else ModelChoice.GRADIENT_BOOSTING)
        if not very_big:
            models.append(ModelChoice.KNN)
        if not big:
            models.append(ModelChoice.SVM)
        return models

    models = [ModelChoice.RIDGE, ModelChoice.LINEAR_REGRESSION,
              ModelChoice.RANDOM_FOREST, ModelChoice.EXTRA_TREES]
    models.append(ModelChoice.HIST_GRADIENT_BOOSTING if big
                  else ModelChoice.GRADIENT_BOOSTING)
    if not very_big:
        models.append(ModelChoice.KNN)
    return models


def proxy_model_for(n_rows: int) -> ModelChoice:
    """The fast, reliable model used to *score* features during the search.

    Feature decisions need a consistent yardstick, not the best possible model.
    A histogram gradient booster is used on larger data because it handles many
    rows quickly; a random forest elsewhere.
    """
    return (ModelChoice.HIST_GRADIENT_BOOSTING
            if n_rows > CONFIG.ml.hist_boosting_row_threshold
            else ModelChoice.RANDOM_FOREST)


# ---------------------------------------------------------------------------
# Preprocessing: build a ColumnTransformer from the dataframe dtypes.
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # min_frequency collapses rare levels into one bucket, which keeps the
        # encoded width sane on real-world columns with a long tail.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 max_categories=20, min_frequency=0.01,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


def build_pipeline(choice: ModelChoice, problem: ProblemType,
                   X: pd.DataFrame, params: dict[str, Any] | None = None) -> Pipeline:
    """Full pipeline: preprocessing + estimator."""
    return Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", build_estimator(choice, problem, params)),
    ])


# ---------------------------------------------------------------------------
# Sampling for the fast search phase
# ---------------------------------------------------------------------------
def subsample(X: pd.DataFrame, y: pd.Series, problem: ProblemType,
              cap: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Sample down to `cap` rows for search-phase evaluations.

    Uses a FIXED seed so every candidate in the search is scored on identical
    rows — otherwise differences between candidates would partly be differences
    between samples. Stratified for classification so rare classes survive.
    """
    cap = cap or CONFIG.ml.fast_search_row_cap
    if len(X) <= cap:
        return X, y

    stratify = None
    if problem == ProblemType.CLASSIFICATION:
        counts = y.value_counts()
        # stratify only if every class has enough members to survive the split
        if counts.min() >= 2 and y.nunique() < cap // 2:
            stratify = y

    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=cap, random_state=CONFIG.ml.random_state,
        stratify=stratify,
    )
    return X_s.reset_index(drop=True), y_s.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The core measurement function
# ---------------------------------------------------------------------------
def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    choice: ModelChoice,
    problem: ProblemType,
    params: dict[str, Any] | None = None,
    fast: bool = True,
) -> TrainingResult:
    """Cross-validate one model configuration and return REAL metrics.

    This is the single source of truth the whole agent trusts. `fast=True`
    (the default for search loops) samples large data down and uses fewer folds.
    """
    t0 = time.time()
    if fast:
        X, y = subsample(X, y, problem)

    pipe = build_pipeline(choice, problem, X, params)
    scoring = _CLF_SCORING if problem == ProblemType.CLASSIFICATION else _REG_SCORING
    primary = CLASSIFICATION_PRIMARY if problem == ProblemType.CLASSIFICATION else REGRESSION_PRIMARY

    n_splits = _safe_cv_folds(y, problem, fast=fast)
    cv = n_splits
    if problem == ProblemType.CLASSIFICATION:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=CONFIG.ml.random_state)

    cv_results = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    all_metrics: dict[str, float] = {}
    for metric_key in scoring:
        vals = cv_results[f"test_{metric_key}"]
        mean = float(np.mean(vals))
        # convert "neg_" metrics back to positive, human-friendly numbers
        if metric_key.startswith("neg_"):
            all_metrics[metric_key.replace("neg_", "")] = -mean
        else:
            all_metrics[metric_key] = mean

    score = float(np.mean(cv_results[f"test_{primary}"]))
    result = TrainingResult(
        model_name=choice.value,
        score=score,
        primary_metric=primary,
        all_metrics=all_metrics,
        params=params or {},
        train_seconds=time.time() - t0,
    )
    log.info("Evaluated %s on %d rows: %s=%.4f (%.1fs)",
             choice.value, len(X), primary, score, result.train_seconds)
    return result


def fit_and_evaluate(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    choice: ModelChoice, problem: ProblemType,
    params: dict[str, Any] | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit on the full training data and score once on the untouched test set.

    The test set reaching this function must never have been seen by any search
    stage — that is what makes these numbers an honest estimate of real-world
    performance rather than a restatement of what the search optimised.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_absolute_error,
        mean_squared_error, precision_score, r2_score, recall_score,
        roc_auc_score,
    )
    pipe = build_pipeline(choice, problem, X_train, params)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    if problem == ProblemType.CLASSIFICATION:
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        }
        # ROC-AUC is the standard headline metric for binary problems; only
        # meaningful when the estimator can produce probabilities.
        if y_test.nunique() == 2 and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except Exception:  # pragma: no cover - metric is a bonus, never fatal
                pass
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_test, preds)),
        }
    return pipe, metrics


def _safe_cv_folds(y: pd.Series, problem: ProblemType, fast: bool = False) -> int:
    """Don't ask for more folds than the smallest class supports."""
    folds = CONFIG.ml.fast_search_cv_folds if fast else CONFIG.ml.cv_folds
    if problem == ProblemType.CLASSIFICATION:
        min_class = int(y.value_counts().min())
        folds = int(max(2, min(folds, min_class)))
    return folds
