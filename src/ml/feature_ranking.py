"""
Feature ranking panel — the evidence layer for feature selection.

Ranking a column is not one question with one answer. A linear correlation
misses a U-shaped relationship; mutual information catches it but ignores
redundancy; a tree's importance is biased toward high-cardinality columns; and
none of them notice that two columns say the same thing. So instead of trusting
one method we run several from genuinely different families and present the
disagreement as evidence:

    filter      variance, pearson, spearman, f-test, mutual information
    embedded    random-forest gain, L1 (lasso/logistic) coefficients
    wrapper     recursive feature elimination, permutation importance
    redundancy  mRMR (relevance minus redundancy)

The output is a table the reasoner reads. It decides *nothing* here — this
module produces numbers only, and every subset it later proposes is verified by
real cross-validation.

Two design notes worth defending:

* **Ordinal, not one-hot, encoding.** For modelling we one-hot encode, but that
  splits one column into many and makes "the importance of `region`" ambiguous.
  Here each original column maps to exactly one encoded column, so every score
  is attributable. The cost is that a linear method sees a fake ordering on
  categorical values — so those methods report N/A for categoricals rather than
  a misleading number.
* **Everything runs on sampled training data.** Ranking on the test set would
  reintroduce exactly the leakage the pipeline is designed to avoid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ..agent.schemas import ProblemType
from ..config import CONFIG
from ..utils.logger import get_logger

log = get_logger(__name__)


#: Methods whose result is meaningless for a categorical column under ordinal
#: encoding, because they assume the numeric order means something.
_LINEAR_METHODS = {"pearson", "spearman", "f_test", "l1_coef"}


def _encode_for_ranking(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One numeric column per original column, plus the list of categoricals.

    Missing values are median/mode filled because most ranking estimators
    cannot accept NaN. This is a ranking-only view of the data; the real
    pipeline does its own imputation inside cross-validation.
    """
    out = pd.DataFrame(index=X.index)
    categorical: list[str] = []

    for col in X.columns:
        s = X[col]
        if ptypes.is_numeric_dtype(s):
            filled = s.astype(float)
            if filled.isna().any():
                filled = filled.fillna(filled.median())
            out[col] = filled.replace([np.inf, -np.inf], np.nan)
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
            if out[col].isna().any():      # column was entirely NaN/inf
                out[col] = 0.0
        else:
            categorical.append(col)
            codes = s.astype("category").cat.codes.astype(float)
            codes[codes < 0] = np.nan      # -1 marks missing
            out[col] = codes.fillna(codes.median() if codes.notna().any() else 0.0)

    return out, categorical


def _encode_target(y: pd.Series, problem: ProblemType) -> np.ndarray:
    if problem == ProblemType.CLASSIFICATION and not ptypes.is_numeric_dtype(y):
        return y.astype("category").cat.codes.to_numpy()
    return pd.to_numeric(y, errors="coerce").fillna(0).to_numpy()


# ---------------------------------------------------------------------------
# Individual ranking methods. Each returns {column: score}, higher = better.
# Each is wrapped by the caller so one failure never stops the panel.
# ---------------------------------------------------------------------------
def _m_variance(Xe: pd.DataFrame, *_args) -> dict[str, float]:
    """Near-constant columns carry no information regardless of the target."""
    std = Xe.std(axis=0).replace(0, np.nan)
    norm = (std / std.abs().max()) if std.notna().any() else std
    return {c: float(v) if pd.notna(v) else 0.0 for c, v in norm.items()}


def _m_pearson(Xe: pd.DataFrame, yv: np.ndarray, *_a) -> dict[str, float]:
    """Strength of a straight-line relationship with the target."""
    return {c: abs(float(np.corrcoef(Xe[c].to_numpy(), yv)[0, 1]))
            if Xe[c].std() > 0 else 0.0 for c in Xe.columns}


def _m_spearman(Xe: pd.DataFrame, yv: np.ndarray, *_a) -> dict[str, float]:
    """Rank correlation — catches any monotonic relationship, not just linear."""
    from scipy.stats import spearmanr
    out: dict[str, float] = {}
    for c in Xe.columns:
        try:
            rho, _ = spearmanr(Xe[c].to_numpy(), yv)
            out[c] = abs(float(rho)) if np.isfinite(rho) else 0.0
        except Exception:
            out[c] = 0.0
    return out


def _m_f_test(Xe: pd.DataFrame, yv: np.ndarray, problem: ProblemType) -> dict[str, float]:
    """Classical ANOVA/regression F statistic."""
    from sklearn.feature_selection import f_classif, f_regression
    fn = f_classif if problem == ProblemType.CLASSIFICATION else f_regression
    scores, _ = fn(Xe.to_numpy(), yv)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
    return dict(zip(Xe.columns, scores.astype(float)))


def _m_mutual_info(Xe: pd.DataFrame, yv: np.ndarray, problem: ProblemType) -> dict[str, float]:
    """Information shared with the target — catches non-linear relationships."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    fn = (mutual_info_classif if problem == ProblemType.CLASSIFICATION
          else mutual_info_regression)
    scores = fn(Xe.to_numpy(), yv, random_state=CONFIG.ml.random_state)
    return dict(zip(Xe.columns, scores.astype(float)))


def _ranking_forest(problem: ProblemType):
    """A small, fast forest used by the embedded/wrapper methods."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    kw = dict(n_estimators=60, max_depth=12, n_jobs=-1,
              random_state=CONFIG.ml.random_state)
    return (RandomForestClassifier(**kw) if problem == ProblemType.CLASSIFICATION
            else RandomForestRegressor(**kw))


def _m_rf_gain(Xe: pd.DataFrame, yv: np.ndarray, problem: ProblemType) -> dict[str, float]:
    """How much each column reduces impurity across a forest (gain)."""
    model = _ranking_forest(problem).fit(Xe.to_numpy(), yv)
    return dict(zip(Xe.columns, model.feature_importances_.astype(float)))


def _m_l1_coef(Xe: pd.DataFrame, yv: np.ndarray, problem: ProblemType) -> dict[str, float]:
    """L1 regularisation drives useless coefficients to exactly zero."""
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(Xe.to_numpy())
    if problem == ProblemType.CLASSIFICATION:
        # saga supports L1 for both binary and multiclass. scikit-learn is
        # migrating from `penalty="l1"` to `l1_ratio=1.0`; try the new spelling
        # first and fall back, so this works either side of that change.
        common = dict(solver="saga", C=0.5, max_iter=300, tol=1e-2,
                      random_state=CONFIG.ml.random_state)
        try:
            m = LogisticRegression(l1_ratio=1.0, **common)
            m.fit(Xs, yv)
        except (TypeError, ValueError):
            m = LogisticRegression(penalty="l1", **common)
            m.fit(Xs, yv)
        coef = np.abs(m.coef_).mean(axis=0)
    else:
        m = Lasso(alpha=0.01, max_iter=3000, random_state=CONFIG.ml.random_state)
        m.fit(Xs, yv)
        coef = np.abs(m.coef_)
    return dict(zip(Xe.columns, coef.astype(float)))


def _m_rfe(Xe: pd.DataFrame, yv: np.ndarray, problem: ProblemType) -> dict[str, float]:
    """Recursive feature elimination: repeatedly drop the weakest column.

    Returns a descending score from the elimination order (rank 1 = best), so
    that higher is better like every other method here.
    """
    from sklearn.feature_selection import RFE
    n = Xe.shape[1]
    sel = RFE(_ranking_forest(problem),
              n_features_to_select=max(1, n // 3),
              step=0.25)
    sel.fit(Xe.to_numpy(), yv)
    return {c: float(n - r + 1) for c, r in zip(Xe.columns, sel.ranking_)}


def _m_permutation(Xe: pd.DataFrame, yv: np.ndarray, problem: ProblemType) -> dict[str, float]:
    """Shuffle a column and see how much a fitted model actually degrades.

    More honest than impurity-based importance, which is biased toward columns
    with many distinct values.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        Xe.to_numpy(), yv, test_size=0.3, random_state=CONFIG.ml.random_state)
    model = _ranking_forest(problem).fit(Xtr, ytr)
    r = permutation_importance(model, Xte, yte, n_repeats=3,
                               random_state=CONFIG.ml.random_state, n_jobs=-1)
    return dict(zip(Xe.columns, np.clip(r.importances_mean, 0, None).astype(float)))


def _m_mrmr(Xe: pd.DataFrame, yv: np.ndarray, problem: ProblemType) -> dict[str, float]:
    """Minimum Redundancy Maximum Relevance.

    The only method here that looks at the columns' relationship to *each
    other*. Two columns can each be highly predictive and still be worth only
    one slot between them if they carry the same information.

        score = relevance(to target) - mean(redundancy with other columns)
    """
    relevance = _m_mutual_info(Xe, yv, problem)
    rel = pd.Series(relevance)
    if rel.max() > 0:
        rel = rel / rel.max()

    # .to_numpy() can hand back a read-only view, so copy before writing to it
    corr = Xe.corr(method="spearman").abs()
    corr_vals = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, np.nan)
    corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
    redundancy = corr.mean(axis=0, skipna=True).fillna(0.0)

    return {c: float(rel.get(c, 0.0) - redundancy.get(c, 0.0)) for c in Xe.columns}


_METHODS = {
    "variance": _m_variance,
    "pearson": _m_pearson,
    "spearman": _m_spearman,
    "f_test": _m_f_test,
    "mutual_info": _m_mutual_info,
    "rf_gain": _m_rf_gain,
    "l1_coef": _m_l1_coef,
    "rfe": _m_rfe,
    "permutation": _m_permutation,
    "mrmr": _m_mrmr,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def rank_features(X: pd.DataFrame, y: pd.Series, problem: ProblemType,
                  methods: list[str] | None = None) -> pd.DataFrame:
    """Score every column with every method; return a tidy ranking table.

    Returns a DataFrame indexed by column name with one `rank_<method>` column
    per method (1 = best) plus `mean_rank`, sorted best-first. Methods that are
    not applicable to a column report NaN and are simply excluded from that
    column's mean, rather than contributing a misleading value.
    """
    from ..ml.trainer import subsample

    X_s, y_s = subsample(X, y, problem)
    Xe, categorical = _encode_for_ranking(X_s)
    yv = _encode_target(y_s, problem)

    if Xe.shape[1] == 0:
        return pd.DataFrame()

    chosen = methods or list(_METHODS)
    raw: dict[str, dict[str, float]] = {}

    for name in chosen:
        fn = _METHODS.get(name)
        if fn is None:
            continue
        try:
            scores = fn(Xe, yv, problem)
            # A linear method on an ordinally-encoded categorical is not
            # meaningful — record N/A instead of a number that looks real.
            if name in _LINEAR_METHODS:
                for c in categorical:
                    scores[c] = float("nan")
            raw[name] = scores
            log.info("Ranking method '%s' completed.", name)
        except Exception as e:      # one method failing must not stop the panel
            log.warning("Ranking method '%s' failed (%s); skipped.", name, e)

    if not raw:
        return pd.DataFrame()

    table = pd.DataFrame(raw, index=Xe.columns)
    for name in table.columns:
        # rank 1 = best; NaN scores stay NaN so they don't pollute the mean
        table[f"rank_{name}"] = table[name].rank(ascending=False, na_option="keep")

    rank_cols = [c for c in table.columns if c.startswith("rank_")]
    table["mean_rank"] = table[rank_cols].mean(axis=1, skipna=True)
    table = table.sort_values("mean_rank")
    table.index.name = "feature"
    return table


def ranking_summary_for_llm(table: pd.DataFrame, max_features: int = 40) -> list[dict]:
    """Compact, token-cheap view of the ranking table for the reasoner.

    Only the per-method ranks are sent (not raw scores, whose scales differ
    wildly and would just confuse a small model), plus a flag marking where the
    methods disagree — which is precisely where a reasoner adds value over a
    simple average.
    """
    if table.empty:
        return []
    rank_cols = [c for c in table.columns if c.startswith("rank_")]
    n = len(table)
    out: list[dict] = []

    for feat, row in table.head(max_features).iterrows():
        ranks = {c.replace("rank_", ""): (None if pd.isna(row[c]) else int(row[c]))
                 for c in rank_cols}
        present = [v for v in ranks.values() if v is not None]
        spread = (max(present) - min(present)) if present else 0
        out.append({
            "feature": str(feat),
            "mean_rank": round(float(row["mean_rank"]), 1),
            "ranks": ranks,
            # methods disagreeing by more than half the field is a signal worth
            # reasoning about (e.g. linear says useless, mutual info says vital)
            "methods_disagree": bool(spread > n / 2),
        })
    return out


def top_k_features(table: pd.DataFrame, k: int) -> list[str]:
    """The k best columns by mean rank — the deterministic control subset."""
    if table.empty:
        return []
    k = max(1, min(k, len(table)))
    return [str(i) for i in table.head(k).index]
