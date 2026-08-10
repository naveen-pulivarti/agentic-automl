"""
Feature-operation validation and execution.

The LLM can only *propose* operations from a whitelist (see schemas.py). This
module is what actually *executes* them on the DataFrame. Because only these
known operations can run, the agent can never execute arbitrary/unsafe code —
a deliberate safety boundary.

Two layers of checking, deliberately separated:

* `agent/schemas.py` enforces **structural** rules that hold for any dataset
  (a ratio needs two columns; date_part needs a component).
* `validate_proposal()` here enforces **dataset-dependent** rules a schema
  cannot see (does the column exist, is it actually numeric, is it the target).

Validating *before* executing matters because every unchecked proposal costs a
full cross-validation to discover it was useless. Rejecting a nonsensical one
up front — extracting a calendar month from a latitude, say — is free.

Every operation returns a new pandas Series (or None if it cannot be built) and
never mutates the input frame in place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ..agent.schemas import FeatureOperation, FeatureProposal
from ..utils.logger import get_logger

log = get_logger(__name__)


#: Operations that need genuinely numeric inputs.
_NUMERIC_OPS = {
    FeatureOperation.RATIO, FeatureOperation.PRODUCT, FeatureOperation.SUM,
    FeatureOperation.DIFFERENCE, FeatureOperation.LOG, FeatureOperation.SQUARE,
    FeatureOperation.BINNING,
}


def validate_proposal(df: pd.DataFrame, proposal: FeatureProposal,
                      target: str | None = None) -> str | None:
    """Return a human-readable rejection reason, or None if the proposal is OK.

    The reason string is fed back to the reasoner in the next round, which is
    how the agent learns not to repeat a mistake.
    """
    cols = proposal.source_columns
    op = proposal.operation

    # 1. The target may never be used to build a feature. Doing so would leak
    #    the answer into the inputs and produce a meaningless perfect score.
    if target is not None and target in cols:
        return f"uses the target column '{target}' — that would leak the answer"

    # 2. Columns must exist.
    for c in cols:
        if c not in df.columns:
            return f"column '{c}' does not exist in the dataset"

    # 3. Don't overwrite something that is already there.
    if proposal.feature_name in df.columns:
        return f"a column named '{proposal.feature_name}' already exists"

    # 4. Numeric operations need numeric columns.
    if op in _NUMERIC_OPS:
        for c in cols:
            if not _is_numeric_like(df[c]):
                return f"operation '{op.value}' needs numeric input but '{c}' is not numeric"

    # 5. date_part needs something that genuinely parses as a date. Without this
    #    check a plain number is silently reinterpreted as a timestamp.
    if op == FeatureOperation.DATE_PART:
        if not _is_datetime_like(df[cols[0]]):
            return f"'{cols[0]}' is not a date column, so date_part cannot apply"

    # 6. Binning needs enough distinct values to form bins.
    if op == FeatureOperation.BINNING:
        if df[cols[0]].nunique(dropna=True) < 4:
            return f"'{cols[0]}' has too few distinct values to bin"

    # 7. Interaction is for categoricals; on continuous columns it produces a
    #    near-unique key per row, which is useless and explodes the encoding.
    if op == FeatureOperation.INTERACTION:
        for c in cols:
            if df[c].nunique(dropna=True) > max(50, 0.2 * len(df)):
                return f"'{c}' has too many distinct values for an interaction"

    return None


def apply_feature(df: pd.DataFrame, proposal: FeatureProposal,
                  reference: pd.DataFrame | None = None) -> pd.Series | None:
    """Create the proposed feature, or return None if it can't be built safely.

    `reference` supplies the *training* frame when rebuilding a feature on unseen
    data. Most operations are row-wise and need it, but two are data-dependent:
    quantile binning needs bin edges, and the log shift needs a minimum. Deriving
    those from the test set would both leak information and produce a feature
    that means something different than it did at training time.
    """
    cols = proposal.source_columns
    op = proposal.operation
    ref = reference if reference is not None else df

    # validate columns exist
    for c in cols:
        if c not in df.columns:
            log.warning("Feature %s skipped: column %r not found",
                        proposal.feature_name, c)
            return None

    try:
        if op in (FeatureOperation.RATIO, FeatureOperation.PRODUCT,
                  FeatureOperation.SUM, FeatureOperation.DIFFERENCE):
            if len(cols) != 2:
                return None
            a, b = _numeric(df[cols[0]]), _numeric(df[cols[1]])
            if a is None or b is None:
                return None
            if op == FeatureOperation.RATIO:
                denom = b.replace(0, np.nan)
                return a / denom
            if op == FeatureOperation.PRODUCT:
                return a * b
            if op == FeatureOperation.SUM:
                return a + b
            if op == FeatureOperation.DIFFERENCE:
                return a - b

        elif op == FeatureOperation.LOG:
            a = _numeric(df[cols[0]])
            if a is None:
                return None
            # log1p needs non-negative; shift by the TRAINING minimum so the
            # transform is identical on unseen data
            a_ref = _numeric(ref[cols[0]])
            shift = 0.0
            ref_min = a_ref.min() if a_ref is not None else None
            if ref_min is not None and pd.notna(ref_min) and ref_min < 0:
                shift = -ref_min
            return np.log1p((a + shift).clip(lower=-0.999999))

        elif op == FeatureOperation.SQUARE:
            a = _numeric(df[cols[0]])
            return a ** 2 if a is not None else None

        elif op == FeatureOperation.BINNING:
            a = _numeric(df[cols[0]])
            a_ref = _numeric(ref[cols[0]])
            if a is None or a_ref is None:
                return None
            # Edges come from the training data; applying them with pd.cut keeps
            # the bins stable rather than re-quantising each new frame.
            edges = np.unique(np.nanquantile(a_ref.to_numpy(dtype=float),
                                             [0.0, 0.25, 0.5, 0.75, 1.0]))
            if len(edges) < 3:
                return None
            edges[0], edges[-1] = -np.inf, np.inf
            return pd.cut(a, bins=edges, labels=False,
                          include_lowest=True).astype("float")

        elif op == FeatureOperation.INTERACTION:
            if len(cols) != 2:
                return None
            return (df[cols[0]].astype(str) + "_" + df[cols[1]].astype(str))

        elif op == FeatureOperation.DATE_PART:
            comp = proposal.date_component or "month"
            dt = pd.to_datetime(df[cols[0]], errors="coerce")
            if dt.isna().all():
                return None
            return getattr(dt.dt, comp).astype("float")

    except Exception as e:  # never let a bad feature crash the pipeline
        log.warning("Feature %s failed: %s", proposal.feature_name, e)
        return None

    return None


# -- helpers ----------------------------------------------------------------
def _numeric(s: pd.Series) -> pd.Series | None:
    """Coerce to numeric; return None if the column isn't usefully numeric."""
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() == 0:
        return None
    return out


def _is_numeric_like(s: pd.Series) -> bool:
    """True if the column is numeric, or is text that cleanly parses as numeric."""
    if ptypes.is_numeric_dtype(s):
        return True
    if ptypes.is_bool_dtype(s):
        return False
    parsed = pd.to_numeric(s, errors="coerce")
    non_null = s.notna().sum()
    # require most non-null values to parse, so an ID-like string column with a
    # couple of numeric-looking entries is not mistaken for a number
    return bool(non_null and parsed.notna().sum() >= 0.9 * non_null)


def _is_datetime_like(s: pd.Series) -> bool:
    """True only for real dates — never for plain numbers.

    Numeric columns are excluded explicitly: pandas will happily read an integer
    as nanoseconds since the epoch, which is how 'the month of a latitude'
    becomes a silently valid feature.
    """
    if ptypes.is_datetime64_any_dtype(s):
        return True
    if ptypes.is_numeric_dtype(s):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().sum() >= 0.9 * len(sample))
