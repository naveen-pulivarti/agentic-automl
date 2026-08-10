"""
Stage 1 — Data profiling & problem-type detection.

This stage is deliberately *factual* (code-driven, not LLM-driven): it computes
ground-truth facts about the dataset that every later stage relies on. Auto-
detecting classification vs regression here determines which algorithms and
metrics the whole pipeline uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ..agent.schemas import ProblemType
from ..config import CONFIG
from ..utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str                 # "numeric" | "categorical" | "datetime" | "text"
    missing_pct: float
    n_unique: int
    sample_values: list[Any] = field(default_factory=list)
    # numeric-only stats
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    target: str
    problem_type: ProblemType
    columns: list[ColumnProfile]
    target_detection_reason: str
    total_missing_cells: int

    def feature_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.name != self.target]

    def numeric_features(self) -> list[str]:
        return [c.name for c in self.columns
                if c.kind == "numeric" and c.name != self.target]

    def categorical_features(self) -> list[str]:
        return [c.name for c in self.columns
                if c.kind == "categorical" and c.name != self.target]

    def datetime_features(self) -> list[str]:
        return [c.name for c in self.columns
                if c.kind == "datetime" and c.name != self.target]

    def to_summary_dict(self) -> dict[str, Any]:
        """Compact dict fed to the LLM as context (kept small on purpose)."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target": self.target,
            "problem_type": self.problem_type.value,
            "columns": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "dtype": c.dtype,
                    "missing_pct": round(c.missing_pct, 1),
                    "n_unique": c.n_unique,
                    "sample": c.sample_values[:3],
                }
                for c in self.columns
            ],
        }


def _column_kind(s: pd.Series) -> str:
    """Classify a column as numeric / categorical / datetime / text / constant /
    identifier.

    The last two exist so the pipeline can drop them automatically. Neither can
    help a model, and both actively hurt: a constant column wastes a slot in
    every search, and an identifier (a row number, an order ID) is often
    *perfectly* correlated with the target in a sorted export — the model learns
    the index instead of the pattern and collapses on new data. A user should not
    have to know to remove these by hand.
    """
    non_null = s.dropna()
    n = len(non_null)
    if n == 0:
        return "constant"
    n_unique = non_null.nunique()
    if n_unique <= 1:
        return "constant"

    if ptypes.is_datetime64_any_dtype(s):
        return "datetime"

    if ptypes.is_numeric_dtype(s):
        # A near-unique integer column that increases steadily is a row id, not
        # a measurement. Requiring monotonicity keeps genuine continuous
        # measurements (which are also near-unique) safe.
        if n >= 20 and n_unique >= 0.99 * n:
            try:
                as_int = np.allclose(non_null % 1, 0)
                if as_int and (non_null.is_monotonic_increasing
                               or non_null.is_monotonic_decreasing):
                    return "identifier"
            except (TypeError, ValueError):
                pass
        return "numeric"

    if ptypes.is_bool_dtype(s):
        return "categorical"

    if s.dtype == object:
        # every value distinct and textual -> a key or free text, not a category
        if n_unique > max(50, 0.5 * n):
            return "text"
        return "categorical"
    return "categorical"


def detect_problem_type(y: pd.Series, ml_cfg=CONFIG.ml) -> tuple[ProblemType, str]:
    """Decide classification vs regression from the target column.

    Heuristics (in order):
    - non-numeric target -> classification
    - numeric but few unique integer-like values -> classification
    - otherwise -> regression
    """
    y_non_null = y.dropna()
    n_unique = y_non_null.nunique()

    if not ptypes.is_numeric_dtype(y_non_null):
        return ProblemType.CLASSIFICATION, (
            f"Target '{y.name}' is non-numeric with {n_unique} categories."
        )

    # numeric target
    is_integer_like = np.allclose(y_non_null.dropna() % 1, 0)
    if n_unique <= 2:
        return ProblemType.CLASSIFICATION, (
            f"Target '{y.name}' is numeric but binary ({n_unique} values)."
        )
    if is_integer_like and n_unique <= ml_cfg.max_classification_classes:
        return ProblemType.CLASSIFICATION, (
            f"Target '{y.name}' has {n_unique} discrete integer values "
            f"(<= {ml_cfg.max_classification_classes}) -> treated as classification."
        )
    return ProblemType.REGRESSION, (
        f"Target '{y.name}' is continuous numeric with {n_unique} unique values."
    )


def profile_dataset(df: pd.DataFrame, target: str) -> DatasetProfile:
    """Produce a full profile of the dataset given the chosen target column."""
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in data.")

    n_rows, n_cols = df.shape
    problem_type, reason = detect_problem_type(df[target])

    columns: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        kind = _column_kind(s)
        prof = ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            kind=kind,
            missing_pct=100.0 * s.isna().mean(),
            n_unique=int(s.nunique(dropna=True)),
            sample_values=[_to_native(v) for v in s.dropna().unique()[:5]],
        )
        if kind == "numeric":
            prof.mean = _safe_float(s.mean())
            prof.std = _safe_float(s.std())
            prof.min = _safe_float(s.min())
            prof.max = _safe_float(s.max())
        columns.append(prof)

    profile = DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        target=target,
        problem_type=problem_type,
        columns=columns,
        target_detection_reason=reason,
        total_missing_cells=int(df.isna().sum().sum()),
    )
    log.info("Profiled dataset: %d rows x %d cols, problem=%s",
             n_rows, n_cols, problem_type.value)
    return profile


def suggest_target_column(df: pd.DataFrame) -> str:
    """Best-guess default target (last column) — user can override in the UI."""
    return df.columns[-1]


# -- small helpers ----------------------------------------------------------
def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _to_native(v):
    """Convert numpy scalars to JSON-serialisable Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
