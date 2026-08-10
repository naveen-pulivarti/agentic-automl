"""
Pydantic schemas.

These do three jobs:
1. Force the LLM to return STRUCTURED output we can trust (validated, typed),
   instead of free-form prose. If the LLM returns something malformed, Pydantic
   raises and we retry — this is the "reliable agent loop" guarantee.
2. Enforce the *structural* rules of each operation (e.g. a ratio needs exactly
   two columns) so an impossible proposal is rejected before any work is done.
   Dataset-dependent rules (does this column exist? is it really a date?) live
   in `ml/feature_ops.py`, because a schema cannot see the data.
3. Provide clean internal data structures passed between pipeline stages.

Note the `salvage()` classmethods. A batch of proposals is all-or-nothing to
Pydantic: one malformed item invalidates the whole response. With small free
models that is common and expensive — a single bad enum value would throw away
three good proposals and force a fallback. `salvage()` validates item by item
and keeps whatever is valid, which measurably increases how much usable output
we get from weak models.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Problem type
# ---------------------------------------------------------------------------
class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# ---------------------------------------------------------------------------
# Stage 2 — Feature engineering proposals (LLM output schema)
# ---------------------------------------------------------------------------
class FeatureOperation(str, Enum):
    """The whitelist of operations the agent is allowed to propose.

    Restricting to a known set is a safety measure: we can only *execute*
    operations we have implemented, so the LLM cannot ask for arbitrary code.
    """
    RATIO = "ratio"                 # col_a / col_b
    PRODUCT = "product"             # col_a * col_b
    SUM = "sum"                     # col_a + col_b
    DIFFERENCE = "difference"       # col_a - col_b
    LOG = "log"                     # log1p(col_a)
    SQUARE = "square"               # col_a ** 2
    BINNING = "binning"             # discretise col_a into quantile bins
    INTERACTION = "interaction"     # categorical A x categorical B (combined key)
    DATE_PART = "date_part"         # extract year/month/day/dayofweek from a date col


#: Operations taking exactly two source columns.
_TWO_COLUMN_OPS = {
    FeatureOperation.RATIO, FeatureOperation.PRODUCT,
    FeatureOperation.SUM, FeatureOperation.DIFFERENCE,
    FeatureOperation.INTERACTION,
}
#: Operations taking exactly one source column.
_ONE_COLUMN_OPS = {
    FeatureOperation.LOG, FeatureOperation.SQUARE,
    FeatureOperation.BINNING, FeatureOperation.DATE_PART,
}


class FeatureProposal(BaseModel):
    """One candidate feature the LLM proposes."""
    feature_name: str = Field(..., description="A short snake_case name for the new feature.")
    operation: FeatureOperation
    source_columns: list[str] = Field(
        ..., min_length=1, max_length=2,
        description="The existing column(s) this feature is derived from.",
    )
    # Only used for DATE_PART: which component to extract.
    date_component: Literal["year", "month", "day", "dayofweek", "hour"] | None = None
    reasoning: str = Field(default="", description="Why this feature might help the model.")

    @model_validator(mode="after")
    def _check_arity(self) -> "FeatureProposal":
        """Reject proposals whose column count cannot work for the operation.

        Previously a `square` with two columns was accepted and the executor
        silently used only the first, producing a feature whose *name* described
        something different from its *contents*. That is exactly the kind of
        quiet wrongness an explainability tool must not have.
        """
        n = len(self.source_columns)
        if self.operation in _TWO_COLUMN_OPS and n != 2:
            raise ValueError(
                f"operation '{self.operation.value}' requires exactly 2 source_columns, got {n}"
            )
        if self.operation in _ONE_COLUMN_OPS and n != 1:
            raise ValueError(
                f"operation '{self.operation.value}' requires exactly 1 source_column, got {n}"
            )
        if self.operation == FeatureOperation.DATE_PART and not self.date_component:
            raise ValueError("operation 'date_part' requires a date_component")
        # Deriving a feature from itself is always redundant.
        if n == 2 and self.source_columns[0] == self.source_columns[1]:
            raise ValueError("source_columns must be two different columns")
        return self


class FeatureProposalBatch(BaseModel):
    """The LLM proposes several features at once; we test each individually."""
    proposals: list[FeatureProposal] = Field(default_factory=list)

    @classmethod
    def salvage(cls, data: Any) -> "FeatureProposalBatch":
        """Validate proposals one at a time, keeping whatever is usable.

        Used when strict validation of the whole batch fails. Raises if nothing
        can be recovered, so the caller still falls back to the heuristic.
        """
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        raw = data.get("proposals")
        if not isinstance(raw, list):
            raise ValueError("'proposals' missing or not a list")
        kept: list[FeatureProposal] = []
        for item in raw:
            try:
                kept.append(FeatureProposal.model_validate(item))
            except Exception:
                continue  # drop just this proposal, keep the rest
        if not kept:
            raise ValueError("no valid proposals in batch")
        return cls(proposals=kept)


# ---------------------------------------------------------------------------
# Stage 3 — Feature selection (LLM output schema)
# ---------------------------------------------------------------------------
class FeatureSubset(BaseModel):
    """One candidate set of columns to keep, composed by the reasoner.

    The LLM sees a ranking table produced by several statistical methods and
    proposes subsets; it never decides alone — every subset is cross-validated
    and compared against deterministic control subsets.
    """
    name: str = Field(..., description="Short label, e.g. 'drop_redundant'.")
    columns: list[str] = Field(..., min_length=1,
                               description="The columns to KEEP in this subset.")
    reasoning: str = Field(default="", description="Why this subset should work.")


class FeatureSubsetBatch(BaseModel):
    subsets: list[FeatureSubset] = Field(default_factory=list)

    @classmethod
    def salvage(cls, data: Any) -> "FeatureSubsetBatch":
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        raw = data.get("subsets")
        if not isinstance(raw, list):
            raise ValueError("'subsets' missing or not a list")
        kept: list[FeatureSubset] = []
        for item in raw:
            try:
                kept.append(FeatureSubset.model_validate(item))
            except Exception:
                continue
        if not kept:
            raise ValueError("no valid subsets in batch")
        return cls(subsets=kept)


# ---------------------------------------------------------------------------
# Stage 4 — Model selection (LLM output schema)
# ---------------------------------------------------------------------------
class ModelChoice(str, Enum):
    """Whitelisted algorithms. Names map to concrete sklearn estimators in ml/."""
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    # Histogram-based boosting: same idea as gradient_boosting but binned, so it
    # scales to hundreds of thousands of rows. Selected automatically on large
    # data (see ml.trainer.valid_models_for).
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    EXTRA_TREES = "extra_trees"
    KNN = "knn"
    SVM = "svm"


class ModelSelectionPlan(BaseModel):
    """The set of algorithms the agent wants to try for this dataset."""
    models_to_try: list[ModelChoice] = Field(..., min_length=1)
    reasoning: str = Field(default="", description="Why these algorithms suit this dataset.")

    @classmethod
    def salvage(cls, data: Any) -> "ModelSelectionPlan":
        """Keep the recognisable model names, drop anything hallucinated."""
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        raw = data.get("models_to_try")
        if not isinstance(raw, list):
            raise ValueError("'models_to_try' missing or not a list")
        kept: list[ModelChoice] = []
        for item in raw:
            try:
                kept.append(ModelChoice(str(item).strip().lower()))
            except Exception:
                continue
        if not kept:
            raise ValueError("no valid model names")
        return cls(models_to_try=kept, reasoning=str(data.get("reasoning", "")))


# ---------------------------------------------------------------------------
# Stage 5 — Hyperparameter tuning (LLM output schema)
# ---------------------------------------------------------------------------
class HyperparameterProposal(BaseModel):
    """One concrete hyperparameter configuration to try next.

    `params` is a free-form dict, but each value is validated at execution time
    against the estimator's accepted parameters, so an invalid suggestion is
    caught and skipped rather than crashing.
    """
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(default="", description="Why try this configuration next, given results so far.")


# ---------------------------------------------------------------------------
# Internal results structures (not LLM output — produced by the ML layer)
# ---------------------------------------------------------------------------
class TrainingResult(BaseModel):
    """The real, measured outcome of training one configuration."""
    model_config = {"protected_namespaces": ()}  # allow field names starting with 'model_'

    model_name: str
    score: float                       # primary metric (higher is always better here)
    primary_metric: str                # e.g. "f1_macro" or "r2"
    all_metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    train_seconds: float = 0.0


class DecisionRecord(BaseModel):
    """A single logged decision for the final explainability report."""
    stage: str
    action: str                        # what was tried
    reasoning: str                     # why (from the LLM)
    score_before: float | None = None
    score_after: float | None = None
    accepted: bool = False
    detail: str = ""                   # human-readable outcome line
    source: str = ""                   # "llm" | "heuristic" | "" — who proposed it
