"""
Shared pipeline state.

A single AutoMLState object flows through every stage (whether orchestrated by
LangGraph or the plain-Python fallback runner). Stages read what they need and
write their results back. Keeping all state in one typed object makes the whole
run inspectable and the final report easy to assemble.

Note the deliberate split between `X`/`y` and `X_test`/`y_test`. The test set is
separated during profiling and no search stage is ever given access to it. Every
number the agent optimises comes from cross-validation on the training portion
only; the test set is opened exactly once, at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .schemas import (
    DecisionRecord, ModelChoice, ProblemType, TrainingResult,
)
from ..data.profiler import DatasetProfile


# A callback the UI can pass in to receive live progress updates.
ProgressFn = Callable[[str, str], None]  # (stage, message) -> None


@dataclass
class AutoMLState:
    # --- inputs -----------------------------------------------------------
    df: pd.DataFrame
    target: str

    # --- filled by Stage 1 ------------------------------------------------
    profile: DatasetProfile | None = None
    problem_type: ProblemType | None = None

    # Training features/target. `X` grows as features are accepted and shrinks
    # when feature selection prunes it. The target is kept separate as `y`.
    X: pd.DataFrame | None = None
    y: pd.Series | None = None

    # Held-out test set — untouched by every search stage.
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None

    # --- Stage 2: feature engineering -------------------------------------
    # The accepted proposals are kept (not just their names) because the same
    # transformations must be rebuilt on the held-out test set at the end.
    accepted_proposals: list[Any] = field(default_factory=list)
    accepted_features: list[str] = field(default_factory=list)
    rejected_features: list[str] = field(default_factory=list)
    baseline_score: float | None = None      # score before any feature work
    engineered_score: float | None = None    # score after engineering

    # --- Stage 3: feature selection ---------------------------------------
    ranking_table: pd.DataFrame | None = None
    selected_features: list[str] = field(default_factory=list)
    dropped_features: list[str] = field(default_factory=list)
    selection_score: float | None = None
    selection_winner: str = ""               # which candidate subset won

    # --- Stage 4: model selection -----------------------------------------
    model_results: list[TrainingResult] = field(default_factory=list)
    best_model: ModelChoice | None = None
    best_model_score: float | None = None

    # --- Stage 5: tuning ---------------------------------------------------
    best_params: dict[str, Any] = field(default_factory=dict)
    tuned_score: float | None = None

    # --- final ------------------------------------------------------------
    final_metrics: dict[str, float] = field(default_factory=dict)
    fitted_pipeline: Any = None

    # --- audit trail (drives the explainability report) -------------------
    decisions: list[DecisionRecord] = field(default_factory=list)
    used_llm_anywhere: bool = False
    llm_stages: list[str] = field(default_factory=list)   # which stages really used the LLM
    stage_seconds: dict[str, float] = field(default_factory=dict)

    # --- infra ------------------------------------------------------------
    progress: ProgressFn | None = None

    # -- helpers -----------------------------------------------------------
    def emit(self, stage: str, message: str) -> None:
        if self.progress:
            self.progress(stage, message)

    def record(self, rec: DecisionRecord) -> None:
        self.decisions.append(rec)

    def note_llm(self, stage: str, used_llm: bool) -> None:
        """Track honestly which stages were driven by the LLM vs a fallback."""
        self.used_llm_anywhere = self.used_llm_anywhere or used_llm
        if used_llm and stage not in self.llm_stages:
            self.llm_stages.append(stage)

    @property
    def current_best_score(self) -> float | None:
        for candidate in (self.tuned_score, self.best_model_score,
                          self.selection_score, self.engineered_score,
                          self.baseline_score):
            if candidate is not None:
                return candidate
        return None

    def score_progression(self) -> list[tuple[str, float]]:
        """Ordered (label, score) pairs for the progression chart/report."""
        steps = [
            ("Baseline", self.baseline_score),
            ("After feature engineering", self.engineered_score),
            ("After feature selection", self.selection_score),
            ("After model selection", self.best_model_score),
            ("After tuning", self.tuned_score),
        ]
        return [(label, float(v)) for label, v in steps if v is not None]
