"""
Stage 4 — Agentic Model Selection.

PROPOSE (LLM picks which algorithms suit this dataset) -> EXECUTE + MEASURE
(cross-validate each) -> DECIDE (the one with the best real score wins).

The LLM's role is to shortlist sensible algorithms given the data shape; the
actual winner is decided purely by measured cross-validation scores.

The candidate list handed to the reasoner is filtered by dataset size before it
is even shown (see `ml.trainer.valid_models_for`). Asking a small model to know
that an SVM is quadratic in samples — and would turn a five-minute run on
500,000 rows into an overnight one — is not a reasonable thing to rely on. That
belongs in code.
"""
from __future__ import annotations

import json

from ..agent.llm import get_reasoner
from ..agent.schemas import (
    DecisionRecord, ModelChoice, ModelSelectionPlan,
)
from ..agent.state import AutoMLState
from ..config import CONFIG
from ..ml.trainer import evaluate_model, valid_models_for
from ..utils.budget import StageTimer
from ..utils.logger import get_logger

log = get_logger(__name__)

STAGE = "Model Selection"

_SYSTEM = """You are an expert ML engineer choosing which algorithms to try.
Given a dataset profile and problem type, choose a shortlist of algorithms to
evaluate. Respond with ONLY valid JSON:
{
  "models_to_try": ["random_forest", "gradient_boosting", ...],
  "reasoning": "short justification"
}
You may ONLY choose from the "allowed_models" list given in the user message —
it has already been filtered to algorithms that are appropriate for this
dataset's size. Pick 3-5 that suit the row count and feature types, favouring
diversity (e.g. one linear, one tree ensemble, one boosting)."""


def _llm_plan(state: AutoMLState) -> tuple[ModelSelectionPlan, bool]:
    n_rows = len(state.X)
    allowed = valid_models_for(state.problem_type, n_rows)
    context = {
        "problem_type": state.problem_type.value,
        "n_rows": n_rows,
        "n_features": len(state.X.columns),
        "n_numeric": len(state.profile.numeric_features()),
        "n_categorical": len(state.profile.categorical_features()),
        "allowed_models": [m.value for m in allowed],
    }
    user = ("Dataset context:\n" + json.dumps(context, indent=2)
            + "\n\nChoose 3-5 algorithms to try as JSON.")

    def fallback() -> ModelSelectionPlan:
        return ModelSelectionPlan(
            models_to_try=allowed[:4],
            reasoning="Default shortlist covering linear, tree-ensemble and "
                      "boosting methods appropriate to this dataset size.",
        )

    plan, used_llm = get_reasoner().propose_json(
        system=_SYSTEM, user=user,
        schema=ModelSelectionPlan, fallback=fallback,
    )
    # sanitise: keep only models valid for this task AND this data size
    allowed_set = set(allowed)
    plan.models_to_try = [m for m in dict.fromkeys(plan.models_to_try)
                          if m in allowed_set]
    if not plan.models_to_try:
        plan.models_to_try = allowed[:4]
    return plan, used_llm


def run_model_selection(state: AutoMLState) -> AutoMLState:
    timer = StageTimer(STAGE)
    state.emit(STAGE, "Asking the agent which algorithms to try...")
    plan, used_llm = _llm_plan(state)
    state.note_llm(STAGE, used_llm)
    source = "llm" if used_llm else "heuristic"

    state.record(DecisionRecord(
        stage=STAGE,
        action=f"Shortlist: {', '.join(m.value for m in plan.models_to_try)}",
        reasoning=plan.reasoning,
        accepted=True,
        detail=f"Chosen from algorithms suitable for {len(state.X)} rows.",
        source=source,
    ))

    budget = CONFIG.budget.max_models
    best: tuple[ModelChoice, float] | None = None
    pending: list[tuple[ModelChoice, float]] = []

    for choice in plan.models_to_try[:budget]:
        if timer.exhausted():
            state.emit(STAGE, "Time budget reached; stopping model comparison.")
            break
        state.emit(STAGE, f"Evaluating {choice.value}...")
        try:
            result = evaluate_model(state.X, state.y, choice, state.problem_type)
        except Exception as e:
            log.warning("Model %s failed: %s", choice.value, e)
            state.record(DecisionRecord(
                stage=STAGE,
                action=f"Evaluate {choice.value}",
                reasoning="Candidate algorithm.",
                accepted=False,
                detail=f"Failed to train: {e}",
                source=source,
            ))
            continue

        state.model_results.append(result)
        pending.append((choice, result.score))
        if best is None or result.score > best[1]:
            best = (choice, result.score)

    if best is None:
        raise RuntimeError("No model could be trained on this dataset.")

    # Record every candidate once the winner is known, so exactly one row is
    # marked accepted. (Previously the first model evaluated was always logged
    # as accepted, even when it turned out to be the worst.)
    for choice, score in pending:
        state.record(DecisionRecord(
            stage=STAGE,
            action=f"Evaluate {choice.value}",
            reasoning="Candidate algorithm.",
            score_after=score,
            accepted=(choice == best[0]),
            detail=f"{state.model_results[0].primary_metric} = {score:.4f}"
                   + (" — best" if choice == best[0] else ""),
            source=source,
        ))

    state.best_model, state.best_model_score = best
    state.stage_seconds[STAGE] = timer.elapsed
    state.emit(STAGE,
               f"Best model: {state.best_model.value} ({state.best_model_score:.4f})")
    log.info("Selected best model: %s (%.4f) (%s)",
             state.best_model.value, state.best_model_score, timer.summary())
    return state
