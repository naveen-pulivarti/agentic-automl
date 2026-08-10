"""
Stage 5 — Agentic Hyperparameter Tuning.

Instead of brute-force grid/random search, the agent reasons about which
configuration to try NEXT based on the results seen so far. Each iteration:
PROPOSE (LLM suggests params given history) -> EXECUTE + MEASURE (cross-validate)
-> DECIDE (keep as best if improved). Bounded by max_tuning_iterations and by a
wall-clock budget.

Three properties this stage has to guarantee, all learned from measurement:

* **No duplicate work.** Every configuration is fingerprinted and never
  evaluated twice. Without this the fallback cycled `iteration % len(space)` and
  simply re-ran the same settings once the budget exceeded the grid size.
* **Stop when there is nothing left to try.** A model with no meaningful
  hyperparameters (plain linear regression) previously consumed the entire
  budget re-measuring an identical empty configuration, filling the
  explainability report with meaningless rows.
* **A search space for every model in the whitelist.** A missing entry silently
  degraded tuning to a no-op.
"""
from __future__ import annotations

import json

from ..agent.llm import get_reasoner
from ..agent.schemas import (
    DecisionRecord, HyperparameterProposal, ModelChoice,
)
from ..agent.state import AutoMLState
from ..config import CONFIG
from ..ml.trainer import evaluate_model
from ..utils.budget import StageTimer
from ..utils.logger import get_logger

log = get_logger(__name__)

STAGE = "Hyperparameter Tuning"

_SYSTEM = """You are an expert ML engineer tuning hyperparameters.
Given the model type and the history of (params -> score) tried so far, propose
the SINGLE next configuration most likely to improve the score. Reason about the
trend (e.g. if deeper trees helped, go deeper; if the score fell, step back).
Respond ONLY with JSON:
{
  "params": {"param_name": value, ...},
  "reasoning": "short why, referencing the trend"
}
Use only valid scikit-learn parameter names for the given model, and never
repeat a configuration that already appears in the history."""

# Reasonable parameter search spaces per model (also used by the heuristic).
# Every model in the whitelist that HAS tunable parameters appears here.
_SEARCH_SPACE: dict[ModelChoice, list[dict]] = {
    ModelChoice.RANDOM_FOREST: [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 300, "max_depth": 20},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": 30, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 4},
    ],
    ModelChoice.EXTRA_TREES: [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 300, "max_depth": 20},
        {"n_estimators": 400, "max_depth": 30, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_features": "sqrt"},
    ],
    ModelChoice.GRADIENT_BOOSTING: [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5},
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3},
    ],
    ModelChoice.HIST_GRADIENT_BOOSTING: [
        {"learning_rate": 0.1, "max_iter": 200},
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 63},
        {"learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 40},
        {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
         "l2_regularization": 1.0},
    ],
    ModelChoice.LOGISTIC_REGRESSION: [
        {"C": 0.1}, {"C": 1.0}, {"C": 10.0}, {"C": 0.01},
    ],
    ModelChoice.RIDGE: [
        {"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0},
    ],
    ModelChoice.KNN: [
        {"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 11},
        {"n_neighbors": 15, "weights": "distance"},
    ],
    ModelChoice.SVM: [
        {"C": 1.0, "kernel": "rbf"}, {"C": 10.0, "kernel": "rbf"},
        {"C": 1.0, "kernel": "linear"},
    ],
    # LINEAR_REGRESSION is deliberately absent: it has no hyperparameters worth
    # searching, and the stage detects that and stops rather than burning budget.
}


def _fingerprint(params: dict) -> str:
    """Canonical form of a configuration, so duplicates are detectable."""
    return json.dumps(params, sort_keys=True, default=str)


def _next_unused(model: ModelChoice, tried: set[str]) -> dict | None:
    """Next configuration from the grid that hasn't been evaluated yet."""
    for params in _SEARCH_SPACE.get(model, []):
        if _fingerprint(params) not in tried:
            return params
    return None


def _llm_next(state: AutoMLState, model: ModelChoice, history: list[dict],
              tried: set[str]) -> tuple[HyperparameterProposal, bool]:
    context = {
        "model": model.value,
        "problem_type": state.problem_type.value,
        "history": history,  # [{"params":..., "score":...}, ...]
        "primary_metric": ("f1_macro"
                           if state.problem_type.value == "classification" else "r2"),
    }
    user = ("Tuning context:\n" + json.dumps(context, indent=2, default=str)
            + "\n\nPropose the next configuration as JSON.")

    def fallback() -> HyperparameterProposal:
        params = _next_unused(model, tried)
        return HyperparameterProposal(
            params=params or {},
            reasoning="Systematic exploration of a sensible parameter grid.",
        )

    return get_reasoner().propose_json(
        system=_SYSTEM, user=user,
        schema=HyperparameterProposal, fallback=fallback,
    )


def run_tuning(state: AutoMLState) -> AutoMLState:
    timer = StageTimer(STAGE)
    model = state.best_model
    state.emit(STAGE, f"Tuning {model.value}...")

    best_params: dict = {}
    best_score = state.best_model_score if state.best_model_score is not None else -1e9
    history: list[dict] = [{"params": {}, "score": round(best_score, 4)}]
    tried: set[str] = {_fingerprint({})}

    state.record(DecisionRecord(
        stage=STAGE,
        action=f"Untuned {model.value}",
        reasoning="Starting point for tuning.",
        score_after=best_score,
        accepted=True,
        detail=f"Baseline score {best_score:.4f}",
        source="heuristic",
    ))

    # Models with nothing to tune: say so and stop, rather than re-measuring the
    # same configuration until the budget runs out.
    if not _SEARCH_SPACE.get(model):
        state.best_params = {}
        state.tuned_score = best_score
        state.record(DecisionRecord(
            stage=STAGE,
            action=f"No tuning performed for {model.value}",
            reasoning="This algorithm has no hyperparameters worth searching, "
                      "so the untuned model is already its best form.",
            accepted=True,
            detail="Tuning budget released back to the run.",
            source="heuristic",
        ))
        state.emit(STAGE, f"{model.value} has no parameters to tune; skipping.")
        return state

    budget = CONFIG.budget.max_tuning_iterations
    for i in range(budget):
        if timer.exhausted():
            state.emit(STAGE, "Time budget reached; stopping tuning.")
            break

        proposal, used_llm = _llm_next(state, model, history, tried)
        state.note_llm(STAGE, used_llm)
        source = "llm" if used_llm else "heuristic"

        # Never evaluate the same configuration twice; fall back to the next
        # unused grid point, and stop entirely once the space is exhausted.
        if _fingerprint(proposal.params) in tried:
            alt = _next_unused(model, tried)
            if alt is None:
                state.emit(STAGE, "Search space exhausted; stopping early.")
                log.info("Tuning stopped early: no unexplored configurations left.")
                break
            proposal = HyperparameterProposal(
                params=alt,
                reasoning="Proposed configuration was already tried; moving to "
                          "the next unexplored point in the grid.",
            )
            source = "heuristic"

        tried.add(_fingerprint(proposal.params))
        state.emit(STAGE, f"Trying config {i + 1}/{budget}: {proposal.params}")

        try:
            result = evaluate_model(state.X, state.y, model,
                                    state.problem_type, proposal.params)
        except Exception as e:
            log.warning("Tuning config failed (%s): %s", proposal.params, e)
            state.record(DecisionRecord(
                stage=STAGE,
                action=f"Config {proposal.params}",
                reasoning=proposal.reasoning,
                accepted=False,
                detail=f"Invalid/failed config: {e}",
                source=source,
            ))
            history.append({"params": proposal.params, "score": None})
            continue

        history.append({"params": proposal.params, "score": round(result.score, 4)})
        improved = result.score > best_score + 1e-6
        state.record(DecisionRecord(
            stage=STAGE,
            action=f"Config {proposal.params}",
            reasoning=proposal.reasoning,
            score_before=best_score,
            score_after=result.score,
            accepted=improved,
            detail=(f"Kept (new best {result.score:.4f})" if improved
                    else f"No improvement ({result.score:.4f} <= {best_score:.4f})"),
            source=source,
        ))
        if improved:
            best_score = result.score
            best_params = proposal.params

    state.best_params = best_params
    state.tuned_score = best_score
    state.stage_seconds[STAGE] = timer.elapsed
    state.emit(STAGE, f"Best score after tuning: {best_score:.4f}")
    log.info("Tuning done. Best params=%s score=%.4f (%s)",
             best_params, best_score, timer.summary())
    return state
