"""
Orchestration.

The five stages are wired into a LangGraph StateGraph: each stage is a node, and
edges define the flow profiling -> feature engineering -> feature selection ->
model selection -> tuning -> finalise. LangGraph gives structured, inspectable
control over the agent's decision flow.

A plain-Python `run_pipeline_sequential` is provided as a fallback so the system
runs even if LangGraph isn't installed — same stages, same order.

The single most important thing this module does is **split the test set before
any stage runs**. Every search stage optimises a cross-validation score, and a
score computed over rows that later serve as the "held-out" test set is not an
independent estimate of anything — the more candidates the agent tries, the more
it fits that particular sample's noise. Splitting first is what makes the final
numbers honest.
"""
from __future__ import annotations

import time

import pandas as pd
from sklearn.model_selection import train_test_split

from .schemas import DecisionRecord, ProblemType
from .state import AutoMLState
from ..config import CONFIG
from ..data.profiler import profile_dataset
from ..ml.feature_ops import apply_feature
from ..ml.trainer import fit_and_evaluate
from ..stages.feature_engineering import run_feature_engineering
from ..stages.feature_selection import run_feature_selection
from ..stages.model_selection import run_model_selection
from ..stages.tuning import run_tuning
from ..utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: profiling + split
# ---------------------------------------------------------------------------
def node_profile(state: AutoMLState) -> AutoMLState:
    t0 = time.time()
    state.emit("Profiling", "Analysing dataset structure...")
    profile = profile_dataset(state.df, state.target)
    state.profile = profile
    state.problem_type = profile.problem_type

    # Split X / y and drop columns that cannot help any model: free text
    # (classical ML can't use it), constants (no information), and identifiers
    # (a row id often correlates perfectly with the target in a sorted export,
    # which produces a model that looks brilliant and generalises to nothing).
    unusable = {"text": [], "constant": [], "identifier": []}
    for c in profile.columns:
        if c.name != state.target and c.kind in unusable:
            unusable[c.kind].append(c.name)

    drop_cols = [state.target] + [n for names in unusable.values() for n in names]
    X = state.df.drop(columns=[c for c in drop_cols if c in state.df.columns])
    y = state.df[state.target]

    # drop rows with a missing target — they cannot teach or test anything
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # --- lock the test set away BEFORE any stage sees the data ------------
    stratify = None
    if profile.problem_type == ProblemType.CLASSIFICATION:
        counts = y.value_counts()
        if counts.min() >= 2:
            stratify = y
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=CONFIG.ml.test_size,
        random_state=CONFIG.ml.random_state, stratify=stratify,
    )
    state.X = X_tr.reset_index(drop=True)
    state.y = y_tr.reset_index(drop=True)
    state.X_test = X_te.reset_index(drop=True)
    state.y_test = y_te.reset_index(drop=True)

    state.record(DecisionRecord(
        stage="Profiling",
        action=f"Detected problem type: {profile.problem_type.value}",
        reasoning=profile.target_detection_reason,
        accepted=True,
        detail=f"{profile.n_rows:,} rows, {profile.n_cols} columns.",
        source="heuristic",
    ))
    _WHY_UNUSABLE = {
        "text": "free text — classical ML cannot use it directly",
        "constant": "the same value in every row — no information",
        "identifier": "a row identifier — it would let the model memorise the "
                      "index rather than learn the pattern",
    }
    for kind, names in unusable.items():
        if names:
            state.record(DecisionRecord(
                stage="Profiling",
                action=f"Dropped {len(names)} {kind} column(s): {', '.join(names)}",
                reasoning=_WHY_UNUSABLE[kind],
                accepted=True,
                detail="Removed automatically before any modelling.",
                source="heuristic",
            ))
    state.record(DecisionRecord(
        stage="Profiling",
        action="Held out a test set before any search began",
        reasoning="Every later stage optimises a cross-validation score. If the "
                  "test rows took part in that search, the final metrics would "
                  "measure the search, not the model.",
        accepted=True,
        detail=f"{len(state.X)} training rows / {len(state.X_test)} test rows "
               f"({int(CONFIG.ml.test_size * 100)}% held out).",
        source="heuristic",
    ))
    state.stage_seconds["Profiling"] = time.time() - t0
    state.emit("Profiling",
               f"{profile.problem_type.value.title()} problem detected "
               f"({len(state.X)} train / {len(state.X_test)} test rows).")
    return state


# ---------------------------------------------------------------------------
# Final stage: rebuild features on the test set, train, and score once
# ---------------------------------------------------------------------------
def node_finalize(state: AutoMLState) -> AutoMLState:
    t0 = time.time()
    state.emit("Finalising", "Rebuilding engineered features on the held-out set...")

    X_test = state.X_test.copy()
    # Rebuild every accepted engineered feature on the test data, using the
    # TRAINING frame as the reference for any data-dependent transform.
    for proposal in state.accepted_proposals:
        if proposal.feature_name in X_test.columns:
            continue
        try:
            col = apply_feature(X_test, proposal, reference=state.X)
        except Exception as e:      # a feature that cannot be rebuilt is fatal
            log.warning("Could not rebuild %s on test set: %s",
                        proposal.feature_name, e)
            col = None
        if col is None:
            # Keep the column present so the pipeline's column set matches;
            # an all-NaN column is imputed by the preprocessor.
            X_test[proposal.feature_name] = float("nan")
        else:
            X_test[proposal.feature_name] = col.to_numpy()

    # align to exactly the columns the model was trained on
    missing = [c for c in state.X.columns if c not in X_test.columns]
    for c in missing:
        X_test[c] = float("nan")
    X_test = X_test[list(state.X.columns)]

    state.emit("Finalising", "Training final model on the full training set...")
    pipe, metrics = fit_and_evaluate(
        state.X, state.y, X_test, state.y_test,
        state.best_model, state.problem_type, state.best_params,
    )
    state.fitted_pipeline = pipe
    state.final_metrics = metrics
    state.X_test = X_test

    state.record(DecisionRecord(
        stage="Finalising",
        action=f"Final model: {state.best_model.value}",
        reasoning="Best model and tuned parameters trained on all training data "
                  "and evaluated once on the untouched test set.",
        accepted=True,
        detail=", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        source="heuristic",
    ))
    state.stage_seconds["Finalising"] = time.time() - t0
    state.emit("Finalising", "Done.")
    return state


# ---------------------------------------------------------------------------
# LangGraph build (optional dependency)
# ---------------------------------------------------------------------------
def build_langgraph():
    """Construct the StateGraph. Raises ImportError if LangGraph missing."""
    from langgraph.graph import END, StateGraph

    # LangGraph works with dict-like state; we wrap our dataclass in a thin dict
    # carrying the object under one key to keep our typed helpers.
    graph = StateGraph(dict)

    def wrap(fn):
        def _inner(s: dict) -> dict:
            state: AutoMLState = s["state"]
            fn(state)
            return {"state": state}
        return _inner

    graph.add_node("profile", wrap(node_profile))
    graph.add_node("feature_engineering", wrap(run_feature_engineering))
    graph.add_node("feature_selection", wrap(run_feature_selection))
    graph.add_node("model_selection", wrap(run_model_selection))
    graph.add_node("tuning", wrap(run_tuning))
    graph.add_node("finalize", wrap(node_finalize))

    graph.set_entry_point("profile")
    graph.add_edge("profile", "feature_engineering")
    graph.add_edge("feature_engineering", "feature_selection")
    graph.add_edge("feature_selection", "model_selection")
    graph.add_edge("model_selection", "tuning")
    graph.add_edge("tuning", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_pipeline(state: AutoMLState, prefer_langgraph: bool = True) -> AutoMLState:
    """Run the full AutoML pipeline. Uses LangGraph if available, else falls
    back to sequential execution (identical stages/order)."""
    if prefer_langgraph:
        try:
            app = build_langgraph()
            result = app.invoke({"state": state})
            return result["state"]
        except ImportError:
            log.warning("LangGraph not installed; using sequential runner.")
        except Exception as e:
            log.warning("LangGraph run failed (%s); using sequential runner.", e)
    return run_pipeline_sequential(state)


def run_pipeline_sequential(state: AutoMLState) -> AutoMLState:
    """Plain-Python orchestration — the reliable fallback."""
    node_profile(state)
    run_feature_engineering(state)
    run_feature_selection(state)
    run_model_selection(state)
    run_tuning(state)
    node_finalize(state)
    return state
