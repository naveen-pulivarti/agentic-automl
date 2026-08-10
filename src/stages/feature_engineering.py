"""
Stage 2 — Agentic Feature Engineering.

A genuine loop, not a single request. Each round:

    PROPOSE  the reasoner suggests a few features, having been told what
             happened to its previous suggestions
    EXECUTE  build each feature (whitelisted operations only)
    MEASURE  train with vs without it
    DECIDE   keep it only if the real score improved

Feeding measured outcomes back between rounds is the point. Given one blind
request a small model will confidently propose extracting a calendar month from
a latitude; told that this was rejected and why, it stops. That feedback is what
makes this stage agentic rather than a filter over a single guess.

Proposals are also validated *before* execution. Every unchecked proposal costs
a full cross-validation to discover it was useless, so rejecting an impossible
one up front is free.
"""
from __future__ import annotations

import json

import pandas as pd

from ..agent.llm import get_reasoner
from ..agent.schemas import (
    DecisionRecord, FeatureOperation, FeatureProposal,
    FeatureProposalBatch,
)
from ..agent.state import AutoMLState
from ..config import CONFIG
from ..ml.feature_ops import apply_feature, validate_proposal
from ..ml.trainer import evaluate_model, proxy_model_for
from ..utils.budget import StageTimer
from ..utils.logger import get_logger

log = get_logger(__name__)

STAGE = "Feature Engineering"

_SYSTEM = """You are an expert data scientist doing feature engineering.
Given a dataset profile, propose new features likely to improve a model.
You MUST respond with ONLY valid JSON matching this schema:
{
  "proposals": [
    {
      "feature_name": "snake_case_name",
      "operation": "ratio|product|sum|difference|log|square|binning|interaction|date_part",
      "source_columns": ["col1", "col2"],
      "date_component": "year|month|day|dayofweek|hour or null",
      "reasoning": "short why"
    }
  ]
}
Hard rules — a proposal breaking any of these is discarded:
- ratio/product/sum/difference/interaction need EXACTLY 2 source_columns.
- log/square/binning need EXACTLY 1 numeric source column.
- date_part needs EXACTLY 1 real date column plus a date_component.
  Never use date_part on a number: a latitude or a count is not a date.
- interaction is for low-cardinality categorical columns only.
- Only reference columns that exist, and NEVER use the target column."""


def _propose(state: AutoMLState, n: int, outcomes: list[dict],
             already: set[str]) -> tuple[FeatureProposalBatch, bool]:
    """Ask the reasoner for the next batch, including what happened before."""
    profile = state.profile
    context = {
        "problem_type": state.problem_type.value,
        "target": state.target,
        "n_rows": profile.n_rows,
        "columns": profile.to_summary_dict()["columns"],
    }
    user = ("Dataset profile:\n" + json.dumps(context, indent=2)
            + f"\n\nPropose exactly {n} high-value features as JSON.")

    if outcomes:
        user += (
            "\n\nMEASURED results of your previous proposals — these are real "
            "cross-validation outcomes, not opinions:\n"
            + json.dumps(outcomes, indent=2)
            + "\n\nLearn from this. Do NOT repeat any feature already listed, "
              "and do NOT repeat an operation/column combination that was "
              "rejected as invalid. Build on what was accepted."
        )

    def fallback() -> FeatureProposalBatch:
        return _heuristic_proposals(state, n, already)

    return get_reasoner().propose_json(
        system=_SYSTEM, user=user,
        schema=FeatureProposalBatch, fallback=fallback,
    )


def _heuristic_proposals(state: AutoMLState, budget: int,
                         already: set[str] | None = None) -> FeatureProposalBatch:
    """Deterministic fallback: build sensible features without any LLM.

    Ensures the stage works with zero LLM availability. `already` lets later
    rounds continue where earlier ones stopped instead of re-proposing the same
    features every time.
    """
    already = already or set()
    profile = state.profile
    num = profile.numeric_features()
    dates = profile.datetime_features()
    cats = profile.categorical_features()
    proposals: list[FeatureProposal] = []

    def add(p: FeatureProposal) -> None:
        if p.feature_name not in already and len(proposals) < budget:
            proposals.append(p)

    # ratios between numeric pairs — cheap and often informative
    for i in range(len(num)):
        for j in range(len(num)):
            if i != j:
                add(FeatureProposal(
                    feature_name=f"{num[i]}_over_{num[j]}",
                    operation=FeatureOperation.RATIO,
                    source_columns=[num[i], num[j]],
                    reasoning="Ratios often expose relationships between magnitudes.",
                ))
    # products capture simple interactions
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            add(FeatureProposal(
                feature_name=f"{num[i]}_x_{num[j]}",
                operation=FeatureOperation.PRODUCT,
                source_columns=[num[i], num[j]],
                reasoning="Products capture multiplicative interactions.",
            ))
    # log-transform skewed numerics
    for c in num:
        add(FeatureProposal(
            feature_name=f"log_{c}",
            operation=FeatureOperation.LOG,
            source_columns=[c],
            reasoning="Log transform can linearise skewed features.",
        ))
    # date parts
    for c in dates:
        for comp in ("month", "dayofweek"):
            add(FeatureProposal(
                feature_name=f"{c}_{comp}",
                operation=FeatureOperation.DATE_PART,
                source_columns=[c],
                date_component=comp,        # type: ignore[arg-type]
                reasoning="Calendar structure often carries seasonal signal.",
            ))
    # categorical interactions
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            add(FeatureProposal(
                feature_name=f"{cats[i]}_{cats[j]}_combo",
                operation=FeatureOperation.INTERACTION,
                source_columns=[cats[i], cats[j]],
                reasoning="Combined categories can separate classes single ones cannot.",
            ))

    return FeatureProposalBatch(proposals=proposals[:budget])


def run_feature_engineering(state: AutoMLState) -> AutoMLState:
    timer = StageTimer(STAGE)
    state.emit(STAGE, "Establishing baseline score...")
    problem = state.problem_type
    X, y = state.X, state.y
    proxy = proxy_model_for(len(X))

    # --- baseline: score with the original features only ---
    baseline = evaluate_model(X, y, proxy, problem)
    state.baseline_score = baseline.score
    state.record(DecisionRecord(
        stage=STAGE,
        action="Baseline (no engineered features)",
        reasoning="Reference point to judge whether new features help.",
        score_after=baseline.score,
        accepted=True,
        detail=f"Baseline {baseline.primary_metric} = {baseline.score:.4f} "
               f"using {proxy.value} on {X.shape[1]} columns",
        source="heuristic",
    ))

    rounds = CONFIG.budget.fe_rounds
    per_round = CONFIG.budget.fe_per_round
    current_score = baseline.score
    outcomes: list[dict] = []          # measured memory, fed back each round
    seen: set[str] = set()             # feature names already proposed
    kept = 0
    considered = 0

    for rnd in range(1, rounds + 1):
        if timer.exhausted():
            state.emit(STAGE, f"Time budget reached; stopping after round {rnd - 1}.")
            break

        state.emit(STAGE, f"Round {rnd}/{rounds}: asking the agent for features...")
        batch, used_llm = _propose(state, per_round, outcomes, seen)
        state.note_llm(STAGE, used_llm)
        source = "llm" if used_llm else "heuristic"
        log.info("Round %d proposed %d features (llm=%s)",
                 rnd, len(batch.proposals), used_llm)

        for proposal in batch.proposals[:per_round]:
            if timer.exhausted():
                break
            if proposal.feature_name in seen:
                continue
            seen.add(proposal.feature_name)
            considered += 1

            # -- validate before spending a cross-validation on it ----------
            reason = validate_proposal(X, proposal, target=state.target)
            if reason is not None:
                state.record(DecisionRecord(
                    stage=STAGE,
                    action=f"Feature '{proposal.feature_name}' ({proposal.operation.value})",
                    reasoning=proposal.reasoning,
                    accepted=False,
                    detail=f"Rejected before testing: {reason}.",
                    source=source,
                ))
                state.rejected_features.append(proposal.feature_name)
                outcomes.append({
                    "feature": proposal.feature_name,
                    "operation": proposal.operation.value,
                    "columns": proposal.source_columns,
                    "accepted": False,
                    "why": f"invalid: {reason}",
                })
                continue

            state.emit(STAGE, f"Testing '{proposal.feature_name}'...")
            new_col = apply_feature(X, proposal)
            if new_col is None or new_col.notna().sum() == 0:
                state.record(DecisionRecord(
                    stage=STAGE,
                    action=f"Feature '{proposal.feature_name}' ({proposal.operation.value})",
                    reasoning=proposal.reasoning,
                    accepted=False,
                    detail="Could not be constructed / all-null; skipped.",
                    source=source,
                ))
                state.rejected_features.append(proposal.feature_name)
                outcomes.append({
                    "feature": proposal.feature_name,
                    "operation": proposal.operation.value,
                    "columns": proposal.source_columns,
                    "accepted": False,
                    "why": "could not be built from this data",
                })
                continue

            # -- measure: train with the candidate feature added -------------
            X_trial = X.copy()
            X_trial[proposal.feature_name] = new_col.to_numpy()
            try:
                trial = evaluate_model(X_trial, y, proxy, problem)
            except Exception as e:
                log.warning("Feature %s evaluation failed: %s",
                            proposal.feature_name, e)
                continue

            improved = trial.score > current_score + 1e-6
            if improved:
                X[proposal.feature_name] = new_col.to_numpy()   # commit
                state.accepted_features.append(proposal.feature_name)
                state.accepted_proposals.append(proposal)
                state.record(DecisionRecord(
                    stage=STAGE,
                    action=f"Feature '{proposal.feature_name}' ({proposal.operation.value})",
                    reasoning=proposal.reasoning,
                    score_before=current_score,
                    score_after=trial.score,
                    accepted=True,
                    detail=f"Kept: {trial.primary_metric} {current_score:.4f} -> {trial.score:.4f}",
                    source=source,
                ))
                current_score = trial.score
                kept += 1
            else:
                state.rejected_features.append(proposal.feature_name)
                state.record(DecisionRecord(
                    stage=STAGE,
                    action=f"Feature '{proposal.feature_name}' ({proposal.operation.value})",
                    reasoning=proposal.reasoning,
                    score_before=current_score,
                    score_after=trial.score,
                    accepted=False,
                    detail=f"Rejected: no improvement ({trial.score:.4f} <= {current_score:.4f})",
                    source=source,
                ))

            outcomes.append({
                "feature": proposal.feature_name,
                "operation": proposal.operation.value,
                "columns": proposal.source_columns,
                "accepted": improved,
                "score": round(trial.score, 4),
            })

    state.X = X
    state.engineered_score = current_score
    state.stage_seconds[STAGE] = timer.elapsed
    state.emit(STAGE,
               f"Done. Kept {kept} of {considered} candidates "
               f"({baseline.score:.4f} -> {current_score:.4f}).")
    log.info("Feature engineering kept %d/%d features; score %.4f -> %.4f (%s)",
             kept, considered, baseline.score, current_score, timer.summary())
    return state
