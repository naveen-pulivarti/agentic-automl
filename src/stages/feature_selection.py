"""
Stage 3 — Agentic Feature Selection.

Feature engineering only ever *adds* columns, and it adds them greedily: a
candidate is judged on its own, in the order it happened to arrive, and is never
reconsidered afterwards. That leaves two blind spots — original columns are
never questioned, and a feature accepted early can be made redundant by one
accepted later.

This stage closes both, and it is the one place the pipeline reasons about
combinations of columns rather than one column at a time:

    EVIDENCE   ten ranking methods score every column (ml/feature_ranking.py)
    PROPOSE    the reasoner reads the ranking table and composes candidate
               subsets, explaining each
    EXECUTE    every candidate is cross-validated for real
    DECIDE     the best measured subset wins

Two deterministic control subsets are *always* evaluated alongside the
reasoner's: the full column set, and plain top-k by mean rank. They serve two
purposes — the stage still works with no LLM at all, and they are the baseline
the reasoner has to beat for its contribution to be demonstrable rather than
assumed.

Ties are broken toward fewer columns. If dropping columns costs nothing
measurable, the simpler model is the better one: less overfitting, faster to
train, and easier to explain.
"""
from __future__ import annotations

import json

import pandas as pd

from ..agent.llm import get_reasoner
from ..agent.schemas import DecisionRecord, FeatureSubset, FeatureSubsetBatch
from ..agent.state import AutoMLState
from ..config import CONFIG
from ..ml.feature_ranking import (
    rank_features, ranking_summary_for_llm, top_k_features,
)
from ..ml.trainer import evaluate_model, proxy_model_for
from ..utils.budget import StageTimer
from ..utils.logger import get_logger

log = get_logger(__name__)

STAGE = "Feature Selection"

_SYSTEM = """You are an expert data scientist choosing which columns to keep.
You are given a table where several independent statistical methods have each
ranked every column (rank 1 = most useful). The methods disagree on purpose:

- pearson / spearman / f_test / l1_coef see LINEAR or monotonic relationships.
  They report null for categorical columns, where they do not apply.
- mutual_info sees NON-LINEAR relationships a correlation would miss.
- rf_gain / permutation / rfe measure real effect on a fitted model.
- mrmr penalises a column that duplicates information already in another.
- variance flags near-constant columns, which are useless whatever else says.

Use the disagreement. A column ranked badly by pearson but well by mutual_info
usually has a non-linear relationship and is worth keeping. A column ranked well
everywhere but poorly by mrmr is probably redundant with another column.

Respond with ONLY valid JSON:
{
  "subsets": [
    {"name": "short_label",
     "columns": ["col_a", "col_b"],
     "reasoning": "why this set should work"}
  ]
}
Propose 3-4 genuinely DIFFERENT subsets — e.g. one aggressive (few columns),
one conservative (most columns), one targeting redundancy. Only use column names
that appear in the table. Never invent a column."""


def _propose_subsets(state: AutoMLState, table: pd.DataFrame,
                     all_cols: list[str]) -> tuple[FeatureSubsetBatch, bool]:
    """Ask the reasoner to compose candidate subsets from the ranking evidence."""
    context = {
        "problem_type": state.problem_type.value,
        "n_rows": int(len(state.X)),
        "n_columns": len(all_cols),
        "ranking_table": ranking_summary_for_llm(table),
    }
    user = ("Column ranking evidence:\n" + json.dumps(context, indent=2, default=str)
            + f"\n\nPropose up to {CONFIG.budget.max_llm_subsets} candidate "
              "subsets as JSON.")

    def fallback() -> FeatureSubsetBatch:
        # The deterministic control subsets already cover the no-LLM case, so
        # the fallback adds nothing rather than inventing something arbitrary.
        return FeatureSubsetBatch(subsets=[])

    return get_reasoner().propose_json(
        system=_SYSTEM, user=user,
        schema=FeatureSubsetBatch, fallback=fallback,
    )


def _control_subsets(table: pd.DataFrame, all_cols: list[str]) -> list[FeatureSubset]:
    """Deterministic candidates: everything, plus top-k by mean rank."""
    subsets = [FeatureSubset(
        name="all_features",
        columns=list(all_cols),
        reasoning="Keep every column — the baseline this stage must beat.",
    )]
    n = len(all_cols)
    seen_sizes = {n}
    for frac in CONFIG.budget.control_subset_fractions:
        k = max(1, int(round(frac * n)))
        if k in seen_sizes:
            continue
        seen_sizes.add(k)
        cols = top_k_features(table, k)
        if cols:
            subsets.append(FeatureSubset(
                name=f"top_{int(frac * 100)}pct_by_rank",
                columns=cols,
                reasoning=f"Statistics only: the {k} best columns by mean rank "
                          f"across all methods, no LLM involved.",
            ))
    return subsets


def _sanitise(subset: FeatureSubset, all_cols: list[str]) -> FeatureSubset | None:
    """Drop invented column names; discard the subset if nothing usable remains."""
    valid = [c for c in dict.fromkeys(subset.columns) if c in all_cols]
    if not valid:
        return None
    return FeatureSubset(name=subset.name, columns=valid, reasoning=subset.reasoning)


def run_feature_selection(state: AutoMLState) -> AutoMLState:
    timer = StageTimer(STAGE)
    X, y = state.X, state.y
    problem = state.problem_type
    all_cols = list(X.columns)

    # Nothing to select from.
    if len(all_cols) < 3:
        state.selected_features = all_cols
        state.selection_score = state.engineered_score
        state.record(DecisionRecord(
            stage=STAGE,
            action="Skipped feature selection",
            reasoning="Too few columns for selection to be meaningful.",
            accepted=True,
            detail=f"{len(all_cols)} column(s) present.",
            source="heuristic",
        ))
        return state

    # --- evidence: rank every column with the full method panel ------------
    state.emit(STAGE, "Ranking columns with 10 statistical methods...")
    table = rank_features(X, y, problem)
    state.ranking_table = table

    if table.empty:
        state.selected_features = all_cols
        state.selection_score = state.engineered_score
        log.warning("Ranking produced no table; skipping selection.")
        return state

    methods = [c.replace("rank_", "") for c in table.columns if c.startswith("rank_")]
    state.record(DecisionRecord(
        stage=STAGE,
        action=f"Ranked {len(table)} columns using {len(methods)} methods",
        reasoning="Different methods detect different kinds of relationship, so "
                  "agreement between them is stronger evidence than any single score.",
        accepted=True,
        detail="Methods: " + ", ".join(methods)
               + ". Best by mean rank: "
               + ", ".join(str(i) for i in table.head(5).index),
        source="heuristic",
    ))

    # --- candidates: controls first, then whatever the reasoner adds -------
    candidates = _control_subsets(table, all_cols)

    state.emit(STAGE, "Asking the agent to compose candidate feature sets...")
    batch, used_llm = _propose_subsets(state, table, all_cols)
    state.note_llm(STAGE, used_llm)
    for s in batch.subsets[:CONFIG.budget.max_llm_subsets]:
        clean = _sanitise(s, all_cols)
        if clean is None:
            continue
        # skip an LLM subset identical to a control we already have
        if any(set(clean.columns) == set(c.columns) for c in candidates):
            continue
        candidates.append(clean)

    if used_llm:
        state.record(DecisionRecord(
            stage=STAGE,
            action=f"Agent proposed {len(batch.subsets)} candidate subsets",
            reasoning=batch.subsets[0].reasoning if batch.subsets else "",
            accepted=True,
            detail="Each is cross-validated against statistics-only controls.",
            source="llm",
        ))

    # --- verify every candidate for real -----------------------------------
    proxy = proxy_model_for(len(X))
    results: list[tuple[FeatureSubset, float]] = []

    for cand in candidates:
        if timer.exhausted():
            state.emit(STAGE, "Time budget reached; stopping subset evaluation.")
            break
        state.emit(STAGE, f"Evaluating subset '{cand.name}' ({len(cand.columns)} columns)...")
        try:
            res = evaluate_model(X[cand.columns], y, proxy, problem)
        except Exception as e:
            log.warning("Subset %s failed: %s", cand.name, e)
            state.record(DecisionRecord(
                stage=STAGE,
                action=f"Subset '{cand.name}' ({len(cand.columns)} columns)",
                reasoning=cand.reasoning,
                accepted=False,
                detail=f"Failed to evaluate: {e}",
                source="llm" if cand.name not in ("all_features",) else "heuristic",
            ))
            continue
        results.append((cand, res.score))

    if not results:
        state.selected_features = all_cols
        state.selection_score = state.engineered_score
        log.warning("No subset could be evaluated; keeping all columns.")
        return state

    # --- decide: best score, ties broken toward fewer columns --------------
    best_score = max(s for _, s in results)
    tolerance = CONFIG.budget.selection_tolerance
    contenders = [(c, s) for c, s in results if s >= best_score - tolerance]
    winner, winner_score = min(contenders, key=lambda cs: len(cs[0].columns))

    for cand, score in results:
        is_winner = cand.name == winner.name
        state.record(DecisionRecord(
            stage=STAGE,
            action=f"Subset '{cand.name}' ({len(cand.columns)} columns)",
            reasoning=cand.reasoning,
            score_before=state.engineered_score,
            score_after=score,
            accepted=is_winner,
            detail=(f"score {score:.4f}" + (" — SELECTED" if is_winner else "")),
            source="heuristic" if cand.name.startswith(("all_", "top_")) else "llm",
        ))

    dropped = [c for c in all_cols if c not in winner.columns]
    state.X = X[winner.columns]
    state.selected_features = list(winner.columns)
    state.dropped_features = dropped
    state.selection_score = winner_score
    state.selection_winner = winner.name
    state.stage_seconds[STAGE] = timer.elapsed

    state.emit(STAGE,
               f"Selected '{winner.name}': kept {len(winner.columns)} of "
               f"{len(all_cols)} columns (score {winner_score:.4f}).")
    log.info("Feature selection winner=%s kept=%d dropped=%d score=%.4f (%s)",
             winner.name, len(winner.columns), len(dropped), winner_score,
             timer.summary())
    return state
