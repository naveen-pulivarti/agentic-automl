"""
Probe the real LLM reasoning layer: latency, JSON validity, and proposal quality.

Runs one feature-engineering proposal and one model-selection proposal against a
real dataset profile, using the project's own Reasoner (so we exercise the exact
prompts and Pydantic schemas the pipeline uses).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

# Derived from this file's location so the harness survives the project
# folder being moved or renamed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.llm import get_reasoner                       # noqa: E402
from src.agent.state import AutoMLState                      # noqa: E402
from src.agent.schemas import FeatureProposalBatch, ModelSelectionPlan  # noqa: E402
from src.data.profiler import profile_dataset                # noqa: E402
from src.stages.feature_engineering import _SYSTEM as FE_SYSTEM, _heuristic_proposals  # noqa: E402
from src.stages.model_selection import _SYSTEM as MS_SYSTEM  # noqa: E402
from src.ml.trainer import valid_models_for                  # noqa: E402
from src.utils.logger import get_token_usage, reset_token_usage  # noqa: E402

DATA = ROOT / "data"


def probe(csv_name: str, target: str) -> None:
    print(f"\n{'='*70}\n{csv_name}  (target={target})\n{'='*70}", flush=True)
    df = pd.read_csv(DATA / csv_name)
    profile = profile_dataset(df, target)
    state = AutoMLState(df=df, target=target)
    state.profile = profile
    state.problem_type = profile.problem_type
    state.X = df.drop(columns=[target])
    state.y = df[target]

    reasoner = get_reasoner()

    # --- health check -------------------------------------------------------
    t0 = time.time()
    ok, info = reasoner.health_check()
    print(f"health_check: ok={ok} info={info}  ({time.time()-t0:.1f}s)", flush=True)

    # --- Stage 2 style call -------------------------------------------------
    reset_token_usage()
    context = {
        "problem_type": state.problem_type.value,
        "target": target,
        "n_rows": profile.n_rows,
        "columns": profile.to_summary_dict()["columns"],
        "max_features_to_propose": 8,
    }
    user = ("Dataset profile:\n" + json.dumps(context, indent=2)
            + "\n\nPropose up to 8 high-value features as JSON.")
    t0 = time.time()
    batch, used_llm = reasoner.propose_json(
        system=FE_SYSTEM, user=user, schema=FeatureProposalBatch,
        fallback=lambda: _heuristic_proposals(state, 8),
    )
    dt = time.time() - t0
    u = get_token_usage()
    print(f"\n[FEATURE ENGINEERING]  used_llm={used_llm}  {dt:.1f}s  "
          f"calls={u.calls} tokens={u.total_tokens}", flush=True)
    valid_cols = set(state.X.columns)
    for p in batch.proposals:
        bad = [c for c in p.source_columns if c not in valid_cols]
        flag = "  <-- INVALID COLUMN" if bad else ""
        tgt = "  <-- USED TARGET" if target in p.source_columns else ""
        print(f"   - {p.feature_name:32s} {p.operation.value:12s} "
              f"{p.source_columns}{flag}{tgt}")
        print(f"       reason: {p.reasoning[:110]}")

    # --- Stage 3 style call -------------------------------------------------
    reset_token_usage()
    valid = [m.value for m in valid_models_for(state.problem_type)]
    ctx2 = {
        "problem_type": state.problem_type.value,
        "n_rows": profile.n_rows,
        "n_features": len(state.X.columns),
        "n_numeric": len(profile.numeric_features()),
        "n_categorical": len(profile.categorical_features()),
        "allowed_models": valid,
    }
    user2 = ("Dataset context:\n" + json.dumps(ctx2, indent=2)
             + "\n\nChoose 3-5 algorithms to try as JSON.")
    t0 = time.time()
    plan, used_llm2 = reasoner.propose_json(
        system=MS_SYSTEM, user=user2, schema=ModelSelectionPlan,
        fallback=lambda: ModelSelectionPlan(
            models_to_try=valid_models_for(state.problem_type)[:4],
            reasoning="fallback"),
    )
    dt2 = time.time() - t0
    u2 = get_token_usage()
    print(f"\n[MODEL SELECTION]  used_llm={used_llm2}  {dt2:.1f}s  "
          f"calls={u2.calls} tokens={u2.total_tokens}", flush=True)
    print(f"   models: {[m.value for m in plan.models_to_try]}")
    print(f"   reason: {plan.reasoning[:200]}")


if __name__ == "__main__":
    probe("synthetic_no_fk.csv", "churn")
    probe("california_regression.csv", "median_house_value")
