"""
Benchmark harness: run the full pipeline over every demo dataset and record
timings + scores to JSON, so we can compare heuristic-vs-LLM runs objectively.

Usage:
    python bench.py <label>
Env:
    AUTOML_LLM_PROVIDER / AUTOML_LLM_MODEL / OLLAMA_BASE_URL / GROQ_API_KEY
    control which reasoning backend is exercised.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# Derived from this file's location so the harness survives the project
# folder being moved or renamed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.state import AutoMLState                      # noqa: E402
from src.agent.graph import run_pipeline_sequential          # noqa: E402
from src.report.generator import generate_summary_dict       # noqa: E402
from src.utils.logger import get_token_usage, reset_token_usage  # noqa: E402

DATA = ROOT / "data"

DATASETS = [
    ("iris_classification.csv", "species"),
    ("wine_classification.csv", "wine_class"),
    ("synthetic_no_fk.csv", "churn"),
    ("diabetes_regression.csv", "disease_progression"),
    ("california_regression.csv", "median_house_value"),
    ("census_income_classification.csv", "income_over_50k"),
]


def main(label: str) -> None:
    out = {"label": label, "runs": []}
    for csv_name, target in DATASETS:
        path = DATA / csv_name
        if not path.exists():
            print(f"SKIP {csv_name} (missing)")
            continue
        print(f"\n=== {csv_name} (target={target}) ===", flush=True)
        reset_token_usage()
        t0 = time.time()
        try:
            df = pd.read_csv(path)
            state = AutoMLState(df=df, target=target)
            result = run_pipeline_sequential(state)
            summary = generate_summary_dict(result)
            elapsed = time.time() - t0
            usage = get_token_usage()
            rec = {
                "dataset": csv_name,
                "target": target,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "seconds": round(elapsed, 1),
                "ok": True,
                "llm_calls": usage.calls,
                "llm_tokens": usage.total_tokens,
                **summary,
            }
            print(f"  -> {summary['problem_type']} | best={summary['best_model']} "
                  f"| baseline={summary['baseline_score']} "
                  f"| tuned={summary['tuned_score']} "
                  f"| final={summary['final_metrics']} "
                  f"| kept={summary['kept_features']} "
                  f"| used_llm={summary['used_llm']} | {elapsed:.1f}s", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            rec = {"dataset": csv_name, "target": target, "ok": False,
                   "seconds": round(elapsed, 1), "error": f"{type(e).__name__}: {e}"}
            print(f"  !! FAILED after {elapsed:.1f}s: {e}", flush=True)
            traceback.print_exc()
        out["runs"].append(rec)

    dest = Path(__file__).parent / "results" / f"bench_{label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
