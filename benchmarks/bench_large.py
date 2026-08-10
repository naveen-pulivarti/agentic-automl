"""
Stress benchmark on large real-world datasets (200,000 rows each).

Small samples prove correctness; these prove usability at realistic scale. Both
datasets come from the public benchmark repositories named in the synopsis:

  covertype_classification.csv   Forest Cover Type (UCI) — 200k x 55, 7 classes
  gpu_performance_regression.csv SGEMM GPU kernel timings (OpenML) — 200k x 15

Usage:
    python benchmarks/bench_large.py <label>
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.state import AutoMLState                      # noqa: E402
from src.agent.graph import run_pipeline_sequential          # noqa: E402
from src.report.generator import generate_summary_dict       # noqa: E402
from src.utils.logger import get_token_usage, reset_token_usage  # noqa: E402

DATA = ROOT / "data"

DATASETS = [
    ("covertype_classification.csv", "cover_type"),
    ("gpu_performance_regression.csv", "target_value"),
]


def main(label: str) -> None:
    out = {"label": label, "runs": []}
    for csv_name, target in DATASETS:
        path = DATA / csv_name
        if not path.exists():
            print(f"SKIP {csv_name} (missing — run data/make_large_samples.py)")
            continue

        print(f"\n{'=' * 70}\n{csv_name}  target={target}\n{'=' * 70}", flush=True)
        reset_token_usage()
        t0 = time.time()
        try:
            df = pd.read_csv(path)
            print(f"  loaded {df.shape[0]:,} rows x {df.shape[1]} cols "
                  f"({time.time() - t0:.1f}s)", flush=True)

            state = AutoMLState(df=df, target=target)
            result = run_pipeline_sequential(state)
            summary = generate_summary_dict(result)
            elapsed = time.time() - t0
            usage = get_token_usage()

            rec = {
                "dataset": csv_name, "target": target,
                "rows": int(df.shape[0]), "cols": int(df.shape[1]),
                "seconds": round(elapsed, 1), "ok": True,
                "llm_calls": usage.calls, "llm_tokens": usage.total_tokens,
                **summary,
            }
            print(f"\n  problem      : {summary['problem_type']}")
            print(f"  best model   : {summary['best_model']} {summary['best_params']}")
            print(f"  baseline     : {summary['baseline_score']}")
            print(f"  engineered   : {summary['engineered_score']}")
            print(f"  selection    : {summary['selection_score']} "
                  f"(winner={summary['selection_winner']})")
            print(f"  tuned        : {summary['tuned_score']}")
            print(f"  HELD-OUT     : {summary['final_metrics']}")
            print(f"  kept feats   : {summary['kept_features']}")
            print(f"  cols kept    : {len(summary['selected_features'])} "
                  f"(dropped {len(summary['dropped_features'])})")
            print(f"  used_llm     : {summary['used_llm']} {summary['llm_stages']}")
            print(f"  stage times  : {summary['stage_seconds']}")
            print(f"  TOTAL        : {elapsed:.1f}s", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            rec = {"dataset": csv_name, "target": target, "ok": False,
                   "seconds": round(elapsed, 1),
                   "error": f"{type(e).__name__}: {e}"}
            print(f"  !! FAILED after {elapsed:.1f}s: {e}", flush=True)
            traceback.print_exc()
        out["runs"].append(rec)

    dest = Path(__file__).parent / "results" / f"bench_large_{label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
