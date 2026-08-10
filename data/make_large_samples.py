"""
Fetch large real-world benchmark datasets (100,000+ rows) for stress testing.

The small bundled samples (iris, wine, diabetes) prove correctness but say
nothing about whether the agent is usable at realistic scale. These do.

Sources are the public benchmark repositories named in the project synopsis —
UCI and OpenML — reached through scikit-learn's fetchers, so no account,
API token or manual download is required.

Datasets are capped at `ROW_CAP` rows. The cap keeps the CSVs a sane size on
disk while staying far above the 100,000-row target, and the agent's own
sub-sampling means a larger file would not change what it does.

Run:  python data/make_large_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROW_CAP = 200_000
SEED = 42


def _save(df: pd.DataFrame, name: str) -> None:
    path = HERE / name
    df.to_csv(path, index=False)
    size_mb = path.stat().st_size / 1e6
    print(f"  wrote {path.name}  ({df.shape[0]:,} x {df.shape[1]})  {size_mb:.1f} MB")


def _cap(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > ROW_CAP:
        df = df.sample(n=ROW_CAP, random_state=SEED)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Classification — Forest Cover Type (UCI, 581,012 rows x 54 features)
# ---------------------------------------------------------------------------
def covertype() -> pd.DataFrame | None:
    """Predict forest cover type from cartographic variables. 7 classes."""
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype(as_frame=True)
    df = d.frame.copy()
    df = df.rename(columns={"Cover_Type": "cover_type"})
    return _cap(df)


# ---------------------------------------------------------------------------
# Regression — tried in order, first success wins
# ---------------------------------------------------------------------------
_REGRESSION_CANDIDATES = [
    # (openml name, version, target column, output name, columns to drop)
    #
    # SGEMM ships four timing runs of the same GPU kernel. Run1 is the target,
    # so Run2-4 are repeat measurements of the answer — leaving them in would
    # hand the model the label under another name and yield a meaningless
    # R² near 1.0. They are dropped to make this a genuine prediction task.
    ("SGEMM_GPU_kernel_performance", 1, None, "gpu_performance_regression.csv",
     ["Run2", "Run3", "Run4"]),
    ("Buzzinsocialmedia_Twitter", 1, None, "social_buzz_regression.csv", []),
    ("nyc-taxi-green-dec-2016", 1, None, "nyc_taxi_regression.csv", []),
]


def openml_regression() -> tuple[pd.DataFrame, str] | None:
    from sklearn.datasets import fetch_openml
    for name, version, target, out, drop in _REGRESSION_CANDIDATES:
        try:
            print(f"  trying OpenML '{name}' ...", flush=True)
            d = fetch_openml(name=name, version=version, as_frame=True,
                             parser="auto")
            df = d.frame.copy()
            tgt = target or (d.target.name if hasattr(d.target, "name") else None)
            if tgt and tgt in df.columns:
                df = df.rename(columns={tgt: "target_value"})
            leaky = [c for c in drop if c in df.columns]
            if leaky:
                df = df.drop(columns=leaky)
                print(f"    dropped leakage columns: {', '.join(leaky)}")
            if len(df) < 100_000:
                print(f"    only {len(df):,} rows — skipping")
                continue
            return _cap(df), out
        except Exception as e:
            print(f"    unavailable ({type(e).__name__}: {str(e)[:90]})")
    return None


# ---------------------------------------------------------------------------
# Fallback — a large synthetic set, so stress testing is always possible
# ---------------------------------------------------------------------------
def synthetic_large(n: int = 150_000) -> pd.DataFrame:
    """Mixed-type regression data with missingness and non-linear structure.

    Only used if the network fetches fail; real data is always preferred.
    """
    import numpy as np
    rng = np.random.default_rng(SEED)
    n_num = 12
    X = rng.normal(0, 1, (n, n_num))
    region = rng.choice(["north", "south", "east", "west", "central"], n)
    grade = rng.choice(list("ABCDE"), n)

    y = (3.0 * X[:, 0]
         + 1.5 * X[:, 1] ** 2
         - 2.0 * X[:, 2] * X[:, 3]
         + np.where(region == "west", 4.0, 0.0)
         + rng.normal(0, 1.2, n))

    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_num)])
    df["region"] = region
    df["grade"] = grade
    # realistic missingness
    for c in ("num_4", "num_7"):
        df.loc[rng.random(n) < 0.04, c] = float("nan")
    df["measured_value"] = y
    return df


def main() -> None:
    print("Fetching large benchmark datasets (this needs internet)...\n")
    made: list[str] = []

    print("Forest Cover Type (UCI) — classification")
    try:
        df = covertype()
        _save(df, "covertype_classification.csv")
        made.append("covertype_classification.csv")
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {str(e)[:120]}")

    print("\nOpenML regression benchmark")
    got = openml_regression()
    if got is not None:
        df, out = got
        _save(df, out)
        made.append(out)
    else:
        print("  no OpenML candidate available — generating synthetic fallback")
        _save(synthetic_large(), "synthetic_large_regression.csv")
        made.append("synthetic_large_regression.csv")

    print(f"\nDone. {len(made)} dataset(s) ready:")
    for m in made:
        print(f"  - data/{m}")
    if not made:
        sys.exit(1)


if __name__ == "__main__":
    main()
