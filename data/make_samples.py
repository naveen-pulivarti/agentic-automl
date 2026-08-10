"""
Generate small sample datasets for local testing and demos.

Creates:
- iris_classification.csv        (classic multiclass classification)
- wine_classification.csv        (multiclass classification)
- diabetes_regression.csv        (regression)
- california_regression.csv      (regression, sampled)
- synthetic_no_fk.csv            (synthetic; deliberately messy for demo)
- demo.db (SQLite)               (so the DB-connection path can be demoed)

Run:  python data/make_samples.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.datasets import (
    fetch_california_housing, load_diabetes, load_iris, load_wine,
)

HERE = Path(__file__).parent


def _iris() -> pd.DataFrame:
    d = load_iris(as_frame=True)
    df = d.frame.copy()
    df["species"] = pd.Categorical.from_codes(d.target, d.target_names)
    df = df.drop(columns=["target"])
    return df


def _wine() -> pd.DataFrame:
    d = load_wine(as_frame=True)
    df = d.frame.copy()
    df["wine_class"] = d.target
    df = df.drop(columns=["target"])
    return df


def _diabetes() -> pd.DataFrame:
    d = load_diabetes(as_frame=True)
    df = d.frame.copy()
    df = df.rename(columns={"target": "disease_progression"})
    return df


def _california() -> pd.DataFrame:
    """California housing, used in full (20,640 rows).

    An earlier version sampled this down to 3,000 rows, which was an arbitrary
    reduction with no justification — the agent's own sub-sampling already keeps
    the search affordable, so shrinking the source data only weakened the
    evidence. The full set is used.
    """
    d = fetch_california_housing(as_frame=True)
    df = d.frame.copy()
    df = df.rename(columns={"MedHouseVal": "median_house_value"})
    return df.reset_index(drop=True)


def _synthetic(n: int = 800) -> pd.DataFrame:
    """A messy-ish synthetic set: mixed types, some missing values, a target
    that depends non-trivially on the features."""
    import numpy as np
    rng = np.random.default_rng(7)
    age = rng.integers(18, 70, n).astype(float)
    income = rng.normal(50000, 15000, n).clip(5000)
    tenure = rng.integers(0, 120, n).astype(float)
    region = rng.choice(["north", "south", "east", "west"], n)
    # introduce missing values
    income[rng.random(n) < 0.05] = np.nan
    # target: churn probability from a nonlinear combo.
    # Intercept tuned to give a reasonably balanced target (~35-45% churn) so
    # the dataset is a clean demo rather than a severe class-imbalance case.
    logit = (
        0.2
        + 0.05 * (age - 40)
        - 0.00003 * (income - 50000)
        - 0.03 * tenure
        + (region == "west") * 0.8
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "age": age, "income": income, "tenure_months": tenure,
        "region": region, "churn": churn,
    })


def _adult() -> pd.DataFrame:
    """Census Income (UCI 'Adult') — 48,842 rows, binary classification.

    Included to fill the gap between the small benchmarks and the 200,000-row
    sets, so results can be compared across three orders of magnitude. It is
    also the most realistically messy of the public sets used here: a mix of
    numeric and categorical columns, genuine missing values, and a moderately
    imbalanced target.
    """
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = d.frame.copy()
    target = "class" if "class" in df.columns else d.target.name
    df = df.rename(columns={target: "income_over_50k"})
    # a duplicate encoding of the target that would hand the model the answer
    df = df.drop(columns=[c for c in ("education-num",) if c in df.columns])
    return df.reset_index(drop=True)


def main() -> None:
    datasets = {
        "iris_classification.csv": _iris(),
        "wine_classification.csv": _wine(),
        "diabetes_regression.csv": _diabetes(),
        "synthetic_no_fk.csv": _synthetic(),
    }
    # these require a network download; include them if available
    try:
        datasets["california_regression.csv"] = _california()
    except Exception as e:
        print(f"(skipping california housing — needs internet: {e})")
    try:
        datasets["census_income_classification.csv"] = _adult()
    except Exception as e:
        print(f"(skipping census income — needs internet: {e})")
    for name, df in datasets.items():
        path = HERE / name
        df.to_csv(path, index=False)
        print(f"wrote {path}  ({df.shape[0]}x{df.shape[1]})")

    # also drop the synthetic set into a SQLite db for the DB-connection demo
    db_path = HERE / "demo.db"
    conn = sqlite3.connect(db_path)
    datasets["synthetic_no_fk.csv"].to_sql("customers", conn,
                                            if_exists="replace", index=False)
    conn.close()
    print(f"wrote {db_path} (table: customers)")


if __name__ == "__main__":
    main()
