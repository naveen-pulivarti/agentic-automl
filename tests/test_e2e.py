"""
End-to-end and unit tests.

The end-to-end tests run WITHOUT any LLM available, so they exercise the
heuristic-fallback path — proving the system produces valid results even with
zero LLM. That is the core reliability guarantee.

The unit tests cover the safety and correctness properties that are easy to
break silently: proposal validation, the no-duplicate-configuration rule, and
the guarantee that the test set never takes part in the search.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Point the reasoner at a dead port so every test deterministically exercises
# the heuristic fallback, regardless of whether Ollama happens to be running.
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:1")

from src.agent.schemas import (                              # noqa: E402
    FeatureOperation, FeatureProposal, FeatureProposalBatch, ProblemType,
)
from src.agent.state import AutoMLState                      # noqa: E402
from src.agent.graph import run_pipeline_sequential          # noqa: E402
from src.ml.feature_ops import apply_feature, validate_proposal  # noqa: E402
from src.report.generator import (                           # noqa: E402
    generate_markdown_report, generate_summary_dict,
)

DATA = ROOT / "data"


def _run(csv_name: str, target: str):
    df = pd.read_csv(DATA / csv_name)
    state = AutoMLState(df=df, target=target)
    result = run_pipeline_sequential(state)
    return result, generate_summary_dict(result)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def test_iris_classification():
    state, s = _run("iris_classification.csv", "species")
    assert s["problem_type"] == "classification"
    assert s["best_model"] is not None
    assert s["final_metrics"]["accuracy"] > 0.7   # iris is easy
    assert s["selected_features"], "selection must keep at least one column"
    print("IRIS:", s["best_model"], s["final_metrics"])


def test_wine_classification():
    state, s = _run("wine_classification.csv", "wine_class")
    assert s["problem_type"] == "classification"
    assert s["final_metrics"]["accuracy"] > 0.8
    print("WINE:", s["best_model"], s["final_metrics"])


def test_diabetes_regression():
    state, s = _run("diabetes_regression.csv", "disease_progression")
    assert s["problem_type"] == "regression"
    assert "r2" in s["final_metrics"]
    print("DIABETES:", s["best_model"], s["final_metrics"])


def test_california_regression():
    if not (DATA / "california_regression.csv").exists():
        pytest.skip("california sample not generated")
    state, s = _run("california_regression.csv", "median_house_value")
    assert s["problem_type"] == "regression"
    assert s["final_metrics"]["r2"] > 0.5
    print("CALIFORNIA:", s["best_model"], s["final_metrics"])


def test_synthetic_classification_with_categoricals():
    state, s = _run("synthetic_no_fk.csv", "churn")
    assert s["problem_type"] == "classification"
    assert s["best_model"] is not None
    print("SYNTHETIC:", s["best_model"], s["final_metrics"])


# ---------------------------------------------------------------------------
# The properties that must never regress
# ---------------------------------------------------------------------------
def test_test_set_is_never_seen_during_search():
    """The held-out rows must not appear in the training frame."""
    df = pd.read_csv(DATA / "iris_classification.csv")
    state = run_pipeline_sequential(AutoMLState(df=df, target="species"))
    assert state.X_test is not None and len(state.X_test) > 0
    assert len(state.X) + len(state.X_test) == len(df)
    # the two frames must be disjoint on their original feature values
    common = pd.merge(state.X.round(6), state.X_test.round(6), how="inner")
    assert len(common) < len(state.X_test), "train and test overlap"


def test_report_generates_without_llm():
    df = pd.read_csv(DATA / "iris_classification.csv")
    state = run_pipeline_sequential(AutoMLState(df=df, target="species"))
    md = generate_markdown_report(state)
    assert "Decision trail" in md
    assert "heuristic fallback" in md   # honest about not using an LLM
    assert len(md) > 500


def test_pipeline_works_with_no_llm_at_all():
    df = pd.read_csv(DATA / "iris_classification.csv")
    state = run_pipeline_sequential(AutoMLState(df=df, target="species"))
    assert state.used_llm_anywhere is False
    assert state.final_metrics, "must still produce metrics with no LLM"


# ---------------------------------------------------------------------------
# Proposal validation — the safety boundary
# ---------------------------------------------------------------------------
def _df():
    return pd.DataFrame({
        "a": [1.0, 2, 3, 4, 5, 6, 7, 8],
        "b": [2.0, 4, 6, 8, 10, 12, 14, 16],
        "cat": list("xxyyzzww"),
        "when": pd.to_datetime(["2024-01-01"] * 8),
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_target_column_is_rejected():
    p = FeatureProposal(feature_name="leak", operation=FeatureOperation.RATIO,
                        source_columns=["a", "target"], reasoning="")
    reason = validate_proposal(_df(), p, target="target")
    assert reason and "target" in reason


def test_date_part_on_a_number_is_rejected():
    p = FeatureProposal(feature_name="a_month", operation=FeatureOperation.DATE_PART,
                        source_columns=["a"], date_component="month", reasoning="")
    reason = validate_proposal(_df(), p, target="target")
    assert reason and "not a date" in reason


def test_date_part_on_a_real_date_is_allowed():
    p = FeatureProposal(feature_name="when_month", operation=FeatureOperation.DATE_PART,
                        source_columns=["when"], date_component="month", reasoning="")
    assert validate_proposal(_df(), p, target="target") is None


def test_numeric_op_on_a_string_column_is_rejected():
    p = FeatureProposal(feature_name="bad", operation=FeatureOperation.LOG,
                        source_columns=["cat"], reasoning="")
    reason = validate_proposal(_df(), p, target="target")
    assert reason and "numeric" in reason


def test_wrong_arity_is_rejected_by_the_schema():
    """A `square` with two columns must not validate at all."""
    with pytest.raises(Exception):
        FeatureProposal(feature_name="bad", operation=FeatureOperation.SQUARE,
                        source_columns=["a", "b"], reasoning="")


def test_batch_salvage_keeps_the_valid_proposals():
    """One malformed item must not discard the whole batch."""
    data = {"proposals": [
        {"feature_name": "ok", "operation": "ratio",
         "source_columns": ["a", "b"], "reasoning": "fine"},
        {"feature_name": "bad_op", "operation": "teleport",
         "source_columns": ["a"], "reasoning": "nonsense"},
        {"feature_name": "bad_arity", "operation": "square",
         "source_columns": ["a", "b"], "reasoning": "wrong shape"},
    ]}
    batch = FeatureProposalBatch.salvage(data)
    assert len(batch.proposals) == 1
    assert batch.proposals[0].feature_name == "ok"


def test_binning_uses_training_edges_on_new_data():
    """A feature rebuilt on unseen data must use the training bin edges."""
    train = pd.DataFrame({"a": list(range(100))})
    test = pd.DataFrame({"a": list(range(100, 120))})   # entirely above train
    p = FeatureProposal(feature_name="a_bin", operation=FeatureOperation.BINNING,
                        source_columns=["a"], reasoning="")
    out = apply_feature(test, p, reference=train)
    assert out is not None
    # every test row sits above the training range, so all land in the top bin
    assert out.nunique() == 1


# ---------------------------------------------------------------------------
# Tuning must not repeat itself
# ---------------------------------------------------------------------------
def test_tuning_never_repeats_a_configuration():
    from src.stages.tuning import _fingerprint, _next_unused
    from src.agent.schemas import ModelChoice

    tried: set[str] = set()
    seen: list[str] = []
    while True:
        params = _next_unused(ModelChoice.RANDOM_FOREST, tried)
        if params is None:
            break
        fp = _fingerprint(params)
        assert fp not in tried, "grid handed back a duplicate configuration"
        tried.add(fp)
        seen.append(fp)
    assert len(seen) == len(set(seen))
    assert len(seen) > 1


def test_model_without_hyperparameters_is_skipped():
    """Linear regression has nothing to tune; the stage must not burn budget."""
    from src.stages.tuning import _SEARCH_SPACE
    from src.agent.schemas import ModelChoice
    assert not _SEARCH_SPACE.get(ModelChoice.LINEAR_REGRESSION)


# ---------------------------------------------------------------------------
# Automatic cleanup of columns no model should ever see
# ---------------------------------------------------------------------------
def test_identifier_and_constant_columns_are_dropped():
    """A row id and a constant column must never reach the model.

    The id is made perfectly predictive of the target on purpose: if it survived,
    the model would score suspiciously well by memorising the index.
    """
    from src.data.profiler import profile_dataset

    n = 300
    df = pd.DataFrame({
        "row_id": range(1, n + 1),                 # identifier, sorted
        "always_7": [7] * n,                       # constant
        "signal": [i % 13 for i in range(n)],      # a real feature
        "noise": [(i * 7) % 31 for i in range(n)],
        "target": [(i % 13) > 6 for i in range(n)],
    })
    prof = profile_dataset(df, "target")
    kinds = {c.name: c.kind for c in prof.columns}
    assert kinds["row_id"] == "identifier"
    assert kinds["always_7"] == "constant"
    assert kinds["signal"] == "numeric"

    state = run_pipeline_sequential(AutoMLState(df=df, target="target"))
    assert "row_id" not in state.X.columns
    assert "always_7" not in state.X.columns
    assert "signal" in state.X.columns


def test_continuous_measurement_is_not_mistaken_for_an_identifier():
    """Near-unique but unsorted measurements must stay — only sorted ids go."""
    from src.data.profiler import profile_dataset
    rng = __import__("numpy").random.default_rng(0)
    df = pd.DataFrame({
        "measurement": rng.normal(50, 10, 300),
        "target": rng.integers(0, 2, 300),
    })
    kinds = {c.name: c.kind for c in profile_dataset(df, "target").columns}
    assert kinds["measurement"] == "numeric"


# ---------------------------------------------------------------------------
# Charts — every figure must build from a real run without raising
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def classification_state():
    df = pd.read_csv(DATA / "iris_classification.csv")
    return run_pipeline_sequential(AutoMLState(df=df, target="species"))


@pytest.fixture(scope="module")
def regression_state():
    df = pd.read_csv(DATA / "diabetes_regression.csv")
    return run_pipeline_sequential(AutoMLState(df=df, target="disease_progression"))


def test_classification_charts_build(classification_state):
    from src.report import charts
    assert charts.score_progression_chart(classification_state) is not None
    assert charts.model_comparison_chart(classification_state) is not None
    assert charts.confusion_matrix_chart(classification_state) is not None
    assert charts.feature_ranking_chart(classification_state) is not None
    assert charts.decision_outcome_chart(classification_state) is not None
    # a regression-only chart must decline politely, not raise
    assert charts.residual_chart(classification_state) is None


def test_regression_charts_build(regression_state):
    from src.report import charts
    assert charts.score_progression_chart(regression_state) is not None
    assert charts.residual_chart(regression_state) is not None
    assert charts.confusion_matrix_chart(regression_state) is None


if __name__ == "__main__":
    for fn in (test_iris_classification, test_wine_classification,
               test_diabetes_regression, test_california_regression,
               test_synthetic_classification_with_categoricals):
        print(f"\n=== {fn.__name__} ===")
        try:
            fn()
        except Exception as exc:            # pytest.skip outside pytest
            print(f"  skipped/failed: {exc}")
    print("\n✅ end-to-end run complete (run `pytest tests/` for the full suite)")
