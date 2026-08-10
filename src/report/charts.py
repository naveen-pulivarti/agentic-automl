"""
Charts for the UI and the report.

Every figure here is built from numbers the pipeline already measured — nothing
is recomputed or estimated for display. A chart that disagreed with the decision
trail would undermine the whole point of the tool.

Plotly is used because it renders interactively in Streamlit and exports cleanly
to a static image for the written report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..agent.state import AutoMLState

# A single palette used across every figure so the app reads as one system.
_ACCENT = "#4C8DFF"
_ACCENT_SOFT = "#9EC1FF"
_GOOD = "#2FB8A0"
_BAD = "#E06C75"
_GRID = "rgba(128,128,128,0.20)"


def _base_layout(fig, title: str, height: int = 340):
    """Shared styling: transparent background so it sits in light or dark theme."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    )
    return fig


def score_progression_chart(state: AutoMLState):
    """How the cross-validated score moved through the pipeline stages."""
    import plotly.graph_objects as go

    steps = state.score_progression()
    if len(steps) < 2:
        return None
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]

    fig = go.Figure(go.Scatter(
        x=labels, y=values, mode="lines+markers+text",
        text=[f"{v:.4f}" for v in values], textposition="top center",
        line=dict(color=_ACCENT, width=3),
        marker=dict(size=10, color=_ACCENT),
    ))
    metric = "F1 (macro)" if state.problem_type.value == "classification" else "R²"
    fig.update_yaxes(title_text=metric)
    span = max(values) - min(values)
    pad = span * 0.35 if span > 0 else 0.01
    fig.update_yaxes(range=[min(values) - pad, max(values) + pad])
    return _base_layout(fig, "Score progression through the pipeline")


def model_comparison_chart(state: AutoMLState):
    """Every algorithm the agent actually cross-validated."""
    import plotly.graph_objects as go

    if not state.model_results:
        return None
    results = sorted(state.model_results, key=lambda r: r.score)
    names = [r.model_name for r in results]
    scores = [r.score for r in results]
    best = state.best_model.value if state.best_model else None
    colors = [_ACCENT if n == best else _ACCENT_SOFT for n in names]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.4f}" for s in scores], textposition="auto",
    ))
    fig.update_xaxes(title_text=results[0].primary_metric)
    return _base_layout(fig, "Algorithms compared (cross-validated)",
                        height=90 + 42 * len(names))


def confusion_matrix_chart(state: AutoMLState):
    """Confusion matrix on the held-out test set (classification only)."""
    import plotly.graph_objects as go
    from sklearn.metrics import confusion_matrix

    if (state.problem_type.value != "classification"
            or state.fitted_pipeline is None or state.X_test is None):
        return None
    try:
        preds = state.fitted_pipeline.predict(state.X_test)
        labels = sorted(pd.unique(pd.concat([
            pd.Series(state.y_test), pd.Series(preds)])))
        cm = confusion_matrix(state.y_test, preds, labels=labels)
    except Exception:
        return None

    text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[str(c) for c in labels], y=[str(c) for c in labels],
        colorscale="Blues", showscale=False,
        text=text, texttemplate="%{text}",
    ))
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", autorange="reversed")
    return _base_layout(fig, "Confusion matrix (held-out test set)",
                        height=120 + 46 * len(labels))


def residual_chart(state: AutoMLState):
    """Predicted vs actual on the held-out test set (regression only)."""
    import plotly.graph_objects as go

    if (state.problem_type.value != "regression"
            or state.fitted_pipeline is None or state.X_test is None):
        return None
    try:
        preds = np.asarray(state.fitted_pipeline.predict(state.X_test), dtype=float)
        actual = np.asarray(state.y_test, dtype=float)
    except Exception:
        return None

    # Large test sets are sampled for the scatter so the browser stays responsive.
    if len(preds) > 4000:
        idx = np.random.default_rng(42).choice(len(preds), 4000, replace=False)
        preds, actual = preds[idx], actual[idx]

    lo, hi = float(min(actual.min(), preds.min())), float(max(actual.max(), preds.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=preds, mode="markers",
        marker=dict(size=5, color=_ACCENT, opacity=0.45),
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=_BAD, width=2, dash="dash"),
    ))
    fig.update_xaxes(title_text="Actual")
    fig.update_yaxes(title_text="Predicted")
    return _base_layout(fig, "Predicted vs actual (held-out test set)")


def feature_ranking_chart(state: AutoMLState, top_n: int = 15):
    """Mean rank across the ranking panel — lower is better, so bars invert."""
    import plotly.graph_objects as go

    table = state.ranking_table
    if table is None or table.empty:
        return None
    head = table.head(top_n).iloc[::-1]
    names = [str(i) for i in head.index]
    ranks = head["mean_rank"].to_numpy(dtype=float)
    # invert so the most useful column has the longest bar
    heights = (ranks.max() + 1) - ranks
    kept = set(state.selected_features or [])
    colors = [_GOOD if n in kept else _BAD for n in names]

    fig = go.Figure(go.Bar(
        x=heights, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"rank {r:.1f}" for r in ranks], textposition="auto",
    ))
    fig.update_xaxes(title_text="more useful →", showticklabels=False)
    return _base_layout(
        fig, "Column ranking — green kept, red dropped by selection",
        height=110 + 30 * len(names))


def decision_outcome_chart(state: AutoMLState):
    """How many proposals each stage accepted vs rejected."""
    import plotly.graph_objects as go

    if not state.decisions:
        return None
    stages: dict[str, list[int]] = {}
    for d in state.decisions:
        row = stages.setdefault(d.stage, [0, 0])
        row[0 if d.accepted else 1] += 1
    names = list(stages)
    accepted = [stages[s][0] for s in names]
    rejected = [stages[s][1] for s in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=accepted, name="accepted",
                         marker=dict(color=_GOOD)))
    fig.add_trace(go.Bar(x=names, y=rejected, name="rejected",
                         marker=dict(color=_BAD)))
    fig.update_layout(barmode="stack", showlegend=True,
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(title_text="decisions")
    return _base_layout(fig, "Decisions per stage")
