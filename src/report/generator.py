"""
Explainability report generator.

Turns the recorded decision trail into a human-readable Markdown report. This is
the "transparency" deliverable: for every stage it shows what was tried, what the
measured result was, and why each decision was made.

Each decision also records who proposed it — the reasoner or the deterministic
fallback — so the report never overstates the LLM's contribution. If the model
was unreachable and heuristics did the work, the report says so.
"""
from __future__ import annotations

from ..agent.state import AutoMLState


def generate_markdown_report(state: AutoMLState) -> str:
    p = state.profile
    lines: list[str] = []
    a = lines.append

    a("# Agentic AutoML — Model Building Report\n")

    # -- summary --
    a("## Summary\n")
    a(f"- **Problem type:** {state.problem_type.value}")
    a(f"- **Dataset:** {p.n_rows:,} rows x {p.n_cols} columns")
    a(f"- **Target column:** `{state.target}`")
    if state.X is not None and state.X_test is not None:
        a(f"- **Training rows:** {len(state.X):,}  ·  "
          f"**Held-out test rows:** {len(state.X_test):,}")
    a(f"- **Best model:** `{state.best_model.value}`")
    if state.best_params:
        a(f"- **Tuned parameters:** `{state.best_params}`")
    a(f"- **Engineered features kept:** {len(state.accepted_features)}")
    a(f"- **Columns after selection:** {len(state.selected_features)}"
      + (f" (dropped {len(state.dropped_features)})" if state.dropped_features else ""))
    if state.used_llm_anywhere:
        a(f"- **Reasoning layer:** LLM (used in: {', '.join(state.llm_stages)})")
    else:
        a("- **Reasoning layer:** heuristic fallback (no LLM was reachable)")
    a("")

    # -- final metrics --
    a("## Final performance (held-out test set)\n")
    a("These rows were separated before any search began, so this is an honest "
      "estimate rather than a restatement of what was optimised.\n")
    if state.final_metrics:
        for k, v in state.final_metrics.items():
            a(f"- **{k}:** {v:.4f}")
    a("")

    # -- score progression --
    a("## Score progression (cross-validated, training data)\n")
    for label, score in state.score_progression():
        a(f"- {label}: {score:.4f}")
    a("")

    # -- engineered features --
    if state.accepted_features:
        a("## Engineered features kept\n")
        for f in state.accepted_features:
            a(f"- `{f}`")
        a("")

    # -- feature selection --
    if state.selected_features:
        a("## Feature selection\n")
        if state.selection_winner:
            a(f"Winning subset: **{state.selection_winner}** "
              f"({len(state.selected_features)} columns).\n")
        a("**Kept:** " + ", ".join(f"`{c}`" for c in state.selected_features))
        if state.dropped_features:
            a("")
            a("**Dropped:** " + ", ".join(f"`{c}`" for c in state.dropped_features))
        a("")

    # -- ranking evidence --
    if state.ranking_table is not None and not state.ranking_table.empty:
        a("## Column ranking evidence\n")
        a("Mean rank across every statistical method (1 = most useful).\n")
        a("| Column | Mean rank |")
        a("|---|---|")
        for feat, row in state.ranking_table.head(15).iterrows():
            a(f"| `{feat}` | {row['mean_rank']:.1f} |")
        a("")

    # -- model comparison --
    if state.model_results:
        a("## Algorithms compared\n")
        a("| Algorithm | Score |")
        a("|---|---|")
        for r in sorted(state.model_results, key=lambda r: r.score, reverse=True):
            mark = " ⭐" if state.best_model and r.model_name == state.best_model.value else ""
            a(f"| {r.model_name}{mark} | {r.score:.4f} |")
        a("")

    # -- full decision trail --
    a("## Decision trail\n")
    a("Every decision below was validated against real measured performance.\n")
    current_stage = None
    for d in state.decisions:
        if d.stage != current_stage:
            a(f"\n### {d.stage}\n")
            current_stage = d.stage
        mark = "✅" if d.accepted else "❌"
        who = f" _({d.source})_" if d.source else ""
        a(f"- {mark} **{d.action}**{who}")
        if d.reasoning:
            a(f"  - *Reasoning:* {d.reasoning}")
        if d.detail:
            a(f"  - *Outcome:* {d.detail}")
    a("")

    # -- timings --
    if state.stage_seconds:
        a("## Stage timings\n")
        for stage, secs in state.stage_seconds.items():
            a(f"- {stage}: {secs:.1f}s")
        a("")

    return "\n".join(line for line in lines if line is not None)


def generate_summary_dict(state: AutoMLState) -> dict:
    """Compact machine-readable summary (useful for tests / the UI)."""
    return {
        "problem_type": state.problem_type.value if state.problem_type else None,
        "best_model": state.best_model.value if state.best_model else None,
        "best_params": state.best_params,
        "baseline_score": state.baseline_score,
        "engineered_score": state.engineered_score,
        "selection_score": state.selection_score,
        "selection_winner": state.selection_winner,
        "model_selection_score": state.best_model_score,
        "tuned_score": state.tuned_score,
        "final_metrics": state.final_metrics,
        "kept_features": list(state.accepted_features),
        "selected_features": list(state.selected_features),
        "dropped_features": list(state.dropped_features),
        "used_llm": state.used_llm_anywhere,
        "llm_stages": list(state.llm_stages),
        "n_decisions": len(state.decisions),
        "stage_seconds": dict(state.stage_seconds),
    }
