"""
Agentic AutoML — Streamlit user interface.

Run with:  streamlit run app.py

Flow: choose a data source -> pick the target column -> run the agent -> watch
the live reasoning trace -> review results, charts, and the decision trail.

The interface deliberately shows *how* each result was reached, not just the
result. That is the whole claim of the project: an AutoML tool you can argue
with, rather than a black box that hands you a number.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.llm import get_reasoner, reset_reasoner
from src.agent.state import AutoMLState
from src.agent.graph import run_pipeline
from src.config import CONFIG
from src.data.loader import load_from_database, load_from_file
from src.data.profiler import profile_dataset, suggest_target_column
from src.report.charts import (
    confusion_matrix_chart, decision_outcome_chart, feature_ranking_chart,
    model_comparison_chart, residual_chart, score_progression_chart,
)
from src.report.generator import generate_markdown_report, generate_summary_dict
from src.utils.logger import get_token_usage, reset_token_usage

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Agentic AutoML", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# Styling — a few small touches so the app doesn't look like a default form
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1250px; }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; }
  .trace-line { font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                font-size: 0.82rem; padding: 1px 0; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px;
          font-size:0.75rem; font-weight:600; margin-right:6px; }
  .pill-llm  { background:rgba(76,141,255,0.18);  color:#4C8DFF; }
  .pill-heur { background:rgba(128,128,128,0.18); color:#9aa0a6; }
  .pill-ok   { background:rgba(47,184,160,0.18);  color:#2FB8A0; }
  .pill-bad  { background:rgba(224,108,117,0.18); color:#E06C75; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("result_state", None)
    st.session_state.setdefault("source_name", "")
    st.session_state.setdefault("health", None)      # cached (ok, info, checked_at)
    st.session_state.setdefault("trace", [])


_init_state()


@st.cache_data(show_spinner=False, ttl=300)
def _cached_health(provider: str, model: str) -> tuple[bool, str]:
    """Probe the reasoning backend at most once per provider/model per 5 min.

    Previously this ran on every single rerun — every click paid for a full LLM
    round trip, and a cold model load made the app look frozen for ~30 seconds.
    Caching on (provider, model) means switching backends still re-probes.
    """
    return get_reasoner().health_check()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🤖 Agentic AutoML")
st.caption("Reasoning-Driven Model Building for Tabular Data — "
           "the LLM proposes, scikit-learn decides.")


# ---------------------------------------------------------------------------
# Sidebar — reasoning backend + budgets
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 Reasoning backend")

    provider = st.selectbox(
        "Provider", ["ollama", "groq", "openai_compatible"],
        index=["ollama", "groq", "openai_compatible"].index(CONFIG.llm.provider)
        if CONFIG.llm.provider in ("ollama", "groq", "openai_compatible") else 0,
        help="Ollama runs locally and free. Groq's free tier is much faster.",
    )
    model = st.text_input("Model", CONFIG.llm.model,
                          help="e.g. llama3.1, qwen2.5:7b (Ollama) · "
                               "llama-3.1-8b-instant (Groq)")
    api_key = ""
    if provider != "ollama":
        api_key = st.text_input("API key", CONFIG.llm.api_key, type="password")

    if st.button("Apply & test", use_container_width=True):
        CONFIG.llm.provider = provider
        CONFIG.llm.model = model
        if api_key:
            CONFIG.llm.api_key = api_key
        reset_reasoner()
        _cached_health.clear()
        st.session_state.health = None

    ok, info = _cached_health(CONFIG.llm.provider, CONFIG.llm.model)
    if ok:
        st.success(f"Reachable · `{info}`")
    else:
        st.warning("LLM not reachable — the agent will use its deterministic "
                   "fallbacks and still produce a full result.")
        st.caption(f"`{info[:120]}`")

    st.divider()
    st.subheader("⚙️ Budgets")
    CONFIG.budget.fe_rounds = st.slider(
        "Feature engineering rounds", 1, 5, CONFIG.budget.fe_rounds,
        help="Each round the agent sees the measured outcome of the last one.")
    CONFIG.budget.fe_per_round = st.slider(
        "Proposals per round", 2, 8, CONFIG.budget.fe_per_round)
    CONFIG.budget.max_tuning_iterations = st.slider(
        "Tuning iterations", 0, 10, CONFIG.budget.max_tuning_iterations)
    CONFIG.budget.max_seconds_per_stage = st.slider(
        "Time budget per stage (s)", 30, 900,
        CONFIG.budget.max_seconds_per_stage, step=30)

    st.divider()
    st.caption("**How it works** — the LLM *proposes* features, column subsets, "
               "algorithms and parameters. scikit-learn *verifies* every one "
               "against real cross-validation. Only measured improvements are "
               "kept, so a weak or absent LLM cannot produce a bad model.")


# ---------------------------------------------------------------------------
# Step 1 — data source
# ---------------------------------------------------------------------------
st.header("1 · Load your data")

tab_sample, tab_upload, tab_db = st.tabs(
    ["📦 Sample dataset", "📤 Upload file", "🗄️ Database"])

with tab_sample:
    samples = sorted(p.name for p in DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    if not samples:
        st.info("No sample datasets found. Run `python data/make_samples.py`.")
    else:
        c1, c2 = st.columns([3, 1])
        choice = c1.selectbox("Bundled dataset", samples, label_visibility="collapsed")
        if c2.button("Load", use_container_width=True, key="load_sample"):
            try:
                st.session_state.df = load_from_file(DATA_DIR / choice, filename=choice)
                st.session_state.source_name = choice
                st.session_state.result_state = None
            except Exception as e:
                st.error(f"Could not load: {e}")

with tab_upload:
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            st.session_state.df = load_from_file(uploaded, filename=uploaded.name)
            st.session_state.source_name = uploaded.name
            st.session_state.result_state = None
        except Exception as e:
            st.error(f"Could not load file: {e}")

with tab_db:
    with st.form("db_form"):
        c1, c2 = st.columns(2)
        dialect = c1.selectbox("Database", ["sqlite", "postgresql", "mysql", "sqlserver"])
        query = c2.text_input("SQL query", "SELECT * FROM customers")
        c3, c4, c5 = st.columns(3)
        host = c3.text_input("Host", "localhost")
        port = c4.text_input("Port", "5432")
        database = c5.text_input("Database name", "")
        c6, c7 = st.columns(2)
        user = c6.text_input("User", "")
        password = c7.text_input("Password", "", type="password")
        sqlite_path = st.text_input("SQLite file path (sqlite only)",
                                    str(DATA_DIR / "demo.db"))
        submitted = st.form_submit_button("Connect & load")
    if submitted:
        try:
            st.session_state.df = load_from_database(
                dialect, query, host=host, port=port, database=database,
                user=user, password=password, sqlite_path=sqlite_path,
            )
            st.session_state.source_name = f"{dialect} query"
            st.session_state.result_state = None
        except Exception as e:
            st.error(f"Database load failed: {e}")


# ---------------------------------------------------------------------------
# Step 2 — preview + target
# ---------------------------------------------------------------------------
df = st.session_state.df
if df is not None:
    st.header("2 · Preview & choose the target")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    m4.metric("Source", st.session_state.source_name or "—")

    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    default_target = suggest_target_column(df)
    target = st.selectbox(
        "Target column (what to predict)",
        options=list(df.columns),
        index=list(df.columns).index(default_target),
    )

    try:
        prof = profile_dataset(df, target)
        st.info(f"Detected **{prof.problem_type.value}** — {prof.target_detection_reason}")
    except Exception as e:
        st.error(f"Profiling error: {e}")

    # -----------------------------------------------------------------------
    # Step 3 — run
    # -----------------------------------------------------------------------
    st.header("3 · Run the agent")
    est = CONFIG.budget.fe_rounds + 1 + 1 + CONFIG.budget.max_tuning_iterations
    st.caption(f"This run will make up to **{est} LLM calls** across 5 stages, "
               "each verified by real cross-validation.")

    if st.button("🚀 Build the model", type="primary", use_container_width=True):
        reset_token_usage()
        st.session_state.trace = []
        trace_box = st.container()
        log_area = trace_box.empty()
        messages: list[str] = []

        def on_progress(stage: str, message: str) -> None:
            messages.append(f"<div class='trace-line'>"
                            f"<b>{stage}</b> — {message}</div>")
            log_area.markdown("".join(messages[-14:]), unsafe_allow_html=True)

        state = AutoMLState(df=df, target=target, progress=on_progress)
        t0 = time.time()
        with st.spinner("The agent is working — proposing, training, measuring..."):
            try:
                result = run_pipeline(state)
                st.session_state.result_state = result
                st.session_state.trace = messages
                st.success(f"✅ Done in {time.time() - t0:.0f}s")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.exception(e)


# ---------------------------------------------------------------------------
# Step 4 — results
# ---------------------------------------------------------------------------
result = st.session_state.result_state
if result is not None:
    st.header("4 · Results")
    summary = generate_summary_dict(result)
    usage = get_token_usage()

    # -- headline numbers ---------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Problem", summary["problem_type"])
    k2.metric("Best model", summary["best_model"])
    if summary["baseline_score"] is not None and summary["tuned_score"] is not None:
        delta = summary["tuned_score"] - summary["baseline_score"]
        k3.metric("Search score", f"{summary['tuned_score']:.4f}", f"{delta:+.4f}")
    k4.metric("Columns kept",
              f"{len(summary['selected_features'])}",
              f"-{len(summary['dropped_features'])}"
              if summary["dropped_features"] else None)
    k5.metric("Reasoning", "LLM" if summary["used_llm"] else "heuristic")

    st.subheader("Held-out test metrics")
    st.caption("Measured on rows separated *before* any search began — "
               "an honest estimate, not a restatement of what was optimised.")
    if result.final_metrics:
        cols = st.columns(len(result.final_metrics))
        for col, (k, v) in zip(cols, result.final_metrics.items()):
            col.metric(k, f"{v:.4f}")

    t_charts, t_features, t_trail, t_export = st.tabs(
        ["📊 Charts", "🧬 Features", "🧾 Decision trail", "⬇️ Export"])

    # -- charts -------------------------------------------------------------
    with t_charts:
        c1, c2 = st.columns(2)
        fig = score_progression_chart(result)
        if fig:
            c1.plotly_chart(fig, use_container_width=True)
        fig = model_comparison_chart(result)
        if fig:
            c2.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        fig = confusion_matrix_chart(result) or residual_chart(result)
        if fig:
            c3.plotly_chart(fig, use_container_width=True)
        fig = decision_outcome_chart(result)
        if fig:
            c4.plotly_chart(fig, use_container_width=True)

        fig = feature_ranking_chart(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # -- features -----------------------------------------------------------
    with t_features:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Engineered features kept**")
            if result.accepted_features:
                for f in result.accepted_features:
                    st.markdown(f"- `{f}`")
            else:
                st.caption("None improved the score — the original columns "
                           "were already sufficient.")
        with c2:
            st.markdown("**Columns dropped by selection**")
            if result.dropped_features:
                for f in result.dropped_features:
                    st.markdown(f"- `{f}`")
            else:
                st.caption("None — every column earned its place.")

        if result.selection_winner:
            st.info(f"Winning subset: **{result.selection_winner}** — "
                    f"{len(result.selected_features)} of "
                    f"{len(result.selected_features) + len(result.dropped_features)} "
                    "columns kept.")

        if result.ranking_table is not None and not result.ranking_table.empty:
            st.markdown("**Ranking evidence** (rank 1 = most useful; "
                        "blank = method not applicable to that column)")
            rank_cols = [c for c in result.ranking_table.columns
                         if c.startswith("rank_")]
            show = result.ranking_table[rank_cols + ["mean_rank"]].copy()
            show.columns = [c.replace("rank_", "") for c in show.columns]
            st.dataframe(show.style.format("{:.0f}", na_rep="—")
                         .format({"mean_rank": "{:.1f}"}),
                         use_container_width=True)

    # -- decision trail -----------------------------------------------------
    with t_trail:
        u1, u2, u3 = st.columns(3)
        u1.metric("LLM calls", usage.calls)
        u2.metric("Tokens", f"{usage.total_tokens:,}")
        u3.metric("Decisions logged", summary["n_decisions"])

        stages = list(dict.fromkeys(d.stage for d in result.decisions))
        chosen = st.multiselect("Filter stages", stages, default=stages)
        show_rejected = st.checkbox("Show rejected proposals", value=True)

        for d in result.decisions:
            if d.stage not in chosen:
                continue
            if not d.accepted and not show_rejected:
                continue
            pill = ("<span class='pill pill-llm'>LLM</span>" if d.source == "llm"
                    else "<span class='pill pill-heur'>heuristic</span>"
                    if d.source else "")
            status = ("<span class='pill pill-ok'>kept</span>" if d.accepted
                      else "<span class='pill pill-bad'>rejected</span>")
            st.markdown(f"{pill}{status} **{d.stage}** · {d.action}",
                        unsafe_allow_html=True)
            if d.reasoning:
                st.caption(f"↳ {d.reasoning}")
            if d.detail:
                st.caption(f"↳ **{d.detail}**")

    # -- export -------------------------------------------------------------
    with t_export:
        report_md = generate_markdown_report(result)
        st.download_button("⬇️ Explainability report (Markdown)", report_md,
                           file_name="automl_report.md", use_container_width=True)

        try:
            import joblib
            buf = io.BytesIO()
            joblib.dump(result.fitted_pipeline, buf)
            st.download_button(
                "⬇️ Trained model (.joblib)", buf.getvalue(),
                file_name=f"automl_{summary['best_model']}.joblib",
                use_container_width=True,
                help="A fitted scikit-learn Pipeline including preprocessing. "
                     "Load with joblib.load(...) and call .predict(df).")
        except Exception as e:
            st.caption(f"Model export unavailable: {e}")

        if result.ranking_table is not None and not result.ranking_table.empty:
            st.download_button(
                "⬇️ Feature ranking table (CSV)",
                result.ranking_table.to_csv().encode("utf-8"),
                file_name="feature_ranking.csv", use_container_width=True)

        with st.expander("Preview the report"):
            st.markdown(report_md)
