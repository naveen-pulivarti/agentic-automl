# CLAUDE.md — Project Context for Agentic AutoML

> This file gives you (Claude, working in VS Code / Claude Code) the full
> context of this project: what it is, why it exists, the academic constraints
> it must satisfy, the architecture, and how to work on it. Read this first.

---

## 1. What this project is

**Agentic AutoML: Reasoning-Driven Model Building for Tabular Data.**

A web application (Streamlit) where a user uploads a tabular dataset (CSV/Excel)
or connects a database, points at a target column, and an AI agent autonomously
builds a good machine-learning model for it — profiling the data, engineering
features, selecting an algorithm, and tuning hyperparameters — while explaining
its reasoning at every step.

It handles **both classification and regression**, auto-detecting which from the
target column.

---

## 2. Why it exists (the academic + conceptual context)

This is an **MCA (Master of Computer Applications) final-year individual
project** for Jain University (Deemed-to-be University), CSIT elective. It must
satisfy specific university requirements (see section 8).

**The core intellectual justification — the "beats ChatGPT" test:**
The project must demonstrate *genuine agentic behaviour*, not just an LLM with a
UI. The defining principle that achieves this:

> **The LLM decides; scikit-learn verifies.**

- The **LLM** is the *reasoning layer*: it proposes what to try and explains why.
- **scikit-learn** is the *verification layer*: it actually trains models and
  computes real metrics.
- A proposal is **kept only if the real measured metric improves**.

This is what makes it defensible: ChatGPT can only generate text; this system
acts on real data and verifies against real results. An LLM suggestion that
doesn't improve a measured score is rejected automatically. This means the
system produces valid results **even with a weak or absent LLM**.

---

## 3. The core pattern

Every stage runs the same loop:

**PROPOSE → EXECUTE → MEASURE → DECIDE**

1. **Propose** — LLM suggests an action (a feature, a model, a param config).
2. **Execute** — code actually does it (trains a real model).
3. **Measure** — compute the real metric via cross-validation.
4. **Decide** — keep it if it improved the score, else discard.

Bounded by **budgets** (max iterations, max time per stage) so it always
terminates. Budgets live in `src/config.py`.

---

## 4. The five pipeline stages

1. **Data Profiling** (`src/data/profiler.py`) — factual, code-driven analysis
   of structure/types/missing values; auto-detects classification vs regression.
   **Also splits off the held-out test set before any other stage runs** — see
   the leakage note below.
2. **Feature Engineering** (`src/stages/feature_engineering.py`) — three rounds
   of proposals; each round receives the *measured* outcome of the previous one,
   so the agent stops repeating operations that already failed. Proposals are
   validated before execution.
3. **Feature Selection** (`src/stages/feature_selection.py`) — ten statistical
   methods rank every column (`src/ml/feature_ranking.py`); the reasoner composes
   candidate subsets from that evidence; every candidate — including
   deterministic controls — is cross-validated and the best measured one wins.
4. **Model Selection** (`src/stages/model_selection.py`) — shortlists and
   compares algorithms by real cross-validation scores. The candidate list is
   filtered by dataset size before the LLM sees it.
5. **Hyperparameter Tuning** (`src/stages/tuning.py`) — reasons about which
   config to try next based on results so far (NOT brute-force grid search).
   Never repeats a configuration; stops when the space is exhausted.

Output: the best pipeline + charts (`src/report/charts.py`) + a human-readable
explainability report (`src/report/generator.py`).

### The leakage rule (do not break this)

Every search stage optimises a cross-validation score. The test set is separated
in **profiling**, before any stage sees the data, and is opened exactly once in
`node_finalize`. Engineered features are rebuilt on the test set using
*training-derived* parameters only (bin edges, log shifts). If you add a stage,
it must operate on `state.X`/`state.y` and never touch `state.X_test`.

---

## 5. Architecture & key files

```
app.py                     Streamlit UI (entry point)
src/
  config.py                central config + budgets (all tunable knobs)
  data/loader.py           unified input: CSV / Excel / SQL -> pandas DataFrame
  data/profiler.py         Stage 1 + problem-type detection
  agent/llm.py             MODEL-AGNOSTIC reasoning layer (Ollama/Groq + fallback)
  agent/schemas.py         Pydantic schemas + whitelists + structural validation
  agent/state.py           AutoMLState — shared state flowing through pipeline
  agent/graph.py           LangGraph orchestration (+ sequential fallback)
  stages/                  the four agentic stages (2-5)
  ml/trainer.py            the VERIFICATION layer (sklearn training + metrics)
  ml/feature_ops.py        whitelisted operations + dataset-aware validation
  ml/feature_ranking.py    ten-method column ranking panel (evidence for Stage 3)
  report/generator.py      explainability report
  report/charts.py         Plotly figures for the UI and the written report
  utils/budget.py          per-stage wall-clock budgets (StageTimer)
  utils/logger.py          logging + local token-usage tracking
data/make_samples.py       generates sample datasets + demo.db (SQLite)
data/make_large_samples.py fetches 200k-row real benchmarks (UCI / OpenML)
tests/test_e2e.py          end-to-end + safety-property tests (run WITHOUT an LLM)
benchmarks/                reproducible measurement harness + saved results
```

### Key design decisions (important — preserve these):

- **Model-agnostic LLM layer.** Nothing calls Ollama/Groq directly. Everything
  goes through `agent/llm.py::Reasoner.propose_json(...)`. Switching
  free↔paid↔any-OpenAI-compatible provider is a config change only
  (`AUTOML_LLM_PROVIDER` / `AUTOML_LLM_MODEL` / `GROQ_API_KEY`).
- **Heuristic fallback everywhere.** Every stage has a deterministic fallback
  used when the LLM is unavailable OR returns invalid output. This guarantees
  the pipeline always produces a result. DO NOT remove these fallbacks — they
  are the core reliability guarantee.
- **Whitelisted operations.** The LLM can only propose feature operations and
  models from fixed enums (`schemas.py`). It can never execute arbitrary code.
  This is a deliberate safety boundary — preserve it.
- **Pydantic validation, in two layers.** `agent/schemas.py` enforces
  *structural* rules that hold for any dataset (a ratio needs two columns);
  `ml/feature_ops.py::validate_proposal` enforces *dataset-dependent* rules a
  schema cannot see (column exists, is really numeric, is not the target).
  Validating before executing matters: every unchecked proposal otherwise costs
  a full cross-validation to discover it was useless.
- **Partial-output salvage.** A batch of proposals is all-or-nothing to Pydantic,
  and small free models routinely emit three good items and one malformed one.
  `salvage()` keeps the valid items instead of discarding the whole call. Do not
  remove it — it is what makes weak local models usable.
- **Size-aware model filtering.** `valid_models_for(problem, n_rows)` drops SVM
  and KNN on large data and swaps in histogram gradient boosting. This is in
  code, not the prompt, because it must hold whether or not the LLM cooperates.
- **LangGraph orchestration** with a plain-Python fallback runner
  (`run_pipeline_sequential`). Both must stay working.
- **Metrics: higher is always better.** Classification primary = f1_macro;
  regression primary = r2. "neg_" sklearn metrics are converted to positive.

---

## 6. Tech stack (and why each is here)

- **Python** — core language.
- **LangGraph (+ langchain-core)** — agent orchestration as a stateful graph.
- **Ollama** (local, free — Llama 3.1 / Qwen2.5) — reasoning layer; **Groq**
  free-tier as fallback.
- **scikit-learn** — training, cross-validation, real metrics (verification).
- **pandas / NumPy** — data handling.
- **openpyxl** — Excel reading.
- **SQLAlchemy** — DB connectivity (PostgreSQL, MySQL, SQL Server, SQLite).
- **Pydantic** — structured/validated LLM output.
- **Streamlit** — UI.
- **Matplotlib / Plotly** — charts.
- Built-in local token tracking (`utils/logger.py`); optional LangSmith later.

Everything is free / open-source. A paid API can substitute via the
model-agnostic layer if free models underperform.

---

## 7. Constraints (scope boundaries — respect these)

- **Tabular data only.** No images, audio, or free text. No deep learning
  (would need GPU). Classical ML via scikit-learn only.
- **Individual project**, buildable solo, **completed in a compressed timeline**
  (originally 8 weeks; actually being built in ~15 days).
- **Zero-budget:** free/open-source tools only (paid API is a last-resort
  fallback, kept behind the model-agnostic layer).
- **Must have a working UI** (Streamlit satisfies this).
- **Scope is intentionally bounded:** a curated set of algorithms and feature
  ops, not exhaustive; not trying to beat commercial AutoML on raw accuracy.
  The originality is the *transparent, reasoning-driven* approach.
- **IMPORTANT — keep separate from the author's day job.** The author works in
  insurance / data engineering; this project must remain an independent academic
  work and must NOT reuse or reference their professional/work projects. Frame
  everything as a general-purpose academic tool.

---

## 8. University requirements (must be satisfied)

- Must be a **working software application with real coding** — not a study,
  survey, or literature review.
- Must have a **user interface** and be **demonstrable** in a video.
- **Existing ideas are acceptable if improved** with original implementation.
- **AI-assisted coding is allowed**, but the author must understand and be able
  to defend every part in a viva.
- **Report structure is fixed** (5 chapters) — do not rename/remove chapters.
- **Plagiarism ceiling: 20%** similarity — code and report must be original.
- Deliverables: Synopsis (submitted), Interim Report (submitted), Final Report
  (45–65 pages), 5-minute demo video.
- The synopsis and interim report are in this repo (`docs/`) for reference.

---

## 9. How to work on this project

### Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/make_samples.py        # generate sample datasets
python tests/test_e2e.py           # verify everything works (no LLM needed)
streamlit run app.py               # launch the UI
```

### For real LLM reasoning (optional but wanted):
```bash
# install Ollama from https://ollama.com then:
ollama pull llama3.1
# defaults already point at local Ollama
```

### Working style expected:
- **Preserve the fallback paths and safety boundaries** described in section 5.
- When adding features, add a real grounded verification step — don't add
  anything that relies purely on the LLM's self-assessment.
- Keep changes explainable — the author must be able to defend them in a viva.
- Match the existing code style (type hints, docstrings explaining *why*,
  conservative error handling that never crashes the pipeline).
- When something needs the author's decision or testing, say so explicitly.

---

## 10. Current status & known next steps

> The living status log is **`context/10_session_progress.md`** — it supersedes
> `context/06_current_status_and_roadmap.md`. Read it first.

**Working and tested:**
- Full 5-stage pipeline end-to-end on 7 datasets — iris, wine, synthetic churn,
  diabetes, california, plus **200,000-row** Forest Cover Type and SGEMM GPU
  performance.
- LangGraph orchestration AND sequential fallback.
- CSV/Excel loading + SQLite DB connection.
- **llama3.1 via Ollama verified working** — valid Pydantic JSON first try.
- Held-out metrics are now honest (test set split before any search).
- Full pytest suite covering end-to-end runs plus the safety properties
  (no target leakage, no duplicate tuning configs, train/test disjoint,
  partial-output salvage).

**Open items:**
1. Groq free-tier validation — needs an API key from the author.
2. Record the 5-minute demo video (local llama3.1 is ~3 min/dataset on 4 GB
   VRAM, so Groq or `llama3.2:3b` is the practical demo backend).
3. Final report (5 chapters) + plagiarism check.
4. Rename the working folder away from `_old` before filming/submission.
5. Real Postgres/MySQL connection has still only been exercised via SQLite.

**Do NOT:**
- Remove heuristic fallbacks, the whitelist safety boundary, or `salvage()`.
- Let any search stage touch `state.X_test`.
- Introduce deep learning or non-tabular data handling.
- Add anything that can't be defended in a viva or that reuses the author's
  work-domain projects.
```
