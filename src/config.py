"""
Central configuration for the Agentic AutoML system.

All tunable knobs live here so the rest of the code never hardcodes magic
numbers. Budgets (max iterations / time) are the mechanism that guarantees the
agentic loops always terminate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# LLM / reasoning-layer configuration
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Configuration for the reasoning layer.

    The design is model-agnostic: `provider` selects the backend and the rest
    of the code only ever calls `agent.llm.get_reasoner()`. Swapping Ollama ->
    Groq -> any OpenAI-compatible endpoint is a one-line change here.
    """
    # "ollama" (local, free) or "groq" (free-tier cloud fallback) or "openai_compatible"
    provider: str = os.getenv("AUTOML_LLM_PROVIDER", "ollama")

    # Model name as understood by the chosen provider.
    #   ollama:  "llama3.1", "qwen2.5:7b", "llama3.2:3b" ...
    #   groq:    "llama-3.1-8b-instant", "llama-3.3-70b-versatile" ...
    model: str = os.getenv("AUTOML_LLM_MODEL", "llama3.1")

    # Base URL for the provider's API.
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # API key (only needed for groq / openai-compatible; ollama needs none).
    api_key: str = os.getenv("GROQ_API_KEY", os.getenv("AUTOML_LLM_API_KEY", ""))

    # Sampling: low temperature -> more deterministic, focused reasoning.
    temperature: float = 0.2
    max_tokens: int = 1024

    # How many times to re-ask the LLM if it returns malformed / invalid output
    # before falling back to a safe default.
    max_parse_retries: int = 3

    # Per-request timeout (seconds). Generous enough for a large local model on
    # a modest GPU, short enough that a hung endpoint cannot stall a whole run.
    request_timeout: int = 180

    # A *short* timeout used only by the UI health probe. The probe must never
    # make the app feel frozen, so it gives up quickly and reports honestly.
    health_timeout: int = 20


# ---------------------------------------------------------------------------
# Agentic-loop budgets — these guarantee termination
# ---------------------------------------------------------------------------
@dataclass
class BudgetConfig:
    """Hard limits so no stage can run forever.

    Both an iteration cap AND a wall-clock cap are enforced, because on a large
    dataset a single cross-validation can take far longer than on a small one —
    iteration counts alone are not a safe guarantee.
    """
    # -- Stage 2: feature engineering ------------------------------------
    # The stage is a genuine loop: `fe_rounds` conversations with the reasoner,
    # each proposing `fe_per_round` candidates, with the MEASURED outcome of
    # previous rounds fed back in. Feeding results back is what lets the agent
    # stop repeating operations that already failed.
    fe_rounds: int = 3
    fe_per_round: int = 4

    # -- Stage 3: feature selection --------------------------------------
    # How many candidate subsets the reasoner may compose from the ranking
    # table. Deterministic control subsets are always evaluated in addition.
    max_llm_subsets: int = 4
    # Fractions of the ranked column list used to build the no-LLM control
    # subsets. These are what the LLM's proposals must beat to prove value.
    control_subset_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    # Accept a smaller feature set when it costs less than this much score.
    # Rationale: if dropping columns loses nothing measurable, prefer the
    # simpler model (less overfitting, faster, easier to explain).
    selection_tolerance: float = 0.002

    # -- Stage 4: model selection ----------------------------------------
    max_models: int = 6

    # -- Stage 5: hyperparameter tuning ----------------------------------
    # Lower than it used to be on purpose: measurement showed tuning delivered
    # tiny gains while consuming most of the LLM budget. That budget now buys
    # feature engineering rounds and feature selection instead.
    max_tuning_iterations: int = 4

    # -- Global -----------------------------------------------------------
    # Wall-clock ceiling per stage (seconds). Checked between iterations, so a
    # stage may overshoot by at most one evaluation.
    max_seconds_per_stage: int = 300


# ---------------------------------------------------------------------------
# Machine-learning configuration
# ---------------------------------------------------------------------------
@dataclass
class MLConfig:
    """Cross-validation and training settings."""
    cv_folds: int = 5
    random_state: int = 42
    # Fraction of data held out as a final, untouched test set.
    test_size: float = 0.2
    # If a categorical target has more than this many unique values it is
    # treated as regression even if the dtype is object (rare safety net).
    max_classification_classes: int = 50

    # -- Large-dataset handling -------------------------------------------
    # The search stages (feature engineering / selection / model comparison)
    # run MANY cross-validations, so on a large dataset they are the bottleneck.
    # Above this row count we sample down for the *search* only; the chosen
    # pipeline is then trained and scored on the full data. This keeps the agent
    # usable on hundreds of thousands of rows without changing what it reports.
    fast_search_row_cap: int = 20000
    # Folds used during the fast search phase. Fewer folds on big data trades a
    # little estimate variance for a large speed-up.
    fast_search_cv_folds: int = 3
    # Above this row count, prefer histogram-based gradient boosting over the
    # classic implementation (orders of magnitude faster, same interface).
    hist_boosting_row_threshold: int = 10000


# ---------------------------------------------------------------------------
# Top-level bundle
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    ml: MLConfig = field(default_factory=MLConfig)


# A single importable instance used across the app.
CONFIG = AppConfig()
