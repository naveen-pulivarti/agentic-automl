"""
Logging + token-usage tracking.

Token accounting is done locally (no cloud dependency) by summing the counts
the providers return. The running totals are exposed so the Streamlit UI can
display a live token/cost counter — satisfying the "transparency" goal without
requiring LangSmith.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from threading import Lock


# ---------------------------------------------------------------------------
# Standard logger
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Token usage tracker (process-wide, thread-safe)
# ---------------------------------------------------------------------------
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class _TokenTracker:
    def __init__(self) -> None:
        self._usage = TokenUsage()
        self._lock = Lock()

    def add(self, prompt_tokens: int, completion_tokens: int,
            provider: str, model: str) -> None:
        with self._lock:
            self._usage.prompt_tokens += int(prompt_tokens or 0)
            self._usage.completion_tokens += int(completion_tokens or 0)
            self._usage.calls += 1
            key = f"{provider}:{model}"
            self._usage.by_provider[key] = (
                self._usage.by_provider.get(key, 0)
                + int(prompt_tokens or 0) + int(completion_tokens or 0)
            )

    def snapshot(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._usage.prompt_tokens,
                completion_tokens=self._usage.completion_tokens,
                calls=self._usage.calls,
                by_provider=dict(self._usage.by_provider),
            )

    def reset(self) -> None:
        with self._lock:
            self._usage = TokenUsage()


_TRACKER = _TokenTracker()


def log_tokens(prompt_tokens: int, completion_tokens: int,
               provider: str, model: str) -> None:
    _TRACKER.add(prompt_tokens, completion_tokens, provider, model)


def get_token_usage() -> TokenUsage:
    return _TRACKER.snapshot()


def reset_token_usage() -> None:
    _TRACKER.reset()
