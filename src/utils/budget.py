"""
Stage budget enforcement.

The project claims — in its reports and in `config.py` — that every agentic loop
is bounded by BOTH an iteration cap and a wall-clock cap. The iteration cap is
trivially enforced by a `for` loop; this module supplies the missing half.

Why a wall-clock cap matters: iteration counts are only a safe termination
guarantee when each iteration costs roughly the same. On a 500,000-row dataset a
single cross-validation can take minutes, so "8 iterations" is not a bound a user
can reason about. A time budget is.

Usage:
    timer = StageTimer("Feature Engineering")
    for candidate in candidates:
        if timer.exhausted():
            break
        ...
"""
from __future__ import annotations

import time

from ..config import CONFIG
from .logger import get_logger

log = get_logger(__name__)


class StageTimer:
    """Wall-clock budget for one pipeline stage.

    Checked *between* iterations rather than interrupting work in progress, so a
    stage may overshoot by at most one evaluation. That is deliberate: killing a
    half-finished `cross_validate` would leave no measurement at all, and a
    partial result is worse than a slightly late one.
    """

    def __init__(self, stage: str, seconds: int | None = None) -> None:
        self.stage = stage
        self.limit = seconds if seconds is not None else CONFIG.budget.max_seconds_per_stage
        self.started = time.time()
        self._reported = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.elapsed)

    def exhausted(self) -> bool:
        """True once the stage has used its wall-clock budget.

        Logs once, so a caller polling in a loop does not spam the log.
        """
        if self.elapsed < self.limit:
            return False
        if not self._reported:
            log.warning("%s hit its %ds time budget after %.1fs; stopping early.",
                        self.stage, self.limit, self.elapsed)
            self._reported = True
        return True

    def summary(self) -> str:
        return f"{self.elapsed:.1f}s of {self.limit}s budget"
