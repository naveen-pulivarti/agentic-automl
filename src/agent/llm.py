"""
Model-agnostic reasoning layer.

The rest of the codebase NEVER talks to Ollama/Groq directly. It calls
`Reasoner.propose_json(...)` and gets back a validated Pydantic object. This
single abstraction is what makes the free->paid swap a one-line config change.

Robustness design (in order of preference):
1. The LLM is asked to return JSON matching a Pydantic schema.
2. We parse + validate strictly.
3. If strict validation fails and the schema knows how to `salvage()`, we keep
   whatever individual items ARE valid. Small free models frequently emit three
   good proposals and one malformed one; discarding all four wastes an
   expensive call for no reason.
4. Failing that we retry, up to `max_parse_retries`.
5. If the LLM is unreachable OR keeps failing, we fall back to a deterministic
   heuristic supplied by the caller. This guarantees the pipeline ALWAYS
   produces a result, even with no LLM at all — the core reliability property.
"""
from __future__ import annotations

import json
import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import CONFIG, LLMConfig
from ..utils.logger import get_logger, log_tokens

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """Raised internally when the backend cannot be reached."""


class Reasoner:
    """Wraps whichever LLM backend is configured behind one method."""

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or CONFIG.llm
        self._client = None
        self._backend = None
        self._init_backend()

    # -- backend setup ------------------------------------------------------
    def _init_backend(self) -> None:
        """Lazily wire up whichever provider is selected. Never raises here —
        availability is checked at call time so the app can still boot."""
        self._backend = self.cfg.provider.lower()

    def _call_raw(self, system: str, user: str, timeout: int | None = None) -> str:
        """Send one prompt, return raw text. Raises LLMUnavailable on failure."""
        timeout = timeout or self.cfg.request_timeout
        if self._backend == "ollama":
            return self._call_ollama(system, user, timeout)
        elif self._backend in ("groq", "openai_compatible"):
            return self._call_openai_compatible(system, user, timeout)
        raise LLMUnavailable(f"Unknown provider: {self._backend}")

    def _call_ollama(self, system: str, user: str, timeout: int) -> str:
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("requests not installed") from e
        url = f"{self.cfg.ollama_base_url}/api/chat"
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
            "format": "json",  # ask Ollama to constrain output to JSON
        }
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
        except Exception as e:
            raise LLMUnavailable(f"Ollama call failed: {e}") from e
        data = r.json()
        # token accounting (Ollama returns eval counts)
        log_tokens(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            provider="ollama",
            model=self.cfg.model,
        )
        return data.get("message", {}).get("content", "")

    def _call_openai_compatible(self, system: str, user: str, timeout: int) -> str:
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("requests not installed") from e
        base = self.cfg.groq_base_url if self._backend == "groq" else self.cfg.ollama_base_url
        if not self.cfg.api_key:
            raise LLMUnavailable("No API key configured for cloud provider")
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
        except Exception as e:
            raise LLMUnavailable(f"Cloud LLM call failed: {e}") from e
        data = r.json()
        usage = data.get("usage", {})
        log_tokens(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            provider=self._backend,
            model=self.cfg.model,
        )
        return data["choices"][0]["message"]["content"]

    # -- public API ---------------------------------------------------------
    def propose_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        fallback: Callable[[], T],
    ) -> tuple[T, bool]:
        """Ask the LLM for JSON validated against `schema`.

        Returns (result, used_llm). `used_llm` is False when we fell back to the
        deterministic heuristic, so the UI can show honestly what happened.
        """
        last_err: Exception | None = None
        original_user = user
        for attempt in range(1, self.cfg.max_parse_retries + 1):
            try:
                raw = self._call_raw(system, user)
            except LLMUnavailable as e:
                log.warning("LLM unavailable (%s); using heuristic fallback.", e)
                return fallback(), False
            try:
                obj = _extract_and_validate(raw, schema)
                return obj, True
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("LLM output invalid (attempt %d/%d): %s",
                            attempt, self.cfg.max_parse_retries, e)
                # tighten the instruction on retry
                user = (
                    original_user
                    + "\n\nYour previous response was not valid JSON matching the "
                    "schema. Respond with ONLY valid JSON, no prose, no markdown."
                )
        log.warning("LLM failed validation %d times (%s); using heuristic fallback.",
                    self.cfg.max_parse_retries, last_err)
        return fallback(), False

    def health_check(self) -> tuple[bool, str]:
        """Quick probe so the UI can tell the user if the LLM is reachable.

        Uses a deliberately short timeout: this runs in the UI thread, and a
        slow probe makes the whole app feel hung. A cold model load can exceed
        it, which is reported honestly rather than hidden.
        """
        try:
            raw = self._call_raw(
                system="You are a helpful assistant. Respond in JSON.",
                user='Return exactly {"ok": true}',
                timeout=self.cfg.health_timeout,
            )
            return ("ok" in raw.lower()), f"{self._backend}:{self.cfg.model}"
        except LLMUnavailable as e:
            return False, str(e)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def _extract_and_validate(raw: str, schema: type[T]) -> T:
    """Pull the first JSON object out of a possibly-noisy string and validate.

    Strict validation first; if the schema provides `salvage()`, fall back to
    per-item validation so a single malformed entry doesn't discard the rest.
    """
    text = raw.strip()
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # find outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response")
    candidate = text[start : end + 1]
    data = json.loads(candidate)

    try:
        return schema.model_validate(data)
    except ValidationError:
        salvage = getattr(schema, "salvage", None)
        if salvage is None:
            raise
        obj = salvage(data)                       # raises if nothing usable
        log.info("Salvaged partial LLM output for %s", schema.__name__)
        return obj


# convenience singleton accessor
_REASONER: Reasoner | None = None


def get_reasoner() -> Reasoner:
    global _REASONER
    if _REASONER is None:
        _REASONER = Reasoner()
    return _REASONER


def reset_reasoner() -> None:
    """Drop the cached reasoner so a changed provider/model takes effect.

    Needed because the config reads environment variables at import time; the
    UI uses this when the user switches backend without restarting.
    """
    global _REASONER
    _REASONER = None
