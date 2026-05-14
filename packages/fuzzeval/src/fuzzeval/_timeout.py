"""Shared LLM-timeout env-var resolver — single source of truth.

Both ``topofuzz.runner`` and ``fuzzeval.judge`` need to consult
``TOPOFUZZ_LLM_TIMEOUT_SECS`` at import time and clamp to a finite
positive ceiling. The two byte-identical copies (except the ``_log``
prefix) were tracked as a v0.5.1 POLISH carryover; this module is the
de-duplicated location they both import from.

Closes BACKLOG: DRY ``_resolve_llm_timeout_secs``.

Security note (closes BACKLOG: truncate env-var value in warning log):
the warning logs include the raw env-var value via ``%r`` so an
operator can see exactly what was set. A malicious / fat-fingered
``TOPOFUZZ_LLM_TIMEOUT_SECS="api_key=sk-..."`` (shell-history paste,
``export`` typo) would leak the secret through any centralised log
shipper that ingests stderr. ``_truncate_for_log`` caps the value at
32 characters with an explicit ellipsis so secrets don't make it past
the local terminal.
"""

from __future__ import annotations

import logging
import math
import os

# 1-hour ceiling matches the prior duplicates. Floats above this are
# almost always operator typos (env-var meant to be in milliseconds, an
# unbounded retry pacer, etc).
DEFAULT_TIMEOUT_SECS: float = 60.0
LLM_TIMEOUT_MAX_SECS: float = 3600.0
ENV_VAR: str = "TOPOFUZZ_LLM_TIMEOUT_SECS"
_LOG_TRUNCATE_AT: int = 32


def _truncate_for_log(raw: str) -> str:
    """Cap the env-var value at 32 chars in log output to avoid leaking
    secrets if the operator pasted an API key into the variable by
    mistake. The repr() the caller wraps this in still escapes embedded
    quotes — we just bound the leaked length."""
    if len(raw) <= _LOG_TRUNCATE_AT:
        return raw
    return raw[:_LOG_TRUNCATE_AT] + "…(truncated)"


def resolve_llm_timeout_secs(_log: logging.Logger, *, prefix: str) -> float:
    """Read ``TOPOFUZZ_LLM_TIMEOUT_SECS`` from the environment and clamp.

    ``prefix`` is the per-caller name written into the warning log
    (``"runner"`` or ``"judge"``) so a single log line surfaces which
    side observed the bad value.
    """
    raw = os.environ.get(ENV_VAR, str(int(DEFAULT_TIMEOUT_SECS)))
    safe = _truncate_for_log(raw)
    try:
        value = float(raw)
    except ValueError:
        _log.warning(
            "%s: %s=%r is not a float; falling back to %s",
            prefix, ENV_VAR, safe, DEFAULT_TIMEOUT_SECS,
        )
        return DEFAULT_TIMEOUT_SECS
    if not math.isfinite(value) or value <= 0:
        _log.warning(
            "%s: %s=%r is not a positive finite float; falling back to %s",
            prefix, ENV_VAR, safe, DEFAULT_TIMEOUT_SECS,
        )
        return DEFAULT_TIMEOUT_SECS
    if value > LLM_TIMEOUT_MAX_SECS:
        _log.warning(
            "%s: %s=%r exceeds %ds ceiling; clamping to ceiling",
            prefix, ENV_VAR, safe, int(LLM_TIMEOUT_MAX_SECS),
        )
        return LLM_TIMEOUT_MAX_SECS
    return value
