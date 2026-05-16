# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Stdlib-logging redaction filter for API keys and bearer tokens.

Closes BLUEPRINT_v5plus.md §A "Structured logging" v0.5 commitment to
scrub provider API keys before any log line hits a handler. Implemented
as a stdlib ``logging.Filter`` rather than a ``structlog`` processor
because the current codebase uses stdlib ``logging`` exclusively
(zero ``structlog.configure(...)`` calls); the structlog migration is
deferred to v0.6 per the same BLUEPRINT section.

Patterns redacted (and the replacement marker that goes in their place
so an operator can still tell that *something* was scrubbed):

| Pattern | Replacement | Provider |
|---------|-------------|----------|
| ``sk-[A-Za-z0-9_-]{20,}`` | ``[REDACTED-OPENAI-KEY]`` | OpenAI / Anthropic dev / xAI variants |
| ``xai-[A-Za-z0-9_-]{20,}`` | ``[REDACTED-XAI-KEY]`` | xAI |
| ``AKIA[0-9A-Z]{16}`` | ``[REDACTED-AWS-ACCESS-KEY]`` | AWS access key id |
| ``AIza[0-9A-Za-z_-]{35}`` | ``[REDACTED-GOOGLE-API-KEY]`` | Google AI Studio / Gemini |
| ``Bearer\\s+[A-Za-z0-9._~+/=-]{16,}`` | ``Bearer [REDACTED-TOKEN]`` | RFC 6750 bearer tokens |

Patterns NOT redacted (out of scope for v0.5; tracked in BACKLOG):

- 32+ char base64-or-hex high-entropy strings (BLUEPRINT line 730
  also names these but the false-positive rate on hashes / canonical
  digests / cache keys is too high without a context heuristic).
- Cookie / `Set-Cookie` headers (no log path emits these in v0.5).
- Custom-prefixed enterprise keys (mistral_, vllm_, etc.) — add as
  needed.

The filter operates on both ``record.msg`` and (after string-formatting)
the rendered message, so call sites like
``_log.warning("body: %r", response_text)`` are covered alongside
direct ``_log.warning("Bearer abc...")`` calls. Records whose ``msg``
is a non-string (rare) are passed through untouched — the filter
errs on the side of NEVER dropping a log record.

Usage (called once at fuzzeval package import time):

    from fuzzeval.log_redaction import install_default_filter
    install_default_filter()

The filter is idempotent: installing twice is a no-op (we tag the
root logger with a private attribute on first install).
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar, Final, Pattern

_INSTALLED_MARKER: Final[str] = "_fuzzeval_log_redaction_installed"


class SecretRedactionFilter(logging.Filter):
    """Stdlib ``logging.Filter`` that scrubs known secret patterns in-place.

    Rewrites ``record.msg`` (when it is a str) and the elements of
    ``record.args`` (when each is a str) so the rendered log line is
    safe regardless of whether the handler later interpolates args
    server-side. Non-string args (ints, floats, dicts) are passed
    through untouched — pattern matching only kicks in on str values.

    The filter ALWAYS returns ``True`` so the record continues through
    the handler chain. Failure to redact must NEVER drop a log record;
    silent log drop is worse than a leaked secret because at least
    the latter is auditable post-hoc.
    """

    # Compile once at class-definition time so per-record overhead is
    # just the regex search, not the regex compile. Order matters:
    # Bearer first so its body doesn't get matched by AIza/sk- inside.
    _PATTERNS: ClassVar[tuple[tuple[Pattern[str], str], ...]] = (
        (
            re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
            "Bearer [REDACTED-TOKEN]",
        ),
        (
            re.compile(r"AIza[0-9A-Za-z_-]{35}"),
            "[REDACTED-GOOGLE-API-KEY]",
        ),
        (
            re.compile(r"AKIA[0-9A-Z]{16}"),
            "[REDACTED-AWS-ACCESS-KEY]",
        ),
        (
            re.compile(r"xai-[A-Za-z0-9_-]{20,}"),
            "[REDACTED-XAI-KEY]",
        ),
        (
            re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            "[REDACTED-OPENAI-KEY]",
        ),
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.msg
        if isinstance(msg, str):
            record.msg = self._scrub(msg)

        args = record.args
        if isinstance(args, tuple) and args:
            new_args = tuple(
                self._scrub(a) if isinstance(a, str) else a for a in args
            )
            # Only swap when something actually changed — keeps tuple
            # identity stable for the common no-secret path so handlers
            # that key on it don't see spurious churn.
            if new_args != args:
                record.args = new_args
        elif isinstance(args, dict) and args:
            # Some `LogRecord.args` are mapping-shaped (when the caller
            # uses `_log.info("%(field)s", {"field": value})`). Walk
            # values; never touch keys.
            scrubbed = {
                k: (self._scrub(v) if isinstance(v, str) else v)
                for k, v in args.items()
            }
            if scrubbed != args:
                record.args = scrubbed

        return True

    @classmethod
    def _scrub(cls, text: str) -> str:
        for pattern, replacement in cls._PATTERNS:
            text = pattern.sub(replacement, text)
        return text


def install_default_filter() -> bool:
    """Install ``SecretRedactionFilter`` on the root logger.

    Idempotent: a second call is a no-op (tagged via a private attr on
    the root logger). Returns True on first install, False on no-op.
    """
    root = logging.getLogger()
    if getattr(root, _INSTALLED_MARKER, False):
        return False
    root.addFilter(SecretRedactionFilter())
    setattr(root, _INSTALLED_MARKER, True)
    return True


def uninstall_default_filter() -> bool:
    """Test-only: undo ``install_default_filter()``.

    Returns True if a filter was actually removed, False otherwise.
    Production code should NEVER call this — the filter exists to
    prevent log leaks and turning it off in production defeats that.
    """
    root = logging.getLogger()
    if not getattr(root, _INSTALLED_MARKER, False):
        return False
    root.filters = [
        f for f in root.filters if not isinstance(f, SecretRedactionFilter)
    ]
    setattr(root, _INSTALLED_MARKER, False)
    return True
