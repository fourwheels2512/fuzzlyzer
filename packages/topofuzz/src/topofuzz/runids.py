"""Date-prefixed ULID for the step 6 run-id scheme.

Per locked decision (user, 2026-05-12): ``runs/<YYYY-MM-DD>-<ULID>/``.

* The date prefix is the UTC date of run kickoff. Sortable by ``ls``.
* The ULID is 26 chars of Crockford base32 — 48 bits of millisecond
  timestamp + 80 bits of randomness, monotonic-safe within the same
  millisecond by sampling fresh randomness per id. We do not bother
  with the strict monotonic-ulid spec (incrementing the random part
  on collision) because at the rates step 6 generates run ids (one
  per CLI invocation) collisions are astronomically improbable.
* Hand-rolled (no new dependency) — ulid-py and python-ulid are both
  small libs but pulling them in just for this would be unfriendly to
  the ``topofuzz[verify]`` slim install.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

# Crockford base32: omits I, L, O, U to avoid look-alike + slur edge cases.
# This is the standard ULID alphabet.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _base32_encode(value: int, *, length: int) -> str:
    """Encode ``value`` as a Crockford base32 string of fixed ``length`` chars.

    Pads with leading zeros (``"0"`` is the lowest digit in Crockford).
    """
    if value < 0:
        raise ValueError(f"_base32_encode: value must be >= 0, got {value}")
    chars: list[str] = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    if value != 0:
        raise ValueError(
            f"_base32_encode: value too large to fit in {length} chars"
        )
    return "".join(reversed(chars))


def _ulid() -> str:
    """Return a fresh 26-char Crockford-base32 ULID.

    Uses ``time.time_ns()`` for the timestamp (millisecond precision
    truncated from nanoseconds) and ``os.urandom(10)`` for the
    randomness. Both are process-safe and need no global state.
    """
    timestamp_ms = time.time_ns() // 1_000_000
    if timestamp_ms < 0 or timestamp_ms >= (1 << 48):
        # The 48-bit ms field overflows in the year ~10889.
        raise OverflowError(
            f"ULID timestamp overflow: {timestamp_ms} doesn't fit in 48 bits."
        )
    rand_bits = int.from_bytes(os.urandom(10), byteorder="big")
    return _base32_encode(timestamp_ms, length=10) + _base32_encode(
        rand_bits, length=16
    )


def new_run_id(*, now: datetime | None = None) -> str:
    """Return ``"<YYYY-MM-DD>-<ULID>"``.

    Args:
        now: Override for tests. Default is ``datetime.now(timezone.utc)``.
            Only the date portion is used; the ULID's internal millis
            timestamp comes from :func:`time.time_ns` regardless.
    """
    when = now if now is not None else datetime.now(timezone.utc)
    return f"{when:%Y-%m-%d}-{_ulid()}"
