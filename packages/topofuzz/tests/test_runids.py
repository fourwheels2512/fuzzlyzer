"""Tests for ``topofuzz.runids`` (date-prefixed ULID generation)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from topofuzz.runids import _CROCKFORD, _base32_encode, new_run_id


def test_new_run_id_format():
    rid = new_run_id()
    # YYYY-MM-DD-26-char-ULID
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[" + _CROCKFORD + "]{26}", rid)


def test_new_run_id_uses_utc_date():
    fixed = datetime(2026, 5, 12, 23, 59, 59, tzinfo=timezone.utc)
    rid = new_run_id(now=fixed)
    assert rid.startswith("2026-05-12-")


def test_new_run_ids_are_unique_in_a_burst():
    """1000 rapid ids should be distinct.

    Spec says monotonic-safe per millisecond is not guaranteed but
    collisions inside one millisecond require simultaneous 80-bit
    random matches — ~2^-40 per call.
    """
    ids = {new_run_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_new_run_ids_sort_in_time_order():
    """Date prefix dominates; within a single day the ULID's timestamp
    half keeps later ids lex-greater than earlier ones.

    Sleep 20 ms — comfortably above the Windows scheduler tick (~15.6 ms)
    so ``time.time_ns() // 1_000_000`` is guaranteed to advance between
    calls. 3 ms previously made the assertion lean entirely on the
    random suffix being lex-greater for rid2 (50% flake rate; audit m-3).
    """
    import time

    rid1 = new_run_id()
    time.sleep(0.020)
    rid2 = new_run_id()
    assert rid1 < rid2


def test_base32_encode_zero():
    assert _base32_encode(0, length=4) == "0000"


def test_base32_encode_max_for_length():
    # 5 bits per char × 4 chars = 20 bits; max value = 2^20 - 1
    assert _base32_encode((1 << 20) - 1, length=4) == "ZZZZ"


def test_base32_encode_rejects_overflow():
    import pytest
    with pytest.raises(ValueError):
        _base32_encode(1 << 20, length=4)  # 1 too big


def test_base32_encode_rejects_negative():
    import pytest
    with pytest.raises(ValueError):
        _base32_encode(-1, length=4)
