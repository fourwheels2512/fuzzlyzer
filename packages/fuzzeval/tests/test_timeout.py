# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for fuzzeval._timeout — shared LLM-timeout env-var resolver."""

from __future__ import annotations

import logging

import pytest

from fuzzeval._timeout import (
    DEFAULT_TIMEOUT_SECS,
    LLM_TIMEOUT_MAX_SECS,
    _truncate_for_log,
    resolve_llm_timeout_secs,
)


def test_truncate_short_value_unchanged():
    assert _truncate_for_log("60") == "60"
    assert _truncate_for_log("a" * 32) == "a" * 32


def test_truncate_long_value_capped():
    """Closes BACKLOG: truncate env-var value to 32 chars in import-time
    warning log so a pasted API key cannot leak through stderr."""
    secret = "api_key=sk-" + "x" * 80
    out = _truncate_for_log(secret)
    assert len(out) <= 32 + len("…(truncated)")
    assert out.endswith("…(truncated)")
    assert "x" * 80 not in out


def test_resolve_llm_timeout_default_when_unset(monkeypatch, caplog):
    monkeypatch.delenv("TOPOFUZZ_LLM_TIMEOUT_SECS", raising=False)
    log = logging.getLogger("test_timeout")
    assert resolve_llm_timeout_secs(log, prefix="test") == DEFAULT_TIMEOUT_SECS


def test_resolve_llm_timeout_secret_truncated_in_warning(monkeypatch, caplog):
    """A malicious / typo'd env value with an API key prefix must not
    appear in the warning log in full."""
    secret = "api_key=sk-" + "y" * 80
    monkeypatch.setenv("TOPOFUZZ_LLM_TIMEOUT_SECS", secret)
    log = logging.getLogger("test_timeout")
    caplog.set_level(logging.WARNING, logger=log.name)
    out = resolve_llm_timeout_secs(log, prefix="judge")
    assert out == DEFAULT_TIMEOUT_SECS  # falls back on non-float
    full_messages = " ".join(r.getMessage() for r in caplog.records)
    assert "y" * 80 not in full_messages
    assert "truncated" in full_messages


def test_resolve_llm_timeout_clamps_above_ceiling(monkeypatch, caplog):
    monkeypatch.setenv("TOPOFUZZ_LLM_TIMEOUT_SECS", str(LLM_TIMEOUT_MAX_SECS * 10))
    log = logging.getLogger("test_timeout")
    caplog.set_level(logging.WARNING, logger=log.name)
    out = resolve_llm_timeout_secs(log, prefix="runner")
    assert out == LLM_TIMEOUT_MAX_SECS


def test_resolve_llm_timeout_rejects_nan_and_inf(monkeypatch):
    log = logging.getLogger("test_timeout")
    monkeypatch.setenv("TOPOFUZZ_LLM_TIMEOUT_SECS", "nan")
    assert resolve_llm_timeout_secs(log, prefix="t") == DEFAULT_TIMEOUT_SECS
    monkeypatch.setenv("TOPOFUZZ_LLM_TIMEOUT_SECS", "inf")
    assert resolve_llm_timeout_secs(log, prefix="t") == DEFAULT_TIMEOUT_SECS
    monkeypatch.setenv("TOPOFUZZ_LLM_TIMEOUT_SECS", "-30")
    assert resolve_llm_timeout_secs(log, prefix="t") == DEFAULT_TIMEOUT_SECS


def test_resolve_llm_timeout_accepts_positive_finite(monkeypatch):
    monkeypatch.setenv("TOPOFUZZ_LLM_TIMEOUT_SECS", "120")
    log = logging.getLogger("test_timeout")
    assert resolve_llm_timeout_secs(log, prefix="t") == 120.0
