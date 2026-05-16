"""Wilson addenda block in report.html — "for anyone who cares" footer.

Wilson 95% score interval on the verdict-flip rate, rendered as a small
"Statistical addenda" block above the page footer. Wilson is a tighter,
boundary-safe binomial CI than the bootstrap interval on the headline
card and answers the orthogonal "is the rate distinguishable from zero?"
question.
"""

from __future__ import annotations

import pytest

from topofuzz.report import _build_wilson_addenda


def test_wilson_addenda_happy_path():
    ctx = {"n_flips": 5, "n_pairs": 10}
    out = _build_wilson_addenda(ctx)
    assert out is not None
    assert out["n_flips"] == 5
    assert out["n_pairs"] == 10
    assert out["point"] == 0.5
    assert out["conf_pct"] == 95
    # Wilson 5/10 ≈ [0.2366, 0.7634].
    assert 0.0 < out["ci_lower"] < 0.5 < out["ci_upper"] < 1.0


def test_wilson_addenda_zero_flips_returns_meaningful_ci():
    """0/N is the boundary case Wilson handles correctly (normal
    approximation collapses there). The point is 0 and the upper CI
    is a real number, not None — that's the entire reason to render it."""
    ctx = {"n_flips": 0, "n_pairs": 20}
    out = _build_wilson_addenda(ctx)
    assert out is not None
    assert out["point"] == 0.0
    assert out["ci_lower"] == 0.0  # clamped at the lower boundary
    assert 0.0 < out["ci_upper"] < 1.0  # bounded — the value of Wilson


def test_wilson_addenda_all_flips_returns_meaningful_ci():
    """Same boundary safety on the other end: N/N → upper at 1.0,
    lower a real number bounded below 1."""
    ctx = {"n_flips": 20, "n_pairs": 20}
    out = _build_wilson_addenda(ctx)
    assert out is not None
    assert out["point"] == 1.0
    assert 0.0 < out["ci_lower"] < 1.0
    assert out["ci_upper"] == 1.0


def test_wilson_addenda_zero_pairs_returns_none():
    """No pairs → no proportion → no addenda (silently omitted from
    the report rather than rendering a placeholder)."""
    ctx = {"n_flips": 0, "n_pairs": 0}
    assert _build_wilson_addenda(ctx) is None


def test_wilson_addenda_missing_context_returns_none():
    assert _build_wilson_addenda(None) is None
    assert _build_wilson_addenda({}) is None
    assert _build_wilson_addenda({"n_flips": 5}) is None  # missing n_pairs


def test_wilson_addenda_non_int_returns_none():
    """Defensive: a malformed flip_composition_ctx with float counts
    should not crash the report — silently omit the block."""
    assert _build_wilson_addenda({"n_flips": 5.0, "n_pairs": 10}) is None
    assert _build_wilson_addenda({"n_flips": "5", "n_pairs": 10}) is None


def test_wilson_addenda_negative_pairs_returns_none():
    """A corrupted pair count shouldn't crash the report renderer."""
    assert _build_wilson_addenda({"n_flips": 0, "n_pairs": -1}) is None
