"""Step 9: §A error-taxonomy boundary tests for classify_run_status."""

from __future__ import annotations

import pytest

from topofuzz.metrics import classify_run_status
from topofuzz.metrics.status import (
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
)


def test_zero_failures_returns_succeeded():
    assert classify_run_status(0, 10) == (STATUS_SUCCEEDED, 0)


def test_just_above_zero_is_degraded_low():
    # 1 / 100 = 1% → degraded but exit 0.
    assert classify_run_status(1, 100) == (STATUS_DEGRADED, 0)


def test_exactly_25_percent_is_degraded_low():
    """§A boundary at exactly 25% — locked in the lower (exit-0) bucket.

    See :mod:`topofuzz.metrics.status` docstring "Boundary semantics"
    for the §A row-text interpretation. The blueprint's "1–25%" text
    is naturally inclusive of 25%, so this is the intended placement.
    25.0001% falls in the next bucket."""
    assert classify_run_status(25, 100) == (STATUS_DEGRADED, 0)


def test_just_above_25_percent_is_degraded_high():
    assert classify_run_status(26, 100) == (STATUS_DEGRADED, 1)


def test_exactly_50_percent_is_degraded_high():
    assert classify_run_status(50, 100) == (STATUS_DEGRADED, 1)


def test_just_above_50_percent_is_failed():
    assert classify_run_status(51, 100) == (STATUS_FAILED, 2)


def test_all_failed_is_failed():
    assert classify_run_status(10, 10) == (STATUS_FAILED, 2)


def test_zero_total_is_succeeded():
    # Empty plan / fully-resumed run: nothing to fail, nothing to
    # degrade. Returning succeeded is the only sensible no-signal
    # default.
    assert classify_run_status(0, 0) == (STATUS_SUCCEEDED, 0)


def test_negative_counts_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        classify_run_status(-1, 10)
    with pytest.raises(ValueError, match="non-negative"):
        classify_run_status(0, -1)


def test_failed_exceeding_total_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        classify_run_status(11, 10)
