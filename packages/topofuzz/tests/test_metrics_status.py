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


# ---------------------------------------------------------------------------
# §8 supplementary gate (judge_failure_rate > 5% → degraded). Polish-audit
# DRIFT-1 fix in status.py.
# ---------------------------------------------------------------------------


def test_judge_failure_subset_constraint_rejected():
    """``n_judge_failure_cells`` cannot exceed ``n_failed_cells``."""
    with pytest.raises(ValueError, match="subset"):
        classify_run_status(5, 100, n_judge_failure_cells=6)


def test_negative_judge_failure_count_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        classify_run_status(5, 100, n_judge_failure_cells=-1)


def test_section8_gate_does_not_fire_when_no_failures():
    """0% judge_failure with 0% total failure stays succeeded."""
    assert classify_run_status(0, 100, n_judge_failure_cells=0) == (
        STATUS_SUCCEEDED, 0
    )


def test_section8_gate_below_threshold_preserves_section_a_status():
    """5% judge_failure_rate is the boundary — only >5% trips §8.

    Here ``n_failed = n_judge_failure = 5 out of 100 = 5%`` exactly.
    §8's gate is strict ``>``; §A still classifies degraded because any
    non-zero failure trips its 0%-floor rule. So the visible status
    here is degraded-via-§A, not degraded-via-§8."""
    status, exit_code = classify_run_status(
        5, 100, n_judge_failure_cells=5
    )
    assert (status, exit_code) == (STATUS_DEGRADED, 0)


def test_section8_gate_above_threshold_forces_degraded_when_section_a_succeeded():
    """The load-bearing §8 test: if §A would say succeeded (impossible
    under the current >0 floor BUT pinned here against a future §A
    relaxation), §8's >5% judge-failure trip downgrades to degraded."""
    # Construct the synthetic case where §A would say succeeded:
    # n_failed=0 (no LLM errors or judge failures rolled in) BUT
    # n_judge_failure_cells > 5%. Per the docstring contract,
    # n_judge_failure_cells must be a SUBSET of n_failed_cells, so this
    # state is unreachable through the orchestrator. We test the §8
    # mechanism by setting both: the orchestrator path always rolls
    # judge_failures into n_failed, so judge-failure > 5% with all
    # failures being judge-failures IS the natural test shape.
    status, exit_code = classify_run_status(
        6, 100, n_judge_failure_cells=6
    )
    # §A also says degraded here (6% > 0 floor); the §8 path is
    # exercised but the status is unchanged.
    assert (status, exit_code) == (STATUS_DEGRADED, 0)


def test_section8_gate_does_not_escalate_failed_runs():
    """§8 ONLY downgrades succeeded → degraded; it never escalates a
    degraded or failed run. A 60% failure rate is FAILED (§A), and §8
    must leave that classification alone even if judge_failures
    dominate."""
    status, exit_code = classify_run_status(
        60, 100, n_judge_failure_cells=60
    )
    assert (status, exit_code) == (STATUS_FAILED, 2)


def test_section8_default_zero_kwarg_preserves_legacy_callers():
    """``n_judge_failure_cells`` defaults to 0; callers that haven't
    been updated yet must see the same (status, exit_code) as before."""
    # Without the kwarg.
    assert classify_run_status(5, 100) == (STATUS_DEGRADED, 0)
    assert classify_run_status(0, 100) == (STATUS_SUCCEEDED, 0)
