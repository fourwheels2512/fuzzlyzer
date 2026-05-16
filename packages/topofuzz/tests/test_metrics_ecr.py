# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 9: §9 metric #2 (ECR) — boundary + degenerate behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from topofuzz.metrics import ECRResult, error_correction_rate


def test_none_upstream_wrong_is_degenerate():
    """v0.5 lending-fixture path: no per-upstream-agent ground truth →
    ``upstream_wrong=None`` → degenerate with a clear reason string."""
    vc = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    r = error_correction_rate(vc, upstream_wrong=None)
    assert isinstance(r, ECRResult)
    assert r.degenerate_flag
    assert r.value is None
    assert "upstream_wrong signal unavailable" in r.degenerate_reason


def test_happy_path_positive_ecr():
    """Toy case: upstream-wrong cells correct ~80% of the time vs
    all-correct cells correct 100% of the time → ECR = 0.8 - 1.0 = -0.2.
    Negative ECR means the topology DEGRADES when upstream errs."""
    # 6 cells each conditional → both clear the 5-cell denominator.
    upstream_wrong = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    verdict_correct = np.array(
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool
    )
    r = error_correction_rate(verdict_correct, upstream_wrong)
    assert not r.degenerate_flag
    assert r.value is not None
    assert r.n_upstream_wrong == 6
    assert r.n_all_correct == 6
    # 5/6 vs 6/6.
    np.testing.assert_allclose(r.value, 5 / 6 - 1.0, atol=1e-12)


def test_denominator_below_threshold_is_degenerate():
    """4 cells in the upstream-wrong bucket → < 5 → degenerate."""
    upstream_wrong = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    verdict_correct = np.ones(10, dtype=bool)
    r = error_correction_rate(verdict_correct, upstream_wrong)
    assert r.degenerate_flag
    assert r.value is None
    assert r.n_upstream_wrong == 4
    assert r.n_all_correct == 6
    assert "denominator too small" in r.degenerate_reason


def test_exactly_5_cells_each_clears_threshold():
    upstream_wrong = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    verdict_correct = np.ones(10, dtype=bool)
    r = error_correction_rate(verdict_correct, upstream_wrong)
    assert not r.degenerate_flag
    # 5/5 in both buckets → 0.0 ECR; no correction signal but the metric
    # is well-defined.
    assert r.value == 0.0


def test_2d_inputs_flattened():
    # (3, 4) shape — flattened to 12 cells. Confirms row/col ordering
    # doesn't matter for the metric.
    verdict_correct = np.ones((3, 4), dtype=bool)
    upstream_wrong = np.array(
        [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool
    )
    r = error_correction_rate(verdict_correct, upstream_wrong)
    assert not r.degenerate_flag
    assert r.value == 0.0  # all verdicts correct in both buckets
    assert r.n_upstream_wrong == 6
    assert r.n_all_correct == 6


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="share shape after flattening"):
        error_correction_rate(
            np.ones(10), np.ones(8)
        )


def test_min_denominator_must_be_positive():
    with pytest.raises(ValueError, match="must be >= 1"):
        error_correction_rate(
            np.ones(10), np.zeros(10), min_denominator=0
        )


def test_tunable_min_denominator():
    """Caller can lower the denominator threshold for forensic loads
    where the ECR signal is weak but still informative."""
    upstream_wrong = np.array([1, 1, 0, 0, 0, 0, 0], dtype=bool)
    verdict_correct = np.ones(7, dtype=bool)
    # min_denominator=2 lets the 2-cell upstream-wrong bucket clear.
    r = error_correction_rate(
        verdict_correct, upstream_wrong, min_denominator=2
    )
    assert not r.degenerate_flag
    assert r.value == 0.0
