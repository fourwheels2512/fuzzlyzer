"""Step 15a (c): verdict-flip rate vs original variant.

Covers the contract in :mod:`topofuzz.metrics.verdict_flip`:

* point matches the BLUEPRINT line 416 definition
  (% of (permuted, eval) pairs that flip vs original)
* zero on identical-variant input
* non-zero on opposite-prediction input
* CI brackets the point on a well-separated fixture
* NaN cells drop from BOTH numerator and denominator (not silently
  counted as non-flips)
* degenerate paths (single variant, no valid pairs, all-same verdict)
  flag without crashing
* original_index validation rejects out-of-range
"""

from __future__ import annotations

import numpy as np
import pytest

from topofuzz.metrics import VerdictFlipResult, compute_verdict_flip_rate


def test_zero_on_identical_variants():
    """All variants score identically → flip rate is exactly 0."""
    scores = np.full((4, 5), 0.72)
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.point == 0.0
    # Degenerate: all-same verdict, no resample can produce a flip.
    assert result.degenerate_flag is True
    assert result.ci_lower is None and result.ci_upper is None


def test_one_on_opposite_predictions():
    """Original predicts 1 on every eval; one permuted predicts 0 → flip rate 1.0
    on the (perm0, *) row. Average over (n_variants - 1 = 1) row is 1.0."""
    scores = np.array(
        [
            [0.80, 0.90, 0.85, 0.95, 0.78],  # original: all >= 0.5 → 1
            [0.10, 0.05, 0.20, 0.15, 0.08],  # permuted: all < 0.5 → 0
        ]
    )
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.point == 1.0
    assert result.degenerate_flag is False
    # CI should be tight around 1.0 (every resampled draw produces 1.0).
    assert result.ci_lower == 1.0
    assert result.ci_upper == 1.0


def test_half_flip_rate_balanced_fixture():
    """Two permuted rows; one flips every eval, the other never flips →
    expected flip rate = 0.5 (5 flips out of 10 pairs)."""
    scores = np.array(
        [
            [0.80, 0.90, 0.85, 0.95, 0.78],  # original: 1,1,1,1,1
            [0.10, 0.05, 0.20, 0.15, 0.08],  # flipper: 0,0,0,0,0
            [0.82, 0.91, 0.87, 0.96, 0.79],  # no flips: 1,1,1,1,1
        ]
    )
    result = compute_verdict_flip_rate(scores, original_index=0, seed=42)
    assert result.point == 0.5
    assert result.degenerate_flag is False
    # CI brackets the point estimate.
    assert result.ci_lower <= result.point <= result.ci_upper


def test_nan_cells_drop_from_both_num_and_denom():
    """A NaN cell shouldn't silently count as a non-flip. Compare two
    matrices that differ only at one NaN cell — flip RATE should be
    identical because the cell drops from both numerator and denominator."""
    # 2 variants, 4 evals. Original predicts 1 on all; permuted flips
    # on eval 0 and eval 2; eval 3 we'll make NaN in one of the matrices.
    base = np.array(
        [
            [0.80, 0.90, 0.85, 0.78],
            [0.10, 0.92, 0.20, 0.79],
        ]
    )
    masked = base.copy()
    masked[1, 3] = np.nan
    r_base = compute_verdict_flip_rate(base, original_index=0, seed=0)
    r_mask = compute_verdict_flip_rate(masked, original_index=0, seed=0)
    # base: 2 flips / 4 pairs = 0.5; masked: 2 flips / 3 pairs ≈ 0.667.
    # If NaN were silently counted as a non-flip, masked would be
    # 2/4 = 0.5 — the assertion catches that bug.
    assert r_base.point == 0.5
    assert abs(r_mask.point - 2.0 / 3.0) < 1e-9


def test_single_variant_is_degenerate():
    """n_variants < 2 has no permuted rows; metric is structurally
    undefined and reported degenerate."""
    scores = np.array([[0.5, 0.6, 0.7]])
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.point is None
    assert result.degenerate_flag is True


def test_original_index_out_of_range_raises():
    scores = np.array([[0.5, 0.6], [0.4, 0.7]])
    with pytest.raises(ValueError, match="original_index"):
        compute_verdict_flip_rate(scores, original_index=5, seed=0)
    with pytest.raises(ValueError, match="original_index"):
        compute_verdict_flip_rate(scores, original_index=-1, seed=0)


def test_threshold_out_of_range_raises():
    scores = np.array([[0.5, 0.6], [0.4, 0.7]])
    with pytest.raises(ValueError, match="threshold"):
        compute_verdict_flip_rate(scores, original_index=0, threshold=1.5)


def test_2d_required():
    with pytest.raises(ValueError, match="2D"):
        compute_verdict_flip_rate(np.array([0.5, 0.6]), original_index=0)


def test_seed_determinism():
    """Same seed → byte-identical CIs across calls."""
    scores = np.array(
        [
            [0.80, 0.30, 0.85, 0.40, 0.70],
            [0.20, 0.91, 0.40, 0.10, 0.85],
            [0.55, 0.40, 0.60, 0.30, 0.45],
        ]
    )
    a = compute_verdict_flip_rate(scores, original_index=0, seed=7)
    b = compute_verdict_flip_rate(scores, original_index=0, seed=7)
    assert a.point == b.point
    assert a.ci_lower == b.ci_lower
    assert a.ci_upper == b.ci_upper


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(123)
    scores = rng.random((6, 30))
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.degenerate_flag is False
    assert result.ci_lower <= result.point <= result.ci_upper


def test_returns_named_result():
    scores = np.array([[0.8, 0.9], [0.1, 0.2]])
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert isinstance(result, VerdictFlipResult)
    assert result.method == "percentile"


def test_original_all_nan_is_degenerate():
    """Audit RT-m3: original row all-NaN → no valid pairs → degenerate
    without crashing, without claiming a zero-width CI."""
    scores = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.80, 0.20, 0.50, 0.30],
            [0.85, 0.15, 0.55, 0.25],
        ]
    )
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.degenerate_flag is True
    assert result.point is None
    assert result.ci_lower is None
    assert result.ci_upper is None


def test_orphan_valid_cells_dont_spoof_non_degenerate():
    """Audit CR-M1: a paired-cell statistic with zero flips must report
    degenerate even when some orphan valid cells exist on either the
    original or permuted row (i.e. where the paired counterpart is NaN).
    The previous ``valid_anywhere`` check could miss this and ship a
    zero-width CI as ``degenerate_flag=False``."""
    # Original row valid only at eval 0; permuted row valid only at eval 1.
    # No paired cells → no flip information → degenerate.
    scores = np.array(
        [
            [0.80, np.nan],
            [np.nan, 0.10],
        ]
    )
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.degenerate_flag is True
    assert result.ci_lower is None and result.ci_upper is None
