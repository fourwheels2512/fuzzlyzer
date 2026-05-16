# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 15a (d) + (e): kappa_original_vs_permuted + balanced_accuracy_drop.

Covers the contracts in :mod:`topofuzz.metrics.kappa`:

* ``cohens_kappa_original_vs_permuted``: κ between original variant's
  binarised verdicts and the union of all permuted variants' verdicts.
* ``balanced_accuracy_drop``: BA(original) − mean BA(permuted) against
  ground-truth labels.

Tests cover the bracketing cases (identical → κ=1; opposite → κ=−1;
mixed → in (−1, 1)), NaN handling, degenerate paths, and the
ground-truth-binary contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from topofuzz.metrics import (
    BalancedAccuracyDropResult,
    balanced_accuracy_drop,
    cohens_kappa_original_vs_permuted,
)


# ---------------------------------------------------------------------------
# cohens_kappa_original_vs_permuted
# ---------------------------------------------------------------------------


def test_kappa_identical_predictions_is_one():
    """All variants predict the same → κ = 1."""
    scores = np.array(
        [
            [0.80, 0.30, 0.85, 0.40, 0.70],
            [0.81, 0.31, 0.86, 0.41, 0.71],
            [0.79, 0.29, 0.84, 0.39, 0.69],
        ]
    )
    kappa = cohens_kappa_original_vs_permuted(scores, original_index=0)
    assert kappa == 1.0


def test_kappa_constant_opposite_is_zero():
    """Original constant=1, permuted constant=0 → p_e is 0, κ=0 by Cohen's
    formula (one rater constant collapses the expected-agreement term —
    paradoxical but correct; the metric has no signal when one rater is
    deterministic)."""
    scores = np.array(
        [
            [0.80, 0.90, 0.85, 0.78, 0.95],  # all 1
            [0.10, 0.05, 0.20, 0.08, 0.15],  # all 0
            [0.12, 0.04, 0.18, 0.09, 0.13],  # all 0
        ]
    )
    kappa = cohens_kappa_original_vs_permuted(scores, original_index=0)
    assert kappa == 0.0


def test_kappa_anticorrelated_is_minus_one():
    """Mixed-distribution raters that anti-correlate perfectly → κ = -1.
    A: 1,0,1,0,1,0; B: 0,1,0,1,0,1; p_o=0, p_e=0.5, κ=-1."""
    scores = np.array(
        [
            [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],  # original: 1,0,1,0,1,0
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],  # permuted: 0,1,0,1,0,1
        ]
    )
    kappa = cohens_kappa_original_vs_permuted(scores, original_index=0)
    assert kappa == -1.0


def test_kappa_partial_disagreement_is_bounded():
    """Mixed predictions land strictly inside (-1, 1)."""
    rng = np.random.default_rng(0)
    scores = rng.random((4, 20))
    kappa = cohens_kappa_original_vs_permuted(scores, original_index=0)
    assert kappa is not None
    assert -1.0 <= kappa <= 1.0


def test_kappa_single_variant_returns_none():
    """No permuted variants → metric undefined → None."""
    scores = np.array([[0.5, 0.6, 0.7]])
    assert cohens_kappa_original_vs_permuted(scores, original_index=0) is None


def test_kappa_all_nan_returns_none():
    scores = np.full((3, 4), np.nan)
    assert cohens_kappa_original_vs_permuted(scores, original_index=0) is None


def test_kappa_partial_nan_drops_unpaired_cells():
    """NaN in original or permuted drops the pair from both raters."""
    # Construct so the surviving pairs are agreement-only → κ should be 1.0
    # if NaN drop is honest; if NaN silently counted as 0, the answer drifts.
    scores = np.array(
        [
            [0.80, 0.30, 0.85, np.nan],  # original: 1,0,1,DROP
            [0.81, 0.29, 0.86, 0.05],   # permuted: 1,0,1,DROP (cell 3 drops)
        ]
    )
    assert cohens_kappa_original_vs_permuted(scores, original_index=0) == 1.0


def test_kappa_threshold_validation():
    scores = np.array([[0.5, 0.6], [0.4, 0.7]])
    with pytest.raises(ValueError, match="threshold"):
        cohens_kappa_original_vs_permuted(scores, original_index=0, threshold=2.0)


def test_kappa_original_index_validation():
    scores = np.array([[0.5, 0.6], [0.4, 0.7]])
    with pytest.raises(ValueError, match="original_index"):
        cohens_kappa_original_vs_permuted(scores, original_index=5)


def test_kappa_2d_required():
    with pytest.raises(ValueError, match="2D"):
        cohens_kappa_original_vs_permuted(
            np.array([0.5, 0.6]), original_index=0
        )


# ---------------------------------------------------------------------------
# balanced_accuracy_drop
# ---------------------------------------------------------------------------


def test_ba_drop_perfect_original_zero_permuted():
    """Original gets every prediction right; permuted gets every one wrong →
    BA(orig) = 1.0, BA(perm) = 0.0, drop = 1.0."""
    gt = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array(
        [
            [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],  # original: 1,0,1,0,1,0 — perfect
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],  # permuted: 0,1,0,1,0,1 — inverse
        ]
    )
    result = balanced_accuracy_drop(scores, gt, original_index=0)
    assert result.degenerate_flag is False
    assert result.original == 1.0
    assert result.mean_permuted == 0.0
    assert result.drop == 1.0


def test_ba_drop_negative_when_permuted_better():
    """If permuted variant accidentally outperforms original, drop is negative.
    The spec doesn't clamp — signed drop is the auditable signal."""
    gt = np.array([1, 0, 1, 0])
    scores = np.array(
        [
            [0.1, 0.9, 0.1, 0.9],  # original: 0,1,0,1 — perfectly wrong → BA=0
            [0.9, 0.1, 0.9, 0.1],  # permuted: 1,0,1,0 — perfect → BA=1
        ]
    )
    result = balanced_accuracy_drop(scores, gt, original_index=0)
    assert result.original == 0.0
    assert result.mean_permuted == 1.0
    assert result.drop == -1.0


def test_ba_drop_constant_ground_truth_is_degenerate():
    gt = np.array([1, 1, 1, 1])
    scores = np.array(
        [
            [0.9, 0.8, 0.7, 0.6],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    result = balanced_accuracy_drop(scores, gt, original_index=0)
    assert result.degenerate_flag is True
    assert result.original is None
    assert result.drop is None


def test_ba_drop_single_variant_is_degenerate():
    gt = np.array([1, 0, 1])
    scores = np.array([[0.9, 0.1, 0.9]])
    result = balanced_accuracy_drop(scores, gt, original_index=0)
    assert result.degenerate_flag is True
    assert result.original is None


def test_ba_drop_nan_rows_skipped_in_mean():
    """A permuted variant with no valid cells is skipped from the mean."""
    gt = np.array([1, 0, 1, 0])
    scores = np.array(
        [
            [0.9, 0.1, 0.9, 0.1],          # original: perfect → BA=1
            [0.1, 0.9, 0.1, 0.9],          # perm A: inverse → BA=0
            [np.nan, np.nan, np.nan, np.nan],  # perm B: all NaN → SKIP
        ]
    )
    result = balanced_accuracy_drop(scores, gt, original_index=0)
    assert result.original == 1.0
    # mean over [perm A=0.0] only — perm B doesn't drag it down.
    assert result.mean_permuted == 0.0
    assert result.drop == 1.0


def test_ba_drop_returns_named_result():
    gt = np.array([1, 0])
    scores = np.array([[0.9, 0.1], [0.8, 0.2]])
    result = balanced_accuracy_drop(scores, gt, original_index=0)
    assert isinstance(result, BalancedAccuracyDropResult)


def test_ba_drop_length_mismatch_raises():
    gt = np.array([1, 0])
    scores = np.array([[0.9, 0.1, 0.8], [0.8, 0.2, 0.7]])
    with pytest.raises(ValueError, match="lengths must match"):
        balanced_accuracy_drop(scores, gt, original_index=0)


def test_ba_drop_non_binary_gt_raises():
    gt = np.array([2, 0])
    scores = np.array([[0.9, 0.1], [0.8, 0.2]])
    with pytest.raises(ValueError, match="ground_truth_binary"):
        balanced_accuracy_drop(scores, gt, original_index=0)
