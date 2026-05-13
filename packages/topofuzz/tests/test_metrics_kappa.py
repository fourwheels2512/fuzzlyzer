"""Step 9: §9 metric #3 (κ + balanced accuracy + collapse_flag)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from topofuzz.metrics import (
    KappaResult,
    balanced_accuracy,
    cohens_kappa,
    cohens_kappa_vs_truth,
)
from topofuzz.metrics.kappa import (
    COLLAPSE_BA_UPPER_THRESHOLD,
    COLLAPSE_KAPPA_UPPER_THRESHOLD,
)


# ---------------------------------------------------------------------------
# cohens_kappa — analytic test vectors
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement():
    a = np.array([1, 0, 1, 0, 1])
    assert cohens_kappa(a, a) == 1.0


def test_kappa_perfect_disagreement():
    a = np.array([1, 0, 1, 0])
    b = 1 - a
    assert cohens_kappa(a, b) == -1.0


def test_kappa_independent_raters_near_zero():
    """Two raters whose marginals match exactly but whose decisions
    are uncorrelated produce κ ≈ 0."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=1000)
    b = rng.integers(0, 2, size=1000)
    k = cohens_kappa(a, b)
    assert abs(k) < 0.1


def test_kappa_constant_raters_agree_returns_1():
    a = np.ones(10, dtype=int)
    assert cohens_kappa(a, a) == 1.0


def test_kappa_constant_raters_disagree_returns_0():
    """Both raters predict a constant but the constants differ: no
    signal to scale, κ defined as 0 by docstring."""
    a = np.ones(10, dtype=int)
    b = np.zeros(10, dtype=int)
    assert cohens_kappa(a, b) == 0.0


def test_kappa_rejects_non_binary():
    with pytest.raises(ValueError, match="only 0/1"):
        cohens_kappa(np.array([0, 1, 2]), np.array([0, 1, 0]))


def test_kappa_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share shape"):
        cohens_kappa(np.ones(5), np.ones(6))


# ---------------------------------------------------------------------------
# balanced_accuracy — analytic test vectors
# ---------------------------------------------------------------------------


def test_balanced_accuracy_perfect_classifier():
    y_true = np.array([1, 0, 1, 0, 1])
    assert balanced_accuracy(y_true, y_true) == 1.0


def test_balanced_accuracy_inverse_classifier():
    y_true = np.array([1, 0, 1, 0])
    y_pred = 1 - y_true
    assert balanced_accuracy(y_true, y_pred) == 0.0


def test_balanced_accuracy_random_is_0_5():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    # TPR = 2/3 (predicted 1 on indices 0 and 4 out of true=1 set {0, 2, 4})
    # TNR = 2/3 (predicted 0 on indices 3 and 5 out of true=0 set {1, 3, 5})
    assert math.isclose(balanced_accuracy(y_true, y_pred), 2 / 3, rel_tol=1e-12)


def test_balanced_accuracy_missing_class_falls_back():
    """If one class is absent from y_true, the missing rate is 0.5 so
    the result remains interpretable (matches sklearn fallback)."""
    y_true = np.ones(5, dtype=int)
    y_pred = np.ones(5, dtype=int)
    # No 0s in y_true → TNR=0.5; TPR=1.0; balanced = 0.75.
    assert balanced_accuracy(y_true, y_pred) == 0.75


# ---------------------------------------------------------------------------
# cohens_kappa_vs_truth — bootstrap + collapse_flag boundaries
# ---------------------------------------------------------------------------


def test_kvt_happy_path_signal_present():
    """Predictions that match truth modulo noise → κ_point > 0 and
    balanced accuracy > 0.5; collapse_flag False."""
    rng = np.random.default_rng(0)
    n_evals = 20
    gt = np.array([1, 0] * (n_evals // 2))
    # 4 variants whose binarised predictions track gt at ~85% accuracy.
    scores = np.empty((4, n_evals), dtype=float)
    for v in range(4):
        noise = rng.random(n_evals) < 0.15
        preds = gt ^ noise  # XOR: flip ~15% of labels
        # Map binary 0/1 → continuous 0.2/0.8 so threshold=0.5 binarises
        # back cleanly.
        scores[v] = np.where(preds == 1, 0.8, 0.2)

    r = cohens_kappa_vs_truth(scores, gt, n_resamples=300, seed=1)
    assert isinstance(r, KappaResult)
    assert not r.degenerate_flag
    assert r.kappa_point is not None and r.kappa_point > 0.5
    assert r.balanced_accuracy_point is not None
    assert r.balanced_accuracy_point > 0.7
    assert r.kappa_ci_lower is not None and r.kappa_ci_upper is not None
    assert r.kappa_ci_lower <= r.kappa_point <= r.kappa_ci_upper
    # Not collapsed.
    assert not r.collapse_flag
    assert r.n_resamples == 300
    # ``method`` mirrors :class:`fuzzeval.stats.BootstrapResult.method`.
    assert r.method == "percentile"


def test_kvt_collapse_flag_when_predictions_constant():
    """All variants predict the same constant → κ degenerate (constant
    rater) → collapse_flag NOT set (we make no claim under degenerate)."""
    n_evals = 20
    gt = np.array([1, 0] * (n_evals // 2))
    scores = np.full((3, n_evals), 0.9, dtype=float)  # all-approve
    r = cohens_kappa_vs_truth(scores, gt, n_resamples=200, seed=0)
    assert r.degenerate_flag
    assert not r.collapse_flag


def test_kvt_collapse_flag_when_signal_is_chance():
    """Deterministic chance-level construction: two variants that each
    predict a single class for every eval. Within each variant the
    predictions are constant → per-variant κ is 0 (one rater is
    constant) and per-variant balanced-accuracy is 0.5 (one TPR or
    TNR is 0.5 by the missing-class fallback). The bootstrap variance
    is zero across both axes → tight CIs at κ=0 / ba=0.5 → ``collapse_flag=True``.

    Using a deterministic construction (instead of random scores) makes
    the assertion non-flaky — small-sample random bootstraps can
    straddle the 0.10 / 0.55 thresholds. Random-noise behaviour is
    covered by ``test_kvt_happy_path_signal_present`` (no collapse)
    and the degenerate tests."""
    n_evals = 30
    # Balanced ground truth.
    gt = np.array(([1, 0] * (n_evals // 2)))
    # Two variants: row 0 always-approve (score=0.9), row 1
    # always-deny (score=0.1). Predictions are constant per variant.
    scores = np.array(
        [
            [0.9] * n_evals,
            [0.1] * n_evals,
        ],
        dtype=float,
    )
    r = cohens_kappa_vs_truth(scores, gt, n_resamples=300, seed=2)
    assert not r.degenerate_flag
    assert r.kappa_ci_upper is not None
    assert r.balanced_accuracy_ci_upper is not None
    assert r.kappa_ci_upper < COLLAPSE_KAPPA_UPPER_THRESHOLD
    assert r.balanced_accuracy_ci_upper < COLLAPSE_BA_UPPER_THRESHOLD
    assert r.collapse_flag


def test_kvt_degenerate_single_variant():
    gt = np.array([1, 0, 1, 0])
    scores = np.array([[0.8, 0.2, 0.8, 0.2]])  # 1 variant
    r = cohens_kappa_vs_truth(scores, gt, n_resamples=100, seed=0)
    assert r.degenerate_flag
    assert r.kappa_ci_lower is None
    # Point estimates can still be computed on the original matrix
    # (the docstring contract — useful even when CI is degenerate).
    assert r.kappa_point is not None


def test_kvt_degenerate_single_eval():
    gt = np.array([1])
    scores = np.array([[0.8], [0.2], [0.8]])
    r = cohens_kappa_vs_truth(scores, gt, n_resamples=100, seed=0)
    assert r.degenerate_flag


def test_kvt_degenerate_constant_ground_truth():
    gt = np.ones(10, dtype=int)
    rng = np.random.default_rng(0)
    scores = rng.random((3, 10))
    r = cohens_kappa_vs_truth(scores, gt, n_resamples=100, seed=0)
    assert r.degenerate_flag


def test_kvt_rejects_threshold_out_of_range():
    gt = np.array([1, 0, 1, 0])
    scores = np.zeros((2, 4))
    with pytest.raises(ValueError, match="threshold must be in"):
        cohens_kappa_vs_truth(scores, gt, threshold=1.5)


def test_kvt_rejects_n_resamples_zero():
    gt = np.array([1, 0, 1, 0])
    scores = np.zeros((2, 4))
    with pytest.raises(ValueError, match="n_resamples must be positive"):
        cohens_kappa_vs_truth(scores, gt, n_resamples=0)


def test_kvt_rejects_non_binary_ground_truth():
    gt = np.array([0, 1, 2])
    scores = np.zeros((2, 3))
    with pytest.raises(ValueError, match="only 0/1"):
        cohens_kappa_vs_truth(scores, gt)


def test_kvt_determinism_with_seed():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 2, size=10)
    scores = rng.random((3, 10))
    r1 = cohens_kappa_vs_truth(scores, gt, n_resamples=200, seed=99)
    r2 = cohens_kappa_vs_truth(scores, gt, n_resamples=200, seed=99)
    assert r1.kappa_ci_lower == r2.kappa_ci_lower
    assert r1.kappa_ci_upper == r2.kappa_ci_upper
    assert r1.balanced_accuracy_ci_lower == r2.balanced_accuracy_ci_lower
