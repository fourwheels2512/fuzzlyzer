# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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

from topofuzz.metrics import (
    VerdictFlipResult,
    compute_verdict_flip_rate,
    wilson_score_interval,
)


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


# ---------------------------------------------------------------------------
# F1: numerator / denominator + Wilson score interval
# ---------------------------------------------------------------------------


def test_numerator_and_denominator_populated_on_clean_run():
    """A non-degenerate run exposes integer (count, total) for the Wilson
    CI consumer. 5 flips across 10 paired cells in the balanced fixture."""
    scores = np.array(
        [
            [0.80, 0.90, 0.85, 0.95, 0.78],  # original: 1,1,1,1,1
            [0.10, 0.05, 0.20, 0.15, 0.08],  # flipper:  0,0,0,0,0  (5 flips)
            [0.82, 0.91, 0.87, 0.96, 0.79],  # no flips: 1,1,1,1,1  (0 flips)
        ]
    )
    result = compute_verdict_flip_rate(scores, original_index=0, seed=42)
    assert result.numerator == 5
    assert result.denominator == 10
    assert isinstance(result.numerator, int)
    assert isinstance(result.denominator, int)


def test_numerator_zero_when_no_flips_but_pairs_exist():
    """All-identical fixture → 0 flips, but the paired denominator is
    still 10 (3 variants → 2 permuted rows × 5 evals)."""
    scores = np.full((3, 5), 0.72)
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.numerator == 0
    assert result.denominator == 10


def test_numerator_denominator_none_on_structurally_degenerate():
    """Single-variant fixture: no permuted rows → no paired observations
    → numerator/denominator are ``None`` (not 0)."""
    scores = np.array([[0.5, 0.5, 0.5]])  # 1 variant
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert result.numerator is None
    assert result.denominator is None


def test_numerator_drops_nan_pairs():
    """NaN pairs drop from BOTH count and total — the paired denominator
    must equal the number of valid (orig, perm) pairs only."""
    scores = np.array(
        [
            [0.80, 0.90, np.nan, 0.78],  # original valid at 3/4 evals
            [0.10, 0.92, 0.20, 0.79],    # 1 flip vs orig at eval 0; 0,0,0
        ]
    )
    result = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    # Paired cells: (e0, e1, e3) since e2 is NaN on original. perm flips
    # only at e0 (1→0). Denominator=3, numerator=1.
    assert result.numerator == 1
    assert result.denominator == 3


# ---------------------------------------------------------------------------
# F1: wilson_score_interval — known-input reference values
# ---------------------------------------------------------------------------


def test_wilson_5_of_10_known_reference():
    """Wilson 95% CI for k=5, n=10 → ~(0.23659, 0.76341).

    Reference: scipy.stats.proportion_confint(5, 10, 0.05, "wilson") =
    (0.23658800818179994, 0.7634119918182001).
    """
    lo, hi = wilson_score_interval(5, 10)
    assert lo is not None and hi is not None
    assert abs(lo - 0.236588) < 1e-5
    assert abs(hi - 0.763412) < 1e-5


def test_wilson_0_of_10_lower_bound_clamped_to_zero():
    """Wilson at p=0: lower clamps to 0; upper ≈ 0.27753."""
    lo, hi = wilson_score_interval(0, 10)
    assert lo == 0.0
    assert abs(hi - 0.277533) < 1e-4


def test_wilson_10_of_10_upper_bound_clamped_to_one():
    """Wilson at p=1: lower ≈ 0.72247; upper clamps to 1.0."""
    lo, hi = wilson_score_interval(10, 10)
    assert abs(lo - 0.722467) < 1e-4
    assert hi == 1.0


def test_wilson_1_of_100_tight_lower_bound():
    """Wilson at very low rate is much tighter than naive normal —
    n=100, k=1 → ~(0.00018, 0.05445)."""
    lo, hi = wilson_score_interval(1, 100)
    assert 0.0 < lo < 0.005
    assert 0.05 < hi < 0.06


def test_wilson_zero_trials_returns_none_pair():
    """No data → no defensible CI. Return ``(None, None)`` not (0, 1)."""
    lo, hi = wilson_score_interval(0, 0)
    assert lo is None and hi is None


def test_wilson_brackets_observed_proportion_under_random_inputs():
    """Property test: for any (k, n) with n > 0, the CI must bracket k/n.

    This is the defining property of any well-formed CI. A regression
    that swaps + and - in the formula would surface here on most inputs.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 500))
        k = int(rng.integers(0, n + 1))
        lo, hi = wilson_score_interval(k, n)
        assert lo is not None and hi is not None
        p = k / n
        assert lo <= p <= hi, f"CI [{lo}, {hi}] does not bracket {p} (k={k}, n={n})"
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_99_percent_wider_than_95_percent():
    """Conf level monotonicity: a 99% CI must be at least as wide as 95%."""
    lo95, hi95 = wilson_score_interval(5, 10, conf=0.95)
    lo99, hi99 = wilson_score_interval(5, 10, conf=0.99)
    assert lo99 <= lo95
    assert hi99 >= hi95


def test_wilson_rejects_invalid_inputs():
    """Boundary contract: out-of-range k, negative n, or conf outside
    (0, 1) raise ValueError rather than returning a silent garbage CI."""
    with pytest.raises(ValueError):
        wilson_score_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_score_interval(11, 10)  # k > n
    with pytest.raises(ValueError):
        wilson_score_interval(0, -5)
    with pytest.raises(ValueError):
        wilson_score_interval(5, 10, conf=0.0)
    with pytest.raises(ValueError):
        wilson_score_interval(5, 10, conf=1.0)
    with pytest.raises(ValueError):
        wilson_score_interval(5, 10, conf=-0.5)
    with pytest.raises(ValueError):
        wilson_score_interval(5, 10, conf=1.5)


def test_wilson_rejects_bool_successes_and_trials():
    """Audit pass-1 Fuzzer-MAJOR-3: ``True`` / ``False`` are
    ``isinstance(_, int)`` and ``0 <= True <= 10`` evaluates True, so
    a producer with ``successes=True`` would silently return a CI for
    ``k=1`` instead of erroring. Reject ``bool`` up front."""
    with pytest.raises(TypeError):
        wilson_score_interval(True, 10)
    with pytest.raises(TypeError):
        wilson_score_interval(5, True)
    with pytest.raises(TypeError):
        wilson_score_interval(False, 10)


def test_wilson_rejects_non_int_successes_and_trials():
    """Docstring says int; a float ``5.0`` slipping through is a
    contract lie even if the math accepts it."""
    with pytest.raises(TypeError):
        wilson_score_interval(5.0, 10)
    with pytest.raises(TypeError):
        wilson_score_interval(5, 10.0)
    with pytest.raises(TypeError):
        wilson_score_interval("5", 10)  # type: ignore[arg-type]


def test_wilson_rejects_bool_conf():
    """``conf=True`` (i.e. 1.0) would crash ``inv_cdf`` AND silently
    treat the type as float. Reject explicitly."""
    with pytest.raises(TypeError):
        wilson_score_interval(5, 10, conf=True)  # type: ignore[arg-type]


def test_wilson_rejects_conf_near_one():
    """Audit pass-1 RT-MAJOR-3: ``conf = 1.0 - eps`` collapses
    ``1.0 - alpha/2`` to exactly 1.0 in float, and
    ``NormalDist().inv_cdf(1.0)`` raises ``StatisticsError``. The
    tightened gate rejects this before the math runs."""
    with pytest.raises(ValueError):
        wilson_score_interval(5, 10, conf=1.0 - 1e-15)
    with pytest.raises(ValueError):
        wilson_score_interval(5, 10, conf=1e-15)


def test_wilson_accepts_safe_conf_boundaries():
    """The tightened gate must still accept conf values just inside
    the new interior ``[1e-9, 1 - 1e-9]`` band."""
    lo, hi = wilson_score_interval(5, 10, conf=1.0 - 1e-9)
    assert lo is not None and hi is not None
    lo2, hi2 = wilson_score_interval(5, 10, conf=1e-9)
    assert lo2 is not None and hi2 is not None


def test_wilson_n_equals_one_does_not_crash():
    """Smallest non-trivial sample size: must produce a valid CI, not
    div-by-zero or NaN."""
    lo, hi = wilson_score_interval(1, 1)
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= 1.0 <= 1.0  # noqa
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_deterministic():
    """Pure stdlib math — no RNG. Calls must return identical bounds."""
    a = wilson_score_interval(7, 23)
    b = wilson_score_interval(7, 23)
    assert a == b


# ---------------------------------------------------------------------------
# 2026-05-30 clear-flip gate: class_matrix path
# ---------------------------------------------------------------------------
# Codes: -1 missing, 0 deny, 1 approve, 2 conditional.


def test_class_matrix_conditional_excluded_from_flip_denominator():
    """A CONDITIONAL cell opposite a clear verdict is NOT a clear flip and
    is excluded from the denominator — tracked in conditional_pairs."""
    scores = np.full((2, 3), 0.5)  # value channel irrelevant on class path
    classes = np.array(
        [
            [1, 1, 1],   # original: approve, approve, approve
            [0, 2, 1],   # permuted: deny (flip), conditional (excluded), approve (no flip)
        ]
    )
    r = compute_verdict_flip_rate(
        scores, original_index=0, seed=0, class_matrix=classes
    )
    # Clear↔clear pairs: eval0 (approve vs deny → flip) and eval2 (approve
    # vs approve → no flip). eval1 is conditional → excluded.
    assert r.numerator == 1
    assert r.denominator == 2
    assert r.clear_pairs == 2
    assert r.conditional_pairs == 1
    assert r.point == pytest.approx(0.5)


def test_class_matrix_all_conditional_is_degenerate():
    """When the original variant is conditional on every eval there is no
    clear↔clear pair → flip rate undefined (degenerate), but the metric
    still reports zero clear pairs and the conditional-pair count."""
    scores = np.full((2, 3), 0.5)
    classes = np.array(
        [
            [2, 2, 2],   # original conditional everywhere
            [2, 1, 0],   # permuted: cond, approve, deny
        ]
    )
    r = compute_verdict_flip_rate(
        scores, original_index=0, seed=0, class_matrix=classes
    )
    # No clear↔clear pair exists → degenerate, point undefined.
    assert r.degenerate_flag is True
    assert r.point is None
    assert r.clear_pairs == 0
    # All three eval columns have the original conditional → 3 conditional
    # pairs (both not-missing, not both clear).
    assert r.conditional_pairs == 3


def test_class_matrix_clear_flip_counts_only_approve_deny():
    """Pure APPROVE↔DENY swing with no conditionals behaves like the
    numeric path: full flip rate, zero conditional pairs."""
    scores = np.full((2, 4), 0.5)
    classes = np.array(
        [
            [1, 1, 1, 1],   # original approve
            [0, 0, 0, 0],   # permuted deny → flips every eval
        ]
    )
    r = compute_verdict_flip_rate(
        scores, original_index=0, seed=0, class_matrix=classes
    )
    assert r.point == 1.0
    assert r.clear_pairs == 4
    assert r.conditional_pairs == 0


def test_class_matrix_missing_code_excluded():
    """A -1 (missing) cell drops from the denominator and is NOT counted as
    a conditional pair."""
    scores = np.full((2, 3), 0.5)
    classes = np.array(
        [
            [1, 1, 1],
            [0, -1, 1],   # eval1 missing
        ]
    )
    r = compute_verdict_flip_rate(
        scores, original_index=0, seed=0, class_matrix=classes
    )
    assert r.denominator == 2          # eval0 + eval2
    assert r.conditional_pairs == 0    # the -1 is missing, not conditional
    assert r.numerator == 1            # eval0 flips


def test_class_matrix_shape_mismatch_raises():
    scores = np.full((2, 3), 0.5)
    bad = np.zeros((2, 2), dtype=int)
    with pytest.raises(ValueError, match="shape"):
        compute_verdict_flip_rate(
            scores, original_index=0, class_matrix=bad
        )


def test_no_class_matrix_leaves_new_fields_none():
    """Legacy numeric path must not populate the 3-state fields."""
    scores = np.array([[0.8, 0.9], [0.1, 0.2]])
    r = compute_verdict_flip_rate(scores, original_index=0, seed=0)
    assert r.clear_pairs is None
    assert r.conditional_pairs is None
