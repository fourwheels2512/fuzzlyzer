import math

import pytest

from fuzzeval.stats import (
    BootstrapResult,
    percentile_bootstrap,
    percentile_cluster_bootstrap,
    wilson_ci,
)

np = pytest.importorskip("numpy")


def test_wilson_ci_50_of_100():
    """Reference: Wilson (1927); 50/100 at 95% CI ≈ (0.4038, 0.5962)."""
    lo, hi = wilson_ci(50, 100, 0.95)
    assert math.isclose(lo, 0.4038, abs_tol=0.001)
    assert math.isclose(hi, 0.5962, abs_tol=0.001)


def test_wilson_ci_zero_successes():
    lo, hi = wilson_ci(0, 100, 0.95)
    assert lo == 0.0
    assert 0.0 < hi < 0.05


def test_wilson_ci_full_successes():
    lo, hi = wilson_ci(100, 100, 0.95)
    assert 0.95 < lo < 1.0
    assert hi == 1.0


def test_wilson_ci_99_wider_than_95():
    lo95, hi95 = wilson_ci(50, 100, 0.95)
    lo99, hi99 = wilson_ci(50, 100, 0.99)
    assert (hi99 - lo99) > (hi95 - lo95)


def test_wilson_ci_n_equals_1_boundary():
    """Smallest valid n; floats must not blow up at the boundary."""
    lo0, hi0 = wilson_ci(0, 1, 0.95)
    assert lo0 == 0.0
    assert 0.0 < hi0 < 1.0
    lo1, hi1 = wilson_ci(1, 1, 0.95)
    assert 0.0 < lo1 < 1.0
    assert hi1 == 1.0


def test_wilson_ci_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_ci(50, 0, 0.95)
    with pytest.raises(ValueError):
        wilson_ci(101, 100, 0.95)
    with pytest.raises(ValueError):
        wilson_ci(50, 100, 0.0)
    with pytest.raises(ValueError):
        wilson_ci(50, 100, 1.0)


def test_percentile_bootstrap_recovers_population_mean():
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=10.0, scale=2.0, size=200)
    result = percentile_bootstrap(sample, np.mean, n_resamples=2000, seed=1)
    assert isinstance(result, BootstrapResult)
    assert math.isclose(result.point_estimate, float(np.mean(sample)), rel_tol=1e-9)
    assert result.ci_lower is not None and result.ci_upper is not None
    assert result.ci_lower < 10.0 < result.ci_upper
    assert not result.degenerate_flag


def test_percentile_bootstrap_degenerate_all_equal():
    sample = np.array([0.5, 0.5, 0.5, 0.5])
    result = percentile_bootstrap(sample, np.mean, n_resamples=1000)
    assert result.degenerate_flag
    assert result.ci_lower is None
    assert result.ci_upper is None
    assert math.isclose(result.point_estimate, 0.5)


def test_percentile_bootstrap_empty():
    result = percentile_bootstrap(np.array([]), np.mean, n_resamples=100)
    assert result.degenerate_flag
    assert math.isnan(result.point_estimate)
    assert result.n_resamples == 0


def test_percentile_bootstrap_ci_narrows_with_n():
    rng = np.random.default_rng(7)
    small = rng.normal(0.0, 1.0, size=20)
    large = rng.normal(0.0, 1.0, size=200)
    small_r = percentile_bootstrap(small, np.mean, n_resamples=1000, seed=2)
    large_r = percentile_bootstrap(large, np.mean, n_resamples=1000, seed=2)
    assert (large_r.ci_upper - large_r.ci_lower) < (small_r.ci_upper - small_r.ci_lower)


def test_percentile_bootstrap_is_deterministic_with_seed():
    sample = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    r1 = percentile_bootstrap(sample, np.mean, n_resamples=500, seed=99)
    r2 = percentile_bootstrap(sample, np.mean, n_resamples=500, seed=99)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


# ---- percentile_cluster_bootstrap (BLUEPRINT §9 metric #1 / §17a) ----


def _voo(m):
    """Canonical variance_across_orderings statistic. Axis 0 = variants."""
    return float(np.mean(np.var(m, axis=0)))


def test_cluster_bootstrap_basic_contract():
    rng = np.random.default_rng(0)
    matrix = rng.random((5, 10))
    result = percentile_cluster_bootstrap(matrix, _voo, n_resamples=200, seed=1)
    assert isinstance(result, BootstrapResult)
    assert result.method == "percentile"
    assert result.n_resamples == 200
    assert not result.degenerate_flag
    assert result.ci_lower is not None and result.ci_upper is not None
    assert math.isclose(result.point_estimate, _voo(matrix), rel_tol=1e-12)
    assert result.ci_lower <= result.point_estimate <= result.ci_upper


def test_cluster_bootstrap_degenerate_all_equal():
    matrix = np.full((4, 8), 0.5)
    result = percentile_cluster_bootstrap(matrix, _voo, n_resamples=100, seed=0)
    assert result.degenerate_flag
    assert result.ci_lower is None
    assert result.ci_upper is None
    # Degenerate path returns the constant value itself (statistic is
    # never called — niche statistics may not tolerate constant input).
    assert math.isclose(result.point_estimate, 0.5)
    assert result.n_resamples == 0
    assert result.method == "percentile"


def test_cluster_bootstrap_empty_along_either_axis_is_degenerate():
    r_no_evals = percentile_cluster_bootstrap(
        np.zeros((4, 0)), _voo, n_resamples=10
    )
    r_no_variants = percentile_cluster_bootstrap(
        np.zeros((0, 4)), _voo, n_resamples=10
    )
    for r in (r_no_evals, r_no_variants):
        assert r.degenerate_flag
        assert r.ci_lower is None and r.ci_upper is None
        assert math.isnan(r.point_estimate)
        assert r.n_resamples == 0
        assert r.method == "percentile"


def test_cluster_bootstrap_nonfinite_is_degenerate():
    """NaN/inf entries route to the degenerate path so the §9 report
    banner renders `None` (rendered as "—") instead of literal "nan"."""
    nan_matrix = np.full((3, 4), np.nan)
    inf_matrix = np.array([[1.0, 2.0, np.inf], [0.1, 0.2, 0.3], [0.5, 0.4, 0.6]])
    for matrix in (nan_matrix, inf_matrix):
        result = percentile_cluster_bootstrap(
            matrix, _voo, n_resamples=50, seed=0
        )
        assert result.degenerate_flag
        assert result.ci_lower is None and result.ci_upper is None
        assert math.isnan(result.point_estimate)
        assert result.n_resamples == 0
        assert result.method == "percentile"


def test_cluster_bootstrap_does_not_call_statistic_on_degenerate_input():
    """Niche statistics may raise on empty / constant / non-finite input.
    The degenerate short-circuit must not invoke `statistic` on them."""
    def hostile(m):
        raise AssertionError(f"statistic should not be called on degenerate input; got shape {m.shape}")
    for matrix in (
        np.zeros((4, 0)),
        np.zeros((0, 4)),
        np.full((3, 5), 0.42),
        np.full((3, 5), np.nan),
    ):
        result = percentile_cluster_bootstrap(matrix, hostile, n_resamples=10)
        assert result.degenerate_flag


def test_cluster_bootstrap_deterministic_with_seed():
    rng = np.random.default_rng(0)
    matrix = rng.random((6, 12))
    r1 = percentile_cluster_bootstrap(matrix, _voo, n_resamples=300, seed=42)
    r2 = percentile_cluster_bootstrap(matrix, _voo, n_resamples=300, seed=42)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper
    assert r1.point_estimate == r2.point_estimate


def test_cluster_bootstrap_idempotent_under_variant_permutation():
    """Property (weak): for a permutation-invariant statistic, the
    bootstrap is invariant under input-cluster permutation.

    - **Point estimate** is exact under any row permutation (the statistic
      itself is symmetric over the inner axis).
    - **CI bounds** are invariant in *distribution* (clusters are
      interchangeable units in the bootstrap). With a fixed seed and
      finite ``n_resamples`` the realised bounds depend on cluster
      identity, so we verify agreement within Monte Carlo tolerance.

    This test is intentionally a weak property check — a broken
    *unpaired* resampler could still pass it, because the canonical
    ``_voo`` statistic is symmetric in both axes. The pairing semantics
    are witnessed separately by
    ``test_cluster_bootstrap_inner_resampling_is_paired``."""
    rng = np.random.default_rng(123)
    matrix = rng.random((7, 15))
    perm = np.array([3, 0, 6, 1, 4, 2, 5])
    r_orig = percentile_cluster_bootstrap(matrix, _voo, n_resamples=1000, seed=2025)
    r_perm = percentile_cluster_bootstrap(matrix[perm, :], _voo, n_resamples=1000, seed=2025)
    assert math.isclose(r_orig.point_estimate, r_perm.point_estimate, rel_tol=1e-12)
    ci_width = r_orig.ci_upper - r_orig.ci_lower
    # MC tolerance: ~5% at n_resamples=1000; cap at 15% for safety margin.
    assert abs(r_orig.ci_lower - r_perm.ci_lower) < 0.15 * ci_width
    assert abs(r_orig.ci_upper - r_perm.ci_upper) < 0.15 * ci_width


def test_cluster_bootstrap_inner_resampling_is_paired():
    """Property (strong): the inner resampling is *paired* across outer
    clusters — one inner-index sample is applied to every chosen outer
    column on each draw. A broken implementation that draws fresh
    inner indices per outer column (unpaired) would never produce a
    bootstrap variance of 0 on this matrix, because no two rows are
    identical and no two columns are identical.

    Constructed witness: rows 0 and 1 are equal; row 2 differs. With a
    paired resampler, any draw where ``inner_idx[0] == inner_idx[1]``
    (~1/3 of draws on a 3-row matrix) makes both reference rows
    collapse to the same row → ``(m[0] - m[1])`` is zero everywhere →
    statistic = 0. So the percentile CI's lower bound MUST be 0.
    An unpaired resampler picks independent rows per column and almost
    never lands on a zero, so its CI lower bound stays well above 0."""
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )

    def row_diff_sq(m):
        # mean squared difference between row 0 and row 1 of the resample
        return float(np.mean((m[0, :] - m[1, :]) ** 2))

    r = percentile_cluster_bootstrap(matrix, row_diff_sq, n_resamples=500, seed=0)
    # With paired inner resampling, ~1/3 of resamples collapse to identical
    # row 0/row 1 → row_diff_sq = 0 → ci_lower must be 0.
    assert r.ci_lower == 0.0
    # And in some draws row 2 lands opposite rows 0/1, producing a large
    # diff. With 5-element rows of value 5/10 vs 10/10, max diff² is 81.
    assert r.ci_upper > 10.0


def test_cluster_bootstrap_resampled_matrix_preserves_shape():
    """Property (length invariance): every resampled matrix passed to
    `statistic` has the same shape as the input — the bootstrap never
    shrinks or grows the cluster counts on a single draw."""
    rng = np.random.default_rng(0)
    matrix = rng.random((4, 11))
    shapes: list[tuple[int, int]] = []

    def tracking_stat(m):
        shapes.append(m.shape)
        return float(np.mean(m))

    percentile_cluster_bootstrap(matrix, tracking_stat, n_resamples=50, seed=1)
    # 1 call for the point estimate + 50 resample calls, all (4, 11).
    assert len(shapes) == 51
    assert all(s == (4, 11) for s in shapes), f"got distinct shapes: {set(shapes)}"


def test_cluster_bootstrap_ci_width_decreases_with_more_evals():
    """Property (CI width monotonic in n_outer): drawing from the same
    homogeneous distribution, a matrix with more eval clusters yields a
    tighter percentile CI for variance_across_orderings."""
    rng = np.random.default_rng(7)
    small = rng.normal(0.5, 0.1, size=(6, 12))
    big = rng.normal(0.5, 0.1, size=(6, 120))
    r_small = percentile_cluster_bootstrap(small, _voo, n_resamples=600, seed=2)
    r_big = percentile_cluster_bootstrap(big, _voo, n_resamples=600, seed=2)
    width_small = r_small.ci_upper - r_small.ci_lower
    width_big = r_big.ci_upper - r_big.ci_lower
    assert width_big < width_small


def test_cluster_bootstrap_axes_swap_equivalent():
    """Passing matrix.T with axes swapped is equivalent to canonical form."""
    rng = np.random.default_rng(0)
    matrix = rng.random((5, 8))
    r_canon = percentile_cluster_bootstrap(matrix, _voo, n_resamples=200, seed=11)
    r_swap = percentile_cluster_bootstrap(
        matrix.T, _voo, n_resamples=200, seed=11,
        outer_axis=0, inner_axis=1,
    )
    assert math.isclose(r_canon.point_estimate, r_swap.point_estimate, rel_tol=1e-12)
    assert math.isclose(r_canon.ci_lower, r_swap.ci_lower, rel_tol=1e-12)
    assert math.isclose(r_canon.ci_upper, r_swap.ci_upper, rel_tol=1e-12)


def test_cluster_bootstrap_accepts_numpy_integer_axes():
    """numpy integer scalars (np.int64 etc.) should be accepted as axes —
    they don't inherit from `int` on modern Python+numpy, but they are
    `numbers.Integral` and the natural type after array-math pipelines."""
    matrix = np.random.default_rng(0).random((4, 6))
    r = percentile_cluster_bootstrap(
        matrix, _voo, n_resamples=100, seed=0,
        outer_axis=np.int64(1), inner_axis=np.int64(0),
    )
    assert not r.degenerate_flag


def test_cluster_bootstrap_rejects_bad_inputs():
    matrix = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="2D"):
        percentile_cluster_bootstrap(np.zeros(5), np.mean)
    with pytest.raises(ValueError, match="2D"):
        percentile_cluster_bootstrap(np.zeros((3, 4, 5)), np.mean)
    with pytest.raises(ValueError, match="axes"):
        percentile_cluster_bootstrap(matrix, np.mean, outer_axis=0, inner_axis=0)
    with pytest.raises(ValueError, match="axes"):
        percentile_cluster_bootstrap(matrix, np.mean, outer_axis=2, inner_axis=0)
    with pytest.raises(ValueError, match="axes"):
        percentile_cluster_bootstrap(matrix, np.mean, outer_axis=-1, inner_axis=0)
    # bool / float axes that hash-equal to {0, 1} would silently bypass
    # a plain `{outer, inner} != {0, 1}` check; verify they're rejected.
    with pytest.raises(ValueError, match="axes"):
        percentile_cluster_bootstrap(matrix, np.mean, outer_axis=True, inner_axis=False)
    with pytest.raises(ValueError, match="axes"):
        percentile_cluster_bootstrap(matrix, np.mean, outer_axis=1.0, inner_axis=0.0)
    with pytest.raises(ValueError, match="conf"):
        percentile_cluster_bootstrap(matrix, np.mean, conf=0.0)
    with pytest.raises(ValueError, match="conf"):
        percentile_cluster_bootstrap(matrix, np.mean, conf=1.0)
    with pytest.raises(ValueError, match="n_resamples"):
        percentile_cluster_bootstrap(matrix, np.mean, n_resamples=0)
    with pytest.raises(ValueError, match="n_resamples"):
        percentile_cluster_bootstrap(matrix, np.mean, n_resamples=-1)
    # Off-contract dtypes raise a clean ValueError instead of leaking a
    # TypeError from np.isfinite.
    obj_matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=object)
    with pytest.raises(ValueError, match="dtype"):
        percentile_cluster_bootstrap(obj_matrix, np.mean)


def test_cluster_bootstrap_wider_ci_at_99_than_95():
    rng = np.random.default_rng(0)
    matrix = rng.random((5, 30))
    r95 = percentile_cluster_bootstrap(matrix, _voo, n_resamples=500, seed=3)
    r99 = percentile_cluster_bootstrap(matrix, _voo, n_resamples=500, seed=3, conf=0.99)
    assert (r99.ci_upper - r99.ci_lower) > (r95.ci_upper - r95.ci_lower)
