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


def test_cluster_bootstrap_raises_until_step_8():
    matrix = np.zeros((4, 5))
    with pytest.raises(NotImplementedError, match="step 8"):
        percentile_cluster_bootstrap(matrix, np.mean)
