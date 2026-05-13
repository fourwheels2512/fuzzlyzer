"""Step 9: §9 metric #1 (variance_across_orderings) — point + CI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzeval.stats import BootstrapResult
from topofuzz.metrics import (
    variance_across_orderings,
    variance_across_orderings_ci,
)


def test_voo_point_estimate_matches_canonical_formula():
    rng = np.random.default_rng(0)
    matrix = rng.random((5, 8))
    expected = float(np.mean(np.var(matrix, axis=0)))
    assert math.isclose(
        variance_across_orderings(matrix), expected, rel_tol=1e-12
    )


def test_voo_zero_on_single_variant():
    # No variation across the variant axis → 0.
    matrix = np.array([[0.1, 0.2, 0.3]])
    assert variance_across_orderings(matrix) == 0.0


def test_voo_zero_on_constant_matrix():
    matrix = np.full((4, 3), 0.5)
    assert variance_across_orderings(matrix) == 0.0


def test_voo_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        variance_across_orderings(np.array([1.0, 2.0, 3.0]))


def test_voo_ci_round_trips_through_cluster_bootstrap():
    rng = np.random.default_rng(0)
    matrix = rng.random((5, 8))
    r = variance_across_orderings_ci(matrix, n_resamples=300, seed=42)
    assert isinstance(r, BootstrapResult)
    assert r.method == "percentile"
    assert r.n_resamples == 300
    assert not r.degenerate_flag
    assert r.ci_lower is not None and r.ci_upper is not None
    # Point estimate equals the bare variance_across_orderings.
    assert math.isclose(
        r.point_estimate, variance_across_orderings(matrix), rel_tol=1e-12
    )
    assert r.ci_lower <= r.point_estimate <= r.ci_upper


def test_voo_ci_degenerate_on_constant_matrix():
    matrix = np.full((4, 6), 0.5)
    r = variance_across_orderings_ci(matrix, n_resamples=200, seed=0)
    assert r.degenerate_flag
    assert r.ci_lower is None and r.ci_upper is None


def test_voo_ci_determinism_with_seed():
    rng = np.random.default_rng(0)
    matrix = rng.random((4, 6))
    r1 = variance_across_orderings_ci(matrix, n_resamples=200, seed=7)
    r2 = variance_across_orderings_ci(matrix, n_resamples=200, seed=7)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper
