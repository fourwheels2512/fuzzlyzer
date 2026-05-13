"""§9 metric #1: variance_across_orderings + cluster-bootstrap CI.

Point estimate: ``np.mean(np.var(scores, axis=0))`` — average per-eval
variance across the variant axis.  Higher = topology is more
order-sensitive (BLUEPRINT §9: "the v4 statistical fix that actually
matters").  The CI helper wraps :func:`fuzzeval.stats.percentile_cluster_bootstrap`
with the canonical statistic closure so the inner-resample is paired
across evals (preserves within-row correlation structure across the
variant axis).

Axis convention is fixed: ``matrix[i, j]`` is the score of variant ``i``
on eval ``j``.  Inner = variant (axis 0), outer = eval (axis 1) — the
defaults of :func:`percentile_cluster_bootstrap`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fuzzeval.stats import BootstrapResult, percentile_cluster_bootstrap

if TYPE_CHECKING:
    import numpy as np


def variance_across_orderings(matrix: "np.ndarray") -> float:
    """Point estimate of §9 metric #1.

    Mirrors the closure pinned in the §17a docstring of
    :func:`percentile_cluster_bootstrap` so a caller computing the
    headline number outside the bootstrap path stays byte-identical
    with the bootstrap's internal point estimate.

    Args:
        matrix: ``(n_variants, n_evals)`` numeric array. NaN / inf
            entries propagate to the return value as ``nan`` — the
            loader is the right place to surface missing cells.

    Returns:
        Scalar float. Always finite for finite input. For a single-
        variant matrix this is ``0.0`` (no variation across the
        singleton axis).

    Raises:
        ImportError: numpy missing (install the ``[stats]`` extra).
        ValueError: matrix is not 2D.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.variance_across_orderings requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(
            f"matrix must be 2D (n_variants, n_evals); got ndim={arr.ndim}"
        )
    # ``np.var`` with ``axis=0`` on a 2D array returns a 1D array of
    # per-eval variances; ``np.mean`` over that is the §9 headline.
    return float(np.mean(np.var(arr, axis=0)))


def variance_across_orderings_ci(
    matrix: "np.ndarray",
    *,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Cluster-bootstrap CI for §9 metric #1.

    Thin wrapper that pins the statistic closure so callers don't have
    to remember the axis-0 convention. Returns the
    :class:`fuzzeval.stats.BootstrapResult` verbatim — including the
    degenerate-path behavior (empty / non-finite / all-equal → ``ci_lower
    = ci_upper = None``, ``degenerate_flag=True``).

    Args:
        matrix: ``(n_variants, n_evals)`` numeric array.
        n_resamples: Defaults to 5,000 per §9.
        conf: Defaults to 0.95.
        seed: RNG seed for reproducibility.

    Returns:
        :class:`BootstrapResult` with ``method="percentile"``.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.variance_across_orderings_ci requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    def _voo(m: "np.ndarray") -> float:
        return float(np.mean(np.var(m, axis=0)))

    return percentile_cluster_bootstrap(
        matrix,
        _voo,
        n_resamples=n_resamples,
        conf=conf,
        outer_axis=1,
        inner_axis=0,
        seed=seed,
    )
