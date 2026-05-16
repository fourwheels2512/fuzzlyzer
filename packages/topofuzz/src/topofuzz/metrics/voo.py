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


# ---------------------------------------------------------------------------
# F3 MAJOR (2026-05-16): jury-aware VOO over per-vendor continuous scores.
#
# Under F3 majority-vote, the score_value persisted to judge_results is
# binarised to {0.0, 1.0} per cell. Computing np.var across variants on
# that binarised matrix is Bernoulli-bounded (max variance 0.25); the §9
# thresholds were calibrated against continuous-judge variance, so the
# headline was effectively saturated under jury runs.
#
# The principled fix is to compute §9 metric #1 ON THE CONTINUOUS
# per-vendor scores (each vendor's own [0, 1] judgement, before majority
# binarisation). That preserves §9 axis semantics — variance is still
# measured ACROSS ORDERINGS, not across judges — by computing the
# standard VOO per vendor and then averaging across vendors. Expanding
# the (vendor, variant) axes into one and running the original closure
# would conflate the two sources of variation and inflate the headline.
#
# Single-judge runs keep the current path: callers dispatch by inspecting
# ``ScoreMatrixResult.per_vendor_matrix is not None``.
# ---------------------------------------------------------------------------


def per_eval_variance(
    matrix: "np.ndarray",
    per_vendor_matrix: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Per-eval variance vector for the report bar + attestation per-eval.

    Single source of truth for "what does §9 metric #1 say about each
    eval column" — dispatches based on whether the run is a jury (per
    F3) or a single judge:

    * **Single judge** (``per_vendor_matrix is None``): returns
      ``np.var(matrix, axis=0, ddof=0)`` — the canonical 2D path used
      everywhere before F3 MAJOR landed.
    * **Jury run** (``per_vendor_matrix is not None``): returns the
      per-vendor-averaged per-eval variance, NaN-masking
      (vendor, eval) cells with fewer than 2 non-NaN votes (audit M1
      partial-roster guard) so a singleton-vote cell doesn't silently
      collapse to "zero variance."

    Both paths return a 1D ``(n_evals,)`` float64 vector. NaN entries
    propagate so the report's bar renderer can hatch them distinctly
    (as it already does for failed-cell columns).

    Args:
        matrix: ``(n_variants, n_evals)`` majority / single-judge scores.
            Used when ``per_vendor_matrix is None``.
        per_vendor_matrix: optional ``(n_vendors, n_variants, n_evals)``
            slab. When provided, takes precedence over ``matrix`` and
            yields the F3 jury-aware per-eval variance.

    Returns:
        1D ``(n_evals,)`` float64 array.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.per_eval_variance requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    if per_vendor_matrix is None:
        if matrix.size == 0 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            return np.empty(0, dtype=np.float64)
        return np.var(matrix, axis=0, ddof=0)

    arr = np.asarray(per_vendor_matrix)
    if arr.ndim != 3:
        raise ValueError(
            "per_vendor_matrix must be 3D "
            f"(n_vendors, n_variants, n_evals); got ndim={arr.ndim}"
        )
    n_vendors, n_variants, n_evals = arr.shape
    if n_vendors == 0 or n_variants == 0 or n_evals == 0:
        return np.empty(0, dtype=np.float64)
    if n_variants < 2:
        # Match :func:`variance_across_orderings_jury` parity with the
        # canonical singleton-variant semantic (return 0, not NaN-mask).
        return np.zeros(n_evals, dtype=np.float64)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        valid_counts = np.sum(~np.isnan(arr), axis=1)  # (n_vendors, n_evals)
        per_vendor_per_eval = np.where(
            valid_counts >= 2, np.nanvar(arr, axis=1), np.nan
        )
        # Average across vendors for each eval. NaN-skip handles the
        # M1 partial-roster mask without poisoning the headline.
        return np.nanmean(per_vendor_per_eval, axis=0)


def variance_across_orderings_jury(per_vendor_matrix: "np.ndarray") -> float:
    """§9 metric #1 over per-vendor continuous scores (F3 jury-aware).

    Computes :func:`variance_across_orderings` on each vendor's
    ``(n_variants, n_evals)`` slice and averages the per-vendor scalars.
    NaN cells (missing judge vote for a cell) propagate via
    ``np.nanvar`` / ``np.nanmean`` — a vendor whose entire slab is NaN
    is silently DROPPED from the headline average (``np.nanmean`` skips
    it). This is the safer default than poisoning the headline because
    one vendor's whole slab failed; the loader's partial-roster WARNING
    is the operator's signal that this happened.

    The per-vendor average preserves §9 semantics: variance is still
    measured across the variant axis, never across the judge axis.

    Args:
        per_vendor_matrix: ``(n_vendors, n_variants, n_evals)`` float64
            array. Typically the ``per_vendor_matrix`` field on
            :class:`topofuzz.metrics.ScoreMatrixResult`.

    Returns:
        Scalar float. ``0.0`` if any axis is degenerate (single vendor,
        single variant, or single eval would all give 0 from the var
        closure on a singleton axis).

    Raises:
        ImportError: numpy missing (install the ``[stats]`` extra).
        ValueError: array is not 3D.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.variance_across_orderings_jury requires "
            "numpy (install via `pip install topofuzz[stats]`)."
        ) from e

    arr = np.asarray(per_vendor_matrix)
    if arr.ndim != 3:
        raise ValueError(
            "per_vendor_matrix must be 3D "
            f"(n_vendors, n_variants, n_evals); got ndim={arr.ndim}"
        )
    n_vendors, n_variants, n_evals = arr.shape
    if n_vendors == 0 or n_variants == 0 or n_evals == 0:
        return 0.0
    # Singleton variant axis: structurally "no orderings to vary" —
    # matches the canonical :func:`variance_across_orderings` 0-return
    # semantic for `test_voo_zero_on_single_variant`. The M1 partial-
    # roster guard below would NaN-mask every cell otherwise, which is
    # mathematically defensible but breaks parity with the 2D path.
    if n_variants < 2:
        return 0.0

    # Per-vendor VOO: nanvar across the variant axis (axis=1 within the
    # 3D slab), then nanmean across evals (axis=1 of the resulting 2D
    # (n_vendors, n_evals) array), then a plain mean across vendors.
    #
    # Audit M1 fix: ``np.nanvar`` on a length-1 non-NaN slice returns 0
    # (treats it as "no variation"), which would let a partial-roster
    # cell with only one vendor reporting silently deflate the headline.
    # Mask per-eval-per-vendor cells with fewer than 2 non-NaN votes to
    # NaN so ``nanmean`` skips them — "1 data point" is genuinely
    # degenerate for the variance estimator, not a measurement of zero.
    # The full-slab singleton (n_variants < 2) is handled above so this
    # mask only fires on partial-roster degeneracy, not the canonical
    # "intentionally single variant" path.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        valid_counts = np.sum(~np.isnan(arr), axis=1)
        per_eval_per_vendor = np.where(
            valid_counts >= 2, np.nanvar(arr, axis=1), np.nan
        )
        per_vendor_voo = np.nanmean(per_eval_per_vendor, axis=1)
        headline = float(np.nanmean(per_vendor_voo))
    return headline


def variance_across_orderings_jury_ci(
    per_vendor_matrix: "np.ndarray",
    *,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Cluster-bootstrap CI for the jury-aware VOO.

    Per-vendor cluster bootstrap: for each bootstrap draw, resample the
    EVAL axis with replacement (shared draw across vendors so the
    "this eval looked stable on judge A but flipped on judge B"
    structure is preserved within a draw), recompute
    :func:`variance_across_orderings_jury` on the resampled slab, and
    take the ``conf``-percentile interval.

    Degenerate paths mirror :class:`fuzzeval.stats.BootstrapResult`:
    a singleton eval axis, an all-NaN slab, or a constant per-vendor
    variance vector all return ``ci_lower = ci_upper = None`` with
    ``degenerate_flag=True``.

    Args:
        per_vendor_matrix: ``(n_vendors, n_variants, n_evals)`` float64.
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
            "topofuzz.metrics.variance_across_orderings_jury_ci requires "
            "numpy (install via `pip install topofuzz[stats]`)."
        ) from e

    arr = np.asarray(per_vendor_matrix)
    if arr.ndim != 3:
        raise ValueError(
            "per_vendor_matrix must be 3D "
            f"(n_vendors, n_variants, n_evals); got ndim={arr.ndim}"
        )

    point = variance_across_orderings_jury(arr)
    n_evals = arr.shape[2]
    # Audit M4: degenerate path matches the project-wide convention
    # (``percentile_cluster_bootstrap`` returns ``n_resamples=0`` when
    # degenerate; the attest path uses ``r.n_resamples == 0`` as a
    # tight degenerate sentinel for terse rendering).
    if n_evals == 0 or not np.isfinite(point):
        return BootstrapResult(
            point_estimate=point if np.isfinite(point) else float("nan"),
            ci_lower=None,
            ci_upper=None,
            method="percentile",
            n_resamples=0,
            degenerate_flag=True,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_evals, size=(n_resamples, n_evals))
    # Resample the eval axis (axis=2 of the slab) per draw and recompute.
    # Loop is fine: ``n_resamples`` is at most 5,000 and each slab is small
    # for realistic jury sizes (3 vendors × 4 variants × 10 evals ~ 120
    # floats per draw).
    #
    # Audit M2: this resamples ONLY the eval axis, intentionally — the
    # variant axis is the "treatment" (different topologies) and the
    # statistic is "how variable are scores across treatments per eval."
    # The canonical 2D :func:`variance_across_orderings_ci` uses
    # ``percentile_cluster_bootstrap`` which paired-resamples both axes
    # because each cell there is one scalar; here each cell is a vector
    # of vendor votes and we want the eval axis as the bootstrap
    # cluster. CIs from the two functions are NOT directly comparable.
    draws = np.empty(n_resamples, dtype=np.float64)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(n_resamples):
            resampled = arr[:, :, indices[k]]
            draws[k] = variance_across_orderings_jury(resampled)

    finite = draws[np.isfinite(draws)]
    if finite.size == 0 or np.all(finite == finite[0]):
        return BootstrapResult(
            point_estimate=point,
            ci_lower=None,
            ci_upper=None,
            method="percentile",
            n_resamples=0,
            degenerate_flag=True,
        )

    alpha = (1.0 - conf) / 2.0
    lo = float(np.quantile(finite, alpha))
    hi = float(np.quantile(finite, 1.0 - alpha))
    return BootstrapResult(
        point_estimate=point,
        ci_lower=lo,
        ci_upper=hi,
        method="percentile",
        n_resamples=n_resamples,
        degenerate_flag=False,
    )
