"""Statistical utilities: Wilson CI, percentile bootstrap, and the
two-level paired cluster bootstrap used by topofuzz's
variance-across-orderings CI (BLUEPRINT §9 metric #1).

Public API frozen per BLUEPRINT_v5plus.md §17a so dependent packages
(pramana-eval and others) can import against the contract on Day 1.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Per BLUEPRINT_v5plus.md §17a."""

    point_estimate: float
    ci_lower: float | None
    ci_upper: float | None
    n_resamples: int
    method: str
    degenerate_flag: bool


# Acklam's inverse-normal CDF — pure-stdlib, accurate to ~1.15e-9.
# Lets wilson_ci honor arbitrary `conf` without pulling in scipy on the
# verify-only install path. Source: P. J. Acklam (2010).
_A = (
    -3.969683028665376e+01,  2.209460984245205e+02,
    -2.759285104469687e+02,  1.383577518672690e+02,
    -3.066479806614716e+01,  2.506628277459239e+00,
)
_B = (
    -5.447609879822406e+01,  1.615858368580409e+02,
    -1.556989798598866e+02,  6.680131188771972e+01,
    -1.328068155288572e+01,
)
_C = (
    -7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
     4.374664141464968e+00,  2.938163982698783e+00,
)
_D = (
    7.784695709041462e-03,  3.224671290700398e-01,
    2.445134137142996e+00,  3.754408661907416e+00,
)


def _inv_normal_cdf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p}")
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
            / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    )


def wilson_ci(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion. Pure stdlib."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n]; got {successes} of {n}")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1); got {conf}")
    z = _inv_normal_cdf(1.0 - (1.0 - conf) / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile_bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Single-axis percentile bootstrap. Requires the [stats] extra (numpy).

    For the two-level cluster variant used by topofuzz's headline metric,
    see `percentile_cluster_bootstrap` (step 8).
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "fuzzeval.stats.percentile_bootstrap requires the [stats] extra: "
            "`pip install fuzzeval[stats]`"
        ) from e

    arr = np.asarray(values).ravel()
    n = arr.size
    if n == 0:
        return BootstrapResult(
            point_estimate=float("nan"),
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )

    point = float(statistic(arr))

    if bool(np.all(arr == arr[0])):
        return BootstrapResult(
            point_estimate=point,
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )

    rng = np.random.default_rng(seed)
    resamples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resamples[i] = statistic(rng.choice(arr, size=n, replace=True))

    alpha = (1.0 - conf) / 2.0
    return BootstrapResult(
        point_estimate=point,
        ci_lower=float(np.quantile(resamples, alpha)),
        ci_upper=float(np.quantile(resamples, 1.0 - alpha)),
        n_resamples=n_resamples,
        method="percentile",
        degenerate_flag=False,
    )


def percentile_cluster_bootstrap(
    matrix: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 5000,
    conf: float = 0.95,
    outer_axis: int = 1,
    inner_axis: int = 0,
    *,
    seed: int | None = None,
) -> BootstrapResult:
    """Two-level paired cluster bootstrap on a 2D score matrix.

    Resamples outer clusters (TSF: ``eval_id``, axis 1) and inner clusters
    (TSF: ``variant_id``, axis 0) each with replacement. The same inner
    resampling is applied across every chosen outer cluster on each draw —
    that's the "paired" part — so within-row correlation structure across
    evals is preserved. ``statistic`` is evaluated on the original matrix
    (point estimate) and on each resampled (n_inner, n_outer)-shaped matrix.
    Returns a ``conf``-level percentile CI from the resample distribution.

    The API is statistic-agnostic: topofuzz's canonical use is
    ``variance_across_orderings`` (BLUEPRINT §9 metric #1) ::

        import numpy as np
        scores = np.random.default_rng(0).random((5, 8))
        stat = lambda m: float(np.mean(np.var(m, axis=0)))
        result = percentile_cluster_bootstrap(scores, stat, seed=0)

    but pramana-eval and other consumers can pass their own closures.

    Signature locked per BLUEPRINT_v5plus.md §17a; ``seed`` is an additive
    keyword-only extension matching the ``percentile_bootstrap`` precedent
    so property tests can pin RNG state.

    Args:
        matrix: 2D score matrix with a numeric dtype (float/int/bool/uint).
            Empty along either axis or containing non-finite entries
            (NaN / inf) is treated as degenerate.
        statistic: maps a 2D array to a scalar. Should be invariant under
            permutation of whichever axis the bootstrap is paired over.
        n_resamples: number of bootstrap draws (must be positive). The
            default of 5000 per §9 is the recommended minimum for a
            meaningful percentile CI.
        conf: confidence level in (0, 1). Default 0.95.
        outer_axis: axis whose clusters are sampled at the outer level
            (TSF: eval_id, default 1).
        inner_axis: axis whose clusters are sampled (paired) at the inner
            level (TSF: variant_id, default 0).
        seed: RNG seed for ``np.random.default_rng``; ``None`` is
            non-deterministic.

    Returns:
        BootstrapResult with ``method="percentile"``. When ``matrix`` is
        empty along either axis, every entry is identical, or any entry
        is non-finite, returns ``ci_lower = ci_upper = None`` with
        ``degenerate_flag=True`` and ``n_resamples=0`` (mirroring
        ``percentile_bootstrap`` and the §9 ECR degenerate-cell convention).
        In the all-equal case ``point_estimate`` is the constant value
        (``arr[0, 0]``) rather than ``statistic(arr)`` — the statistic is
        never invoked on degenerate input. For empty / non-finite matrices,
        ``point_estimate`` is ``nan``.

        A single-row or single-column matrix is **not** flagged degenerate —
        the bootstrap is mathematically valid but degenerates content-wise
        (zero variance across the singleton axis). For the (1, N) row case
        this produces ``point=0, ci=[0, 0], degenerate_flag=False``, which
        will render in §9's report as ``0.0 [0.0, 0.0]`` rather than as an
        ECR-style dash. Callers that need the dash behaviour should check
        ``matrix.shape[inner_axis] >= 2`` upstream.

    Performance:
        Pre-draws the resample indices as two ``(n_resamples, ·)`` int64
        arrays — roughly ``8 · n_resamples · (n_inner + n_outer)`` bytes.
        For §9 defaults (5,000 × 5 × 50) this is ~4 MB. Above
        ``n_resamples ≈ 100_000`` on large matrices, callers should chunk.

    Raises:
        ValueError: matrix is not 2D, has a non-numeric dtype, ``conf`` is
            outside (0, 1), ``n_resamples`` is not positive, or axes are
            not a permutation of ``{0, 1}`` (with ``numbers.Integral``
            values only — ``bool``/``float``/out-of-range are rejected).
        ImportError: numpy is not installed (install the ``[stats]`` extra).
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "fuzzeval.stats.percentile_cluster_bootstrap requires the "
            "[stats] extra: `pip install fuzzeval[stats]`"
        ) from e

    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D; got ndim={arr.ndim}")
    # dtype kind: f=float, i=int, u=uint, b=bool. Anything else (object,
    # str, datetime, complex) is off-contract and `np.isfinite` would
    # raise a confusing TypeError downstream — refuse here, loudly.
    if arr.dtype.kind not in "fiub":
        raise ValueError(
            f"matrix dtype must be numeric (float/int/uint/bool); "
            f"got dtype={arr.dtype!s}"
        )
    # Reject bool/float/np.int subclass-mismatch axes explicitly. Python
    # treats True/1.0 as set-equal to 1, which would silently bypass the
    # {0, 1} check; `numbers.Integral` covers `int` + `np.int64` + ...,
    # excluding bool which we reject separately.
    def _bad_axis(a: object) -> bool:
        return isinstance(a, bool) or not isinstance(a, numbers.Integral)

    if (
        _bad_axis(outer_axis) or _bad_axis(inner_axis)
        or {int(outer_axis), int(inner_axis)} != {0, 1}
    ):
        raise ValueError(
            "axes (outer_axis, inner_axis) must be a permutation of "
            f"{{0, 1}}; got outer_axis={outer_axis!r} "
            f"inner_axis={inner_axis!r}"
        )
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1); got {conf}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive; got {n_resamples}")

    # Canonicalize: axis 0 = inner (variant), axis 1 = outer (eval).
    if int(inner_axis) == 1:
        arr = arr.T
    n_inner, n_outer = arr.shape

    # Degenerate paths: empty / non-finite / all-equal. All resolved
    # BEFORE the point-estimate `statistic(arr)` call so niche statistics
    # that don't tolerate degenerate input never see it.
    if n_inner == 0 or n_outer == 0:
        return BootstrapResult(
            point_estimate=float("nan"),
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )
    if not bool(np.all(np.isfinite(arr))):
        return BootstrapResult(
            point_estimate=float("nan"),
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )
    if bool(np.all(arr == arr[0, 0])):
        return BootstrapResult(
            point_estimate=float(arr[0, 0]),
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )

    point = float(statistic(arr))

    # Pre-draw all resample indices in two RNG calls for ~2× speedup
    # over a per-iteration draw. Same seed → same byte stream → same
    # bootstrap distribution (different from a per-iteration scheme,
    # but deterministically so).
    rng = np.random.default_rng(seed)
    outer_indices = rng.integers(0, n_outer, size=(n_resamples, n_outer))
    inner_indices = rng.integers(0, n_inner, size=(n_resamples, n_inner))
    resamples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = arr[inner_indices[i][:, None], outer_indices[i][None, :]]
        resamples[i] = float(statistic(sampled))

    alpha = (1.0 - conf) / 2.0
    return BootstrapResult(
        point_estimate=point,
        ci_lower=float(np.quantile(resamples, alpha)),
        ci_upper=float(np.quantile(resamples, 1.0 - alpha)),
        n_resamples=n_resamples,
        method="percentile",
        degenerate_flag=False,
    )
