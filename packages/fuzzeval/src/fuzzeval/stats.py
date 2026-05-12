"""Statistical utilities: Wilson CI + percentile bootstrap.

Cluster bootstrap (the two-level paired resampler used by topofuzz for its
variance-across-orderings CI) lands in Weekend 2 step 8. The signature is
defined here so dependent packages can import against the §17a contract.
"""

from __future__ import annotations

import math
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
) -> BootstrapResult:
    """Two-level paired cluster bootstrap (variant × eval). Step 8.

    Signature locked per BLUEPRINT_v5plus.md §17a so dependent packages can
    import against it Day 1.
    """
    raise NotImplementedError(
        "percentile_cluster_bootstrap is implemented in step 8. "
        "For step 2, use percentile_bootstrap on flattened data."
    )
