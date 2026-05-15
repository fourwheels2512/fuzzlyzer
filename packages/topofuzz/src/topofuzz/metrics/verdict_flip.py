"""§14 step 15a (c): verdict-flip rate vs the original variant.

BLUEPRINT line 416 definition::

    % of (variant != original, eval_id) pairs where the binary verdict
    (score ≥ threshold) differs from the original variant's binary
    verdict on the same eval_id.

The "original variant" is identified by manifest insertion order — i.e.
``manifest.variant_canonical_hashes[0]`` (the §6 contract). The
score-matrix axis is canonical_hash lex (handoff decision C of step 9),
so callers resolve the original's row index via
:func:`topofuzz.metrics.align_orderings_to_matrix` (or equivalent) and
pass the resulting integer in as ``original_index``.

CI via cluster bootstrap on the eval-id axis (n=5000 per §9). Resampling
along the eval axis preserves the per-eval flip structure; resampling
permuted variants would mix the comparison's reference frame.

The point estimate is THE pull-quote number for step 22's CEO outreach
PDF: *"swapping agents X and Y flips the verdict on Z% of inputs."*
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# Shared verdict-flip threshold for the binarisation
# ``score >= DEFAULT_FLIP_THRESHOLD``. Pulled out of report.py (was
# ``_FLIP_THRESHOLD``) and the per-call kwarg defaults below so the
# threshold has a single source of truth. Closes BACKLOG: lift
# _FLIP_THRESHOLD constant.
DEFAULT_FLIP_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class VerdictFlipResult:
    """Output of :func:`compute_verdict_flip_rate`.

    All CI fields are ``None`` when the metric is degenerate (single
    variant — no permuted rows to compare against; < 2 evals after
    NaN filtering; all valid cells share the same binarised verdict).
    ``degenerate_flag`` is the boolean gate so callers don't have to
    string-match the absence of a CI.

    ``method`` mirrors :class:`fuzzeval.stats.BootstrapResult.method`
    for reporting symmetry — always ``"percentile"`` in v0.5.

    ``numerator`` / ``denominator`` are the integer flip count and the
    integer count of valid (original, permuted) pairs the point estimate
    is computed from. They are ``None`` only on the structurally-
    degenerate paths (no paired cells at all). Exposed so downstream
    callers can compute alternative confidence intervals (Wilson, etc.)
    without re-walking the score matrix.
    """

    point: float | None
    ci_lower: float | None
    ci_upper: float | None
    n_resamples: int
    method: str
    degenerate_flag: bool
    numerator: int | None = None
    denominator: int | None = None


def wilson_score_interval(
    successes: int, trials: int, *, conf: float = 0.95,
) -> tuple[float | None, float | None]:
    """95% (default) Wilson score interval for a binomial proportion.

    Wilson (1927) is the standard well-behaved CI for a binomial
    proportion ``p = k/n``: it stays inside ``[0, 1]``, behaves
    correctly at the boundaries (``p=0`` and ``p=1``) where the normal
    approximation collapses, and is tighter than Clopper-Pearson on
    realistic sample sizes. Used for the ``headline.verdict_flip_rate``
    user-facing CI alongside the cluster-bootstrap CI already on the
    same dict — Wilson is the "is the rate non-zero?" question; the
    bootstrap is the "how much would re-sampling the eval set move
    it?" question. Both are in scope.

    Args:
        successes: Integer count of "flips" (or whichever 1-class).
            Must satisfy ``0 <= successes <= trials``.
        trials: Integer total of paired observations. ``0`` returns
            ``(None, None)`` — there is no defensible CI on a zero-
            sample proportion.
        conf: Confidence level in ``(0, 1)``. Defaults to 0.95.

    Returns:
        ``(lower, upper)`` floats clamped to ``[0.0, 1.0]``, or
        ``(None, None)`` when ``trials == 0``.

    Raises:
        TypeError: ``successes`` / ``trials`` is ``bool`` or non-int;
            ``conf`` is ``bool``.
        ValueError: ``successes`` out of ``[0, trials]``; ``trials``
            negative; ``conf`` out of the usable range
            ``[1e-9, 1 - 1e-9]``. The interior tightening defends
            against ``inv_cdf`` underflow at the float boundary —
            ``conf = 1.0 - eps`` collapses ``1 - alpha/2`` to exactly
            ``1.0`` which raises ``StatisticsError`` from
            :meth:`statistics.NormalDist.inv_cdf`.
    """
    # F1 audit pass-1 Fuzzer-MAJOR-3 + RT-MAJOR-3 hardening.
    # Bool is a subclass of int so ``True`` and ``False`` slip through
    # ``0 <= True <= 10`` and the docstring's ``successes: int`` claim
    # silently lies. Same for ``trials``. ``conf`` is a float; only the
    # bool form is the relevant trap because ``isinstance(True, float)``
    # is False but ``0.0 < True < 1.0`` evaluates True.
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError(
            "successes and trials must be int, not bool "
            f"(got successes={successes!r}, trials={trials!r})"
        )
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError(
            "successes and trials must be int "
            f"(got successes={type(successes).__name__}, "
            f"trials={type(trials).__name__})"
        )
    if isinstance(conf, bool):
        raise TypeError(f"conf must be float, not bool; got {conf!r}")

    if trials < 0:
        raise ValueError(f"trials must be >= 0; got {trials!r}")
    if not (0 <= successes <= trials):
        raise ValueError(
            f"successes must be in [0, trials={trials}]; got {successes!r}"
        )
    # Tightened interior range: NormalDist.inv_cdf(1.0) raises
    # StatisticsError; with conf = 1 - eps the formula collapses to that.
    _CONF_INTERIOR_EPS = 1e-9
    if not (_CONF_INTERIOR_EPS <= conf <= 1.0 - _CONF_INTERIOR_EPS):
        raise ValueError(
            f"conf must be in [{_CONF_INTERIOR_EPS}, "
            f"{1.0 - _CONF_INTERIOR_EPS}]; got {conf!r}"
        )
    if trials == 0:
        return (None, None)

    alpha = 1.0 - conf
    # NormalDist.inv_cdf(1 - alpha/2) is the two-sided z-score. For 95%
    # this is ~1.959963984540054, matching scipy.stats.norm.ppf(0.975).
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z2 = z * z
    n = float(trials)
    p_hat = successes / n
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    ) / denom
    # Clamp to [0, 1] — center ± half can sit a hair outside on
    # degenerate inputs (p_hat == 0 or 1), and the headline display is
    # a probability, not a real-line statistic.
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    # Snap residual float-rounding noise at the boundaries: at p_hat=0
    # the formula gives ``center == half`` in real arithmetic but float
    # subtraction can leave ~2.8e-17. Without this snap the rendered
    # "0%" floor is "0.000000000000000028", which leaks through any
    # equality test against 0.0 and is jarring in raw json.
    _BOUNDARY_EPS = 1e-12
    if low < _BOUNDARY_EPS:
        low = 0.0
    if high > 1.0 - _BOUNDARY_EPS:
        high = 1.0
    return (low, high)


def compute_verdict_flip_rate(
    score_matrix: "np.ndarray",
    original_index: int,
    *,
    threshold: float = DEFAULT_FLIP_THRESHOLD,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = None,
) -> VerdictFlipResult:
    """Verdict-flip rate of permuted variants vs the original variant.

    Args:
        score_matrix: ``(n_variants, n_evals)`` float matrix of
            continuous judge scores in ``[0, 1]``. NaN cells are
            dropped from BOTH the numerator and denominator — the
            denominator counts only pairs where the original cell and
            the permuted cell are both finite. This matches the
            §9 metric convention (failed cells are missing data, not
            "non-flips").
        original_index: Row index of the "original" variant in
            ``score_matrix``. Caller resolves this from
            ``manifest.variant_canonical_hashes[0]`` via
            ``align_orderings_to_matrix`` (or equivalent) and passes
            the resulting axis-0 index here.
        threshold: Cutoff for binarising continuous scores. Defaults
            to 0.5 per handoff decision A.
        n_resamples: Cluster-bootstrap draws on the eval axis.
            Default 5000 per §9.
        conf: Confidence level in (0, 1).
        seed: RNG seed.

    Returns:
        :class:`VerdictFlipResult`. Degenerate cases:

        * ``n_variants < 2`` — no permuted rows exist (single-agent
          fixtures hit this; their flip rate is structurally 0 but
          we report it as degenerate so the report doesn't quote a
          CI on a structural zero).
        * fewer than 2 eval columns with at least one valid
          (original, permuted) pair — no statistical power for the
          cluster bootstrap.
        * All valid cells share the same binarised verdict — no
          flips, and no resample can produce one. Point is 0.0;
          ``degenerate_flag=True``.

    Raises:
        ImportError: numpy missing.
        ValueError: ``original_index`` out of range; ``threshold``
            out of [0, 1]; ``n_resamples`` not positive; ``conf``
            out of (0, 1); ``score_matrix`` not 2D.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.compute_verdict_flip_rate requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    arr = np.asarray(score_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2D (n_variants, n_evals); got "
            f"ndim={arr.ndim}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0, 1); got {conf!r}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive; got {n_resamples!r}")

    n_variants, n_evals = arr.shape
    if not (0 <= original_index < n_variants):
        raise ValueError(
            f"original_index={original_index!r} out of range for "
            f"n_variants={n_variants}"
        )

    if n_variants < 2 or n_evals < 1:
        return VerdictFlipResult(
            point=None,
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
            numerator=None,
            denominator=None,
        )

    # Track valid cells explicitly. ``arr >= threshold`` returns False
    # for NaN under numpy's defaults, which is the same answer the
    # binary "did this cell pass" question deserves — but the cell
    # didn't actually produce a verdict, so we drop it from the
    # denominator rather than counting it as a non-flip.
    valid = np.isfinite(arr)
    preds = arr >= threshold

    orig_preds = preds[original_index, :]
    orig_valid = valid[original_index, :]

    perm_mask = np.ones(n_variants, dtype=bool)
    perm_mask[original_index] = False
    perm_preds = preds[perm_mask, :]
    perm_valid = valid[perm_mask, :]

    denom_mask = perm_valid & orig_valid[None, :]
    # Audit RT-M1: ``arr >= threshold`` evaluates NaN as False, so
    # ``preds[i, j]`` for a NaN cell is structurally False. Without
    # ``denom_mask`` the resulting "False == False == agreement"
    # would silently count failed cells as non-flips and deflate the
    # headline. ``np.where(denom_mask, ..., False)`` makes the
    # validity-gate structural (the boolean dependency is encoded in
    # the array shape rather than in a downstream `& denom_mask`).
    flip = np.where(
        denom_mask, perm_preds != orig_preds[None, :], False
    )

    col_num = flip.sum(axis=0).astype(np.float64)
    col_denom = denom_mask.sum(axis=0).astype(np.float64)
    finite_cols = col_denom > 0
    if not finite_cols.any():
        return VerdictFlipResult(
            point=None,
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
            numerator=None,
            denominator=None,
        )

    total_num = float(col_num[finite_cols].sum())
    total_denom = float(col_denom[finite_cols].sum())
    if total_denom == 0:
        # Defensive: ``finite_cols`` was non-empty so col_denom > 0 on
        # at least one column — total_denom should be > 0. Keep the
        # guard so a future refactor that broadens finite_cols doesn't
        # silently divide by zero.
        return VerdictFlipResult(
            point=None,
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
            numerator=None,
            denominator=None,
        )
    point = total_num / total_denom
    # ``flip`` and ``denom_mask`` are integer counts living inside a
    # float64 sum for vectorisation; cast back to int for the public
    # surface so downstream consumers (Wilson CI, JSON mirror, the §17
    # paragraph) get unambiguous integer arithmetic.
    num_int = int(round(total_num))
    denom_int = int(round(total_denom))

    # Audit CR-M1: no-flip degenerate path is fully determined by the
    # paired-cell statistic. The previous ``valid_anywhere`` term
    # inspected orphan valid cells (e.g. an original-row cell where
    # the matching permuted-row cell is NaN), which could fail to fire
    # for genuinely-degenerate runs and silently emit a zero-width CI.
    # ``total_num == 0`` (over the strict pair_mask) is the contract.
    if total_num == 0:
        return VerdictFlipResult(
            point=0.0,
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
            numerator=0,
            denominator=denom_int,
        )

    finite_idx = np.flatnonzero(finite_cols)
    if finite_idx.size < 2:
        # Single valid eval column — bootstrap collapses to a point.
        return VerdictFlipResult(
            point=point,
            ci_lower=None,
            ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
            numerator=num_int,
            denominator=denom_int,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, finite_idx.size, size=(n_resamples, finite_idx.size)
    )
    resampled_num = col_num[finite_idx][indices]
    resampled_denom = col_denom[finite_idx][indices]
    # All columns in ``finite_idx`` have col_denom > 0 by construction,
    # so resampled_denom.sum(axis=1) is strictly positive on every draw.
    sample_rates = resampled_num.sum(axis=1) / resampled_denom.sum(axis=1)

    alpha = (1.0 - conf) / 2.0
    return VerdictFlipResult(
        point=point,
        ci_lower=float(np.quantile(sample_rates, alpha)),
        ci_upper=float(np.quantile(sample_rates, 1.0 - alpha)),
        n_resamples=n_resamples,
        method="percentile",
        degenerate_flag=False,
        numerator=num_int,
        denominator=denom_int,
    )
