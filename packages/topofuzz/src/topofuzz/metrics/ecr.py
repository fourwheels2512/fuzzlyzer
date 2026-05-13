"""§9 metric #2: Error Correction Rate (ECR).

::

    ECR = P(verdict_correct | ≥1 upstream agent wrong)
        - P(verdict_correct | all upstream correct)

Point estimate only (no CI per §9).  Degenerate when either conditional
has fewer than ``min_denominator=5`` cells contributing to it — the
caller surfaces the value as ``None`` and the report shows the
"ECR unavailable on this fixture" banner.

**v0.5 lending-fixture reality.** The shipped lending fixture's eval
CSV columns are ``id, input, ground_truth_decision`` only — there is no
per-upstream-agent ground-truth column.  Without one, ``upstream_wrong``
must be passed as ``None``, and ECR reports degenerate. The function
signature accepts a caller-supplied mask so v0.6 (or third-party
fixtures with richer eval rows) can wire a real signal without a
contract change.

Three reasonable v0.6 sources for the mask, all reachable by extending
:func:`topofuzz.metrics.read_score_matrix` (not done here — that would
expand the §9 metrics layer past its handoff scope):

1. **Per-agent CSV columns**: ``upstream_intake_correct``,
   ``upstream_risk_correct``, ... — the loader derives ``upstream_wrong``
   as the OR across columns.
2. **Per-agent secondary judge**: a second :class:`fuzzeval.judge.Judge`
   scores each intermediate ``NodeOutput`` against an intermediate-
   rubric; the per-agent binary feeds the mask.
3. **Heuristic from text**: regex on each ``NodeOutput.response_text``
   for the decision token vs ``ground_truth_decision``. Cheap, noisy;
   acceptable for spot-checks but not headline reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ECRResult:
    """Output of :func:`error_correction_rate`.

    ``value`` is the point estimate when ``not degenerate_flag``,
    otherwise ``None``. ``n_upstream_wrong`` / ``n_all_correct`` are
    the per-conditional cell counts (the bootstrap denominators).
    ``degenerate_reason`` is a short human-readable diagnosis when
    degenerate; empty string otherwise.
    """

    value: float | None
    n_upstream_wrong: int
    n_all_correct: int
    degenerate_flag: bool
    degenerate_reason: str


def error_correction_rate(
    verdict_correct: "np.ndarray",
    upstream_wrong: "np.ndarray | None",
    *,
    min_denominator: int = 5,
) -> ECRResult:
    """§9 metric #2: ECR with degenerate-cell handling.

    Both inputs may be 1D (per-cell) or 2D (per-(variant, eval)).
    They are flattened in row-major order before evaluation; the
    metric is conditional on cells, not on rows, so the original shape
    only matters for the caller's bookkeeping.

    Args:
        verdict_correct: Boolean (or 0/1) array indicating whether
            the final verdict on each cell matches ``ground_truth``.
            Typically derived via :func:`fuzzeval.judge.binary_from_continuous`
            on the continuous judge score, compared to the eval row's
            ``ground_truth_decision``.
        upstream_wrong: Boolean (or 0/1) mask of the same shape: True
            when ≥1 upstream agent (any node before the verdict node)
            produced a wrong output on that cell. ``None`` signals that
            the fixture has no upstream-correctness signal — see module
            docstring; returns degenerate.
        min_denominator: §9 calls for at least 5 cells in each
            conditional.  Exposed for tests that want to validate
            boundary behavior at exactly 5.

    Returns:
        :class:`ECRResult`. ``degenerate_flag`` is True when:

        * ``upstream_wrong is None`` (no signal in this fixture), OR
        * either conditional contributes fewer than ``min_denominator``
          cells, OR
        * the input arrays are empty.

        Otherwise ``value`` is in ``[-1, 1]``.

    Raises:
        ImportError: numpy missing.
        ValueError: shape mismatch between ``verdict_correct`` and
            ``upstream_wrong``; or ``min_denominator`` < 1.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.error_correction_rate requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    if min_denominator < 1:
        raise ValueError(
            f"min_denominator must be >= 1; got {min_denominator!r}"
        )

    if upstream_wrong is None:
        # No signal — v0.5 lending-fixture path.
        return ECRResult(
            value=None,
            n_upstream_wrong=0,
            n_all_correct=0,
            degenerate_flag=True,
            degenerate_reason=(
                "upstream_wrong signal unavailable: the eval fixture has "
                "no per-upstream-agent ground-truth column."
            ),
        )

    vc = np.asarray(verdict_correct).ravel().astype(bool)
    uw = np.asarray(upstream_wrong).ravel().astype(bool)
    if vc.shape != uw.shape:
        raise ValueError(
            f"verdict_correct and upstream_wrong must share shape after "
            f"flattening; got {vc.shape!r} vs {uw.shape!r}"
        )

    n_uw = int(uw.sum())
    n_ac = int((~uw).sum())

    if n_uw < min_denominator or n_ac < min_denominator:
        return ECRResult(
            value=None,
            n_upstream_wrong=n_uw,
            n_all_correct=n_ac,
            degenerate_flag=True,
            degenerate_reason=(
                f"denominator too small: {n_uw} cells with ≥1 upstream "
                f"wrong, {n_ac} cells with all upstream correct "
                f"(need ≥{min_denominator} in each)."
            ),
        )

    p_with = float(vc[uw].mean())
    p_without = float(vc[~uw].mean())
    return ECRResult(
        value=p_with - p_without,
        n_upstream_wrong=n_uw,
        n_all_correct=n_ac,
        degenerate_flag=False,
        degenerate_reason="",
    )
