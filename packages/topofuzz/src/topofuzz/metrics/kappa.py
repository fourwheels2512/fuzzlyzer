# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""§9 metric #3: Cohen's κ + balanced accuracy + ``collapse_flag``.

Inline numpy (handoff decision A): sklearn would pull ~50 MB transitive
for two functions of ~10 LOC each, and the topofuzz install path is
already wary of extras (``[full]`` already pulls litellm + langgraph).

Headline statistic per §9:

* ``kappa_point = mean over variants of cohens_kappa(predicted_variant,
  ground_truth)`` — predicted is the binarisation of the continuous
  judge score at ``threshold=0.5`` (v0.5 hard-coded per handoff
  decision A); ground_truth is the binarised ``ground_truth_decision``
  column.
* ``ba_point`` is the same mean but for balanced accuracy.
* CI via a two-level paired cluster bootstrap (outer = eval columns,
  inner = variant rows) — same shape as §9 metric #1 but for a
  different statistic. The bootstrap is implemented locally (rather
  than via :func:`fuzzeval.stats.percentile_cluster_bootstrap`) because
  the κ statistic needs access to ground_truth indexed by the resampled
  eval columns; the bootstrap helper's closure-only interface doesn't
  expose those indices.
* ``collapse_flag = (kappa_ci_upper < 0.10) OR (ba_ci_upper < 0.55)``
  per §9: a topology whose CIs don't even reach those thresholds is
  effectively rating at chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class KappaResult:
    """Output of :func:`cohens_kappa_vs_truth`.

    All CI fields are ``None`` when the metric is degenerate (single
    variant, all predictions identical, < 2 evals, etc.). The
    ``collapse_flag`` is False in the degenerate case (we make no
    claim about collapse when we couldn't measure).

    ``method`` mirrors :class:`fuzzeval.stats.BootstrapResult.method`
    for reporting symmetry — always ``"percentile"`` in v0.5 (κ uses
    the same paired-cluster percentile bootstrap as §9 metric #1).
    Surfaced separately so a future variant (BCa, basic) can be
    added without changing the result type.
    """

    kappa_point: float | None
    kappa_ci_lower: float | None
    kappa_ci_upper: float | None
    balanced_accuracy_point: float | None
    balanced_accuracy_ci_lower: float | None
    balanced_accuracy_ci_upper: float | None
    n_resamples: int
    method: str
    degenerate_flag: bool
    collapse_flag: bool


# §9 thresholds — exposed as module constants so a future blueprint
# revision can edit one place and the tests catch any drift.
COLLAPSE_KAPPA_UPPER_THRESHOLD = 0.10
COLLAPSE_BA_UPPER_THRESHOLD = 0.55


def cohens_kappa(rater_a: "np.ndarray", rater_b: "np.ndarray") -> float:
    """Cohen's κ between two binary raters.

    ::

        κ = (p_o - p_e) / (1 - p_e)

    where ``p_o`` is observed agreement and ``p_e`` is expected
    agreement under independence.  Returns ``1.0`` for perfect
    agreement, ``-1.0`` for perfect disagreement, ``0.0`` for chance.

    Degenerate boundary: ``p_e == 1`` (one class collapses — both
    raters always predict the same single class).  Two cases:

    * If observed agreement is also 1 (every prediction matches),
      return 1.0 — they agree perfectly on a constant.
    * Otherwise return 0.0 — there is no signal to scale.

    Args:
        rater_a: 1D array of 0/1 (or boolean) values.
        rater_b: 1D array of 0/1 (or boolean) values, same length.

    Returns:
        κ in ``[-1, 1]``. Always finite.

    Raises:
        ImportError: numpy missing.
        ValueError: arrays differ in length or contain non-binary
            values.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.cohens_kappa requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    a = np.asarray(rater_a).ravel().astype(int)
    b = np.asarray(rater_b).ravel().astype(int)
    if a.shape != b.shape:
        raise ValueError(
            f"rater_a and rater_b must share shape; got {a.shape!r} "
            f"vs {b.shape!r}"
        )
    if a.size == 0:
        # Empty — caller should handle as degenerate upstream; returning
        # 0.0 lets the bootstrap closure not branch on this case.
        return 0.0
    if not np.all((a == 0) | (a == 1)) or not np.all((b == 0) | (b == 1)):
        raise ValueError(
            "rater_a and rater_b must contain only 0/1 values; got "
            f"a values {set(a.tolist())!r}, b values {set(b.tolist())!r}"
        )

    n = a.size
    p_o = float((a == b).mean())
    # Marginal probabilities of class 1 for each rater.
    pa1 = float(a.sum()) / n
    pb1 = float(b.sum()) / n
    p_e = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)

    if p_e >= 1.0:
        # Both raters predict a constant; κ is undefined as a ratio.
        # See docstring for the resolution: return 1.0 iff they also
        # observed-agree on every cell, else 0.0.
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def balanced_accuracy(
    y_true: "np.ndarray", y_pred: "np.ndarray"
) -> float:
    """Balanced accuracy for a binary classifier: ``(TPR + TNR) / 2``.

    Degenerate boundary: if either class is absent from ``y_true``,
    the missing rate is taken as 0.5 (chance) so the result remains
    interpretable. This mirrors sklearn's ``balanced_accuracy_score``
    behavior with ``adjusted=False`` plus the standard fallback for an
    all-one-class fold.

    Args:
        y_true: 1D array of 0/1 (or boolean) ground-truth values.
        y_pred: 1D array of 0/1 (or boolean) predictions, same length.

    Returns:
        Balanced accuracy in ``[0, 1]``.

    Raises:
        ImportError: numpy missing.
        ValueError: arrays differ in length or contain non-binary
            values.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.balanced_accuracy requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must share shape; got {yt.shape!r} "
            f"vs {yp.shape!r}"
        )
    if yt.size == 0:
        return 0.5
    if not np.all((yt == 0) | (yt == 1)) or not np.all((yp == 0) | (yp == 1)):
        raise ValueError(
            "y_true and y_pred must contain only 0/1 values"
        )

    pos_mask = yt == 1
    neg_mask = yt == 0
    tpr = float(yp[pos_mask].mean()) if pos_mask.any() else 0.5
    tnr = float((1 - yp[neg_mask]).mean()) if neg_mask.any() else 0.5
    return 0.5 * (tpr + tnr)


def _kappa_and_ba_over_variants(
    scores: "np.ndarray",
    ground_truth_binary: "np.ndarray",
    *,
    threshold: float,
) -> tuple[float, float]:
    """Compute (mean-κ-over-variants, mean-BA-over-variants).

    Pulled out as a helper so the bootstrap loop calls the same code
    path as the point-estimate call — guarantees the resample-vs-point
    distinction never silently diverges.
    """
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover — runtime dep handled above
        raise ImportError("numpy required") from e

    preds = (scores >= threshold).astype(int)
    n_variants = preds.shape[0]
    kappas = np.empty(n_variants, dtype=float)
    bas = np.empty(n_variants, dtype=float)
    for r in range(n_variants):
        kappas[r] = cohens_kappa(preds[r], ground_truth_binary)
        bas[r] = balanced_accuracy(ground_truth_binary, preds[r])
    return float(kappas.mean()), float(bas.mean())


def cohens_kappa_vs_truth(
    score_matrix: "np.ndarray",
    ground_truth_binary: "np.ndarray",
    *,
    threshold: float = 0.5,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = None,
) -> KappaResult:
    """§9 metric #3: κ + balanced accuracy + CIs + collapse_flag.

    Args:
        score_matrix: ``(n_variants, n_evals)`` float matrix of
            continuous judge scores in ``[0, 1]``. Binarised against
            ``threshold`` before comparison with ``ground_truth_binary``.
        ground_truth_binary: ``(n_evals,)`` binary array (0/1 or bool).
            Typically derived from the eval CSV's
            ``ground_truth_decision`` column (e.g. ``"approve" → 1``).
        threshold: Cutoff for binarising continuous scores. Defaults to
            0.5 per handoff decision A.
        n_resamples: Cluster-bootstrap draws. Default 5,000 per §9.
        conf: Confidence level.
        seed: RNG seed.

    Returns:
        :class:`KappaResult`.  Degenerate when:

        * score_matrix has < 2 variants or < 2 evals, or
        * all predictions in score_matrix are identical (no variation
          to bootstrap over), or
        * ground_truth_binary has fewer than 2 distinct values.

        ``collapse_flag`` is True iff ``kappa_ci_upper <
        COLLAPSE_KAPPA_UPPER_THRESHOLD`` OR ``balanced_accuracy_ci_upper
        < COLLAPSE_BA_UPPER_THRESHOLD`` (and the result is not
        degenerate).

    Raises:
        ImportError: numpy missing.
        ValueError: shape mismatch; threshold out of [0, 1];
            n_resamples or conf out of range; ground_truth not binary.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.cohens_kappa_vs_truth requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    arr = np.asarray(score_matrix, dtype=np.float64)
    gt = np.asarray(ground_truth_binary).ravel().astype(int)

    if arr.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2D (n_variants, n_evals); "
            f"got ndim={arr.ndim}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0, 1); got {conf!r}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive; got {n_resamples!r}")
    if not np.all((gt == 0) | (gt == 1)):
        raise ValueError(
            "ground_truth_binary must contain only 0/1 values"
        )

    n_variants, n_evals = arr.shape
    if arr.shape[1] != gt.shape[0]:
        raise ValueError(
            f"score_matrix has {arr.shape[1]} evals but ground_truth_binary "
            f"has {gt.shape[0]}; lengths must match."
        )

    # Degenerate fast paths. n_variants==1 means κ has no variance
    # across the resampling axis — the bootstrap CI would be a point.
    # n_evals < 2 means no statistical power. Constant ground_truth
    # means κ is structurally degenerate (one rater is constant).
    if (
        n_variants < 2
        or n_evals < 2
        or not np.all(np.isfinite(arr))
        or len(set(gt.tolist())) < 2
        or bool(np.all(arr == arr[0, 0]))
    ):
        kappa_point: float | None = None
        ba_point: float | None = None
        if n_variants >= 1 and n_evals >= 1 and bool(np.all(np.isfinite(arr))):
            kappa_point, ba_point = _kappa_and_ba_over_variants(
                arr, gt, threshold=threshold
            )
        return KappaResult(
            kappa_point=kappa_point,
            kappa_ci_lower=None,
            kappa_ci_upper=None,
            balanced_accuracy_point=ba_point,
            balanced_accuracy_ci_lower=None,
            balanced_accuracy_ci_upper=None,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
            collapse_flag=False,
        )

    # Point estimates on the original matrix.
    kappa_point, ba_point = _kappa_and_ba_over_variants(
        arr, gt, threshold=threshold
    )

    # Two-level paired cluster bootstrap. We can't reuse
    # :func:`percentile_cluster_bootstrap` here because the κ statistic
    # needs the resampled eval indices to subset ground_truth in
    # lockstep. The resampling structure is otherwise identical:
    # ``outer_indices[i]`` permutes evals (columns); ``inner_indices[i]``
    # permutes variants (rows) with replacement.
    rng = np.random.default_rng(seed)
    outer_indices = rng.integers(0, n_evals, size=(n_resamples, n_evals))
    inner_indices = rng.integers(0, n_variants, size=(n_resamples, n_variants))

    kappa_samples = np.empty(n_resamples, dtype=float)
    ba_samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        # The paired resample. Note the broadcast indexing pattern
        # mirrors :func:`percentile_cluster_bootstrap` so the bootstrap
        # math stays cross-checkable.
        sampled = arr[inner_indices[i][:, None], outer_indices[i][None, :]]
        # ground_truth must follow the eval resampling — that's the
        # whole reason this loop doesn't fit `percentile_cluster_bootstrap`.
        sampled_gt = gt[outer_indices[i]]
        k_i, ba_i = _kappa_and_ba_over_variants(
            sampled, sampled_gt, threshold=threshold
        )
        kappa_samples[i] = k_i
        ba_samples[i] = ba_i

    alpha = (1.0 - conf) / 2.0
    kappa_lo = float(np.quantile(kappa_samples, alpha))
    kappa_hi = float(np.quantile(kappa_samples, 1.0 - alpha))
    ba_lo = float(np.quantile(ba_samples, alpha))
    ba_hi = float(np.quantile(ba_samples, 1.0 - alpha))

    collapse_flag = (
        kappa_hi < COLLAPSE_KAPPA_UPPER_THRESHOLD
        or ba_hi < COLLAPSE_BA_UPPER_THRESHOLD
    )

    return KappaResult(
        kappa_point=kappa_point,
        kappa_ci_lower=kappa_lo,
        kappa_ci_upper=kappa_hi,
        balanced_accuracy_point=ba_point,
        balanced_accuracy_ci_lower=ba_lo,
        balanced_accuracy_ci_upper=ba_hi,
        n_resamples=n_resamples,
        method="percentile",
        degenerate_flag=False,
        collapse_flag=collapse_flag,
    )


# ---------------------------------------------------------------------------
# §14 step 15a (d) + (e): original-vs-permuted comparisons
#
# These are distinct from ``cohens_kappa_vs_truth`` above:
#   * ``cohens_kappa_vs_truth`` compares every variant against ground
#     truth, then averages — measures "do the variants agree with
#     reality, on average?"
#   * ``cohens_kappa_original_vs_permuted`` compares the original
#     variant's verdicts against the union of all permuted variants'
#     verdicts — measures "does the original agree with its own
#     semantically-equivalent permutations?" (Landis-Koch < 0.4 → poor
#     agreement is the headline gate for step 22.)
#   * ``balanced_accuracy_drop`` is the difference between the
#     original's BA-vs-truth and the mean permuted BA-vs-truth —
#     measures "does permuting the topology degrade correctness?"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BalancedAccuracyDropResult:
    """Output of :func:`balanced_accuracy_drop`.

    ``original`` and ``mean_permuted`` are the two BA-vs-truth values
    being compared. ``drop`` is ``original - mean_permuted`` — a
    positive drop means permutation degrades correctness (the
    BLUEPRINT §22 framing). All three are ``None`` when the metric is
    degenerate (single variant, < 2 evals, all-NaN, ground_truth
    not binary, etc.).
    """

    original: float | None
    mean_permuted: float | None
    drop: float | None
    degenerate_flag: bool


def cohens_kappa_original_vs_permuted(
    score_matrix: "np.ndarray",
    original_index: int,
    *,
    threshold: float = 0.5,
) -> float | None:
    """§14 step 15a (d): κ between original-variant and permuted-variant verdicts.

    **Locked semantics — "union" pairing, NOT per-eval majority vote:**
    pairs the original variant's binarised verdicts (length n_evals)
    against the FLATTENED concatenation of all permuted variants'
    binarised verdicts (length (n_variants-1) × n_evals_valid). The
    original column is repeated to match length so Cohen's κ sees
    paired raters. Cells where either side is NaN are dropped from
    both sides in lockstep so the pair count is honest.

    Audit Pass-1 CR-M2 noted that an alternative reading — collapse
    permuted variants to a per-eval majority verdict and then compare
    one-to-one with the original — would also satisfy the BLUEPRINT
    §14 (d) phrasing "union of all permuted variants' binary
    verdicts." We deliberately chose UNION pairing because (i)
    majority-vote collapse discards the per-variant disagreement
    signal that motivates topofuzz in the first place — a topology
    where 3 permuted variants tie 2-1 vs the original is materially
    different from one where they tie 3-0, and only union pairing
    surfaces that, and (ii) the resulting κ is the standard
    pairwise-raters statistic auditors will recognise. The choice
    is locked in this docstring; do not change without re-aligning
    step 22's pull-quote interpretation.

    Args:
        score_matrix: ``(n_variants, n_evals)`` float matrix of
            continuous judge scores in ``[0, 1]``. NaN cells are
            dropped from both raters in lockstep.
        original_index: Row index of the "original" variant in
            ``score_matrix``. Caller resolves from
            ``manifest.variant_canonical_hashes[0]``.
        threshold: Cutoff for binarising. Defaults to 0.5.

    Returns:
        κ in ``[-1, 1]``, or ``None`` when the metric is degenerate
        (n_variants < 2, no valid pairs after NaN drop, ground_truth
        not provided here — that's the BA-drop function's domain).

    Raises:
        ImportError: numpy missing.
        ValueError: ``score_matrix`` not 2D, ``original_index`` out
            of range, ``threshold`` out of [0, 1].
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.cohens_kappa_original_vs_permuted requires "
            "numpy (install via `pip install topofuzz[stats]`)."
        ) from e

    arr = np.asarray(score_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2D (n_variants, n_evals); got "
            f"ndim={arr.ndim}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")

    n_variants, n_evals = arr.shape
    if not (0 <= original_index < n_variants):
        raise ValueError(
            f"original_index={original_index!r} out of range for "
            f"n_variants={n_variants}"
        )
    if n_variants < 2 or n_evals < 1:
        return None

    valid = np.isfinite(arr)
    preds = (arr >= threshold).astype(int)

    orig_preds = preds[original_index, :]
    orig_valid = valid[original_index, :]

    perm_mask = np.ones(n_variants, dtype=bool)
    perm_mask[original_index] = False
    perm_preds = preds[perm_mask, :]
    perm_valid = valid[perm_mask, :]

    # For each (perm_row, eval) cell, keep iff BOTH original and
    # permuted cells are valid. Build the paired-raters arrays from
    # the surviving cells. Use ``perm_preds.shape`` directly for the
    # broadcast target (audit CR-m2: avoid an unused ``n_perm`` local).
    pair_mask = perm_valid & orig_valid[None, :]
    if pair_mask.sum() == 0:
        return None

    a = np.broadcast_to(orig_preds[None, :], perm_preds.shape)[pair_mask]
    b = perm_preds[pair_mask]
    if a.size == 0:
        return None
    return cohens_kappa(a, b)


def balanced_accuracy_drop(
    score_matrix: "np.ndarray",
    ground_truth_binary: "np.ndarray",
    original_index: int,
    *,
    threshold: float = 0.5,
) -> BalancedAccuracyDropResult:
    """§14 step 15a (e): BA(original) − mean BA(permuted) against ground truth.

    Args:
        score_matrix: ``(n_variants, n_evals)`` continuous-score matrix.
            NaN cells in a given variant row drop that variant's
            BA-vs-truth from the comparison (the variant is skipped,
            not zeroed).
        ground_truth_binary: ``(n_evals,)`` 0/1 ground-truth array.
            Length must match ``score_matrix.shape[1]``. NaN-aware
            indexing keeps the alignment.
        original_index: Row index of the "original" variant.
        threshold: Cutoff for binarising.

    Returns:
        :class:`BalancedAccuracyDropResult`. Degenerate when
        n_variants < 2, ground_truth has < 2 distinct values, or
        the original variant's row has no valid cells.

    Raises:
        ImportError: numpy missing.
        ValueError: shape mismatch; ``original_index`` out of range;
            ``threshold`` out of [0, 1]; ground_truth not binary.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.balanced_accuracy_drop requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    arr = np.asarray(score_matrix, dtype=np.float64)
    gt = np.asarray(ground_truth_binary).ravel().astype(int)

    if arr.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2D; got ndim={arr.ndim}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
    if not np.all((gt == 0) | (gt == 1)):
        raise ValueError("ground_truth_binary must contain only 0/1 values")

    n_variants, n_evals = arr.shape
    if arr.shape[1] != gt.shape[0]:
        raise ValueError(
            f"score_matrix has {arr.shape[1]} evals but ground_truth_binary "
            f"has {gt.shape[0]}; lengths must match."
        )
    if not (0 <= original_index < n_variants):
        raise ValueError(
            f"original_index={original_index!r} out of range for "
            f"n_variants={n_variants}"
        )
    if n_variants < 2 or n_evals < 1 or len(set(gt.tolist())) < 2:
        return BalancedAccuracyDropResult(
            original=None,
            mean_permuted=None,
            drop=None,
            degenerate_flag=True,
        )

    valid = np.isfinite(arr)
    preds = (arr >= threshold).astype(int)

    def _ba_one_row(row_idx: int) -> float | None:
        m = valid[row_idx]
        if not m.any():
            return None
        return balanced_accuracy(gt[m], preds[row_idx, m])

    ba_orig = _ba_one_row(original_index)
    if ba_orig is None:
        return BalancedAccuracyDropResult(
            original=None,
            mean_permuted=None,
            drop=None,
            degenerate_flag=True,
        )

    perm_indices = [i for i in range(n_variants) if i != original_index]
    perm_bas: list[float] = []
    for i in perm_indices:
        ba_i = _ba_one_row(i)
        if ba_i is not None:
            perm_bas.append(ba_i)
    if not perm_bas:
        return BalancedAccuracyDropResult(
            original=float(ba_orig),
            mean_permuted=None,
            drop=None,
            degenerate_flag=True,
        )
    mean_perm = float(sum(perm_bas) / len(perm_bas))
    return BalancedAccuracyDropResult(
        original=float(ba_orig),
        mean_permuted=mean_perm,
        drop=float(ba_orig) - mean_perm,
        degenerate_flag=False,
    )
