# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Noise-corrected metric: ``confirmed_verdict_flip_rate``.

The raw ``verdict_flip_rate`` (:mod:`topofuzz.metrics.verdict_flip`)
conflates the topology effect with LLM noise — the pipeline re-runs the
SAME variant and flips ~26% of the time (see
:mod:`topofuzz.metrics.verdict_stability`). The ``topofuzz confirm`` pass
re-runs every screened flip candidate K times and re-runs the original
variant K times per relevant eval; this module turns those repeats into
a noise-corrected rate via majority-vote-vs-majority-vote.

Logic (v0.5 — majority-vote, NOT a formal hypothesis test):

* **Reference verdict** per eval_id = majority vote of the K
  ``role='reference'`` repeats' binarised verdicts. No strict majority
  (a tie) → that eval is ``unstable_baseline``: its noise floor is too
  high to call, so every candidate on it is excluded. These are counted.
* **Candidate verdict** per ``(eval_id, variant_id)`` = majority vote of
  that candidate's K ``role='candidate'`` repeats.
* A flip is **confirmed** iff the candidate-majority differs from the
  reference-majority AND the eval has a stable baseline.
* A candidate that the single-pass screen flagged but whose majority now
  EQUALS the reference is ``flaky`` — excluded from the numerator,
  counted separately.

    confirmed_verdict_flip_rate = confirmed
                                  / (candidate pairs with stable baseline)

The CI is an eval-cluster bootstrap: resample the eval_id clusters with
replacement (NOT the variant axis — resampling variants would mix the
comparison's reference frame, same rationale as
:mod:`topofuzz.metrics.verdict_flip`). Mirrors that module's manual
percentile-bootstrap shape rather than
:func:`fuzzeval.stats.percentile_cluster_bootstrap` because the statistic
is a ratio of per-eval integer counts, not a matrix statistic.

numpy contract: like :mod:`topofuzz.metrics.verdict_flip`, this module
``raise``\\ s ``ImportError`` when numpy is missing (it cannot produce a
CI without it). :mod:`topofuzz.metrics.voo_noise_floor` returns ``None``
instead — that divergence is deliberate: the attest path gates on numpy
presence *before* calling either, so neither contract is reached in
practice; the divergence only matters to a direct caller.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from topofuzz.metrics.verdict_flip import DEFAULT_FLIP_THRESHOLD


@dataclass(frozen=True)
class ConfirmedVerdictFlipResult:
    """Output of :func:`compute_confirmed_verdict_flip_rate`.

    ``point`` is confirmed / (candidate pairs with a stable baseline), in
    ``[0, 1]``, or ``None`` when degenerate (no candidate pair has a
    stable baseline). CI fields are ``None`` when degenerate or when
    fewer than 2 eval clusters carry a stable-baseline candidate.

    Disposition counts — every screened candidate pair lands in exactly
    one bucket, so ``confirmed + flaky + unstable_baseline +
    missing_baseline + judge_failure_pairs == n_screened_pairs`` holds
    *when ``candidates`` is supplied* (the path :mod:`topofuzz.attest`
    uses — the screened pair list is walked, so judge-failure-only pairs
    are enumerated). When only ``n_screened_pairs=<int>`` is given the
    pair universe is the pairs observed in ``candidate_rows``: a pair
    whose every repeat was a judge failure contributes no row and cannot
    be enumerated, so the buckets sum to ``len(cand_by_pair)``, not
    necessarily ``n_screened_pairs`` — the invariant is NOT guaranteed on
    that path (see the ``n_screened_pairs`` arg docstring). The buckets:

    * ``confirmed`` — a real noise-corrected flip (candidate majority
      differs from a stable reference majority).
    * ``flaky`` — the single-pass screen flagged it, but majority-vote
      reclassified it as noise (candidate majority equals the reference,
      or the candidate's own repeats tied).
    * ``unstable_baseline`` — the eval's reference repeats *tied* (a
      genuine baseline-too-noisy-to-call signal).
    * ``missing_baseline`` — the eval had NO scored reference repeat at
      all (the original variant produced zero callable reference rows —
      a data-coverage gap, distinct from a tied baseline).
    * ``judge_failure_pairs`` — every one of the pair's K candidate
      repeats was a judge failure (``score_value is None``), so the pair
      has no callable candidate verdict. Without this bucket such a pair
      would silently vanish and the sum invariant would break.

    ``n_screened_pairs`` echoes the input ``n_screened_pairs`` (or the
    count derived from ``candidates``) so a caller can assert the sum
    invariant directly.
    """

    point: float | None
    ci_lower: float | None
    ci_upper: float | None
    confirmed: int
    flaky: int
    unstable_baseline: int
    missing_baseline: int
    judge_failure_pairs: int
    n_screened_pairs: int
    n_resamples: int
    method: str
    degenerate_flag: bool


def _majority(verdicts: list[bool]) -> bool | None:
    """Strict majority of a list of bools; ``None`` on a tie or empty."""
    if not verdicts:
        return None
    counts = Counter(verdicts)
    true_n = counts.get(True, 0)
    false_n = counts.get(False, 0)
    if true_n > false_n:
        return True
    if false_n > true_n:
        return False
    return None  # tie — no strict majority


def compute_confirmed_verdict_flip_rate(
    reference_rows: list[dict],
    candidate_rows: list[dict],
    *,
    candidates: list[tuple[str, str]] | None = None,
    n_screened_pairs: int | None = None,
    repeats: int | None = None,
    threshold: float = DEFAULT_FLIP_THRESHOLD,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = None,
) -> ConfirmedVerdictFlipResult:
    """Noise-corrected verdict-flip rate from ``flip_confirmations`` rows.

    Args:
        reference_rows: ``flip_confirmations`` rows with
            ``role='reference'`` (K re-runs of the original variant).
            Each dict carries ``eval_id`` + ``score_value``.
        candidate_rows: rows with ``role='candidate'`` (K re-runs of each
            screened flip candidate). Each dict carries ``eval_id``,
            ``variant_id`` + ``score_value``.
        candidates: the screened candidate-pair set — the
            ``(eval_id, variant_id)`` list :func:`topofuzz.repro.
            find_all_flip_candidates` produced. Supplying it lets the
            metric account for pairs whose every repeat was a judge
            failure (which otherwise contribute zero rows and silently
            vanish). Mutually informative with ``n_screened_pairs``;
            ``candidates`` takes precedence when both are given.
        n_screened_pairs: explicit count of screened candidate pairs,
            for callers that don't have the pair list handy. When
            neither this nor ``candidates`` is given the metric falls
            back to the set of pairs observed in ``candidate_rows`` (the
            historical behaviour — judge-failure pairs cannot then be
            counted and ``judge_failure_pairs`` stays 0).
        repeats: K — the planned repeat count. When given, any reference
            eval or candidate pair whose scored-repeat count is NOT
            exactly K is dropped from the metric (a ragged unit cannot be
            majority-voted honestly — see B-RT-1; a unit with MORE than K
            scored repeats is stale/contaminated and equally untrustable
            — RT-2, the ``!= K`` drop is exact-count consistent with
            :mod:`topofuzz.metrics.voo_noise_floor`). ``None`` keeps the
            historical behaviour (majority-vote over whatever scored
            repeats exist).
        threshold: continuous-score binarisation cutoff.
        n_resamples: eval-cluster bootstrap draws. Default 5000 per §9.
        conf: confidence level in (0, 1).
        seed: RNG seed.

    Returns:
        :class:`ConfirmedVerdictFlipResult`. Judge-failure repeats
        (``score_value is None``) are dropped before the majority vote;
        an eval / candidate left with zero scored repeats has no majority
        and is routed to ``missing_baseline`` / ``judge_failure_pairs``.

    Raises:
        ImportError: numpy missing (install the ``[stats]`` extra).
        ValueError: ``threshold`` / ``conf`` / ``n_resamples`` /
            ``repeats`` invalid.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.compute_confirmed_verdict_flip_rate requires "
            "numpy (install via `pip install topofuzz[stats]`)."
        ) from e

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0, 1); got {conf!r}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive; got {n_resamples!r}")
    if repeats is not None and repeats <= 0:
        raise ValueError(f"repeats must be positive; got {repeats!r}")

    # The screened candidate-pair universe. When the caller hands us the
    # pair list (or a count) we honour it so the disposition buckets
    # account for EVERY screened pair — including pairs whose every
    # repeat was a judge failure, which contribute no candidate row.
    screened_pairs: set[tuple[str, str]] | None = None
    if candidates is not None:
        screened_pairs = {
            (eid, vid)
            for eid, vid in candidates
            if isinstance(eid, str) and eid and isinstance(vid, str) and vid
        }

    # Reference verdicts grouped per eval_id.
    ref_by_eval: dict[str, list[bool]] = {}
    for row in reference_rows:
        eid = row.get("eval_id")
        if not isinstance(eid, str) or not eid:
            continue
        score = row.get("score_value")
        if score is None:
            continue
        ref_by_eval.setdefault(eid, []).append(bool(score >= threshold))

    # Reference majority per eval_id; tie → unstable baseline. A
    # reference eval whose scored-repeat count is not exactly K (when
    # ``repeats`` is given) cannot be majority-voted honestly — treat its
    # baseline as uncallable (the candidate pair is then routed to
    # missing_baseline). The drop is ``!= K`` not ``< K``: a block with
    # MORE than K scored repeats is stale/contaminated (RT-2).
    ref_majority: dict[str, bool] = {}
    unstable_baseline_evals: set[str] = set()
    for eid, verdicts in ref_by_eval.items():
        if repeats is not None and len(verdicts) != repeats:
            # Ragged or stale baseline — drop it; downstream pairs land
            # in missing_baseline (no callable reference frame).
            continue
        maj = _majority(verdicts)
        if maj is None:
            unstable_baseline_evals.add(eid)
        else:
            ref_majority[eid] = maj

    # Candidate verdicts grouped per (eval_id, variant_id).
    cand_by_pair: dict[tuple[str, str], list[bool]] = {}
    for row in candidate_rows:
        eid = row.get("eval_id")
        vid = row.get("variant_id")
        if not isinstance(eid, str) or not eid:
            continue
        if not isinstance(vid, str) or not vid:
            continue
        score = row.get("score_value")
        if score is None:
            continue
        cand_by_pair.setdefault((eid, vid), []).append(
            bool(score >= threshold)
        )

    # The pair universe we iterate. With a screened set we walk THAT (so
    # judge-failure-only pairs are seen); without one we fall back to the
    # pairs that produced at least one scored candidate row.
    if screened_pairs is not None:
        pair_universe = sorted(screened_pairs)
    else:
        pair_universe = sorted(cand_by_pair)

    # Disposition each candidate pair. Per-eval lists of 0/1 confirmed
    # indicators feed the cluster bootstrap (resample the eval cluster).
    confirmed = 0
    flaky = 0
    unstable_baseline = 0
    missing_baseline = 0
    judge_failure_pairs = 0
    # eval_id -> list of confirmed-indicator ints for its stable-baseline
    # candidate pairs.
    per_eval_flags: dict[str, list[int]] = {}
    for (eid, _vid) in pair_universe:
        verdicts = cand_by_pair.get((eid, _vid), [])
        if not verdicts or (
            repeats is not None and len(verdicts) != repeats
        ):
            # The pair has no scored candidate repeat at all (every
            # repeat was a judge failure) OR its scored-repeat count is
            # not exactly K — ragged (cost-cap hard-stop mid-pair) or
            # stale (more than K rows, RT-2). Either way it cannot be
            # majority-voted. Counted in its own bucket so the sum
            # invariant holds.
            judge_failure_pairs += 1
            continue
        if eid in unstable_baseline_evals:
            # The eval's reference repeats tied — baseline too noisy.
            unstable_baseline += 1
            continue
        if eid not in ref_majority:
            # The original variant produced no callable reference repeat
            # for this eval (zero scored rows, or — when ``repeats`` is
            # given — a ragged reference block). Distinct from a tied
            # baseline: this is a data-coverage gap.
            missing_baseline += 1
            continue
        cand_maj = _majority(verdicts)
        if cand_maj is None:
            # Candidate's own repeats tied — no strict majority, so the
            # screen flip cannot be confirmed. Treat as flaky (the
            # screen flagged it; the repeats don't back it up).
            flaky += 1
            per_eval_flags.setdefault(eid, []).append(0)
            continue
        if cand_maj != ref_majority[eid]:
            confirmed += 1
            per_eval_flags.setdefault(eid, []).append(1)
        else:
            flaky += 1
            per_eval_flags.setdefault(eid, []).append(0)

    # Resolve the screened-pair count for the sum invariant. Priority:
    # the pair list, then the explicit count, then the observed universe.
    if screened_pairs is not None:
        screened_count = len(screened_pairs)
    elif n_screened_pairs is not None:
        screened_count = int(n_screened_pairs)
    else:
        screened_count = len(pair_universe)

    denom = confirmed + flaky  # candidate pairs with a stable baseline
    if denom == 0:
        return ConfirmedVerdictFlipResult(
            point=None,
            ci_lower=None,
            ci_upper=None,
            confirmed=confirmed,
            flaky=flaky,
            unstable_baseline=unstable_baseline,
            missing_baseline=missing_baseline,
            judge_failure_pairs=judge_failure_pairs,
            n_screened_pairs=screened_count,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )

    point = confirmed / denom

    # Eval-cluster bootstrap. Each cluster contributes (n_confirmed,
    # n_pairs) for that eval; resample clusters with replacement and
    # recompute the pooled ratio.
    eids = sorted(per_eval_flags)
    col_num = np.array(
        [sum(per_eval_flags[e]) for e in eids], dtype=np.float64
    )
    col_denom = np.array(
        [len(per_eval_flags[e]) for e in eids], dtype=np.float64
    )

    if len(eids) < 2:
        # Single eval cluster — bootstrap collapses to a point.
        return ConfirmedVerdictFlipResult(
            point=point,
            ci_lower=None,
            ci_upper=None,
            confirmed=confirmed,
            flaky=flaky,
            unstable_baseline=unstable_baseline,
            missing_baseline=missing_baseline,
            judge_failure_pairs=judge_failure_pairs,
            n_screened_pairs=screened_count,
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(eids), size=(n_resamples, len(eids)))
    resampled_num = col_num[indices].sum(axis=1)
    resampled_denom = col_denom[indices].sum(axis=1)
    # Every cluster has col_denom >= 1 by construction, so a resample's
    # denominator sum is strictly positive on every draw.
    sample_rates = resampled_num / resampled_denom

    alpha = (1.0 - conf) / 2.0
    return ConfirmedVerdictFlipResult(
        point=point,
        ci_lower=float(np.quantile(sample_rates, alpha)),
        ci_upper=float(np.quantile(sample_rates, 1.0 - alpha)),
        confirmed=confirmed,
        flaky=flaky,
        unstable_baseline=unstable_baseline,
        missing_baseline=missing_baseline,
        judge_failure_pairs=judge_failure_pairs,
        n_screened_pairs=screened_count,
        n_resamples=n_resamples,
        method="percentile",
        degenerate_flag=False,
    )
