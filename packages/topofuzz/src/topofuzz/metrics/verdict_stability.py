# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Noise-floor metric: ``verdict_stability`` (the AFL "stability %" steal).

Negative-control number for the ``topofuzz confirm`` pass. The pipeline
(Gemini Flash, temp=0) is non-deterministic — re-running the SAME variant
flips the binarised verdict some fraction of the time. ``verdict_stability``
quantifies that floor directly:

    verdict_stability = (# flip-candidate eval_ids whose K reference repeats
                         are unanimous)
                        / (# flip-candidate eval_ids with >= 2 scored
                           reference repeats)

**Scope (read this before quoting the number).** The ``topofuzz confirm``
pass intentionally runs reference repeats ONLY for the eval_ids that
carry a screened flip candidate (re-running every eval would multiply the
re-run bill by the full eval-set size for no extra signal on the flips
under test). So ``verdict_stability`` is stability over *flip-candidate
eval_ids only* — a **lower bound on run-wide stability**, biased low
because flip-prone evals are over-represented in that subset by
construction. It is NOT the stability of the whole run; do not present it
as such.

A run with ``verdict_stability = 1.0`` means every flip-candidate eval's
reference repeats agreed — the floor is clean for the evals that matter.
A run with ``verdict_stability = 0.74`` means ~26% of flip-candidate
eval_ids flip their verdict on re-run of the *same* ordering, so the raw
``verdict_flip_rate`` conflates topology with noise on those evals — that
is exactly the finding this metric exists to surface (see
``confirmed_verdict_flip.py`` for the corrected rate).

"Reference" repeats are the ``flip_confirmations`` rows with
``role='reference'`` — K re-runs of the *original* variant. This module
consumes those rows; it does not touch the score matrix or the
``cells`` / ``judge_results`` tables.

Structure mirrors :mod:`topofuzz.metrics.verdict_flip` and
:mod:`topofuzz.metrics.voo`: a frozen result dataclass + a pure compute
function that takes already-loaded rows (no DB dependency — the attest
path owns the read).
"""

from __future__ import annotations

from dataclasses import dataclass

from topofuzz.metrics.verdict_flip import DEFAULT_FLIP_THRESHOLD


@dataclass(frozen=True)
class VerdictStabilityResult:
    """Output of :func:`compute_verdict_stability`.

    ``point`` is the unanimity fraction in ``[0, 1]``, or ``None`` when
    the metric is degenerate (no eval_id has >= 2 scored reference
    repeats). ``n_evals`` is the number of distinct flip-candidate
    eval_ids that contributed at least 2 scored reference repeats;
    ``n_unanimous`` is how many of those were unanimous across their
    repeats.

    ``degenerate_flag`` is the boolean gate so callers don't have to
    string-match the absence of a point estimate.
    """

    point: float | None
    n_evals: int
    n_unanimous: int
    degenerate_flag: bool


def compute_verdict_stability(
    reference_rows: list[dict],
    *,
    threshold: float = DEFAULT_FLIP_THRESHOLD,
    repeats: int | None = None,
) -> VerdictStabilityResult:
    """Fraction of flip-candidate eval_ids whose reference repeats agree.

    See the module docstring for the scope caveat: this is a lower bound
    on run-wide stability, computed over flip-candidate eval_ids only.

    Args:
        reference_rows: ``flip_confirmations`` rows with
            ``role='reference'``. Each row is a dict carrying at least
            ``eval_id`` and the judge fields ``judge_kind`` /
            ``score_value`` (binary judges are not used by the v0.5
            confirm pass, but a ``passed``-style row is tolerated via
            ``score_value``). Rows whose judge failed — ``score_value``
            is ``None`` *and* there is no usable verdict — are dropped
            from BOTH numerator and denominator (a failed repeat is
            missing data, not a "non-flip"), matching the
            :mod:`topofuzz.metrics.verdict_flip` convention.
        threshold: Cutoff for binarising continuous scores
            (``score_value >= threshold`` → APPROVE). Defaults to
            :data:`topofuzz.metrics.verdict_flip.DEFAULT_FLIP_THRESHOLD`.
        repeats: K — the planned repeat count. When given, an eval_id
            whose scored-reference-repeat count is NOT exactly K is
            dropped from BOTH numerator and denominator (a ragged block
            from a cost-cap hard-stop, or a stale block with more than K
            rows, cannot be scored honestly — see B-RT-1 / RT-2; the
            ``!= K`` drop is exact-count consistent with
            :mod:`topofuzz.metrics.voo_noise_floor`). Additionally, a
            ``repeats < 2`` is degenerate for the whole metric: a single
            repeat is trivially "unanimous", so K=1 cannot measure
            agreement at all (``point=None``, ``degenerate_flag=True``).
            ``None`` keeps the looser ">= 2 scored repeats" floor.

    Returns:
        :class:`VerdictStabilityResult`. Degenerate (``point=None``,
        ``degenerate_flag=True``) when no eval_id has >= 2 scored
        reference repeats (or, with ``repeats`` set, none has exactly K),
        and also when ``repeats`` is given but ``< 2`` (K=1 cannot
        measure agreement — see the ``repeats`` arg).

    Note:
        An eval_id with exactly ONE scored reference repeat is dropped:
        ``len(set([v])) == 1`` is trivially true for a singleton, which
        would inflate the unanimity fraction with evals that were never
        actually tested for agreement. Unanimity needs >= 2 observations.
    """
    if repeats is not None and repeats <= 0:
        raise ValueError(f"repeats must be positive; got {repeats!r}")

    # RT-2 nuance: K=1 is degenerate for the WHOLE metric. A single
    # repeat is trivially "unanimous" (len(set([v])) == 1), so a K=1
    # confirm pass cannot measure agreement at all — refuse rather than
    # report a meaningless 1.0. (Do not reintroduce pass-1's m-RT-1 bug.)
    if repeats is not None and repeats < 2:
        return VerdictStabilityResult(
            point=None,
            n_evals=0,
            n_unanimous=0,
            degenerate_flag=True,
        )

    # Group binarised verdicts per eval_id. A judge-failure repeat
    # (score_value is None) contributes nothing.
    by_eval: dict[str, list[bool]] = {}
    for row in reference_rows:
        eid = row.get("eval_id")
        if not isinstance(eid, str) or not eid:
            continue
        score = row.get("score_value")
        if score is None:
            continue  # judge failure / no verdict — missing data
        verdict = bool(score >= threshold)
        by_eval.setdefault(eid, []).append(verdict)

    # An eval needs >= 2 scored repeats to be meaningfully "unanimous";
    # a singleton is trivially unanimous and would inflate the headline.
    # With ``repeats`` set, require EXACTLY K (a ragged block from a
    # cost-cap hard-stop, or a stale block with > K rows, is not a clean
    # unanimity test — RT-2; exact-K is consistent with voo_noise_floor).
    # ``repeats < 2`` was already handled as a whole-metric degenerate
    # above, so here ``repeats`` is either None or >= 2.
    if repeats is None:
        scorable = {
            eid: verdicts
            for eid, verdicts in by_eval.items()
            if len(verdicts) >= 2
        }
    else:
        scorable = {
            eid: verdicts
            for eid, verdicts in by_eval.items()
            if len(verdicts) == repeats
        }

    if not scorable:
        return VerdictStabilityResult(
            point=None,
            n_evals=0,
            n_unanimous=0,
            degenerate_flag=True,
        )

    n_evals = len(scorable)
    n_unanimous = sum(
        1 for verdicts in scorable.values() if len(set(verdicts)) == 1
    )
    return VerdictStabilityResult(
        point=n_unanimous / n_evals,
        n_evals=n_evals,
        n_unanimous=n_unanimous,
        degenerate_flag=False,
    )
