"""VOO noise floor + the §22 falsifiability gate inputs.

:mod:`topofuzz.metrics.voo` computes ``variance_across_orderings`` (VOO)
across the *variant* axis — but the statistician found VOO is itself
noise-corrupted: re-running the SAME ordering produces non-zero variance
because the pipeline (Gemini Flash, temp=0) is non-deterministic. So a
``voo_ci_lower > 0`` gate fires even on a topology with zero real
order-sensitivity.

This module computes ``voo_noise_floor`` = VOO over the K
``role='reference'`` repeats of the *original* variant (variance across
re-runs of the same ordering), and :func:`voo_gate_passes` is the
canonical helper for the *updated* §22 gate::

    voo_ci_lower > voo_noise_floor

instead of ``voo_ci_lower > 0`` — i.e. the measured order-sensitivity
must clear the noise floor, not merely clear zero.

**Scope.** This module only *emits the inputs* for the updated §22 gate;
no in-tree caller applies the gate yet (the §22 CEO-PDF gate enforcement
site is not built). :func:`voo_gate_passes` is kept as the canonical
helper for that future site so the gate logic has one home when §22
lands.

The reference repeats are ``flip_confirmations`` rows with
``role='reference'``; this module builds a ``(K, n_evals)`` matrix from
them (K repeats on axis 0, evals on axis 1) and reuses
:func:`topofuzz.metrics.voo.variance_across_orderings` — the K-repeat
axis is structurally identical to the variant axis for the variance
statistic.
"""

from __future__ import annotations


def compute_voo_noise_floor(
    reference_rows: list[dict],
    *,
    repeats: int | None = None,
) -> float | None:
    """VOO over the K reference repeats of the original variant.

    Args:
        reference_rows: ``flip_confirmations`` rows with
            ``role='reference'``. Each dict carries ``eval_id``,
            ``repeat_idx`` + ``score_value``.
        repeats: K — the planned repeat count. When given, the function
            returns ``None`` unless EVERY eval_id has exactly K scored
            reference repeats (a ragged block from a cost-cap hard-stop
            cannot be averaged into an honest floor — see B-RT-1).
            ``None`` keeps the looser "every eval scored on every
            observed repeat_idx" completeness check.

    Returns:
        The mean per-eval variance across the K repeats — the same
        statistic as :func:`topofuzz.metrics.voo.variance_across_orderings`
        but computed over the repeat axis instead of the variant axis.
        ``None`` (never raises) when the floor isn't computable: numpy
        missing, no reference rows, fewer than 2 repeats, any eval_id
        has a judge-failure repeat (a ragged matrix — the floor is only
        meaningful over a complete K×n_evals block), or — when
        ``repeats`` is given — any eval_id has fewer than K scored
        repeats.
    """
    if repeats is not None and repeats <= 0:
        raise ValueError(f"repeats must be positive; got {repeats!r}")
    try:
        import numpy as np
    except ImportError:
        return None

    from topofuzz.metrics.voo import variance_across_orderings

    # Group score_value per (eval_id, repeat_idx).
    by_eval: dict[str, dict[int, float]] = {}
    repeat_idxs: set[int] = set()
    for row in reference_rows:
        eid = row.get("eval_id")
        ridx = row.get("repeat_idx")
        score = row.get("score_value")
        if not isinstance(eid, str) or not eid:
            continue
        if not isinstance(ridx, int):
            continue
        if score is None:
            # Judge failure on a reference repeat → ragged block; the
            # noise floor over an incomplete K×n_evals matrix is not
            # well-defined. Bail rather than impute.
            return None
        by_eval.setdefault(eid, {})[ridx] = float(score)
        repeat_idxs.add(ridx)

    if not by_eval or len(repeat_idxs) < 2:
        return None
    # With a declared K, the floor is only honest over the full K×n_evals
    # block — a cost-cap hard-stop that left some evals with < K repeats
    # produces a ragged matrix the mean-variance statistic would silently
    # bias. Refuse rather than average a partial block (B-RT-1).
    if repeats is not None and len(repeat_idxs) < repeats:
        return None

    eids = sorted(by_eval)
    ridxs = sorted(repeat_idxs)
    # Require a complete block: every eval scored on every repeat_idx
    # (and, when ``repeats`` is given, on the full K repeats).
    expected_repeats = len(ridxs) if repeats is None else repeats
    matrix = np.empty((len(ridxs), len(eids)), dtype=np.float64)
    for j, eid in enumerate(eids):
        col = by_eval[eid]
        if len(col) != expected_repeats:
            return None
        for i, ridx in enumerate(ridxs):
            if ridx not in col:
                return None
            matrix[i, j] = col[ridx]

    return variance_across_orderings(matrix)


def voo_gate_passes(
    voo_ci_lower: float | None,
    voo_noise_floor: float | None,
    *,
    confirm_partial: bool = False,
) -> bool:
    """The §22 falsifiability gate: ``voo_ci_lower > voo_noise_floor``.

    Canonical helper for the updated §22 gate. No in-tree caller applies
    it yet — the §22 CEO-PDF gate enforcement site is not built — but the
    gate logic lives here so it has one home when §22 lands.

    Replaces the historical ``voo_ci_lower > 0`` gate. The measured
    order-sensitivity (lower CI bound on VOO across variants) must clear
    the noise floor — the variance the pipeline produces re-running the
    SAME ordering — not merely clear zero.

    Returns ``False`` when ``voo_ci_lower`` is ``None`` (VOO degenerate /
    not computed — nothing to assert).

    ``voo_noise_floor is None`` is ambiguous on its own — it can mean
    "no confirm pass run" OR "the confirm pass ran but was partial /
    ragged so the floor is uncomputable". ``confirm_partial`` disambiguates:

    * ``confirm_partial=False`` + ``floor is None`` → no confirm pass:
      fall back to the historical ``voo_ci_lower > 0`` so a run without a
      confirm pass is not silently un-gated.
    * ``confirm_partial=True`` + ``floor is None`` → a confirm pass ran
      but did not produce a usable floor. The pre-correction ``> 0`` gate
      is NOT a safe fallback here (the operator asked for noise
      correction and the data is incomplete): return ``False`` — the
      gate cannot pass on a partial confirm pass; re-run ``topofuzz
      confirm`` to complete it.
    """
    if voo_ci_lower is None:
        return False
    if voo_noise_floor is None:
        if confirm_partial:
            # A partial confirm pass must not silently fall back to the
            # pre-correction gate — refuse until the pass is completed.
            return False
        # No confirm pass at all — historical ``> 0`` fallback.
        return voo_ci_lower > 0.0
    return voo_ci_lower > voo_noise_floor
