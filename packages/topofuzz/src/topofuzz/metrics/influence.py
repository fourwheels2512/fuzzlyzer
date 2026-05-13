"""§6a per-agent influence score (the actionable line).


BLUEPRINT_v5plus.md §6a defines influence as::

    influence(A) = P(answer changes | A's position changed)
                 - P(answer changes | only other agents' positions changed)

The "answer changes" event is the binary flip between two variants on the
same eval row. The conditioning event is whether agent ``A`` occupies a
different slot in those two variants. We expand the expectation across
all unordered variant pairs and all evals with both cells scored:

* numerator_moved   = #(i, j, e) where A's slot differs in i vs j AND
                      binarised score differs
* denom_moved       = #(i, j, e) where A's slot differs in i vs j
* numerator_same    = #(i, j, e) where A's slot is identical in i vs j
                      AND binarised score differs
* denom_same        = #(i, j, e) where A's slot is identical in i vs j
* influence(A)      = numerator_moved / denom_moved
                    - numerator_same  / denom_same

Negative influence means moving ``A`` stabilises the answer relative to
holding it in place — surfaced signed per the step-15 kickoff decision.

CI: percentile bootstrap over eval clusters, ``n_bootstrap=1000`` per
§6a, matching the existing :func:`fuzzeval.stats.percentile_bootstrap`
estimator. Resampling along the eval axis (not the variant axis) keeps
the pair-structure intact: pairs depend on which agents share slots,
which is a property of the variant set rather than the eval set.

Verify-path safety: this module is imported only by
:mod:`topofuzz.metrics` (kept lazy at the CLI / attest entry points
already, see :mod:`topofuzz.attest` ``compute_voo_for_attestation``).
It depends on numpy and on stdlib only — no Pydantic, no
:mod:`topofuzz.ir`, no LiteLLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class AgentInfluence:
    """One agent's influence row in :class:`InfluenceComputation`.

    ``influence`` / ``ci_lower`` / ``ci_upper`` are ``None`` when the
    agent's row is degenerate (no moved pairs OR no same-slot pairs OR
    no scored cells in the eval axis to bootstrap over). ``degenerate``
    is the boolean gate; ``degenerate_reason`` is a short human string
    surfaced in the HTML report so a "—" cell isn't silent.
    """

    node_id: str
    influence: float | None
    ci_lower: float | None
    ci_upper: float | None
    degenerate: bool
    degenerate_reason: str | None


@dataclass(frozen=True)
class InfluenceComputation:
    """Per-agent influence rollup for one run.

    ``results`` is ranked **descending by signed influence** (NaN /
    degenerate rows sink to the bottom in a deterministic node_id-lex
    order so the report's bar chart is stable across re-renders).
    """

    results: tuple[AgentInfluence, ...]
    n_bootstrap: int
    conf: float
    flip_threshold: float


# Sentinel for "not in this variant" in the per-variant slot map.
_ABSENT = -1


class OrderingAlignmentError(ValueError):
    """Manifest ``variant_orderings`` could not be aligned to the score
    matrix axis.

    Carries a short structured ``reason`` slug so callers can both log
    the human message (``str(exc)``) and route on the kind of failure
    (``exc.reason``) without string-matching. Reasons:

    * ``"missing"`` — manifest has no ``variant_orderings`` field
    * ``"malformed"`` — manifest field is not a list, has the wrong
      length, or carries non-string entries
    * ``"duplicate_hashes"`` — manifest ``variant_canonical_hashes``
      contains a duplicate; aligning to a lex-sorted matrix axis would
      silently pick one ordering and signing the result would lie
    * ``"variant_count_mismatch"`` — score matrix and manifest disagree
      on the number of variants (e.g. partial-resume run); influence on
      a subset would silently drop agents
    * ``"unknown_hash"`` — score matrix references a variant the
      manifest never claimed; this is a corruption signal, not an
      old-run signal
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def align_orderings_to_matrix(
    manifest: "dict",
    variant_axis_ids: Sequence[str],
) -> list[list[str]]:
    """Realign manifest ``variant_orderings`` against a score-matrix axis.

    ``manifest['variant_canonical_hashes']`` is in kickoff (insertion)
    order; ``variant_axis_ids`` is the matrix-axis order (which the
    score-matrix loader emits in canonical_hash lex). Pairs the two
    via the hash field so the returned list of orderings has the same
    row order as ``variant_axis_ids``.

    Strict by design (audit RT-B-1 / RT-B-2 fixes): duplicate hashes,
    variant-count mismatches, and unknown hashes all raise
    :class:`OrderingAlignmentError` rather than silently picking a
    plausible-looking answer. Callers in the silent-degrade paths
    (attestation, HTML report) catch the exception, log, and emit a
    placeholder so the operator sees an explanation rather than a
    wrong-axis influence chart.
    """
    orderings = manifest.get("variant_orderings")
    if not orderings or not isinstance(orderings, list):
        raise OrderingAlignmentError(
            "missing",
            "manifest has no variant_orderings field "
            "(likely a pre-step-15 run)",
        )

    hashes = manifest.get("variant_canonical_hashes")
    if not isinstance(hashes, list) or len(hashes) != len(orderings):
        raise OrderingAlignmentError(
            "malformed",
            "manifest variant_canonical_hashes / variant_orderings "
            "are missing or mismatched in length",
        )

    if len(set(hashes)) != len(hashes):
        raise OrderingAlignmentError(
            "duplicate_hashes",
            "manifest variant_canonical_hashes has duplicates; aligning "
            "to a lex-sorted matrix axis would silently pick one ordering",
        )

    axis_ids = list(variant_axis_ids)
    if len(axis_ids) != len(hashes):
        raise OrderingAlignmentError(
            "variant_count_mismatch",
            f"score matrix has {len(axis_ids)} variants but manifest "
            f"claims {len(hashes)}; influence on a subset would silently "
            "drop agents",
        )

    hash_to_ordering = dict(zip(hashes, orderings, strict=True))
    aligned: list[list[str]] = []
    for h in axis_ids:
        try:
            o = hash_to_ordering[h]
        except KeyError as e:
            raise OrderingAlignmentError(
                "unknown_hash",
                f"score matrix references variant {h!r} not in manifest "
                "— possible run-dir corruption",
            ) from e
        if not all(isinstance(s, str) for s in o):
            raise OrderingAlignmentError(
                "malformed",
                f"variant_orderings entry for {h!r} contains a non-string "
                "node_id",
            )
        aligned.append(list(o))
    return aligned


def compute_influence(
    score_matrix: "np.ndarray",
    variant_orderings: Sequence[Sequence[str]],
    *,
    flip_threshold: float = 0.5,
    n_bootstrap: int = 1000,
    conf: float = 0.95,
    seed: int | None = None,
) -> InfluenceComputation:
    """Per-agent influence + 95% percentile-bootstrap CIs.

    Args:
        score_matrix: ``(n_variants, n_evals)`` float array. NaN cells
            are treated as "judge failure" and excluded from any pair
            involving that cell. Variant order MUST match
            ``variant_orderings``.
        variant_orderings: per-variant tuple/list of ``node_id`` strings
            in slot order. Lengths may differ across variants (the
            decoding_param_swap generator preserves length, but the
            module does not assume that). A node_id that appears in some
            variants and not others is allowed; its influence is then
            computed only over pairs where it appears in BOTH variants.
        flip_threshold: continuous → binary cutoff for the "answer
            changes" event. Default ``0.5`` matches the lending fixture
            judge convention (BLUEPRINT §8 — scores normalised to [0, 1]).
        n_bootstrap: bootstrap resample count. Default ``1000`` is the
            §6a literal. Set ``0`` to disable CI (returns ``None`` for
            both bounds, ``degenerate=False`` if the point estimate
            itself is computable).
        conf: confidence level in (0, 1). Default 0.95.
        seed: RNG seed for ``np.random.default_rng``; ``None`` is
            non-deterministic.

    Returns:
        :class:`InfluenceComputation` with one :class:`AgentInfluence`
        per distinct node_id seen across ``variant_orderings``.

    Raises:
        ImportError: numpy is not installed.
        ValueError: ``flip_threshold`` / ``conf`` / ``n_bootstrap`` out of
            range, ``score_matrix`` is not 2D, or ``len(variant_orderings)
            != score_matrix.shape[0]``.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.compute_influence requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1); got {conf}")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0; got {n_bootstrap}")
    if not np.isfinite(flip_threshold):
        raise ValueError(
            f"flip_threshold must be finite; got {flip_threshold}"
        )

    matrix = np.asarray(score_matrix)
    if matrix.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2D (n_variants, n_evals); got "
            f"ndim={matrix.ndim}"
        )
    n_variants, n_evals = matrix.shape
    if len(variant_orderings) != n_variants:
        raise ValueError(
            f"variant_orderings has {len(variant_orderings)} rows but "
            f"score_matrix has {n_variants} variants"
        )

    # Per-variant duplicate-id check (audit CR-m1): the same node_id in
    # two slots of one variant would silently overwrite slot_map below
    # and produce wrong-axis influence for that agent. Hard error.
    for v_idx, ordering in enumerate(variant_orderings):
        if len(set(ordering)) != len(ordering):
            dupes = sorted({nid for nid in ordering if list(ordering).count(nid) > 1})
            raise ValueError(
                f"variant_orderings[{v_idx}] contains duplicate node_ids "
                f"{dupes!r}; the slot_map would silently lose one slot"
            )

    # Collect distinct node_ids in stable first-seen order so the
    # downstream output's degenerate-row tie-break is reproducible
    # across calls with the same input shape.
    node_ids: list[str] = []
    seen: set[str] = set()
    for ordering in variant_orderings:
        for nid in ordering:
            if nid not in seen:
                seen.add(nid)
                node_ids.append(nid)

    # Slot map: slot_map[a, v] = slot index of agent a in variant v, or
    # _ABSENT when a is not in variant v. int16 covers any plausible
    # v0.5 topology (max ~16 variants × ~10 agents — int8 would also
    # fit, but the saved 1 byte/cell is irrelevant and int16 sidesteps
    # the boundary-case overflow at slot == 128).
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    slot_map = np.full((len(node_ids), n_variants), _ABSENT, dtype=np.int16)
    for v_idx, ordering in enumerate(variant_orderings):
        for slot, nid in enumerate(ordering):
            slot_map[node_to_idx[nid], v_idx] = slot

    # Binarise. NaN propagates: comparison with a finite threshold
    # returns False but we mask those cells before counting flips so
    # NaN never contributes to either numerator or denominator.
    valid_mask = np.isfinite(matrix)
    answers = (matrix >= flip_threshold) & valid_mask

    # Pre-compute eval-axis bootstrap indices once for the whole call so
    # every agent's CI is bootstrapped from the same eval resample. Two
    # benefits: (a) cheaper than n_agents × n_bootstrap draws, (b) cross-
    # agent ranking under the bootstrap is comparable because each draw
    # uses an identical eval permutation.
    rng = np.random.default_rng(seed)
    if n_bootstrap > 0 and n_evals > 0:
        boot_idx = rng.integers(0, n_evals, size=(n_bootstrap, n_evals))
    else:
        boot_idx = None

    out_rows: list[AgentInfluence] = []
    alpha = (1.0 - conf) / 2.0
    for a_idx, node_id in enumerate(node_ids):
        row = _influence_one_agent(
            slot_row=slot_map[a_idx],
            answers=answers,
            valid_mask=valid_mask,
            node_id=node_id,
            boot_idx=boot_idx,
            alpha=alpha,
        )
        out_rows.append(row)

    out_rows.sort(key=_rank_key)

    return InfluenceComputation(
        results=tuple(out_rows),
        n_bootstrap=int(n_bootstrap),
        conf=float(conf),
        flip_threshold=float(flip_threshold),
    )


def _rank_key(row: AgentInfluence) -> tuple[int, float, str]:
    """Sort key: degenerate rows last, then by ``-influence``, then node_id."""
    if row.degenerate or row.influence is None:
        return (1, 0.0, row.node_id)
    # Negative influence (stabilising agents) is meaningful; the §6a UX
    # ranks by absolute headline magnitude but the panel chart wants
    # signed bars descending. We rank by signed value here; the report
    # caller can re-sort by |influence| if it ever wants the bare
    # magnitude ranking.
    return (0, -float(row.influence), row.node_id)


def _influence_one_agent(
    *,
    slot_row: "np.ndarray",
    answers: "np.ndarray",
    valid_mask: "np.ndarray",
    node_id: str,
    boot_idx: "np.ndarray | None",
    alpha: float,
) -> AgentInfluence:
    """Per-agent influence point estimate + percentile CI.

    Builds the moved / same-slot pair masks once, then walks each
    bootstrap resample to recompute the conditional flip rates. The
    bootstrap is over eval columns: a resample is an int array of eval
    indices (with replacement). The point estimate uses the literal
    eval axis ``range(n_evals)``.
    """
    import numpy as np

    n_variants = slot_row.size

    present = slot_row != _ABSENT
    if int(present.sum()) < 2:
        return AgentInfluence(
            node_id=node_id,
            influence=None,
            ci_lower=None,
            ci_upper=None,
            degenerate=True,
            degenerate_reason="agent appears in fewer than 2 variants",
        )

    # Upper-triangular pair iterator over variants where A is present.
    i_idx, j_idx = np.triu_indices(n_variants, k=1)
    both_present = present[i_idx] & present[j_idx]
    i_idx = i_idx[both_present]
    j_idx = j_idx[both_present]
    if i_idx.size == 0:
        return AgentInfluence(
            node_id=node_id,
            influence=None,
            ci_lower=None,
            ci_upper=None,
            degenerate=True,
            degenerate_reason="no variant pairs share this agent",
        )

    moved = slot_row[i_idx] != slot_row[j_idx]
    same = ~moved
    if not moved.any() or not same.any():
        return AgentInfluence(
            node_id=node_id,
            influence=None,
            ci_lower=None,
            ci_upper=None,
            degenerate=True,
            degenerate_reason=(
                "no moved pairs"
                if not moved.any()
                else "no same-slot pairs"
            ),
        )

    # Per-pair × per-eval flip + valid matrices (shape: n_pairs, n_evals).
    flip = answers[i_idx] != answers[j_idx]
    pair_valid = valid_mask[i_idx] & valid_mask[j_idx]

    point = _conditional_diff(flip, pair_valid, moved)
    if point is None:
        return AgentInfluence(
            node_id=node_id,
            influence=None,
            ci_lower=None,
            ci_upper=None,
            degenerate=True,
            degenerate_reason="no scored cells contribute to influence",
        )

    if boot_idx is None or boot_idx.shape[0] == 0:
        return AgentInfluence(
            node_id=node_id,
            influence=float(point),
            ci_lower=None,
            ci_upper=None,
            degenerate=False,
            degenerate_reason=None,
        )

    # Vectorised bootstrap: index the eval axis with the (B, n_evals)
    # bootstrap matrix. ``flip[:, boot_idx]`` has shape (n_pairs, B,
    # n_evals); sum over the eval axis per bootstrap is cheap.
    flip_b = flip[:, boot_idx]
    valid_b = pair_valid[:, boot_idx]
    contrib_b = (flip_b & valid_b).astype(np.int64)
    valid_count_b = valid_b.astype(np.int64)

    moved_num = contrib_b[moved].sum(axis=(0, 2))
    moved_den = valid_count_b[moved].sum(axis=(0, 2))
    same_num = contrib_b[same].sum(axis=(0, 2))
    same_den = valid_count_b[same].sum(axis=(0, 2))

    valid_b_rows = (moved_den > 0) & (same_den > 0)
    if not bool(valid_b_rows.any()):
        return AgentInfluence(
            node_id=node_id,
            influence=float(point),
            ci_lower=None,
            ci_upper=None,
            degenerate=False,
            degenerate_reason="bootstrap degenerate (all resamples empty)",
        )

    safe_moved_den = np.where(moved_den > 0, moved_den, 1)
    safe_same_den = np.where(same_den > 0, same_den, 1)
    boot_influence = (
        moved_num / safe_moved_den - same_num / safe_same_den
    )[valid_b_rows]

    return AgentInfluence(
        node_id=node_id,
        influence=float(point),
        ci_lower=float(np.quantile(boot_influence, alpha)),
        ci_upper=float(np.quantile(boot_influence, 1.0 - alpha)),
        degenerate=False,
        degenerate_reason=None,
    )


def _conditional_diff(
    flip: "np.ndarray",
    pair_valid: "np.ndarray",
    moved: "np.ndarray",
) -> float | None:
    """``P(flip | A moved) - P(flip | A same slot)`` on the literal axis.

    Returns ``None`` if either conditional has zero valid (pair, eval)
    cells — the point estimate is undefined.
    """
    import numpy as np

    contrib = (flip & pair_valid).astype(np.int64)
    valid_count = pair_valid.astype(np.int64)

    moved_num = int(contrib[moved].sum())
    moved_den = int(valid_count[moved].sum())
    same_num = int(contrib[~moved].sum())
    same_den = int(valid_count[~moved].sum())
    if moved_den == 0 or same_den == 0:
        return None
    return moved_num / moved_den - same_num / same_den
