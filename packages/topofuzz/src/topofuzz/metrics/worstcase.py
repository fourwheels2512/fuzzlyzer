"""Step 16 (Weekend 5): §10a worst-case viewer data layer.

BLUEPRINT_v5plus.md §10a / DoD step 16 (line 430):

    Top-3-disagreement selection + Jinja partial with side-by-side
    trace excerpts. ~100 LOC. Disabled-state fallback ("no instability
    detected"). Visual regression test on a fixture run.

Locked step-16 kickoff decisions (handoff in
``project_step15a_handoff.md``):

1. **Top-K = 3** per BLUEPRINT line 430.
2. **Per-eval worst-case score** = max binary-flip count over variant
   pairs (BLUEPRINT line 477 "largest verdict gap" → binary-flip
   reading; aligns with the §14 ``verdict_flip_rate`` metric shipped
   in step 15a (c)).
3. **Trace excerpt source** = per-cell ``cells.node_outputs_json``
   already in ``run.sqlite``. No re-read of ``traces.jsonl`` — single
   source of truth with :mod:`topofuzz.metrics.score_matrix`.
4. **Disabled-state** = banner in ``report.html`` only (no
   ``attestation.json`` finding).

Two-gate disabled contract (Pass-1 RT BLOCKER-1 fix):

* **Variance gate (BLUEPRINT DoD #13, line 533).** The panel renders
  only when ``np.nanmax(np.nanvar(matrix, axis=0)) > 0.1``. The
  ranker (flip count) is the user-locked decision #2 above; the
  variance gate is a separate spec contract that says "below this
  threshold there is no story to tell, full stop." Both must hold.
* **Flip-count selection.** Among eval columns that pass the variance
  gate, rank by binary-flip pair count and surface the top K.

Without the variance gate, a column with scores like
``[0.51, 0.49, 0.51, 0.49]`` (variance ≈ 0.0001) would still surface
as WORST #1 simply because every pair flips on the 0.5 cutoff. DoD #13
explicitly excludes that case.

Note on the eval *input* bubble: BLUEPRINT line 272 mentions
truncating the eval row's input to 200 chars. The eval input is NOT
currently persisted to ``run.sqlite`` or ``manifest.json`` (only
``eval_ids`` is — see :func:`topofuzz.orchestrator.write_manifest`).
Surfacing it would require a manifest schema extension which changes
the attestation payload bytes; per the step-16 minimalist scope this
is deferred. The eval_id is rendered in the card header instead and
side-by-side node outputs do the visceral work.

Pure data layer: takes a score matrix + run dir, returns a frozen
:class:`WorstCasePanel` ready for the report renderer to feed into
Jinja. No matplotlib / no jinja2 imports — heavy report deps stay
inside :mod:`topofuzz.report`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_log = logging.getLogger(__name__)

# Default Top-K per BLUEPRINT line 430. Overridable so the report path
# can dial K down for tiny fixtures without rewriting the contract.
DEFAULT_K: int = 3

# Default binarisation cutoff — same 0.5 the §14 verdict_flip_rate uses
# (handoff decision A of step 15a). Keep them in lockstep so the
# headline metric and the rendered worst-case agree on what counts as
# a flip.
DEFAULT_THRESHOLD: float = 0.5

# DoD #13 variance gate (BLUEPRINT line 533). The panel renders only
# when at least one eval column has ``np.var(col) > 0.1`` over the
# variant axis. Below this, "the topology produced consistent answers"
# is the correct framing regardless of how many cells happen to land
# on opposite sides of the 0.5 binarisation threshold.
DEFAULT_VARIANCE_GATE: float = 0.1

# Truncation caps. The BLUEPRINT calls for "scrollable" side-by-side
# trace excerpts (§10a line 274). We don't ship a JS scroller; instead
# we cap node-output and rationale text so the rendered cards stay
# legible on a single screen. The full trace lives in
# ``run.sqlite.cells.node_outputs_json`` for the curious operator.
DEFAULT_NODE_TEXT_CAP: int = 400
DEFAULT_RATIONALE_CAP: int = 400


@dataclass(frozen=True)
class NodeOutputView:
    """One agent's contribution to a side of a worst-case card.

    ``response_text`` is truncated to :data:`DEFAULT_NODE_TEXT_CAP`
    chars by default with an ellipsis suffix when clipping fires.
    """

    node_id: str
    role: str
    response_text: str


@dataclass(frozen=True)
class VariantSideView:
    """One ordering's column of a worst-case card.

    ``ordering`` is the slot-ordered node-id sequence pulled from
    ``manifest.variant_orderings`` (parallel to
    ``manifest.variant_canonical_hashes``). Pre-step-15 runs without
    that field render an empty tuple — callers can still display the
    canonical hash.
    """

    variant_id: str
    ordering: tuple[str, ...]
    score: float
    verdict: bool
    rationale: str | None
    node_outputs: tuple[NodeOutputView, ...]


@dataclass(frozen=True)
class WorstCaseEntry:
    """One eval-level worst-case card.

    ``flip_count`` = number of (variant_a, variant_b) pairs where both
    cells are valid and the binarised verdicts differ. ``n_pairs`` =
    total number of valid pairs at that eval (the denominator for a
    "X of Y" caption). The selected (variant_a, variant_b) pair is the
    one with the largest ``|score_a - score_b|`` among differing-verdict
    pairs at this eval.
    """

    eval_id: str
    flip_count: int
    n_pairs: int
    variant_a: VariantSideView
    variant_b: VariantSideView


@dataclass(frozen=True)
class WorstCasePanel:
    """Output of :func:`compute_worstcase_panel`.

    ``disabled=True`` carries an empty ``cases`` tuple and a
    ``disabled_reason`` slug for the rendered banner. Reasons:

    * ``"matrix empty"`` — score matrix is empty / < 2 variants /
      < 1 eval.
    * ``"no instability detected"`` — all valid cells in every eval
      column share the same binarised verdict (the BLUEPRINT banner
      copy at line 300).

    ``k`` records the Top-K the panel was computed under, for
    diagnostics / test assertions.
    """

    disabled: bool
    disabled_reason: str | None
    cases: tuple[WorstCaseEntry, ...]
    k: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_worstcase_panel(
    score_matrix: "np.ndarray",
    variant_ids: tuple[str, ...],
    eval_ids: tuple[str, ...],
    *,
    run_dir: Path | str,
    run_id: str,
    manifest: dict[str, Any] | None = None,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
    variance_gate: float = DEFAULT_VARIANCE_GATE,
    node_text_cap: int = DEFAULT_NODE_TEXT_CAP,
    rationale_cap: int = DEFAULT_RATIONALE_CAP,
) -> WorstCasePanel:
    """Select the Top-K worst-disagreement eval rows + load side traces.

    Args:
        score_matrix: ``(n_variants, n_evals)`` float64. NaN cells are
            excluded from BOTH the per-eval flip count and the pair
            selection. Convention matches
            :func:`topofuzz.metrics.read_score_matrix`.
        variant_ids: Axis-0 labels (canonical_hash lex order).
        eval_ids: Axis-1 labels (manifest insertion order).
        run_dir: Run directory containing ``run.sqlite``. Cells and
            judge rationales are read from there.
        run_id: Disambiguates the ``cells`` / ``judge_results`` rows
            (the v0.5 contract is one run per file but the schema
            keys on ``run_id`` so we filter explicitly).
        manifest: Parsed ``manifest.json``. Used to resolve the
            per-variant ordering (node-id sequence) for display.
            Optional — when missing or shape-mismatched the rendered
            ``VariantSideView.ordering`` is empty.
        k: Number of cards to surface. Default 3 per BLUEPRINT line 430.
        threshold: Binarisation cutoff. Same default (0.5) as
            :func:`topofuzz.metrics.compute_verdict_flip_rate` so the
            headline metric and the rendered cards agree on flips.
        node_text_cap: Truncate each ``NodeOutput.response_text`` to
            this many chars.
        rationale_cap: Truncate each judge rationale to this many
            chars.

    Returns:
        :class:`WorstCasePanel`. ``disabled=True`` with a reason slug
        when nothing to show.

    Raises:
        ImportError: numpy missing (install ``topofuzz[stats]``).
        ValueError: ``score_matrix.ndim != 2``; ``threshold`` out of
            ``[0, 1]``; ``k < 1``; ``variant_ids`` / ``eval_ids``
            shape mismatch.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.compute_worstcase_panel requires numpy "
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
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k!r}")

    n_variants, n_evals = arr.shape
    if len(variant_ids) != n_variants:
        raise ValueError(
            f"variant_ids length {len(variant_ids)} != matrix.shape[0] "
            f"{n_variants}"
        )
    if len(eval_ids) != n_evals:
        raise ValueError(
            f"eval_ids length {len(eval_ids)} != matrix.shape[1] {n_evals}"
        )

    # Soft lex-invariant check. ``_pick_best_pair`` and the rendered
    # tiebreak depend on ``variant_ids`` being in canonical_hash lex
    # order — the contract from
    # :func:`topofuzz.metrics.read_score_matrix` (handoff decision C of
    # step 9). A future refactor that changes that ordering would
    # silently break determinism; this warning makes the regression
    # observable in the operator log (audit pass-2 CR-M2 / RT-NIT-2).
    sorted_variant_ids = sorted(variant_ids)
    if list(variant_ids) != sorted_variant_ids:
        _log.warning(
            "worstcase: variant_ids not in canonical_hash lex order — "
            "pair-tiebreak determinism is not guaranteed"
        )

    # Structural disables.
    if arr.size == 0 or n_variants < 2 or n_evals < 1:
        return WorstCasePanel(
            disabled=True,
            disabled_reason="matrix empty",
            cases=(),
            k=k,
        )

    valid = np.isfinite(arr)
    if not valid.any():
        # Every cell is NaN — there's nothing to rank, but it's not the
        # "all agreed" case (we don't know what they would have said).
        return WorstCasePanel(
            disabled=True,
            disabled_reason="no continuous-judge cells",
            cases=(),
            k=k,
        )

    # DoD #13 variance gate (BLUEPRINT line 533, **strict** ``>``). Use
    # ``np.nanvar`` so NaN cells are ignored per column. A column with
    # < 2 valid cells contributes NaN to the per-column variance and is
    # excluded from candidate selection — partial-degeneracy is
    # surfaced as a single WARNING (audit pass-2 RT-MINOR-1) so the
    # operator sees the cell-count problem without a per-column burst.
    with warnings.catch_warnings():
        # ``np.nanvar`` on an all-NaN column emits "Degrees of freedom
        # <= 0 for slice" RuntimeWarning. We surface partial degeneracy
        # ourselves via the explicit ``n_nan_cols`` log below; the
        # numpy warning would just be noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        per_col_var = np.nanvar(arr, axis=0, ddof=0)
    finite_var_mask = np.isfinite(per_col_var)
    n_nan_cols = int((~finite_var_mask).sum())
    if n_nan_cols:
        _log.warning(
            "worstcase: %d of %d eval columns have < 2 valid cells "
            "and were excluded from the variance ranker",
            n_nan_cols, n_evals,
        )
    max_var = (
        float(per_col_var[finite_var_mask].max())
        if finite_var_mask.any() else 0.0
    )
    if max_var < float(variance_gate):
        # BLUEPRINT line 533 says "> 0.1"; we test `< gate` so the
        # disabled branch is the strict negation. Equality to the
        # threshold falls through and may surface as a case.
        return WorstCasePanel(
            disabled=True,
            disabled_reason="no instability detected",
            cases=(),
            k=k,
        )

    # ``arr >= threshold`` evaluates NaN-cells to False; we mask via
    # ``valid`` everywhere we consume ``preds`` so an invalid cell never
    # silently counts as a "fail" verdict (the same hardening the §14
    # verdict_flip_rate landed in step 15a — audit RT-M1).
    preds = arr >= threshold

    flip_counts, n_pairs_arr = _per_eval_flip_counts(valid, preds)

    if int(flip_counts.sum()) == 0:
        # Variance was above the gate but no column actually flips at
        # the 0.5 cutoff — every variant lands on the same side even
        # though the spread is wide. Same banner copy as the variance-
        # gated case from the BLUEPRINT line 300 perspective.
        return WorstCasePanel(
            disabled=True,
            disabled_reason="no instability detected",
            cases=(),
            k=k,
        )

    # Top-K eval indices. Audit pass-2 RT-BLOCKER-1: candidate filter
    # is per-column — both flip_count > 0 AND per-column variance >=
    # gate. Without the per-column variance check, a low-variance
    # noise-flip column (e.g. [0.51, 0.49, 0.51, 0.49] with var ≈
    # 0.0001) would surface as a card whenever ANY other column
    # passes the global gate. DoD #13 explicitly excludes those.
    # Primary sort key: descending flip count; secondary: ascending
    # eval_id (lex) for deterministic rendering across runs.
    candidates = [
        j for j in range(n_evals)
        if flip_counts[j] > 0
        and finite_var_mask[j]
        and float(per_col_var[j]) >= float(variance_gate)
    ]
    if not candidates:
        return WorstCasePanel(
            disabled=True,
            disabled_reason="no instability detected",
            cases=(),
            k=k,
        )
    candidates.sort(key=lambda j: (-int(flip_counts[j]), eval_ids[j]))
    selected = candidates[: max(1, int(k))]

    # Best-pair selection per eval: among differing-verdict valid pairs,
    # pick the one with the largest ``|score_a - score_b|``. The pair
    # iteration is ascending over ``valid_idx``; since
    # :func:`topofuzz.metrics.read_score_matrix` returns ``variant_ids``
    # in canonical_hash lex order (handoff decision C of step 9), the
    # FIRST equal-gap pair encountered is already the lex-min pair —
    # ``>`` (not ``>=``) inside the helper preserves it without a
    # separate tiebreak swap.
    pair_by_eval: dict[int, tuple[int, int]] = {}
    for j in selected:
        pair = _pick_best_pair(arr, valid, preds, variant_ids, j)
        if pair is not None:
            pair_by_eval[j] = pair

    if not pair_by_eval:
        # Defensive: every selected eval lost its differing pair after
        # the valid-mask filter. ``flip_counts[j] > 0`` implies at least
        # one valid (PASS, FAIL) pair exists at that column so this
        # path is structurally unreachable from a consistent matrix /
        # valid grid — but a malformed caller bypassing the validator
        # could trip it. Surface as the no-valid-pair reason so the
        # operator log distinguishes it from the genuinely-clean case.
        return WorstCasePanel(
            disabled=True,
            disabled_reason="no valid differing-verdict pair",
            cases=(),
            k=k,
        )

    # Batch-load all required (variant_id, eval_id) cell + rationale
    # rows in a single sqlite connection. K=3 → at most 6 cells; even
    # at K=20 this is one short scan, not 2K round-trips.
    needed: list[tuple[str, str]] = []
    for j, (ai, bi) in pair_by_eval.items():
        needed.append((variant_ids[ai], eval_ids[j]))
        needed.append((variant_ids[bi], eval_ids[j]))
    cell_lookup = _load_cells_and_rationales(run_dir, run_id, needed)

    ordering_by_variant = _build_ordering_lookup(manifest)

    cases: list[WorstCaseEntry] = []
    for j in selected:
        if j not in pair_by_eval:
            continue
        ai, bi = pair_by_eval[j]
        a_view = _build_side(
            variant_ids[ai],
            eval_ids[j],
            float(arr[ai, j]),
            bool(preds[ai, j]),
            cell_lookup,
            ordering_by_variant,
            node_text_cap=node_text_cap,
            rationale_cap=rationale_cap,
        )
        b_view = _build_side(
            variant_ids[bi],
            eval_ids[j],
            float(arr[bi, j]),
            bool(preds[bi, j]),
            cell_lookup,
            ordering_by_variant,
            node_text_cap=node_text_cap,
            rationale_cap=rationale_cap,
        )
        cases.append(
            WorstCaseEntry(
                eval_id=eval_ids[j],
                flip_count=int(flip_counts[j]),
                n_pairs=int(n_pairs_arr[j]),
                variant_a=a_view,
                variant_b=b_view,
            )
        )

    return WorstCasePanel(
        disabled=False,
        disabled_reason=None,
        cases=tuple(cases),
        k=k,
    )


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _per_eval_flip_counts(
    valid: "np.ndarray", preds: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return ``(flip_counts, n_pairs)`` arrays of length ``n_evals``.

    For each eval column j: among rows where ``valid[i, j]`` is True,
    flip_counts[j] = ``n_true * n_false`` (every (True, False) ordered
    pair is one differing-verdict pair; equivalent to counting unordered
    pairs with differing verdicts). n_pairs[j] = C(n_valid, 2).
    """
    import numpy as np

    n_valid = valid.sum(axis=0).astype(np.int64)
    # ``preds & valid`` zeros out NaN-cells so the per-column sum counts
    # only valid passes.
    n_true = (preds & valid).sum(axis=0).astype(np.int64)
    n_false = n_valid - n_true
    flip_counts = n_true * n_false
    # C(n, 2) = n * (n - 1) / 2 with integer division (n >= 0 always).
    n_pairs = n_valid * (n_valid - 1) // 2
    return flip_counts, n_pairs


def _pick_best_pair(
    arr: "np.ndarray",
    valid: "np.ndarray",
    preds: "np.ndarray",
    variant_ids: tuple[str, ...],
    j: int,
) -> tuple[int, int] | None:
    """Return ``(ai, bi)`` with max score gap among differing-verdict pairs.

    **Invariant the caller must hold:** ``variant_ids`` is in
    canonical_hash lex order (the contract from
    :func:`topofuzz.metrics.read_score_matrix`, handoff decision C of
    step 9). Under that invariant, iterating ``(ai_pos, bi_pos)`` with
    ``ai_pos < bi_pos`` visits ``(variant_ids[ai], variant_ids[bi])``
    keys in ascending lex order, so the FIRST equal-gap pair is
    already the lex-min one — ``>`` (not ``>=``) preserves it without
    needing a separate tiebreak-swap clause (audit RT-BLOCKER-2: the
    swap clause that previously sat here was structurally dead).

    Returns ``None`` when no differing-verdict pair exists at this
    eval (caller should skip — only reachable from a malformed flip-
    counts driver).
    """
    import numpy as np

    valid_idx = np.flatnonzero(valid[:, j])
    if valid_idx.size < 2:
        return None

    best_pair: tuple[int, int] | None = None
    best_gap: float = -1.0
    col_pred = preds[:, j]
    col_score = arr[:, j]

    # The candidate set is O(n_variants^2) per eval but n_variants is
    # bounded at ~50 in the v0.5 plan (BLUEPRINT line 333: ``--variants
    # 8`` default). At K=3 the total is ~7500 pair compares —
    # comfortably under one millisecond.
    for ai_pos in range(valid_idx.size):
        ai = int(valid_idx[ai_pos])
        for bi_pos in range(ai_pos + 1, valid_idx.size):
            bi = int(valid_idx[bi_pos])
            if col_pred[ai] == col_pred[bi]:
                continue
            gap = float(abs(col_score[ai] - col_score[bi]))
            if gap > best_gap:
                best_pair = (ai, bi)
                best_gap = gap
    return best_pair


# ---------------------------------------------------------------------------
# DB read + side view assembly
# ---------------------------------------------------------------------------


def _load_cells_and_rationales(
    run_dir: Path | str,
    run_id: str,
    needed: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Batch-load ``cells.node_outputs_json`` + ``judge_results.rationale``.

    Returns a dict keyed ``(variant_id, eval_id)`` carrying:

    * ``node_outputs``: list of dicts as parsed from
      ``cells.node_outputs_json`` (may be ``[]`` if the row is missing
      or the JSON is malformed — both logged WARNING).
    * ``rationale``: string from ``judge_results.rationale`` (may be
      ``None``).

    A single sqlite connection is opened for the whole batch. Missing
    keys in the returned dict mean the cell row was absent — the
    caller renders the card with empty node-outputs but the score +
    verdict it already had from the matrix.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not needed:
        return out

    db_path = Path(run_dir) / "run.sqlite"
    if not db_path.exists():
        # The score matrix was already loaded successfully, so the DB
        # was present at that point. A missing file here is a real
        # filesystem race; surface a WARNING but don't crash the
        # report path.
        _log.warning(
            "worstcase: run.sqlite missing at %s — rendering empty traces",
            db_path,
        )
        return out

    db = sqlite3.connect(str(db_path))
    # Row name lookup so the SELECT column-order isn't load-bearing
    # (audit RT-MINOR-3).
    db.row_factory = sqlite3.Row
    try:
        # Preflight: probe table existence ONCE per kind so a pre-
        # step-N DB (no ``cells`` / no ``judge_results``) doesn't emit
        # N warnings per missing table (audit RT-MINOR-4).
        has_cells = _table_exists(db, "cells")
        has_judge = _table_exists(db, "judge_results")
        if not has_cells:
            _log.warning(
                "worstcase: run.sqlite at %s has no 'cells' table — "
                "trace excerpts unavailable", db_path,
            )
        if not has_judge:
            _log.warning(
                "worstcase: run.sqlite at %s has no 'judge_results' "
                "table — rationales unavailable", db_path,
            )

        for vid, eid in needed:
            cell_row = None
            if has_cells:
                try:
                    cell_row = db.execute(
                        "SELECT node_outputs_json FROM cells "
                        "WHERE run_id = ? AND variant_id = ? "
                        "AND eval_id = ?",
                        (run_id, vid, eid),
                    ).fetchone()
                except sqlite3.DatabaseError as e:
                    _log.warning(
                        "worstcase: cells read failed for (%s, %s): %s",
                        vid, eid, e,
                    )

            node_dicts: list[dict[str, Any]] = []
            if cell_row is not None and cell_row["node_outputs_json"]:
                try:
                    parsed = json.loads(cell_row["node_outputs_json"])
                    if isinstance(parsed, list):
                        node_dicts = [
                            d for d in parsed if isinstance(d, dict)
                        ]
                except (json.JSONDecodeError, ValueError) as e:
                    # Audit MINOR-2 fix: log the exception type only, not
                    # `e` directly. JSONDecodeError.msg can include the
                    # leading bytes of the offending payload, which on
                    # this code path is user-controlled (cells.
                    # node_outputs_json carries LLM-agent text); centralised
                    # log shippers would otherwise see eval/agent content.
                    _log.warning(
                        "worstcase: node_outputs_json invalid for "
                        "(%s, %s): %s", vid, eid, type(e).__name__,
                    )

            judge_row = None
            if has_judge:
                try:
                    judge_row = db.execute(
                        "SELECT rationale FROM judge_results "
                        "WHERE run_id = ? AND variant_id = ? "
                        "AND eval_id = ?",
                        (run_id, vid, eid),
                    ).fetchone()
                except sqlite3.DatabaseError as e:
                    _log.warning(
                        "worstcase: judge_results read failed for "
                        "(%s, %s): %s", vid, eid, e,
                    )
            rationale = (
                judge_row["rationale"] if judge_row is not None else None
            )
            out[(vid, eid)] = {
                "node_outputs": node_dicts,
                "rationale": rationale,
            }
    finally:
        db.close()
    return out


def _table_exists(db: sqlite3.Connection, name: str) -> bool:
    """Return True iff ``name`` is a table in ``db``'s main schema.

    Wrapped in try/except so a corrupted DB (which can raise
    :class:`sqlite3.OperationalError` on the ``sqlite_master`` SELECT
    itself) degrades gracefully — the caller will see ``False``, log
    a WARNING, and render the report without trace excerpts rather
    than crash the whole report path (audit pass-2 CR-M3).
    """
    try:
        row = db.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = ? LIMIT 1",
            (name,),
        ).fetchone()
    except sqlite3.DatabaseError as e:
        _log.warning(
            "worstcase: sqlite_master probe for %r failed: %s — "
            "assuming table absent", name, e,
        )
        return False
    return row is not None


def _build_ordering_lookup(
    manifest: dict[str, Any] | None,
) -> dict[str, tuple[str, ...]]:
    """Map ``variant_canonical_hash -> (node_id, node_id, ...)``.

    Returns an empty dict on pre-step-15 manifests (no
    ``variant_orderings``) or on shape mismatch — the caller renders
    an empty ordering tuple in that case.
    """
    if not isinstance(manifest, dict):
        return {}
    hashes = manifest.get("variant_canonical_hashes")
    orderings = manifest.get("variant_orderings")
    if not isinstance(hashes, list) or not isinstance(orderings, list):
        return {}
    if len(hashes) != len(orderings):
        _log.warning(
            "worstcase: manifest variant_canonical_hashes (n=%d) and "
            "variant_orderings (n=%d) length mismatch — orderings "
            "skipped", len(hashes), len(orderings),
        )
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for h, order in zip(hashes, orderings):
        if not isinstance(h, str) or not isinstance(order, list):
            continue
        out[h] = tuple(str(n) for n in order)
    return out


def _build_side(
    variant_id: str,
    eval_id: str,
    score: float,
    verdict: bool,
    cell_lookup: dict[tuple[str, str], dict[str, Any]],
    ordering_by_variant: dict[str, tuple[str, ...]],
    *,
    node_text_cap: int,
    rationale_cap: int,
) -> VariantSideView:
    """Assemble one :class:`VariantSideView` from the batched DB rows."""
    row = cell_lookup.get((variant_id, eval_id))
    node_dicts = row["node_outputs"] if row is not None else []
    rationale = row["rationale"] if row is not None else None

    node_outputs = tuple(
        NodeOutputView(
            node_id=str(d.get("node_id", "") or ""),
            role=str(d.get("role", "") or ""),
            response_text=_truncate(
                str(d.get("response_text", "") or ""), node_text_cap
            ),
        )
        for d in node_dicts
    )
    rationale_text = (
        _truncate(rationale, rationale_cap) if rationale else None
    )
    return VariantSideView(
        variant_id=variant_id,
        ordering=ordering_by_variant.get(variant_id, ()),
        score=score,
        verdict=verdict,
        rationale=rationale_text,
        node_outputs=node_outputs,
    )


def _truncate(text: str, cap: int) -> str:
    """Truncate ``text`` to ``cap`` chars with a trailing ellipsis.

    ``cap <= 0`` returns the empty string. Trailing whitespace inside
    the cut window is stripped before the ellipsis so the rendered
    excerpt doesn't end with a stray space + ellipsis.
    """
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + "…"
