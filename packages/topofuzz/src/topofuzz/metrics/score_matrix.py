# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Load a 2D score matrix from one run's ``run.sqlite``.

Read pipeline:
    run.sqlite ``judge_results`` × ``runs.rubric_hash`` →
    (n_variants, n_evals) np.ndarray + variant_ids + eval_ids + rubric_hash
    [+ optional (n_vendors, n_variants, n_evals) per-vendor jury matrix]

Variants ordered by ``canonical_hash`` lex (handoff decision C). Evals
ordered by SQL insertion order (rowid). Both are deterministic across
re-runs of the same plan; the canonical_hash ordering specifically
survives changes to the orchestrator's wave scheduling, so step-10a's
worst-case viewer + golden-file tests can rely on it.

Cells with a non-NULL ``failure_reason`` or with
``judge_kind != 'continuous'`` are excluded from the matrix.  Step 9
only consumes continuous scores — §9 metric #1 (variance) is undefined
for binary scores; the binary path lives in :mod:`topofuzz.metrics.kappa`
which binarises continuous scores via :func:`fuzzeval.judge.binary_from_continuous`.

Mixed-rubric refusal (handoff decision D): if a single ``run_dir``
contains rows from multiple ``runs.rubric_hash`` values (only possible
if a caller manually mutated the DB), the loader raises
:class:`ValueError`.  An empty ``rubric_hash`` (legacy v3 row migrated
forward, or an explicit empty-string from a no-judge run) is treated as
"rubric unknown" and refused unless ``allow_empty_rubric_hash=True`` —
which exists for forensic tooling, not the headline report path.

Per-vendor jury plumbing (F3 MAJOR closure, 2026-05-16): when every
continuous cell has a parseable ``judge_results.votes_json`` payload
with a consistent vendor set, the loader also exposes a
``(n_vendors, n_variants, n_evals)`` per-vendor matrix on the result
so :mod:`topofuzz.metrics.voo` can compute §9 metric #1 over continuous
per-vendor scores instead of the Bernoulli-bounded majority. Legacy v5
DBs (no column), single-judge runs (column NULL), and mixed-protocol
runs (inconsistent vendor sets) keep ``per_vendor_matrix=None`` — the
existing 2D ``matrix`` always populates and stays byte-identical.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ScoreMatrixResult:
    """Output of :func:`read_score_matrix`.

    ``matrix`` is ``(n_variants, n_evals)`` float64 with NaN entries
    where a cell's judge row was missing or filtered out.  Downstream
    metrics that don't tolerate NaN (the bootstrap, κ) MUST check
    ``np.isnan(matrix).any()`` and report degenerate; the loader does
    not silently impute.

    ``variant_ids`` is in canonical_hash lex order; ``eval_ids`` in
    SQL insertion order.

    ``per_vendor_matrix`` is the F3 jury-aware companion: a
    ``(n_vendors, n_variants, n_evals)`` float64 array of continuous
    per-vendor judge scores (NaN where a vendor's vote was missing for
    that cell). Populated only when EVERY kept continuous cell has a
    parseable ``votes_json`` payload AND the vendor set is consistent
    across cells; ``None`` otherwise (SingleJudge runs, legacy v5 DBs
    without the column, or jury runs with mixed-vendor cells).
    ``vendor_ids`` is sorted lex so the leading axis is deterministic;
    empty tuple when ``per_vendor_matrix is None``.

    Axis convention: the per-vendor slab agrees with ``matrix`` on the
    (variant_ids, eval_ids) axes, so ``per_vendor_matrix.mean(axis=0)``
    is NOT the majority — it is the continuous per-vendor average. The
    majority lives in ``matrix`` exactly as before.

    ``class_matrix`` (2026-05-30, v7) is the 3-state companion: a
    ``(n_variants, n_evals)`` int8 array aligned cell-for-cell with
    ``matrix``, coding ``-1`` missing, ``0`` deny, ``1`` approve, ``2``
    conditional. Populated only when the DB carries the v7
    ``verdict_class`` column AND at least one cell has an explicit class;
    ``None`` otherwise (pre-v7 DBs, or runs where no judgement set a
    class). When ``None``, the flip metric stays on the legacy
    score-threshold path — byte-identical to before. When present,
    ``compute_verdict_flip_rate(class_matrix=...)`` applies the clear-flip
    gate (APPROVE-vs-DENY only; conditional tracked separately).
    """

    matrix: "np.ndarray"
    variant_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]
    rubric_hash: str
    per_vendor_matrix: "np.ndarray | None" = None
    vendor_ids: tuple[str, ...] = ()
    class_matrix: "np.ndarray | None" = None


def read_score_matrix(
    run_dir: Path | str,
    *,
    run_id: str | None = None,
    allow_empty_rubric_hash: bool = False,
) -> ScoreMatrixResult:
    """Build a continuous-score matrix from one run's ``run.sqlite``.

    Args:
        run_dir: Path to the run directory containing ``run.sqlite``.
        run_id: If supplied, restrict the query to this run_id. When
            ``None`` and the DB holds exactly one ``runs`` row the
            single run is used; multi-run DBs (not produced by v0.5
            today but possible if a user copies multiple ``run.sqlite``
            payloads into one file) require an explicit ``run_id``.
        allow_empty_rubric_hash: When ``True``, accept rows with an
            empty ``rubric_hash`` (the v3-migrated default).  The
            headline report path passes ``False``; forensic tooling that
            wants to look at legacy data passes ``True``.

    Returns:
        :class:`ScoreMatrixResult`. ``matrix.shape == (n_variants,
        n_evals)``. Missing / failed cells appear as ``np.nan``.

    Raises:
        FileNotFoundError: ``run.sqlite`` does not exist under ``run_dir``.
        ValueError: multiple ``rubric_hash`` values present; or the
            ``rubric_hash`` is empty and ``allow_empty_rubric_hash`` is
            False; or ``run_id`` was not supplied and the DB holds
            multiple runs; or ``run_id`` was supplied but not found.
        ImportError: numpy is not installed (install the topofuzz
            ``[stats]`` extra).
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "topofuzz.metrics.read_score_matrix requires numpy "
            "(install via `pip install topofuzz[stats]`)."
        ) from e

    db_path = Path(run_dir) / "run.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(
            f"run.sqlite not found under run_dir={run_dir!r} "
            f"(expected at {db_path!s})"
        )

    db = sqlite3.connect(str(db_path))
    try:
        # Resolve which run_id to read. The v0.5 contract is one run
        # per run.sqlite; the multi-run branch exists so forensic
        # tooling can split apart a manually-merged file without
        # silently picking one.
        rows = db.execute(
            "SELECT run_id, rubric_hash FROM runs"
        ).fetchall()
        if not rows:
            raise ValueError(
                f"run.sqlite at {db_path!s} has no rows in the runs table."
            )
        runs_by_id = {rid: rhash for rid, rhash in rows}
        if run_id is None:
            if len(runs_by_id) != 1:
                raise ValueError(
                    f"run.sqlite at {db_path!s} contains {len(runs_by_id)} "
                    f"runs; pass run_id= to disambiguate. Found: "
                    f"{sorted(runs_by_id)!r}"
                )
            (run_id, rubric_hash), = runs_by_id.items()
        else:
            if run_id not in runs_by_id:
                raise ValueError(
                    f"run_id={run_id!r} not found in {db_path!s}. "
                    f"Available: {sorted(runs_by_id)!r}"
                )
            rubric_hash = runs_by_id[run_id]

        # Reject empty rubric_hash early. The v3-migrated legacy value is
        # the empty string; any caller asking for headline metrics on a
        # rubric-unknown run is almost certainly making a mistake.
        if not rubric_hash and not allow_empty_rubric_hash:
            raise ValueError(
                f"run_id={run_id!r} has empty rubric_hash (likely a v3-"
                f"migrated legacy run, or a run with no judge). "
                "Headline metrics need a known rubric to be meaningful; "
                "pass allow_empty_rubric_hash=True for forensic loads."
            )

        # Probe the schema for the v6 ``votes_json`` column. Legacy v5
        # DBs predate the column; we treat them as "no per-vendor data"
        # rather than failing the load. The probe is one PRAGMA, not a
        # try/except on the SELECT — that would catch real runtime errors.
        judge_cols = {
            row[1]
            for row in db.execute(
                "PRAGMA table_info(judge_results)"
            ).fetchall()
        }
        has_votes_json = "votes_json" in judge_cols
        # v7 ``verdict_class`` probe — same conditional-read shape as
        # votes_json so a pre-v7 DB still loads (class_matrix stays None →
        # flip metric uses the legacy threshold path).
        has_verdict_class = "verdict_class" in judge_cols

        # Pull the continuous-judge cells. ORDER BY canonical_hash on the
        # variant axis (handoff decision C) — variant_id IS the canonical
        # hash per :class:`topofuzz.runner.Cell`. eval_id ordering is by
        # rowid (insertion order) which mirrors the orchestrator's plan
        # order for the typical single-writer case.
        if has_votes_json:
            cell_rows = db.execute(
                "SELECT variant_id, eval_id, score_value, votes_json "
                "FROM judge_results "
                "WHERE run_id = ? "
                "  AND judge_kind = 'continuous' "
                "  AND failure_reason IS NULL "
                "  AND score_value IS NOT NULL "
                "ORDER BY variant_id, rowid",
                (run_id,),
            ).fetchall()
        else:
            cell_rows = [
                (vid, eid, sv, None)
                for vid, eid, sv in db.execute(
                    "SELECT variant_id, eval_id, score_value "
                    "FROM judge_results "
                    "WHERE run_id = ? "
                    "  AND judge_kind = 'continuous' "
                    "  AND failure_reason IS NULL "
                    "  AND score_value IS NOT NULL "
                    "ORDER BY variant_id, rowid",
                    (run_id,),
                ).fetchall()
            ]
        if has_verdict_class:
            vclass_rows = db.execute(
                "SELECT variant_id, eval_id, verdict_class "
                "FROM judge_results "
                "WHERE run_id = ? "
                "  AND judge_kind = 'continuous' "
                "  AND failure_reason IS NULL "
                "  AND score_value IS NOT NULL "
                "ORDER BY variant_id, rowid",
                (run_id,),
            ).fetchall()
        else:
            vclass_rows = []
    finally:
        db.close()

    if not cell_rows:
        return ScoreMatrixResult(
            matrix=np.empty((0, 0), dtype=np.float64),
            variant_ids=(),
            eval_ids=(),
            rubric_hash=rubric_hash,
        )

    # Discover variant + eval axes from the rows. Deterministic ordering:
    # variants by canonical_hash lex (already guaranteed by SQL); evals
    # by first-seen rowid (preserves insertion order across variants
    # because the orchestrator schedules round-robin across evals for
    # a given variant, then moves to the next variant).
    variant_ids: list[str] = []
    eval_ids: list[str] = []
    variant_index: dict[str, int] = {}
    eval_index: dict[str, int] = {}
    for vid, eid, _score, _vj in cell_rows:
        if vid not in variant_index:
            variant_index[vid] = len(variant_ids)
            variant_ids.append(vid)
        if eid not in eval_index:
            eval_index[eid] = len(eval_ids)
            eval_ids.append(eid)

    matrix = np.full(
        (len(variant_ids), len(eval_ids)), np.nan, dtype=np.float64
    )
    for vid, eid, score, _vj in cell_rows:
        matrix[variant_index[vid], eval_index[eid]] = float(score)

    per_vendor_matrix, vendor_ids = _try_build_per_vendor_matrix(
        cell_rows,
        variant_index=variant_index,
        eval_index=eval_index,
    )

    # Build the 3-state class_matrix aligned to ``matrix``. Codes: -1
    # missing, 0 deny, 1 approve, 2 conditional. An explicit class wins;
    # a "" / "dryrun" / NULL class on a scored cell falls back to the 0.5
    # threshold (so a numeric-only run gets clear 0/1 codes and the flip
    # metric matches the legacy path). Returns None unless at least one
    # cell carried an EXPLICIT class — that keeps pre-3-state runs on the
    # untouched threshold path (None → legacy in compute_verdict_flip_rate).
    _CODE = {"approve": 1, "deny": 0, "conditional": 2}
    class_matrix: "np.ndarray | None" = None
    if vclass_rows and any(
        (vc or "") in _CODE for _v, _e, vc in vclass_rows
    ):
        class_matrix = np.full(
            (len(variant_ids), len(eval_ids)), -1, dtype=np.int8
        )
        for vid, eid, vclass in vclass_rows:
            if vid not in variant_index or eid not in eval_index:
                continue
            i, j = variant_index[vid], eval_index[eid]
            code = _CODE.get(vclass or "")
            if code is None:
                # "" / "dryrun" / unknown on a scored cell → threshold the
                # continuous value (matches the legacy binarisation).
                val = matrix[i, j]
                code = 1 if (val == val and val >= 0.5) else 0  # NaN-safe
            class_matrix[i, j] = code

    return ScoreMatrixResult(
        matrix=matrix,
        variant_ids=tuple(variant_ids),
        eval_ids=tuple(eval_ids),
        rubric_hash=rubric_hash,
        per_vendor_matrix=per_vendor_matrix,
        vendor_ids=vendor_ids,
        class_matrix=class_matrix,
    )


# ---------------------------------------------------------------------------
# F3 MAJOR (2026-05-16): per-vendor jury matrix extraction.
# ---------------------------------------------------------------------------


def _try_build_per_vendor_matrix(
    cell_rows: list,
    *,
    variant_index: dict,
    eval_index: dict,
) -> "tuple[np.ndarray | None, tuple[str, ...]]":
    """Return ``(per_vendor_matrix, vendor_ids)`` or ``(None, ())``.

    Populated only when EVERY supplied cell has a parseable
    ``votes_json`` payload AND the vendor set is consistent across cells
    (modulo missing-vote NaN for an individual cell). Anything else —
    mixed protocol (some cells with votes, some without), legacy v5 DB
    (every cell's ``votes_json`` is ``None``), malformed JSON, vendor
    drift mid-run — degrades silently to ``(None, ())``. The caller
    still gets the ordinary 2D ``matrix``; the jury-aware code path in
    :mod:`topofuzz.metrics.voo` just stays on the existing majority
    path.

    Vendor axis is sorted lex so the leading axis is deterministic
    across re-runs (matches the variant-axis canonical_hash convention).

    A WARNING is logged only when votes_json data exists but is
    structurally inconsistent (mixed protocol or malformed) — the
    operator should see the signal. Pure single-judge / legacy paths
    log nothing (they're the common case).
    """
    try:
        import numpy as np
    except ImportError:
        # Mirrors the caller's numpy gate — keep behavior identical
        # when numpy is unavailable (the caller has already imported).
        return None, ()

    log = logging.getLogger(__name__)

    # Quick scan: do we have ANY votes_json payloads at all?
    any_votes = any(vj for *_h, vj in cell_rows)
    if not any_votes:
        # Legacy v5 DB or pure single-judge run — silent degrade.
        return None, ()

    all_votes = all(vj for *_h, vj in cell_rows)
    if not all_votes:
        # Mixed-protocol run: some cells have votes, some don't. Refuse
        # to build a half-populated matrix (the F3 audit's "mixed-judge
        # cells degrades silently to None" contract); a downstream
        # consumer should not see ambiguity here.
        log.warning(
            "score_matrix: %d/%d continuous cells have votes_json; "
            "per_vendor_matrix skipped (mixed-protocol run)",
            sum(1 for *_h, vj in cell_rows if vj),
            len(cell_rows),
        )
        return None, ()

    # Parse vote payloads and collect the union of vendor names. Refuse
    # on malformed JSON or non-string/non-numeric entries — a corrupted
    # column should surface, not silently fall through to a wrong-shape
    # matrix.
    parsed: list[tuple[str, str, dict[str, float]]] = []
    vendor_set: set[str] = set()
    for vid, eid, _score, vj in cell_rows:
        try:
            votes = json.loads(vj)
        except (TypeError, ValueError) as e:
            log.warning(
                "score_matrix: votes_json parse failed for "
                "variant=%s eval=%s; per_vendor_matrix skipped (%s)",
                vid, eid, e,
            )
            return None, ()
        if not isinstance(votes, list) or not votes:
            log.warning(
                "score_matrix: votes_json shape invalid for "
                "variant=%s eval=%s; per_vendor_matrix skipped",
                vid, eid,
            )
            return None, ()
        scores_by_vendor: dict[str, float] = {}
        for entry in votes:
            if not isinstance(entry, dict):
                log.warning(
                    "score_matrix: votes_json entry not a dict for "
                    "variant=%s eval=%s; per_vendor_matrix skipped",
                    vid, eid,
                )
                return None, ()
            v = entry.get("vendor")
            s = entry.get("score")
            # Audit M3 + m5: validate vendor name and score value.
            # ``bool`` is a subclass of ``int`` in Python — accepting it
            # would silently coerce True→1.0; refuse explicitly. Vendor
            # name length cap is defense-in-depth against a hand-edited
            # DB carrying a multi-MB string.
            if (
                not isinstance(v, str)
                or not v
                or len(v) > 256
                or isinstance(s, bool)
                or not isinstance(s, (int, float))
            ):
                log.warning(
                    "score_matrix: votes_json missing/invalid "
                    "vendor/score for variant=%s eval=%s; "
                    "per_vendor_matrix skipped",
                    vid, eid,
                )
                return None, ()
            s_float = float(s)
            # ``inf`` / ``NaN`` / out-of-range scores would poison the
            # downstream jury VOO (entire vendor's slab inflates to inf
            # or propagates NaN). Per the JuryJudge contract, score is
            # always in [0, 1]; a value outside the range signals a
            # corrupted / hand-edited / cross-project DB.
            if not math.isfinite(s_float) or not 0.0 <= s_float <= 1.0:
                log.warning(
                    "score_matrix: votes_json score=%r out of [0,1] for "
                    "variant=%s eval=%s; per_vendor_matrix skipped",
                    s_float, vid, eid,
                )
                return None, ()
            # Duplicate vendor in one cell's votes is ambiguous — refuse.
            if v in scores_by_vendor:
                log.warning(
                    "score_matrix: duplicate vendor %r in votes_json "
                    "for variant=%s eval=%s; per_vendor_matrix skipped",
                    v, vid, eid,
                )
                return None, ()
            scores_by_vendor[v] = s_float
        parsed.append((vid, eid, scores_by_vendor))
        vendor_set.update(scores_by_vendor.keys())

    # Per-cell vendor-roster subset is tolerated (mirrors the 2D
    # ``matrix``'s NaN-on-missing-cell semantics): a partial roster comes
    # from a per-vendor judge failure inside the jury (e.g. one vendor
    # rate-limited or returned an unparseable score), but the cell's
    # ``score_value`` majority is still meaningful and the other vendors'
    # votes are real measurements. The slab carries NaN for the missing
    # (vendor, variant, eval) entry and :func:`variance_across_orderings_jury`
    # handles it via ``np.nanvar`` / ``np.nanmean``. We log a one-line
    # summary when ANY cell has a partial roster so the operator sees the
    # signal — but we do not refuse, because refusing throws away the rest
    # of the per-vendor data the run did successfully collect.
    vendor_ids = tuple(sorted(vendor_set))
    partial_roster_count = sum(
        1
        for _vid, _eid, scores_by_vendor in parsed
        if set(scores_by_vendor.keys()) != vendor_set
    )
    if partial_roster_count:
        log.warning(
            "score_matrix: %d/%d jury cells have a partial vendor "
            "roster (run vendors=%r); per_vendor_matrix populated with "
            "NaN for missing (vendor, variant, eval) entries — jury "
            "VOO uses nanvar/nanmean so the missing votes are "
            "structurally dropped, not silently imputed.",
            partial_roster_count,
            len(parsed),
            sorted(vendor_set),
        )

    n_vendors = len(vendor_ids)
    n_variants = len(variant_index)
    n_evals = len(eval_index)
    pv = np.full(
        (n_vendors, n_variants, n_evals), np.nan, dtype=np.float64
    )
    vendor_index = {v: j for j, v in enumerate(vendor_ids)}
    for vid, eid, scores_by_vendor in parsed:
        vi = variant_index[vid]
        ei = eval_index[eid]
        for v, s in scores_by_vendor.items():
            pv[vendor_index[v], vi, ei] = s

    return pv, vendor_ids
