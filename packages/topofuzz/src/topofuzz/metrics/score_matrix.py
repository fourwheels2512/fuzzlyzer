"""Load a 2D score matrix from one run's ``run.sqlite``.

Read pipeline:
    run.sqlite ``judge_results`` × ``runs.rubric_hash`` →
    (n_variants, n_evals) np.ndarray + variant_ids + eval_ids + rubric_hash

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
"""

from __future__ import annotations

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
    """

    matrix: "np.ndarray"
    variant_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]
    rubric_hash: str


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

        # Pull the continuous-judge cells. ORDER BY canonical_hash on the
        # variant axis (handoff decision C) — variant_id IS the canonical
        # hash per :class:`topofuzz.runner.Cell`. eval_id ordering is by
        # rowid (insertion order) which mirrors the orchestrator's plan
        # order for the typical single-writer case.
        cell_rows = db.execute(
            "SELECT variant_id, eval_id, score_value "
            "FROM judge_results "
            "WHERE run_id = ? "
            "  AND judge_kind = 'continuous' "
            "  AND failure_reason IS NULL "
            "  AND score_value IS NOT NULL "
            "ORDER BY variant_id, rowid",
            (run_id,),
        ).fetchall()
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
    for vid, eid, _score in cell_rows:
        if vid not in variant_index:
            variant_index[vid] = len(variant_ids)
            variant_ids.append(vid)
        if eid not in eval_index:
            eval_index[eid] = len(eval_ids)
            eval_ids.append(eid)

    matrix = np.full(
        (len(variant_ids), len(eval_ids)), np.nan, dtype=np.float64
    )
    for vid, eid, score in cell_rows:
        matrix[variant_index[vid], eval_index[eid]] = float(score)

    return ScoreMatrixResult(
        matrix=matrix,
        variant_ids=tuple(variant_ids),
        eval_ids=tuple(eval_ids),
        rubric_hash=rubric_hash,
    )
