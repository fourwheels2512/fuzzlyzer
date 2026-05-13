"""§A error taxonomy + §8 judge-rate gate: → (status, exit code).

The orchestrator calls :func:`classify_run_status` after the writer
drain finishes, then passes the resulting status into
:func:`topofuzz.runner.RunWriter.finalize`. The CLI maps the exit code
into ``raise typer.Exit(code=...)``.

§A thresholds (BLUEPRINT_v5plus.md lines 634-643)::

    0 cells failed   → succeeded, exit 0
    0 < x ≤ 25%      → degraded,  exit 0
    25% < x ≤ 50%    → degraded,  exit 1
    50% < x          → failed,    exit 2

**Boundary semantics (intentional, locked in step 9).** §A spells the
ranges as "1–25% / 25–50% / >50%" — hyphenated ranges are ambiguous at
the shared edge. The §9 audit flagged this. The interpretation locked
in step 9 is:

* The TEXT of row 1 ("1–25% failed → exit 0") naturally reads INCLUSIVE
  of both endpoints; so **25% lands in exit 0**.
* Row 2 ("25–50% → exit 1") therefore starts strictly above 25% and
  is inclusive at 50%; so **50% lands in exit 1**.
* Row 3 (">50%") is written with a strict ``>`` in the blueprint and so
  excludes 50% explicitly.

Implementation uses ``rate <= boundary`` for each row's upper limit.
The chosen boundary policy is pinned by the boundary tests in
``tests/test_metrics_status.py`` — flipping the convention requires
flipping those tests AND updating the table text in the blueprint
(don't do one without the other).

**§8 supplementary rule (BLUEPRINT_v5plus.md line 221).** Independently
of §A's cell-failure rate, ``judge_failure_rate > 5%`` (cell-level —
fraction of cells with ``judge_failure=True``) flags the run as
``degraded``. Under the current §A's "any failure → degraded" reading
the §8 rule is operationally redundant (a single judge_failure cell
already trips §A's > 0 rate); the §8 gate is wired EXPLICITLY here so
that a future §A relaxation (e.g., "small failures OK") cannot quietly
swallow the §8 judge-quality signal. Polish-audit DRIFT-1 fix.

``cost_cap_hit`` (exit 3) and ``interrupted`` (exit 130) are wired
elsewhere — the cost-cap branch in :func:`topofuzz.orchestrator.execute_run`
short-circuits before this function is called.
"""

from __future__ import annotations

# §A thresholds as module constants so a future blueprint revision
# updates one place. The 0.25 / 0.50 split is the §A row break, not a
# rounded heuristic — keep them exact.
_DEGRADED_LOW_FRACTION = 0.0   # > this → degraded
_DEGRADED_HIGH_FRACTION = 0.25  # > this → exit 1
_FAILED_FRACTION = 0.50         # > this → failed (exit 2)

# §8 gate (BLUEPRINT line 221). Cell-level rate threshold — a separate
# axis from §A's combined cell-failure rate. Trips degraded without
# changing the exit code (§A still owns exit-code escalation).
_JUDGE_FAILURE_DEGRADED_FRACTION = 0.05

STATUS_SUCCEEDED = "succeeded"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"


def classify_run_status(
    n_failed_cells: int,
    n_total_cells: int,
    *,
    n_judge_failure_cells: int = 0,
) -> tuple[str, int]:
    """Map cell-failure rate to (status, exit_code) per §A + §8.

    Args:
        n_failed_cells: Cells that failed (judge_failure OR
            ``Cell.error is not None``).  The orchestrator counts both
            classes into one number — §A's "judge_failure or LLM error"
            phrasing intentionally collapses them.
        n_total_cells: Planned cells (``n_variants * n_evals``).  When
            zero (an empty plan), returns ``('succeeded', 0)`` — there
            is no signal to interpret as failure.
        n_judge_failure_cells: Cells with ``judge_failure=True``
            specifically (subset of ``n_failed_cells``). Drives the §8
            supplementary "judge_failure_rate > 5% → degraded" gate.
            Default 0 keeps callers that don't track the breakdown on
            the pre-§8 behavior.

    Returns:
        ``(status, exit_code)`` tuple.  ``status`` is one of
        ``"succeeded"``, ``"degraded"``, or ``"failed"``.

    Raises:
        ValueError: any of ``n_failed_cells`` / ``n_total_cells`` /
            ``n_judge_failure_cells`` < 0; ``n_failed_cells >
            n_total_cells``; or ``n_judge_failure_cells > n_failed_cells``
            (the judge-failure count is a subset of the total-failure
            count by construction).
    """
    if n_failed_cells < 0 or n_total_cells < 0 or n_judge_failure_cells < 0:
        raise ValueError(
            f"counts must be non-negative; got n_failed_cells="
            f"{n_failed_cells!r}, n_total_cells={n_total_cells!r}, "
            f"n_judge_failure_cells={n_judge_failure_cells!r}"
        )
    if n_failed_cells > n_total_cells:
        raise ValueError(
            f"n_failed_cells ({n_failed_cells}) cannot exceed "
            f"n_total_cells ({n_total_cells})"
        )
    if n_judge_failure_cells > n_failed_cells:
        raise ValueError(
            f"n_judge_failure_cells ({n_judge_failure_cells}) cannot "
            f"exceed n_failed_cells ({n_failed_cells}); judge failures "
            "are a subset of the total-failure bucket"
        )

    if n_total_cells == 0:
        # No cells planned (or all cells were resumed-already). Nothing
        # to fail; nothing to degrade.
        return STATUS_SUCCEEDED, 0

    rate = n_failed_cells / n_total_cells
    judge_rate = n_judge_failure_cells / n_total_cells

    # §A primary gate.
    if rate <= _DEGRADED_LOW_FRACTION:
        status, exit_code = STATUS_SUCCEEDED, 0
    elif rate <= _DEGRADED_HIGH_FRACTION:
        status, exit_code = STATUS_DEGRADED, 0
    elif rate <= _FAILED_FRACTION:
        status, exit_code = STATUS_DEGRADED, 1
    else:
        status, exit_code = STATUS_FAILED, 2

    # §8 supplementary gate: a high judge-failure rate downgrades a
    # would-be ``succeeded`` run to ``degraded`` (exit 0). It NEVER
    # escalates an already-degraded or failed run — §A owns escalation.
    if status == STATUS_SUCCEEDED and judge_rate > _JUDGE_FAILURE_DEGRADED_FRACTION:
        return STATUS_DEGRADED, 0

    return status, exit_code
