"""§A error taxonomy: cell-failure-rate → (status, exit code).

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

STATUS_SUCCEEDED = "succeeded"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"


def classify_run_status(
    n_failed_cells: int, n_total_cells: int
) -> tuple[str, int]:
    """Map cell-failure rate to (status, exit_code) per §A.

    Args:
        n_failed_cells: Cells that failed (judge_failure OR
            ``Cell.error is not None``).  The orchestrator counts both
            classes into one number — §A's "judge_failure or LLM error"
            phrasing intentionally collapses them.
        n_total_cells: Planned cells (``n_variants * n_evals``).  When
            zero (an empty plan), returns ``('succeeded', 0)`` — there
            is no signal to interpret as failure.

    Returns:
        ``(status, exit_code)`` tuple.  ``status`` is one of
        ``"succeeded"``, ``"degraded"``, or ``"failed"``.

    Raises:
        ValueError: ``n_failed_cells < 0`` or ``n_total_cells < 0`` or
            ``n_failed_cells > n_total_cells``.
    """
    if n_failed_cells < 0 or n_total_cells < 0:
        raise ValueError(
            f"counts must be non-negative; got n_failed_cells="
            f"{n_failed_cells!r}, n_total_cells={n_total_cells!r}"
        )
    if n_failed_cells > n_total_cells:
        raise ValueError(
            f"n_failed_cells ({n_failed_cells}) cannot exceed "
            f"n_total_cells ({n_total_cells})"
        )

    if n_total_cells == 0:
        # No cells planned (or all cells were resumed-already). Nothing
        # to fail; nothing to degrade.
        return STATUS_SUCCEEDED, 0

    rate = n_failed_cells / n_total_cells
    if rate <= _DEGRADED_LOW_FRACTION:
        return STATUS_SUCCEEDED, 0
    if rate <= _DEGRADED_HIGH_FRACTION:
        return STATUS_DEGRADED, 0
    if rate <= _FAILED_FRACTION:
        return STATUS_DEGRADED, 1
    return STATUS_FAILED, 2
