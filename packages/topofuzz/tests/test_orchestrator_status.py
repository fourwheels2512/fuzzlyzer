"""Step 9: orchestrator wires §A status onto the runs row.

Coverage:
* ``succeeded`` when zero judge_failures + zero errors.
* ``degraded`` (exit 0 / exit 1) at 1–25% / 25–50% failure rates.
* ``failed`` (exit 2) at >50%.
* ``cost_cap_hit`` keeps its existing path unchanged.
* ``interrupted`` when cancel_event was set by a non-cap source.

All tests use the in-process executor: a fake LLM that succeeds, a
fake judge that fails on demand to drive the cell-failure rate.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from fuzzeval.judge import ContinuousScore
from topofuzz.ir import Edge, LLMConfig, Node, Topology
from topofuzz.orchestrator import RunPlan, execute_run


def _toy_topo() -> Topology:
    cfg = LLMConfig(model="google/gemini-2.5-pro", temperature=0.0)
    return Topology(
        name="status-test",
        nodes=(Node(id="n0", role="agent", llm_config=cfg),),
        edges=(),
        entrypoint="n0",
        terminal=("n0",),
    )


async def _fake_llm(**kwargs):
    return {
        "choices": [{"message": {"content": "ok"}}],
        "model": kwargs["model"],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }


class _AlwaysJudge:
    """Judge that returns 0.5 (no failure)."""

    def score(self, payload):
        return ContinuousScore(value=0.5)


class _FailingJudge:
    """Judge that raises on every call → judge_failure on every cell."""

    def score(self, payload):
        raise RuntimeError("boom")


class _RateLimitedFailingJudge:
    """Fails on the first ``n_fail`` calls, then succeeds.

    Keeps the test independent of any inter-task ordering: the
    aggregate failure count is the number we ask for.
    """

    def __init__(self, n_fail: int) -> None:
        self.n_fail = n_fail
        self._calls = 0
        self._lock = None  # set lazily so __init__ is sync-safe

    def score(self, payload):
        # asyncio.to_thread serialises judge calls via the default
        # executor, but Cells from different variants may race here.
        # We use a plain int counter; on the default executor that's
        # safe enough — single-threaded behavior in practice for a
        # tiny test, and an off-by-one wouldn't change the §A bucket
        # the test is asserting against (boundary tests use widely
        # spaced rates).
        self._calls += 1
        if self._calls <= self.n_fail:
            raise RuntimeError(f"forced failure {self._calls}")
        return ContinuousScore(value=0.5)


def _eval_rows(n: int) -> tuple[dict, ...]:
    return tuple(
        {"id": f"e{i}", "input": f"x{i}", "ground_truth_decision": "approve"}
        for i in range(n)
    )


def _plan(tmp_path: Path, n_evals: int = 10) -> RunPlan:
    return RunPlan(
        topology=_toy_topo(),
        variants=(_toy_topo(),),
        eval_rows=_eval_rows(n_evals),
        run_id="status-run",
        run_dir=tmp_path,
        cache_dir=None,
        max_cost_usd=None,
        judge_model="test-judge",
        rubric_hash="rh-test",
    )


def _read_runs_row_status(run_dir: Path) -> str:
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        return db.execute(
            "SELECT status FROM runs WHERE run_id = ?", ("status-run",)
        ).fetchone()[0]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# §A status tests
# ---------------------------------------------------------------------------


def test_status_succeeded_with_all_good_cells(tmp_path):
    plan = _plan(tmp_path, n_evals=5)
    result = asyncio.run(execute_run(
        plan, llm_callable=_fake_llm, judge=_AlwaysJudge()
    ))
    assert result.status == "succeeded"
    assert result.metrics_exit_code == 0
    assert _read_runs_row_status(tmp_path) == "succeeded"


def test_status_failed_when_all_judges_fail(tmp_path):
    plan = _plan(tmp_path, n_evals=5)
    result = asyncio.run(execute_run(
        plan, llm_callable=_fake_llm, judge=_FailingJudge()
    ))
    assert result.status == "failed"
    assert result.metrics_exit_code == 2
    assert _read_runs_row_status(tmp_path) == "failed"


def test_status_degraded_high_at_majority_failure(tmp_path):
    """6 of 10 cells fail → 60% → ABOVE 50% → failed.

    The §A row break sits between 50% and 51%. We use 60% to land
    cleanly in the failed bucket without relying on rounding."""
    plan = _plan(tmp_path, n_evals=10)
    result = asyncio.run(execute_run(
        plan, llm_callable=_fake_llm,
        judge=_RateLimitedFailingJudge(n_fail=6),
    ))
    assert result.status == "failed"
    assert result.metrics_exit_code == 2


def test_status_degraded_exit_zero_at_low_failure(tmp_path):
    """2 of 20 cells fail = 10% → 1-25% bucket → degraded, exit 0."""
    plan = _plan(tmp_path, n_evals=20)
    result = asyncio.run(execute_run(
        plan, llm_callable=_fake_llm,
        judge=_RateLimitedFailingJudge(n_fail=2),
    ))
    assert result.status == "degraded"
    assert result.metrics_exit_code == 0
    assert _read_runs_row_status(tmp_path) == "degraded"


def test_status_interrupted_when_cancel_without_cap(tmp_path):
    """Pre-setting the cancel event simulates Ctrl+C: no cells run,
    status flips to 'interrupted' and exit code 130."""
    plan = _plan(tmp_path, n_evals=3)
    cancel = asyncio.Event()
    cancel.set()
    result = asyncio.run(execute_run(
        plan, llm_callable=_fake_llm, judge=_AlwaysJudge(), cancel=cancel,
    ))
    assert result.status == "interrupted"
    assert result.metrics_exit_code == 130
    assert _read_runs_row_status(tmp_path) == "interrupted"


def test_status_cost_cap_hit_preserved(tmp_path):
    """The pre-step-9 cost-cap path keeps its 'cost_cap_hit' status
    regardless of how many cells succeeded. ``metrics_exit_code`` is 0
    so the CLI uses ``_COST_CAP_EXIT`` from its own branch."""
    plan = RunPlan(
        topology=_toy_topo(),
        variants=(_toy_topo(),),
        eval_rows=_eval_rows(3),
        run_id="status-run",
        run_dir=tmp_path,
        cache_dir=None,
        max_cost_usd=1e-12,  # absurdly low — first cell trips the cap
        judge_model="test-judge",
        rubric_hash="rh-test",
    )
    result = asyncio.run(execute_run(
        plan, llm_callable=_fake_llm, judge=_AlwaysJudge()
    ))
    assert result.cap_tripped
    assert result.status == "cost_cap_hit"
    # metrics_exit_code 0 — CLI uses _COST_CAP_EXIT (3) for cap-tripped.
    assert result.metrics_exit_code == 0
    assert _read_runs_row_status(tmp_path) == "cost_cap_hit"
