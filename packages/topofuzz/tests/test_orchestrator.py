"""Tests for ``topofuzz.orchestrator`` (step 6 run loop)."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("diskcache")

from fuzzeval.judge import ContinuousScore  # noqa: E402

from topofuzz.ir import Edge, LLMConfig, Node, Topology  # noqa: E402
from topofuzz.orchestrator import (  # noqa: E402
    RunPlan,
    _ResolverMismatchDedupFilter,
    execute_run,
    install_resolver_mismatch_dedup,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_topology(n: int = 2, *, model: str = "google/gemini-2.5-pro") -> Topology:
    cfg = LLMConfig(model=model, temperature=0.0)
    nodes = tuple(
        Node(id=f"n{i}", role=f"r{i}", llm_config=cfg) for i in range(n)
    )
    edges = tuple(Edge(from_id=f"n{i}", to_id=f"n{i+1}") for i in range(n - 1))
    return Topology(
        name="toy",
        nodes=nodes,
        edges=edges,
        entrypoint="n0",
        terminal=("n0",) if n == 1 else (f"n{n-1}",),
    )


@dataclass
class FakeLLMWithUsage:
    prompt_tokens: int = 200
    completion_tokens: int = 100
    delay_s: float = 0.0
    calls: list[dict] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> dict:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": f"resp#{len(self.calls)}"}}],
            "model": kwargs["model"],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }


def _plan(
    tmp_path: Path,
    *,
    topology: Topology | None = None,
    n_rows: int = 3,
    max_cost_usd: float | None = None,
) -> RunPlan:
    topo = topology or _toy_topology(n=2)
    eval_rows = tuple(
        {"id": f"e{i}", "input": f"input{i}"} for i in range(n_rows)
    )
    return RunPlan(
        topology=topo,
        variants=(topo,),
        eval_rows=eval_rows,
        run_id="test-run-001",
        run_dir=tmp_path / "runs" / "test-run-001",
        cache_dir=tmp_path / ".cache",
        max_cost_usd=max_cost_usd,
        judge_model="google/gemini-2.5-pro",
        semaphore_size=4,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_orchestrator_persists_all_cells(tmp_path):
    plan = _plan(tmp_path, n_rows=3)
    fake = FakeLLMWithUsage()
    result = _run(execute_run(plan, llm_callable=fake))

    assert result.total_cells == 3
    assert result.successful_cells == 3
    assert result.cap_tripped is False
    assert result.total_cost_usd > 0.0

    db = sqlite3.connect(str(plan.run_dir / "run.sqlite"))
    try:
        rows = db.execute(
            "SELECT eval_id, cell_cost_usd, error FROM cells "
            "WHERE run_id = ? ORDER BY eval_id", (plan.run_id,)
        ).fetchall()
    finally:
        db.close()
    assert len(rows) == 3
    for eid, cost, err in rows:
        assert err is None
        assert cost is not None and cost > 0.0


def test_orchestrator_writes_manifest(tmp_path):
    plan = _plan(tmp_path, n_rows=2)
    path = write_manifest(plan, cli_args={"flag": "value"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == plan.run_id
    assert payload["topology_canonical_hash"] == plan.topology.canonical_hash()
    assert payload["n_variants"] == 1
    assert payload["n_eval_rows"] == 2
    assert payload["cli_args"]["flag"] == "value"
    assert payload["max_cost_usd"] is None


def test_orchestrator_writes_traces_jsonl(tmp_path):
    plan = _plan(tmp_path, n_rows=2)
    _run(execute_run(plan, llm_callable=FakeLLMWithUsage()))
    lines = (plan.run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec["run_id"] == plan.run_id
        assert rec["cell_cost_usd"] is not None


# ---------------------------------------------------------------------------
# Cost cap
# ---------------------------------------------------------------------------


def test_orchestrator_cost_cap_trips_and_cancels(tmp_path):
    """2 nodes × 300 tokens × $0.020/1K = $0.012 per priced cell.

    Cap at $0.005 → first persisted cell pushes total over the cap → drain
    sets cancel → workers see cancel at node boundaries and return
    cancelled cells (error='cancelled', no cost). The runner's
    per-node cancel granularity is documented in
    :mod:`topofuzz.runner` — exact counts depend on scheduling, so the
    assertion is on the *outcome shape*: cap tripped + some successful
    cells + some cancelled cells. All `total_cells` rows are persisted
    (cancelled cells still get a row); only `successful_cells` differs.
    """
    # Delay each LLM call enough that workers reliably interleave with
    # the writer drain on a slow CI box. 50 ms × 2 nodes × 10 rows / 4
    # semaphore slots ≈ 250 ms total — comfortably longer than the
    # drain's per-cell latency, so the cap trips before the queue is
    # full (audit m-4 / m-10 — bump from 10 ms to 50 ms).
    plan = _plan(tmp_path, n_rows=10, max_cost_usd=0.005)
    fake = FakeLLMWithUsage(prompt_tokens=200, completion_tokens=100, delay_s=0.05)

    result = _run(execute_run(plan, llm_callable=fake))

    assert result.cap_tripped is True
    assert result.successful_cells >= 1
    assert result.successful_cells < result.total_cells

    # Verify cap-trip actually produced cancelled cells in the DB — the
    # mechanism under test, not just the result shape (audit m-4).
    db = sqlite3.connect(str(plan.run_dir / "run.sqlite"))
    try:
        cancelled = db.execute(
            "SELECT COUNT(*) FROM cells WHERE error = 'cancelled' AND run_id = ?",
            (plan.run_id,),
        ).fetchone()[0]
    finally:
        db.close()
    assert cancelled >= 1


def test_orchestrator_no_cap_means_full_run(tmp_path):
    plan = _plan(tmp_path, n_rows=5, max_cost_usd=None)
    result = _run(execute_run(plan, llm_callable=FakeLLMWithUsage()))
    assert result.cap_tripped is False
    assert result.successful_cells == 5


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_orchestrator_resume_skips_already_completed(tmp_path):
    plan = _plan(tmp_path, n_rows=3)
    fake1 = FakeLLMWithUsage()
    _run(execute_run(plan, llm_callable=fake1))
    assert len(fake1.calls) == 3 * 2  # 3 rows × 2 nodes

    # Second run with same plan / run_id: every cell is already done,
    # so no LLM call should fire.
    fake2 = FakeLLMWithUsage()
    result2 = _run(execute_run(plan, llm_callable=fake2))
    assert len(fake2.calls) == 0
    assert result2.successful_cells == 3
    assert result2.completed_cells == 3


# ---------------------------------------------------------------------------
# Worker exception isolation
# ---------------------------------------------------------------------------


def test_orchestrator_continues_after_worker_exception(tmp_path):
    """One bad eval row must not kill the run.

    Failure trigger is purely prompt-content based — the test is
    asyncio-concurrent, so a counter-based trigger races with other
    workers and produces flaky failures.
    """
    plan = _plan(tmp_path, n_rows=3)

    async def flaky(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        # Fail every call for eval row "e1" (which carries "input1" in
        # its prompt). The runner catches the first node's exception,
        # the worker translates it to an error-marked Cell, and the
        # other rows continue unaffected.
        if "input1" in prompt:
            raise RuntimeError("transient")
        return {
            "choices": [{"message": {"content": "ok"}}],
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    result = _run(execute_run(plan, llm_callable=flaky))

    db = sqlite3.connect(str(plan.run_dir / "run.sqlite"))
    try:
        rows = db.execute(
            "SELECT eval_id, error FROM cells WHERE run_id = ? ORDER BY eval_id",
            (plan.run_id,),
        ).fetchall()
    finally:
        db.close()
    assert {r[0] for r in rows} == {"e0", "e1", "e2"}
    # e1 should be the failed one (with non-None error)
    errors = {r[0]: r[1] for r in rows}
    assert errors["e0"] is None
    assert errors["e2"] is None
    assert errors["e1"] is not None and "transient" in errors["e1"]
    assert result.successful_cells == 2


# ---------------------------------------------------------------------------
# Resolver mismatch dedup filter
# ---------------------------------------------------------------------------


def test_install_resolver_dedup_idempotent():
    f1 = install_resolver_mismatch_dedup()
    try:
        f2 = install_resolver_mismatch_dedup()
        assert f1 is f2
    finally:
        # Cleanup must run even if the assertion fires — otherwise the
        # filter leaks into the next test in this session and any test
        # asserting on resolver-warning log records sees fewer than
        # expected (audit m-11).
        logging.getLogger("topofuzz.runner").removeFilter(f1)


def test_resolver_dedup_filter_drops_duplicates(caplog):
    f = _ResolverMismatchDedupFilter()
    runner_log = logging.getLogger("topofuzz.runner")
    runner_log.addFilter(f)
    try:
        with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
            for _ in range(10):
                runner_log.warning(
                    "model_served_version_resolver returned %r for %r but "
                    "served %r.",
                    "pred-X", "req-Y", "actual-Z",
                )
            runner_log.warning(
                "model_served_version_resolver returned %r for %r but "
                "served %r.",
                "pred-A", "req-B", "actual-C",
            )
    finally:
        runner_log.removeFilter(f)
    # 10 identical + 1 different message → 2 records survive the filter.
    surviving = [
        r for r in caplog.records if "resolver" in r.getMessage().lower()
    ]
    assert len(surviving) == 2


def test_resolver_dedup_filter_does_not_touch_unrelated_messages(caplog):
    f = _ResolverMismatchDedupFilter()
    runner_log = logging.getLogger("topofuzz.runner")
    runner_log.addFilter(f)
    try:
        with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
            for _ in range(3):
                runner_log.warning("some unrelated warning %d", 1)
    finally:
        runner_log.removeFilter(f)
    assert len([r for r in caplog.records if "unrelated" in r.getMessage()]) == 3


# ---------------------------------------------------------------------------
# Manifest captures all variants
# ---------------------------------------------------------------------------


def test_orchestrator_cancels_workers_when_drain_dies(tmp_path):
    """Second-pass audit B-1: if RunWriter.write raises, the drain
    coroutine dies. Without the supervisor fix, workers kept running
    and billing while the drain exception was silently swallowed.

    Monkey-patch RunWriter.write to raise on the FIRST cell. The
    supervisor must:
      1. Cancel pending workers (they should NOT bill further calls)
      2. Re-raise the drain's exception so the caller sees it
    """
    from topofuzz import runner as _runner_mod

    plan = _plan(tmp_path, n_rows=10)

    write_count = 0
    original_write = _runner_mod.RunWriter.write

    def boom_on_first_write(self, cell):
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            raise RuntimeError("simulated drain failure (e.g. SQLite locked)")
        return original_write(self, cell)

    _runner_mod.RunWriter.write = boom_on_first_write
    fake = FakeLLMWithUsage(prompt_tokens=10, completion_tokens=5, delay_s=0.02)
    try:
        with pytest.raises(RuntimeError, match="simulated drain failure"):
            _run(execute_run(plan, llm_callable=fake))
    finally:
        _runner_mod.RunWriter.write = original_write

    # Critical assertion: the drain's death must have stopped further
    # billing. Without the supervisor fix, fake.calls grew to ~20
    # (10 rows × 2 nodes). With the supervisor, workers cancel after
    # the drain dies — at most a few in-flight calls complete before
    # cancellation propagates. Worst-case bound: semaphore_size × nodes
    # already-acquired = 4 × 2 = 8. Anything less than the full 20
    # demonstrates the supervisor worked.
    assert len(fake.calls) < 20


def test_manifest_records_variant_hashes(tmp_path):
    topo_a = _toy_topology(n=2, model="google/gemini-2.5-pro")
    topo_b = _toy_topology(n=2, model="google/gemini-2.5-flash")
    plan = RunPlan(
        topology=topo_a,
        variants=(topo_a, topo_b),
        eval_rows=({"id": "e1", "input": "x"},),
        run_id="rid",
        run_dir=tmp_path / "rid",
        cache_dir=None,
        max_cost_usd=None,
        judge_model="m",
    )
    path = write_manifest(plan, cli_args={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["variant_canonical_hashes"] == [
        topo_a.canonical_hash(),
        topo_b.canonical_hash(),
    ]
    assert payload["eval_ids"] == ["e1"]
