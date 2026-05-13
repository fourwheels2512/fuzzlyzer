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


# ---------------------------------------------------------------------------
# Step 7: rubric_hash + judge_prompt_renderer threading
# ---------------------------------------------------------------------------


class _CapturingJudge:
    """Sync judge that records what it was asked to score.

    Returns ContinuousScore(0.7) for every payload so the runner's
    happy-path code paths exercise the cache (cells with same
    response_text would hit on second call).
    """

    def __init__(self):
        self.payloads: list[dict] = []

    def score(self, payload: dict) -> ContinuousScore:
        self.payloads.append(payload)
        return ContinuousScore(value=0.7)


def test_run_plan_rubric_hash_defaults_to_empty():
    topo = _toy_topology(n=1)
    plan = RunPlan(
        topology=topo,
        variants=(topo,),
        eval_rows=({"id": "e", "input": "x"},),
        run_id="r",
        run_dir=Path("/tmp/r"),
        cache_dir=None,
        max_cost_usd=None,
        judge_model="m",
    )
    assert plan.rubric_hash == ""


def test_run_plan_rubric_hash_round_trip():
    topo = _toy_topology(n=1)
    plan = RunPlan(
        topology=topo,
        variants=(topo,),
        eval_rows=({"id": "e", "input": "x"},),
        run_id="r",
        run_dir=Path("/tmp/r"),
        cache_dir=None,
        max_cost_usd=None,
        judge_model="m",
        rubric_hash="deadbeef" * 8,
    )
    assert plan.rubric_hash == "deadbeef" * 8


def test_orchestrator_threads_judge_and_records_result(tmp_path):
    plan = _plan(tmp_path, n_rows=2)
    fake = FakeLLMWithUsage()
    judge = _CapturingJudge()
    _run(execute_run(plan, llm_callable=fake, judge=judge))

    # Judge was invoked for each cell.
    assert len(judge.payloads) == 2
    # Every payload has the §17a-contract keys topofuzz uses.
    for p in judge.payloads:
        assert set(p.keys()) == {"variant_id", "eval_row", "agent_output"}

    # Cell rows record the judge result via judge_results table.
    db = sqlite3.connect(str(plan.run_dir / "run.sqlite"))
    try:
        rows = db.execute(
            "SELECT judge_kind, score_value FROM judge_results "
            "WHERE run_id = ? ORDER BY eval_id", (plan.run_id,)
        ).fetchall()
    finally:
        db.close()
    assert {(k, v) for k, v in rows} == {("continuous", 0.7)}


def test_orchestrator_uses_renderer_output_for_judge_cache_key(tmp_path):
    """Two cells whose RENDERED prompts are byte-identical must hit
    the JudgeCache after the first one writes — even if their underlying
    payloads differ. This is the contract the renderer-keying refinement
    is here to enforce: the cache key is the prompt the judge saw.

    Setup:
      * Two distinct (variant, eval_row) cells with different ``id`` and
        different ``agent_output`` reaching the judge.
      * A renderer that DELIBERATELY collapses both cells onto the same
        rendered text (constant return).
      * The first call writes ContinuousScore(0.7) to JudgeCache; the
        second call MUST hit cache and NOT invoke judge.score again.

    Failure mode this guards against: if the runner reverted to the
    eval_row hash for the prompt_hash slot, the two cells would have
    different cache keys and judge.score would be called twice — the
    assertion ``len(payloads) == 1`` would fail.
    """
    pytest.importorskip("diskcache")

    # Two variants of the same topology that hash-distinctly so the
    # `(variant, eval)` keys differ; the eval_rows also differ on `id`.
    # Only the rendered prompt collapses to the same string.
    topo_a = _toy_topology(n=1, model="google/gemini-2.5-pro")
    topo_b = _toy_topology(n=1, model="google/gemini-2.5-flash")
    eval_rows = (
        {"id": "e1", "input": "alpha"},
        {"id": "e2", "input": "beta"},
    )
    plan = RunPlan(
        topology=topo_a,
        variants=(topo_a, topo_b),
        eval_rows=eval_rows,
        run_id="renderer-thread-001",
        run_dir=tmp_path / "renderer-thread-001",
        cache_dir=tmp_path / ".cache",
        max_cost_usd=None,
        judge_model="judge-m",
        rubric_hash="rubric-X",
    )

    judge = _CapturingJudge()

    # The judge cache key has three other slots beyond prompt_hash:
    # response_hash, rubric_hash, judge_model_version. We force same
    # response_hash by giving every LLM call the same response text.
    class _ConstResponseLLM:
        def __init__(self):
            self.calls: list[dict] = []

        async def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "choices": [{"message": {"content": "FIXED_AGENT_OUTPUT"}}],
                "model": kwargs["model"],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }

    fake = _ConstResponseLLM()

    # Renderer returns a constant string regardless of payload — so any
    # two cells collapse to the same prompt_hash. With same rubric_hash,
    # same judge_model_version, same response_hash, AND now same
    # prompt_hash, the JudgeCache row IS the same → second call hits.
    def renderer(payload: dict) -> str:
        return "FIXED_RENDERED_PROMPT"

    _run(
        execute_run(
            plan,
            llm_callable=fake,
            judge=judge,
            judge_prompt_renderer=renderer,
        )
    )

    # 4 cells total (2 variants × 2 eval rows), 4 agent rollouts, but
    # only ONE judge invocation — the other three hit JudgeCache.
    assert len(fake.calls) == 4
    assert len(judge.payloads) == 1, (
        f"Expected JudgeCache to short-circuit 3 of 4 calls "
        f"(same rendered prompt → same cache key); "
        f"got {len(judge.payloads)} judge invocations."
    )


def test_orchestrator_marks_cell_judge_failure_when_renderer_raises(tmp_path):
    plan = _plan(tmp_path, n_rows=1)

    def bad_renderer(payload: dict) -> str:
        raise RuntimeError("renderer broke")

    judge = _CapturingJudge()
    _run(
        execute_run(
            plan,
            llm_callable=FakeLLMWithUsage(),
            judge=judge,
            judge_prompt_renderer=bad_renderer,
        )
    )

    db = sqlite3.connect(str(plan.run_dir / "run.sqlite"))
    try:
        kind, reason = db.execute(
            "SELECT judge_kind, failure_reason FROM judge_results "
            "WHERE run_id = ?", (plan.run_id,)
        ).fetchone()
    finally:
        db.close()
    assert kind == "failure"
    assert "renderer broke" in reason
    # Judge.score must NOT be called when the renderer raised.
    assert len(judge.payloads) == 0


def test_orchestrator_marks_cell_judge_failure_when_renderer_returns_non_str(tmp_path):
    plan = _plan(tmp_path, n_rows=1)

    judge = _CapturingJudge()
    _run(
        execute_run(
            plan,
            llm_callable=FakeLLMWithUsage(),
            judge=judge,
            judge_prompt_renderer=lambda p: 42,  # type: ignore[arg-type,return-value]
        )
    )

    db = sqlite3.connect(str(plan.run_dir / "run.sqlite"))
    try:
        kind, reason = db.execute(
            "SELECT judge_kind, failure_reason FROM judge_results "
            "WHERE run_id = ?", (plan.run_id,)
        ).fetchone()
    finally:
        db.close()
    assert kind == "failure"
    assert "int" in reason  # the offending type name appears in the message


def test_orchestrator_falls_back_to_eval_row_hash_without_renderer(tmp_path, caplog):
    """Step-5 callers (judge but no renderer) must still get cells judged.

    Backwards-compat check — the legacy path still works AND emits the
    audit-M6 deprecation warning so callers see they're on the stopgap.
    """
    plan = _plan(tmp_path, n_rows=2)
    judge = _CapturingJudge()
    with caplog.at_level(logging.WARNING, logger="topofuzz.orchestrator"):
        _run(execute_run(plan, llm_callable=FakeLLMWithUsage(), judge=judge))
    db = sqlite3.connect(str(plan.run_dir / "run.sqlite"))
    try:
        kinds = [
            r[0] for r in db.execute(
                "SELECT judge_kind FROM judge_results WHERE run_id = ?",
                (plan.run_id,),
            ).fetchall()
        ]
    finally:
        db.close()
    assert kinds == ["continuous", "continuous"]
    # M6: caller is told they're on the legacy path.
    assert any(
        "judge_prompt_renderer" in r.getMessage() and "stopgap" in r.getMessage()
        for r in caplog.records
    ), "expected audit-M6 warning about missing judge_prompt_renderer"


def test_orchestrator_does_not_warn_when_renderer_supplied(tmp_path, caplog):
    plan = _plan(tmp_path, n_rows=1)
    judge = _CapturingJudge()
    with caplog.at_level(logging.WARNING, logger="topofuzz.orchestrator"):
        _run(
            execute_run(
                plan,
                llm_callable=FakeLLMWithUsage(),
                judge=judge,
                judge_prompt_renderer=lambda p: "rendered",
            )
        )
    assert not any(
        "judge_prompt_renderer" in r.getMessage() and "stopgap" in r.getMessage()
        for r in caplog.records
    )


def test_orchestrator_renderer_mutation_does_not_leak_to_judge(tmp_path):
    """A buggy renderer that mutates payload['eval_row'] in place must
    not affect what the judge receives (audit MINOR: renderer mutation).
    """
    plan = _plan(tmp_path, n_rows=1)
    judge = _CapturingJudge()

    def mutating_renderer(payload: dict) -> str:
        payload["eval_row"]["__poisoned__"] = "yes"
        payload["eval_row"].pop("input", None)
        return "rendered"

    _run(
        execute_run(
            plan,
            llm_callable=FakeLLMWithUsage(),
            judge=judge,
            judge_prompt_renderer=mutating_renderer,
        )
    )
    # Judge sees a clean payload (the deepcopy in the runner protects it).
    assert len(judge.payloads) == 1
    received = judge.payloads[0]["eval_row"]
    assert "__poisoned__" not in received
    assert received.get("input") == "input0"


def test_manifest_records_rubric_hash_at_top_level(tmp_path):
    """Manifest carries reproducibility-relevant state at the top level.

    rubric_hash sits alongside topology_canonical_hash and
    variant_canonical_hashes so post-mortem tooling and step 11's
    report HTML can read it without parsing cli_args. Empty string is
    the explicit "no rubric configured" sentinel.
    """
    topo = _toy_topology(n=1)
    plan = RunPlan(
        topology=topo, variants=(topo,),
        eval_rows=({"id": "e", "input": "x"},),
        run_id="r", run_dir=tmp_path / "r",
        cache_dir=None, max_cost_usd=None, judge_model="m",
        rubric_hash="deadbeef",
    )
    path = write_manifest(plan, cli_args={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["rubric_hash"] == "deadbeef"
    # Empty-rubric runs (no judge) preserve the empty sentinel.
    plan_no_rubric = RunPlan(
        topology=topo, variants=(topo,),
        eval_rows=({"id": "e", "input": "x"},),
        run_id="r2", run_dir=tmp_path / "r2",
        cache_dir=None, max_cost_usd=None, judge_model="m",
    )
    payload2 = json.loads(
        write_manifest(plan_no_rubric, cli_args={}).read_text(encoding="utf-8")
    )
    assert payload2["rubric_hash"] == ""
