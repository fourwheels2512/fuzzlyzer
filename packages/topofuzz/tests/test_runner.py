"""Tests for BLUEPRINT_v5plus.md §7 runner + Cell + RunWriter."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("diskcache")  # LLMCache / JudgeCache need it

from fuzzeval.cache import JudgeCache, LLMCache  # noqa: E402
from fuzzeval.canonical import canonical_hash  # noqa: E402
from fuzzeval.judge import BinaryScore, ContinuousScore  # noqa: E402

from topofuzz.ir import Edge, LLMConfig, Node, Topology  # noqa: E402
from topofuzz.runner import (  # noqa: E402
    Cell,
    NodeOutput,
    RunWriter,
    run_variant,
)


# ---------------------------------------------------------------------------
# Helpers — in-process LLM/judge fakes
# ---------------------------------------------------------------------------


def _toy_topology(n: int = 3, *, model: str = "google/gemini-2.5-pro") -> Topology:
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
class FakeLLM:
    """Records every call; returns a deterministic envelope per node."""

    calls: list[dict] = field(default_factory=list)
    served_override: str | None = None
    delay_s: float = 0.0
    raise_on_call: Exception | None = None

    async def __call__(self, **kwargs: Any) -> dict:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.calls.append(kwargs)
        served = self.served_override or kwargs["model"]
        # Echo a deterministic response that includes the requested model
        # so tests can verify ordering by inspecting later prompts.
        text = f"<response from {kwargs['model']} call#{len(self.calls)}>"
        return {
            "choices": [{"message": {"content": text}}],
            "model": served,
        }


@dataclass
class FakeJudge:
    """Returns a fixed score; optionally records each payload."""

    score_to_return: Any = field(default_factory=lambda: ContinuousScore(value=0.5))
    seen_payloads: list[dict] = field(default_factory=list)
    raise_exc: Exception | None = None

    def score(self, payload: dict) -> Any:
        self.seen_payloads.append(payload)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.score_to_return


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Sequential walk + prior-output propagation
# ---------------------------------------------------------------------------


def test_runner_walks_nodes_in_tuple_order():
    topo = _toy_topology(n=3)
    fake = FakeLLM()
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
        )
    )
    assert [c["model"] for c in fake.calls] == [
        "google/gemini-2.5-pro"
    ] * 3
    assert [o.node_id for o in cell.node_outputs] == ["n0", "n1", "n2"]


def test_runner_propagates_prior_outputs_into_prompts():
    topo = _toy_topology(n=2)
    fake = FakeLLM()
    _run(
        run_variant(
            topo,
            {"id": "e1", "input": "task input"},
            llm_callable=fake,
        )
    )
    first_prompt = fake.calls[0]["messages"][0]["content"]
    second_prompt = fake.calls[1]["messages"][0]["content"]
    assert "Prior agent outputs" not in first_prompt
    assert "Prior agent outputs" in second_prompt
    # The second prompt must surface n0's response text.
    assert "google/gemini-2.5-pro call#1" in second_prompt


def test_runner_variant_id_is_topology_canonical_hash():
    topo = _toy_topology(n=2)
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
        )
    )
    assert cell.variant_id == topo.canonical_hash()
    assert cell.ir_canonical_hash == topo.canonical_hash()


def test_runner_does_not_mutate_topology():
    topo = _toy_topology(n=3)
    before = topo.canonical_hash()
    _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
        )
    )
    after = topo.canonical_hash()
    assert before == after


# ---------------------------------------------------------------------------
# §7 cache key — six-tuple identity
# ---------------------------------------------------------------------------


def test_runner_caches_llm_responses(tmp_path):
    topo = _toy_topology(n=2)
    llm_cache = LLMCache(tmp_path)
    fake = FakeLLM()

    cell1 = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
        )
    )
    assert all(not o.cache_hit for o in cell1.node_outputs)
    assert len(fake.calls) == 2

    # Second run with same IR + same eval_row — every node should hit cache.
    cell2 = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
        )
    )
    assert all(o.cache_hit for o in cell2.node_outputs)
    assert len(fake.calls) == 2  # no new calls
    assert [o.response_text for o in cell1.node_outputs] == [
        o.response_text for o in cell2.node_outputs
    ]
    llm_cache.close()


def test_runner_cache_key_includes_section_7_six_tuple(tmp_path):
    """White-box: the cache key is the §7 6-tuple via canonical_hash."""
    from fuzzeval.cache import _canonical_key  # private but contractually pinned

    topo = _toy_topology(n=1)
    node = topo.nodes[0]
    eval_row = {"id": "e1", "input": "x"}

    fake = FakeLLM()
    llm_cache = LLMCache(tmp_path)
    _run(
        run_variant(
            topo,
            eval_row,
            llm_callable=fake,
            llm_cache=llm_cache,
        )
    )

    # Reconstruct what the runner *should* have stored under, independently.
    from topofuzz.runner import _default_prompt

    prompt = _default_prompt(node, eval_row, ())
    expected_identity = {
        "ir_canonical_hash": topo.canonical_hash(),
        "node_id": node.id,
        "llm_config_hash": node.llm_config.canonical_hash(),
        "prompt_hash": canonical_hash(prompt),
        "model_served_version": node.llm_config.model,
    }
    assert llm_cache.get(expected_identity) is not None
    # The exact key lookup uses _canonical_key — verify both yield the same.
    expected_key = _canonical_key(expected_identity, "local")
    assert llm_cache._cache.get(expected_key) is not None  # noqa: SLF001
    llm_cache.close()


def test_runner_cache_key_changes_when_llm_config_changes(tmp_path):
    topo_a = _toy_topology(n=1, model="google/gemini-2.5-pro")
    topo_b = _toy_topology(n=1, model="google/gemini-2.5-flash")
    llm_cache = LLMCache(tmp_path)
    fake = FakeLLM()

    _run(run_variant(topo_a, {"id": "e1", "input": "x"},
                     llm_callable=fake, llm_cache=llm_cache))
    _run(run_variant(topo_b, {"id": "e1", "input": "x"},
                     llm_callable=fake, llm_cache=llm_cache))
    # Different model → different llm_config_hash → different cache key
    assert len(fake.calls) == 2
    llm_cache.close()


def test_runner_cache_key_changes_when_prompt_changes(tmp_path):
    topo = _toy_topology(n=1)
    llm_cache = LLMCache(tmp_path)
    fake = FakeLLM()

    _run(run_variant(topo, {"id": "e1", "input": "input1"},
                     llm_callable=fake, llm_cache=llm_cache))
    _run(run_variant(topo, {"id": "e2", "input": "input2"},
                     llm_callable=fake, llm_cache=llm_cache))
    assert len(fake.calls) == 2
    llm_cache.close()


# ---------------------------------------------------------------------------
# Served-version resolver behavior
# ---------------------------------------------------------------------------


def test_runner_warns_when_served_version_mismatches_but_caches_under_predicted(
    tmp_path, caplog
):
    """When resolver disagrees with provider, warn — but cache MUST be
    keyed on the resolver's prediction so the next run hits, not on the
    actual served id (which next run's lookup would never query)."""
    topo = _toy_topology(n=1)
    fake = FakeLLM(served_override="google/gemini-2.5-pro-actual-built-id")
    llm_cache = LLMCache(tmp_path)

    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        cell1 = _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=fake,
                llm_cache=llm_cache,
                model_served_version_resolver=lambda m: m,
            )
        )

    # Forensics: NodeOutput records the actual served id.
    assert cell1.node_outputs[0].model_served_version == (
        "google/gemini-2.5-pro-actual-built-id"
    )
    assert any(
        "served" in rec.message.lower() and "resolver" in rec.message.lower()
        for rec in caplog.records
    )

    # The critical regression test: a SECOND run with the same predicted
    # key MUST hit cache. If we'd keyed under actual served version, it
    # would miss every time and the cache would fill with junk.
    cell2 = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
            model_served_version_resolver=lambda m: m,
        )
    )
    assert cell2.node_outputs[0].cache_hit is True
    assert len(fake.calls) == 1  # second run was a cache hit, no new LLM call
    llm_cache.close()


def test_runner_does_not_warn_when_served_version_matches(tmp_path, caplog):
    """Negative test: identity-resolver happy path emits no warning."""
    topo = _toy_topology(n=1)
    fake = FakeLLM()  # served defaults to requested model
    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=fake,
            )
        )
    assert not any(
        "served" in rec.message.lower() for rec in caplog.records
    )


def test_runner_resolver_exception_falls_back_to_request_id(tmp_path, caplog):
    topo = _toy_topology(n=1)
    fake = FakeLLM()

    def buggy_resolver(model: str) -> str:
        raise KeyError("unknown alias")

    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        cell = _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=fake,
                model_served_version_resolver=buggy_resolver,
            )
        )
    assert any(
        "resolver raised" in rec.message.lower()
        for rec in caplog.records
    )
    assert cell.node_outputs[0].model_served_version == "google/gemini-2.5-pro"


def test_runner_resolver_returning_garbage_falls_back(caplog):
    topo = _toy_topology(n=1)

    def garbage_resolver(model: str) -> str:
        return None  # type: ignore[return-value]

    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        cell = _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=FakeLLM(),
                model_served_version_resolver=garbage_resolver,
            )
        )
    assert cell.node_outputs[0].model_served_version == "google/gemini-2.5-pro"


def test_runner_uses_resolver_for_cache_lookup(tmp_path):
    topo = _toy_topology(n=1)
    llm_cache = LLMCache(tmp_path)
    fake = FakeLLM()

    # Run once with resolver returning "X-served"
    _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
            model_served_version_resolver=lambda m: "X-served",
        )
    )
    assert len(fake.calls) == 1

    # Different resolver → different lookup key → cache miss
    _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
            model_served_version_resolver=lambda m: "Y-served",
        )
    )
    assert len(fake.calls) == 2
    llm_cache.close()


# ---------------------------------------------------------------------------
# LLMConfig knob threading
# ---------------------------------------------------------------------------


def test_runner_passes_seed_when_set():
    cfg = LLMConfig(model="google/gemini-2.5-pro", seed=42)
    node = Node(id="a", role="a", llm_config=cfg)
    topo = Topology(
        name="t", nodes=(node,), edges=(), entrypoint="a", terminal=("a",)
    )
    fake = FakeLLM()
    _run(run_variant(topo, {"id": "e1", "input": "x"}, llm_callable=fake))
    assert fake.calls[0].get("seed") == 42


def test_runner_omits_seed_when_none():
    cfg = LLMConfig(model="google/gemini-2.5-pro", seed=None)
    node = Node(id="a", role="a", llm_config=cfg)
    topo = Topology(
        name="t", nodes=(node,), edges=(), entrypoint="a", terminal=("a",)
    )
    fake = FakeLLM()
    _run(run_variant(topo, {"id": "e1", "input": "x"}, llm_callable=fake))
    assert "seed" not in fake.calls[0]


def test_runner_passes_temperature_and_max_tokens():
    cfg = LLMConfig(model="m", temperature=0.3, max_tokens=128)
    node = Node(id="a", role="a", llm_config=cfg)
    topo = Topology(
        name="t", nodes=(node,), edges=(), entrypoint="a", terminal=("a",)
    )
    fake = FakeLLM()
    _run(run_variant(topo, {"id": "e1", "input": "x"}, llm_callable=fake))
    call = fake.calls[0]
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 128


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_runner_requires_eval_id_str():
    topo = _toy_topology(n=1)
    with pytest.raises(ValueError, match="eval_row\\['id'\\]"):
        _run(run_variant(topo, {"input": "x"}, llm_callable=FakeLLM()))
    with pytest.raises(ValueError, match="eval_row\\['id'\\]"):
        _run(run_variant(topo, {"id": "", "input": "x"},
                         llm_callable=FakeLLM()))
    with pytest.raises(ValueError, match="eval_row\\['id'\\]"):
        _run(run_variant(topo, {"id": 42, "input": "x"},
                         llm_callable=FakeLLM()))


def test_runner_rejects_node_without_llm_config():
    node = Node(id="a", role="a", llm_config=None)
    topo = Topology(
        name="t", nodes=(node,), edges=(), entrypoint="a", terminal=("a",)
    )
    with pytest.raises(ValueError, match="llm_config=None"):
        _run(run_variant(topo, {"id": "e1", "input": "x"},
                         llm_callable=FakeLLM()))


def test_runner_rejects_non_str_prompt_builder_output():
    topo = _toy_topology(n=1)

    def bad_builder(node, eval_row, prior):
        return 123  # not a string

    with pytest.raises(TypeError, match="prompt_builder"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=FakeLLM(),
                prompt_builder=bad_builder,
            )
        )


def test_runner_rejects_non_str_llm_response():
    topo = _toy_topology(n=1)

    async def bad_llm(**kwargs):
        return {
            "choices": [{"message": {"content": {"not": "a string"}}}],
            "model": "m",
        }

    with pytest.raises(TypeError, match="not.*str|expected str"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=bad_llm,
            )
        )


def test_runner_rejects_unparseable_llm_response_shape():
    topo = _toy_topology(n=1)

    async def bad_shape(**kwargs):
        return {"unexpected": "shape"}

    with pytest.raises(TypeError, match="LLM response shape"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=bad_shape,
            )
        )


# ---------------------------------------------------------------------------
# Lazy litellm import
# ---------------------------------------------------------------------------


def test_runner_module_does_not_import_litellm_at_top_level():
    """Static contract: the litellm import lives inside ``_call_llm``, not
    at module scope. Hoisting it to the top would silently break the
    ``topofuzz[verify]`` slim install — and the previous monkeypatch-based
    test would still pass. Scan the source instead."""
    import inspect

    import topofuzz.runner as runner_mod

    src = inspect.getsource(runner_mod)
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.lstrip()
        if not (
            stripped.startswith("import litellm")
            or stripped.startswith("from litellm")
        ):
            continue
        # An import is "lazy" iff its line is indented (i.e. inside a
        # function / try block, not at module scope).
        assert line[:1] in (" ", "\t"), (
            f"runner.py:{lineno} hoists litellm import to module scope: "
            f"{line!r}. Move it inside _call_llm."
        )


def test_runner_raises_importerror_when_litellm_missing(monkeypatch):
    """Behavioral test: when default path runs and litellm isn't
    importable, raise a clear ImportError pointing at topofuzz[full]."""
    monkeypatch.setitem(sys.modules, "litellm", None)
    topo = _toy_topology(n=1)
    with pytest.raises(ImportError, match="litellm"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                # llm_callable=None → default litellm path
            )
        )


# ---------------------------------------------------------------------------
# Judge integration
# ---------------------------------------------------------------------------


def test_runner_calls_judge_with_final_agent_output():
    topo = _toy_topology(n=2)
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.7))
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=judge,
        )
    )
    assert isinstance(cell.judge_result, ContinuousScore)
    assert cell.judge_result.value == 0.7
    assert not cell.judge_failure
    assert len(judge.seen_payloads) == 1
    pld = judge.seen_payloads[0]
    assert pld["variant_id"] == topo.canonical_hash()
    assert pld["agent_output"] == cell.node_outputs[-1].response_text


def test_runner_marks_judge_failure_when_judge_raises(tmp_path):
    topo = _toy_topology(n=1)
    judge = FakeJudge(raise_exc=RuntimeError("kaboom"))
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=judge,
        )
    )
    assert cell.judge_result is None
    assert cell.judge_failure
    assert "RuntimeError" in cell.judge_failure_reason
    assert "kaboom" in cell.judge_failure_reason


def test_runner_marks_judge_failure_when_judge_returns_garbage():
    topo = _toy_topology(n=1)
    judge = FakeJudge(score_to_return="not a JudgeResult")
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=judge,
        )
    )
    assert cell.judge_failure
    assert "non-JudgeResult" in cell.judge_failure_reason


def test_runner_caches_judge_results(tmp_path):
    topo = _toy_topology(n=1)
    # An LLM cache so the response_text is stable across runs — judge
    # cache identity depends on response_hash, which would otherwise drift
    # since FakeLLM embeds a call counter in its echoed text.
    llm_cache = LLMCache(tmp_path / "llm")
    judge_cache = JudgeCache(tmp_path / "judge")
    judge = FakeJudge(score_to_return=BinaryScore(passed=True, rationale="ok"))
    fake = FakeLLM()

    cell1 = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
            judge=judge,
            judge_cache=judge_cache,
            rubric_hash="r1",
            judge_model_version="jm1",
        )
    )
    assert len(judge.seen_payloads) == 1
    assert isinstance(cell1.judge_result, BinaryScore)

    # Same IR + same input → LLM hit (no new fake.calls), same response →
    # judge cache hit (no new judge.seen_payloads).
    cell2 = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
            judge=judge,
            judge_cache=judge_cache,
            rubric_hash="r1",
            judge_model_version="jm1",
        )
    )
    assert len(fake.calls) == 1
    assert len(judge.seen_payloads) == 1  # not invoked again
    assert isinstance(cell2.judge_result, BinaryScore)
    assert cell2.judge_result.passed
    llm_cache.close()
    judge_cache.close()


def test_runner_judge_cache_key_changes_when_rubric_changes(tmp_path):
    topo = _toy_topology(n=1)
    llm_cache = LLMCache(tmp_path / "llm")
    judge_cache = JudgeCache(tmp_path / "judge")
    judge = FakeJudge()
    fake = FakeLLM()

    _run(run_variant(topo, {"id": "e1", "input": "x"},
                     llm_callable=fake, llm_cache=llm_cache,
                     judge=judge, judge_cache=judge_cache,
                     rubric_hash="r1", judge_model_version="jm1"))
    _run(run_variant(topo, {"id": "e1", "input": "x"},
                     llm_callable=fake, llm_cache=llm_cache,
                     judge=judge, judge_cache=judge_cache,
                     rubric_hash="r2", judge_model_version="jm1"))
    # Same response (LLM cache hit), different rubric → judge invoked twice
    assert len(judge.seen_payloads) == 2
    llm_cache.close()
    judge_cache.close()


def test_runner_no_judge_means_judge_result_none_no_failure():
    topo = _toy_topology(n=1)
    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"}, llm_callable=FakeLLM()
        )
    )
    assert cell.judge_result is None
    assert cell.judge_failure is False
    assert cell.judge_failure_reason is None


# ---------------------------------------------------------------------------
# Concurrency / semaphore plumbing
# ---------------------------------------------------------------------------


def test_runner_uses_supplied_semaphore_for_single_node_variants():
    """Shared Semaphore(1) caps LLM calls — two single-node variants
    serialize."""
    topo = _toy_topology(n=1)
    sem = asyncio.Semaphore(1)
    order: list[str] = []

    async def slow_llm(**kwargs):
        order.append("enter")
        await asyncio.sleep(0.05)
        order.append("exit")
        return {
            "choices": [{"message": {"content": "x"}}],
            "model": kwargs["model"],
        }

    async def runner_pair():
        await asyncio.gather(
            run_variant(topo, {"id": "e1", "input": "x"},
                        llm_callable=slow_llm, semaphore=sem),
            run_variant(topo, {"id": "e2", "input": "x"},
                        llm_callable=slow_llm, semaphore=sem),
        )

    asyncio.run(runner_pair())
    assert order == ["enter", "exit", "enter", "exit"]


def test_runner_semaphore_caps_llm_calls_not_variants_for_multinode():
    """The semaphore caps simultaneous **LLM calls** (per the documented
    contract), NOT whole variants. With Semaphore(1) and two N=2 variants
    in parallel, the second variant's first node may interleave between
    the first variant's nodes — which is fine and intended."""
    topo = _toy_topology(n=2)
    sem = asyncio.Semaphore(1)
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def slow_llm(**kwargs):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.02)
        finally:
            async with lock:
                in_flight -= 1
        return {
            "choices": [{"message": {"content": "x"}}],
            "model": kwargs["model"],
        }

    async def runner_pair():
        await asyncio.gather(
            run_variant(topo, {"id": "e1", "input": "x"},
                        llm_callable=slow_llm, semaphore=sem),
            run_variant(topo, {"id": "e2", "input": "x"},
                        llm_callable=slow_llm, semaphore=sem),
        )

    asyncio.run(runner_pair())
    assert max_in_flight == 1  # sem(1) → only one LLM call at a time, ever


# ---------------------------------------------------------------------------
# Cancellation (step 6 cost-cap / Ctrl+C readiness)
# ---------------------------------------------------------------------------


def test_runner_returns_cancelled_cell_when_event_set_before_first_node():
    topo = _toy_topology(n=3)
    cancel = asyncio.Event()
    cancel.set()
    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            cancel=cancel,
        )
    )
    assert cell.error == "cancelled"
    assert cell.node_outputs == ()
    assert cell.judge_result is None


def test_runner_returns_partial_cell_on_mid_walk_cancel():
    topo = _toy_topology(n=3)
    cancel = asyncio.Event()
    seen_models: list[str] = []

    async def cancelling_llm(**kwargs):
        seen_models.append(kwargs["model"])
        # Trip cancel after the first call so the runner stops before n2.
        if len(seen_models) == 1:
            cancel.set()
        return {
            "choices": [{"message": {"content": "x"}}],
            "model": kwargs["model"],
        }

    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=cancelling_llm,
            cancel=cancel,
        )
    )
    assert cell.error == "cancelled"
    # First node ran; second + third never started.
    assert len(cell.node_outputs) == 1
    assert len(seen_models) == 1


def test_runner_skips_judge_when_cancelled_after_walk():
    topo = _toy_topology(n=1)
    cancel = asyncio.Event()
    judge = FakeJudge()

    async def llm_then_cancel(**kwargs):
        cancel.set()
        return {
            "choices": [{"message": {"content": "x"}}],
            "model": kwargs["model"],
        }

    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=llm_then_cancel,
            judge=judge,
            cancel=cancel,
        )
    )
    assert cell.error == "cancelled"
    assert cell.judge_result is None
    # Critical: judge must NOT have been invoked after cancel was set.
    assert judge.seen_payloads == []


# ---------------------------------------------------------------------------
# Judge-cache isolation: judge.score not invoked when JudgeCache hits alone
# ---------------------------------------------------------------------------


def test_judge_cache_hit_skips_judge_invocation_even_without_llm_cache(tmp_path):
    """Even with LLMCache disabled, an existing JudgeCache entry must
    bypass judge.score. (Independence of the two caches per §7.)"""
    topo = _toy_topology(n=1)
    judge_cache = JudgeCache(tmp_path)
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.9))

    # Use a deterministic LLM so response_hash is stable across runs.
    async def deterministic_llm(**kwargs):
        return {
            "choices": [{"message": {"content": "stable_response"}}],
            "model": kwargs["model"],
        }

    _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=deterministic_llm,
            judge=judge, judge_cache=judge_cache,
            rubric_hash="r1", judge_model_version="jm1",
        )
    )
    assert len(judge.seen_payloads) == 1

    _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=deterministic_llm,
            judge=judge, judge_cache=judge_cache,
            rubric_hash="r1", judge_model_version="jm1",
        )
    )
    # Judge cache hit → judge.score NOT invoked again
    assert len(judge.seen_payloads) == 1
    judge_cache.close()


def test_judge_cache_keyed_on_full_eval_row_not_just_input(tmp_path):
    """Two eval rows with same 'input' but different 'expected_answer'
    must NOT share a judge-cache entry — that's the audit-driven
    eval_row-hash fix (B-RT-1)."""
    topo = _toy_topology(n=1)
    judge_cache = JudgeCache(tmp_path)
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.5))

    async def deterministic_llm(**kwargs):
        return {
            "choices": [{"message": {"content": "stable"}}],
            "model": kwargs["model"],
        }

    # Critical: SAME id, SAME input, ONLY expected_answer differs.
    # If _eval_row_hash regressed to "hash only id" or "hash only input",
    # this test would still see one judge call and pass. Using a single
    # field difference isolates the contribution of the secondary field
    # exactly as the B-RT-1 audit fix promises.
    _run(
        run_variant(
            topo, {"id": "e_same", "input": "x", "expected_answer": "A"},
            llm_callable=deterministic_llm,
            judge=judge, judge_cache=judge_cache,
            rubric_hash="r1", judge_model_version="jm1",
        )
    )
    _run(
        run_variant(
            topo, {"id": "e_same", "input": "x", "expected_answer": "B"},
            llm_callable=deterministic_llm,
            judge=judge, judge_cache=judge_cache,
            rubric_hash="r1", judge_model_version="jm1",
        )
    )
    assert len(judge.seen_payloads) == 2
    judge_cache.close()


def test_runner_eval_row_with_unjsonable_value_raises_with_context():
    """The _eval_row_hash error message must name the eval id so a user
    with 1000 rows can find the bad one."""
    topo = _toy_topology(n=1)
    judge = FakeJudge()
    bad_row = {"id": "e_bad", "input": "x", "tags": {"a", "b"}}  # set
    with pytest.raises(TypeError, match="e_bad"):
        _run(
            run_variant(
                topo, bad_row,
                llm_callable=FakeLLM(),
                judge=judge,
            )
        )


def test_runner_handles_object_shaped_llm_response():
    """The non-dict (object-attribute) extraction branch was uncovered
    in pass 2. Real litellm returns an object with .choices/.model
    attributes, not a dict envelope. Verify the branch works."""
    import types

    topo = _toy_topology(n=1)

    async def object_llm(**kwargs):
        msg = types.SimpleNamespace(content="object-response-text")
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(
            choices=[choice],
            model="google/gemini-2.5-pro-served-id",
        )

    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=object_llm,
        )
    )
    assert cell.node_outputs[0].response_text == "object-response-text"
    assert cell.node_outputs[0].model_served_version == (
        "google/gemini-2.5-pro-served-id"
    )


# ---------------------------------------------------------------------------
# RunWriter: audit-driven hardening
# ---------------------------------------------------------------------------


def test_run_writer_refuses_filesystem_root(tmp_path):
    from pathlib import Path
    root = Path(tmp_path).resolve().anchor  # "C:\\" on Win, "/" on Unix
    topo = _toy_topology(n=1)
    with pytest.raises(ValueError, match="filesystem root"):
        RunWriter(root, "run-1", topo)


def test_run_writer_persists_schema_version_pragma(tmp_path):
    topo = _toy_topology(n=1)
    with RunWriter(tmp_path, "run-1", topo) as w:
        del w
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        ver = db.execute("PRAGMA user_version").fetchone()[0]
    finally:
        db.close()
    assert ver == 4  # step 9: rubric_hash on runs (was 3 after step 8 polish)


def test_run_writer_resume_preserves_original_created_at(tmp_path):
    """INSERT OR IGNORE on runs table → first kickoff time wins."""
    topo = _toy_topology(n=1)
    with RunWriter(tmp_path, "run-1", topo) as w:
        del w
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        first_created = db.execute(
            "SELECT created_at FROM runs WHERE run_id = ?", ("run-1",)
        ).fetchone()[0]
    finally:
        db.close()

    # Resume after a delay — created_at must NOT advance.
    import time as _time
    _time.sleep(0.05)
    with RunWriter(tmp_path, "run-1", topo) as w:
        del w
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        second_created = db.execute(
            "SELECT created_at FROM runs WHERE run_id = ?", ("run-1",)
        ).fetchone()[0]
    finally:
        db.close()
    assert second_created == first_created


def test_run_writer_init_does_not_leak_db_on_jsonl_open_failure(
    tmp_path, monkeypatch
):
    """If jsonl open raises mid-init, the SQLite handle must not leak —
    otherwise the next Windows attempt to open run.sqlite would fail with
    a file lock."""
    topo = _toy_topology(n=1)

    real_open = type(tmp_path).open

    def bad_open(self, *args, **kwargs):
        if self.name == "traces.jsonl":
            raise OSError("simulated jsonl open failure")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "open", bad_open)
    with pytest.raises(OSError, match="simulated jsonl"):
        RunWriter(tmp_path, "run-1", topo)
    monkeypatch.undo()

    # If the connection had leaked, this open would observe a locked DB
    # on Windows. On Unix it's mostly cosmetic, but the test still
    # verifies the second construction succeeds cleanly.
    with RunWriter(tmp_path, "run-1", topo) as w:
        w.write(_run(
            run_variant(topo, {"id": "e1", "input": "x"},
                        llm_callable=FakeLLM())
        ))


def test_run_writer_write_after_close_raises(tmp_path):
    topo = _toy_topology(n=1)
    cell = _run(
        run_variant(topo, {"id": "e1", "input": "x"},
                    llm_callable=FakeLLM())
    )
    w = RunWriter(tmp_path, "run-1", topo)
    w.close()
    with pytest.raises(RuntimeError, match="closed"):
        w.write(cell)


def test_run_writer_persists_cancelled_cell_with_error(tmp_path):
    topo = _toy_topology(n=2)
    cancel = asyncio.Event()
    cancel.set()
    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            cancel=cancel,
        )
    )
    with RunWriter(tmp_path, "run-1", topo) as w:
        w.write(cell)
        # Cancelled cells have error="cancelled" → NOT in completed_cells
        assert w.completed_cells() == set()
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        row = db.execute(
            "SELECT error FROM cells WHERE eval_id = 'e1'"
        ).fetchone()
    finally:
        db.close()
    assert row[0] == "cancelled"


# ---------------------------------------------------------------------------
# tenant_id parameter removed (B1 fix) — verify the kwarg is rejected
# ---------------------------------------------------------------------------


def test_runner_does_not_accept_tenant_id_kwarg():
    """tenant_id was a footgun (silently ignored). The fix is to remove
    the parameter — tenant scoping lives on cache constructors. Future
    addition would need to actually thread the value, so guard against
    accidental re-introduction with a TypeError-on-unknown-kwarg test."""
    topo = _toy_topology(n=1)
    with pytest.raises(TypeError, match="tenant_id"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=FakeLLM(),
                tenant_id="customer_42",  # type: ignore[call-arg]
            )
        )


# ---------------------------------------------------------------------------
# RunWriter
# ---------------------------------------------------------------------------


def test_run_writer_creates_jsonl_and_sqlite(tmp_path):
    topo = _toy_topology(n=1)
    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"}, llm_callable=FakeLLM()
        )
    )
    with RunWriter(tmp_path, "run-1", topo) as w:
        w.write(cell)
    assert (tmp_path / "traces.jsonl").exists()
    assert (tmp_path / "run.sqlite").exists()
    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["variant_id"] == cell.variant_id
    assert record["eval_id"] == "e1"


def test_run_writer_one_jsonl_line_per_cell(tmp_path):
    topo = _toy_topology(n=1)
    cells = [
        _run(run_variant(topo, {"id": f"e{i}", "input": "x"},
                         llm_callable=FakeLLM()))
        for i in range(5)
    ]
    with RunWriter(tmp_path, "run-1", topo) as w:
        for c in cells:
            w.write(c)
    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5


def test_run_writer_idempotent_on_duplicate_cells(tmp_path):
    topo = _toy_topology(n=1)
    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"}, llm_callable=FakeLLM()
        )
    )
    with RunWriter(tmp_path, "run-1", topo) as w:
        w.write(cell)
        w.write(cell)  # same key → INSERT OR REPLACE keeps single row
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        cnt = db.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
        assert cnt == 1
        cnt_judge = db.execute("SELECT COUNT(*) FROM judge_results").fetchone()[0]
        assert cnt_judge == 1
    finally:
        db.close()


def test_run_writer_completed_cells_for_resume(tmp_path):
    topo = _toy_topology(n=1)
    cells = [
        _run(run_variant(topo, {"id": f"e{i}", "input": "x"},
                         llm_callable=FakeLLM()))
        for i in range(3)
    ]
    with RunWriter(tmp_path, "run-1", topo) as w:
        for c in cells:
            w.write(c)

    # New writer (simulating resume) on the same dir reads the already-done set.
    with RunWriter(tmp_path, "run-1", topo) as w2:
        completed = w2.completed_cells()
    assert completed == {(cells[i].variant_id, f"e{i}") for i in range(3)}


def test_run_writer_persists_judge_kinds(tmp_path):
    topo = _toy_topology(n=1)

    cont_cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(score_to_return=ContinuousScore(value=0.4)),
        )
    )
    bin_cell = _run(
        run_variant(
            topo, {"id": "e2", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(score_to_return=BinaryScore(passed=False, rationale="r")),
        )
    )
    fail_cell = _run(
        run_variant(
            topo, {"id": "e3", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(raise_exc=RuntimeError("boom")),
        )
    )
    absent_cell = _run(
        run_variant(
            topo, {"id": "e4", "input": "x"},
            llm_callable=FakeLLM(),
        )
    )

    with RunWriter(tmp_path, "run-1", topo) as w:
        for c in (cont_cell, bin_cell, fail_cell, absent_cell):
            w.write(c)

    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        rows = db.execute(
            "SELECT eval_id, judge_kind, score_value, passed, "
            "rationale, failure_reason FROM judge_results ORDER BY eval_id"
        ).fetchall()
    finally:
        db.close()

    by_eval = {r[0]: r for r in rows}
    assert by_eval["e1"][1] == "continuous"
    assert by_eval["e1"][2] == 0.4
    assert by_eval["e2"][1] == "binary"
    assert by_eval["e2"][3] == 0
    assert by_eval["e2"][4] == "r"
    assert by_eval["e3"][1] == "failure"
    assert "boom" in by_eval["e3"][5]
    assert by_eval["e4"][1] == "absent"


def test_run_writer_records_topology_in_runs_table(tmp_path):
    topo = _toy_topology(n=2)
    with RunWriter(tmp_path, "run-xyz", topo) as w:
        del w  # only constructor matters
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        row = db.execute(
            "SELECT topology_canonical_hash, topology_name, tds_version "
            "FROM runs WHERE run_id = ?",
            ("run-xyz",),
        ).fetchone()
    finally:
        db.close()
    assert row is not None
    assert row[0] == topo.canonical_hash()
    assert row[1] == "toy"
    assert row[2] == "0.5.0"


def test_run_writer_rejects_empty_run_id(tmp_path):
    topo = _toy_topology(n=1)
    with pytest.raises(ValueError, match="run_id"):
        RunWriter(tmp_path, "", topo)


def test_run_writer_refuses_resume_with_topology_mismatch(tmp_path):
    """If a caller resumes the same run_id with a *different* topology,
    the runs row keeps the original hash but new cells would carry new
    variant_ids — silent data corruption. Refuse with a clear error."""
    topo_a = _toy_topology(n=2, model="google/gemini-2.5-pro")
    topo_b = _toy_topology(n=2, model="google/gemini-2.5-flash")
    assert topo_a.canonical_hash() != topo_b.canonical_hash()

    with RunWriter(tmp_path, "run-x", topo_a) as w:
        del w  # only constructor matters for this test

    with pytest.raises(ValueError, match="topology mismatch"):
        RunWriter(tmp_path, "run-x", topo_b)


def test_run_writer_resume_with_same_topology_succeeds(tmp_path):
    """Negative pair to the mismatch test: same topology resume is fine."""
    topo = _toy_topology(n=2)
    with RunWriter(tmp_path, "run-x", topo) as w:
        del w
    # Should not raise
    with RunWriter(tmp_path, "run-x", topo) as w:
        del w


def test_run_writer_safe_close_swallows_exception_during_close(tmp_path):
    """If sqlite close raises (e.g. corrupted DB on disk), _safe_close
    must not propagate — the writer must still reach a closed state so
    a retry can attempt cleanly. Exercises the `except Exception: pass`
    branches in _safe_close that pass-2 audit flagged as untested.

    sqlite3.Connection.close is a C-extension method and cannot be
    monkey-patched directly; wrap it in a proxy that raises instead.
    """
    topo = _toy_topology(n=1)
    w = RunWriter(tmp_path, "run-1", topo)

    class BoomCloseProxy:
        def __init__(self, real):
            self._real = real

        def close(self):
            raise sqlite3.Error("simulated corrupt DB on close")

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_db = w._db  # noqa: SLF001
    assert real_db is not None
    w._db = BoomCloseProxy(real_db)  # noqa: SLF001
    try:
        # close() must not propagate the simulated sqlite error.
        w.close()
        assert w._db is None  # noqa: SLF001 — _safe_close cleared it
        assert w._jsonl is None  # noqa: SLF001
    finally:
        # Make sure the real connection is actually closed so tmp_path
        # cleanup doesn't trip on a held file handle on Windows.
        real_db.close()


# ---------------------------------------------------------------------------
# Step 6.0 — cost telemetry
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMWithUsage:
    """FakeLLM variant that echoes a ``usage`` block in the response."""

    prompt_tokens: int = 100
    completion_tokens: int = 50
    served_override: str | None = None
    omit_usage: bool = False
    use_object_shape: bool = False

    async def __call__(self, **kwargs: Any) -> Any:
        served = self.served_override or kwargs["model"]
        text = f"<resp from {kwargs['model']}>"
        if self.use_object_shape:
            import types
            msg = types.SimpleNamespace(content=text)
            choice = types.SimpleNamespace(message=msg)
            obj = types.SimpleNamespace(
                choices=[choice],
                model=served,
            )
            if not self.omit_usage:
                obj.usage = types.SimpleNamespace(
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                )
            return obj
        envelope: dict = {
            "choices": [{"message": {"content": text}}],
            "model": served,
        }
        if not self.omit_usage:
            envelope["usage"] = {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        return envelope


def test_node_output_token_cost_fields_default_to_none():
    """Additive change to NodeOutput: existing positional usage still works."""
    from topofuzz.runner import NodeOutput as NO
    no = NO(
        node_id="n0",
        role="r",
        response_text="x",
        model_served_version="m",
        cache_hit=False,
        latency_ms=10,
    )
    assert no.input_tokens is None
    assert no.output_tokens is None
    assert no.cost_usd is None


def test_runner_records_token_counts_from_dict_usage():
    topo = _toy_topology(n=1, model="google/gemini-2.5-pro")
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLMWithUsage(prompt_tokens=200, completion_tokens=80),
        )
    )
    no = cell.node_outputs[0]
    assert no.input_tokens == 200
    assert no.output_tokens == 80
    # gemini-2.5-pro rate is 0.020 / 1K → 280 * 0.020 / 1000 = 0.0056
    assert no.cost_usd is not None
    assert abs(no.cost_usd - 0.0056) < 1e-9


def test_runner_records_token_counts_from_object_usage():
    topo = _toy_topology(n=1, model="google/gemini-2.5-pro")
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLMWithUsage(
                prompt_tokens=10, completion_tokens=5, use_object_shape=True
            ),
        )
    )
    no = cell.node_outputs[0]
    assert no.input_tokens == 10
    assert no.output_tokens == 5
    assert no.cost_usd is not None


def test_runner_leaves_tokens_none_when_usage_absent():
    topo = _toy_topology(n=1, model="google/gemini-2.5-pro")
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLMWithUsage(omit_usage=True),
        )
    )
    no = cell.node_outputs[0]
    assert no.input_tokens is None
    assert no.output_tokens is None
    assert no.cost_usd is None


def test_runner_cost_none_when_model_not_in_rate_table():
    """Unknown model → usage captured but cost_usd None.

    Don't silently substitute the default rate; the cap would understate
    cost. The §7 deviation is documented in NodeOutput's docstring.
    """
    topo = _toy_topology(n=1, model="unknown/model-xyz")
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLMWithUsage(prompt_tokens=100, completion_tokens=50),
        )
    )
    no = cell.node_outputs[0]
    assert no.input_tokens == 100
    assert no.output_tokens == 50
    assert no.cost_usd is None


def test_runner_cache_hit_leaves_cost_none(tmp_path):
    """Cache hits are not billed → cost_usd stays None even though the
    cold call recorded tokens. Test it explicitly so a future regression
    that 'helpfully' carries forward the cold cost gets caught."""
    topo = _toy_topology(n=1, model="google/gemini-2.5-pro")
    llm_cache = LLMCache(tmp_path)
    fake = FakeLLMWithUsage(prompt_tokens=100, completion_tokens=50)

    cold = _run(run_variant(
        topo, {"id": "e1", "input": "x"},
        llm_callable=fake, llm_cache=llm_cache,
    ))
    assert cold.node_outputs[0].cost_usd is not None

    warm = _run(run_variant(
        topo, {"id": "e1", "input": "x"},
        llm_callable=fake, llm_cache=llm_cache,
    ))
    assert warm.node_outputs[0].cache_hit is True
    assert warm.node_outputs[0].cost_usd is None
    assert warm.node_outputs[0].input_tokens is None
    assert warm.node_outputs[0].output_tokens is None
    llm_cache.close()


def test_runner_extract_usage_rejects_negative_and_bool():
    """``_extract_usage`` strips garbage (bool, negatives, non-int) →
    NodeOutput.input_tokens / output_tokens stay None on misbehaving
    providers rather than slipping a poison value into the cost cap.

    Stringified ints (``"100"``) ARE accepted — some older provider
    proxies return them as strings (audit RT m-8). Anything that
    fails ``int()`` is rejected.
    """
    from topofuzz.runner import _extract_usage

    bad_envelopes = [
        {"usage": {"prompt_tokens": -1, "completion_tokens": 5}},
        {"usage": {"prompt_tokens": True, "completion_tokens": 5}},
        {"usage": {"prompt_tokens": 1.5, "completion_tokens": 5}},
        {"usage": {"prompt_tokens": "not-a-number", "completion_tokens": 5}},
    ]
    for env in bad_envelopes:
        i, o = _extract_usage(env)
        assert i is None, f"expected None for {env}, got {i}"
        # output remains valid in these cases
        assert o == 5

    # New positive: stringified int is coerced.
    i, o = _extract_usage(
        {"usage": {"prompt_tokens": "100", "completion_tokens": "50"}}
    )
    assert i == 100
    assert o == 50


def test_runwriter_persists_cell_cost_usd(tmp_path):
    """``cells.cell_cost_usd`` is the aggregated cost across NodeOutputs.

    Step 6's orchestrator sums this column to drive ``--max-cost-usd``.
    The sum across a fresh run = sum of per-node cost_usd values.
    """
    topo = _toy_topology(n=2, model="google/gemini-2.5-pro")
    cell = _run(run_variant(
        topo, {"id": "e1", "input": "x"},
        llm_callable=FakeLLMWithUsage(prompt_tokens=100, completion_tokens=50),
    ))
    with RunWriter(tmp_path, "run-1", topo) as w:
        w.write(cell)

    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        row = db.execute(
            "SELECT cell_cost_usd FROM cells WHERE eval_id = ?", ("e1",)
        ).fetchone()
    finally:
        db.close()
    expected = sum(o.cost_usd for o in cell.node_outputs)
    assert row[0] is not None
    assert abs(row[0] - expected) < 1e-9


def test_runwriter_cell_cost_null_when_no_tokens(tmp_path):
    topo = _toy_topology(n=1, model="google/gemini-2.5-pro")
    cell = _run(run_variant(
        topo, {"id": "e1", "input": "x"},
        llm_callable=FakeLLMWithUsage(omit_usage=True),
    ))
    with RunWriter(tmp_path, "run-1", topo) as w:
        w.write(cell)

    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        row = db.execute(
            "SELECT cell_cost_usd FROM cells WHERE eval_id = ?", ("e1",)
        ).fetchone()
    finally:
        db.close()
    assert row[0] is None


def test_runwriter_migrates_v1_db_in_place(tmp_path):
    """An on-disk schema-v1 DB (no cell_cost_usd column) opened by today's
    RunWriter must be ALTERed to v2 in place, NOT rebuilt. Forensic
    preservation of any pre-existing rows matters.

    The DDL below mirrors the v1 schema shipped at commit ``6da01c8``
    (verified via ``git show 6da01c8:packages/topofuzz/src/topofuzz/
    runner.py``). If the v1 shipped DDL ever changes, this fixture
    must too — audit B-1 raised the concern that a hand-crafted v1
    that drifts from reality would let a real-DB migration regression
    slip past CI. Pinning the cells/runs/judge_results column lists
    here is the cheap defense.
    """
    topo = _toy_topology(n=1, model="google/gemini-2.5-pro")
    db_path = tmp_path / "run.sqlite"

    # Hand-craft a v1 DB matching the pre-step-6 schema literally.
    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.executescript("""
            PRAGMA user_version = 1;
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                topology_canonical_hash TEXT NOT NULL,
                topology_name TEXT NOT NULL,
                tds_version TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE cells (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                eval_id TEXT NOT NULL,
                ir_canonical_hash TEXT NOT NULL,
                node_outputs_json TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY (run_id, variant_id, eval_id)
            );
            CREATE TABLE judge_results (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                eval_id TEXT NOT NULL,
                judge_kind TEXT NOT NULL,
                score_value REAL,
                passed INTEGER,
                rationale TEXT,
                failure_reason TEXT,
                PRIMARY KEY (run_id, variant_id, eval_id)
            );
        """)
        legacy.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("legacy-run", topo.canonical_hash(), topo.name,
             topo.tds_version, 0.0),
        )
        legacy.execute(
            "INSERT INTO cells (run_id, variant_id, eval_id, "
            "ir_canonical_hash, node_outputs_json, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-run", "old-variant", "old-eval",
             topo.canonical_hash(), "[]", None),
        )
        legacy.commit()
    finally:
        legacy.close()

    # Re-open via RunWriter → migration runs, legacy row survives.
    with RunWriter(tmp_path, "legacy-run", topo) as w:
        # Write a fresh cell to confirm the new column accepts writes.
        cell = _run(run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=FakeLLMWithUsage(prompt_tokens=10, completion_tokens=5),
        ))
        w.write(cell)

    db = sqlite3.connect(str(db_path))
    try:
        # Schema bumped v1 → v4 cascade: cost telemetry (v2) + §A runs
        # cols (v3) + rubric_hash on runs (v4 / step 9).
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        # v3 + v4 runs columns landed. ``status`` defaults to
        # 'running', tenant_id to 'local', rubric_hash to ''.
        runs_cols = {
            row[1]
            for row in db.execute("PRAGMA table_info(runs)").fetchall()
        }
        for col in (
            "status",
            "finished_at",
            "n_variants",
            "n_evals",
            "judge_protocol",
            "cost_actual_usd",
            "tenant_id",
            "rubric_hash",
        ):
            assert col in runs_cols, f"v3/v4 column {col!r} missing post-migrate"
        legacy_status, legacy_tenant, legacy_rubric = db.execute(
            "SELECT status, tenant_id, rubric_hash FROM runs "
            "WHERE run_id = 'legacy-run'"
        ).fetchone()
        assert legacy_status == "running"
        assert legacy_tenant == "local"
        # v4 default for the migrated-forward row: empty rubric_hash
        # (the safe "rubric unknown" signal for read_score_matrix).
        assert legacy_rubric == ""
        # Legacy row preserved with NULL in the new column.
        legacy_row = db.execute(
            "SELECT cell_cost_usd FROM cells WHERE eval_id = 'old-eval'"
        ).fetchone()
        assert legacy_row is not None
        assert legacy_row[0] is None
        # New row carries the aggregated cost.
        new_row = db.execute(
            "SELECT cell_cost_usd FROM cells WHERE eval_id = 'e1'"
        ).fetchone()
        assert new_row[0] is not None
    finally:
        db.close()


def test_runwriter_refuses_future_schema(tmp_path):
    """A run.sqlite written by a newer topofuzz (PRAGMA user_version > 2)
    must NOT be silently downgraded — refuse with a clear error."""
    topo = _toy_topology(n=1)
    db_path = tmp_path / "run.sqlite"

    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.executescript("""
            PRAGMA user_version = 99;
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                topology_canonical_hash TEXT NOT NULL,
                topology_name TEXT NOT NULL,
                tds_version TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE cells (run_id TEXT, variant_id TEXT, eval_id TEXT,
                ir_canonical_hash TEXT, node_outputs_json TEXT, error TEXT,
                cell_cost_usd REAL, future_column TEXT,
                PRIMARY KEY (run_id, variant_id, eval_id));
            CREATE TABLE judge_results (run_id TEXT, variant_id TEXT, eval_id TEXT,
                judge_kind TEXT, score_value REAL, passed INTEGER, rationale TEXT,
                failure_reason TEXT,
                PRIMARY KEY (run_id, variant_id, eval_id));
        """)
        legacy.commit()
    finally:
        legacy.close()

    with pytest.raises(RuntimeError, match="newer topofuzz"):
        RunWriter(tmp_path, "run-x", topo)


def test_cost_for_call_blended_rate():
    from fuzzeval.costest import cost_for_call

    # gemini-2.5-pro: 0.020 / 1K
    assert abs(
        cost_for_call("google/gemini-2.5-pro", 1000, 500) - (1500 * 0.020 / 1000.0)
    ) < 1e-9


def test_cost_for_call_unknown_model_returns_none():
    from fuzzeval.costest import cost_for_call

    assert cost_for_call("unknown/model-xyz", 100, 50) is None


def test_cost_for_call_missing_tokens_returns_none():
    from fuzzeval.costest import cost_for_call

    assert cost_for_call("google/gemini-2.5-pro", None, 50) is None
    assert cost_for_call("google/gemini-2.5-pro", 100, None) is None


def test_cost_for_call_rejects_bool_and_negative():
    from fuzzeval.costest import cost_for_call

    assert cost_for_call("google/gemini-2.5-pro", True, 50) is None
    assert cost_for_call("google/gemini-2.5-pro", 100, -1) is None
    assert cost_for_call("google/gemini-2.5-pro", 100, 1.5) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step 12b soft-DoD integration: run_variant + JuryJudge with stub clients
#
# Per BLUEPRINT_v5plus.md §12b line 400 and DoD #14 line 445: prove the
# jury infrastructure plugs into the existing runner contract without
# special-casing — the runner sees a Judge, JuryJudge IS a Judge, and
# all three failure-mode buckets (3/3 parseable, 1-or-2 fail, 3/3 fail)
# route through the same _run_judge path.
#
# Stub clients only — only Gemini key is confirmed in v0.5 per
# [[project_llm_api_access]]; real-keys integration is deferred to
# v0.5.1.
# ---------------------------------------------------------------------------


def _stub_jury(
    tmp_path,
    *,
    responses_by_vendor: dict[str, str],
    dry_run: bool = False,
):
    """Build a JuryJudge whose 3 SingleJudges each have a one-response stub.

    Inline the StubLLM shape (mirrors fuzzeval.tests.test_judge._StubLLM)
    so this integration test has no cross-test dependency.
    """
    from fuzzeval.judge import JuryJudge

    class _StubLLM:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls: list[dict] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            text = self._responses.pop(0)
            return {
                "choices": [{"message": {"content": text}}],
                "model": kwargs["model"],
            }

    rubric = tmp_path / "rubric.txt"
    rubric.write_text(
        "Variant: {variant_id}\n"
        "Applicant: {applicant_blurb}\n"
        "Truth: {ground_truth_decision}\n"
        "Output: {agent_output}\n"
        "Score: <int 0-10>\nRationale: <text>\n",
        encoding="utf-8",
    )
    callables = {v: _StubLLM([r]) for v, r in responses_by_vendor.items()}
    return JuryJudge(
        rubric_path=rubric,
        llm_callables=callables,
        dry_run=dry_run,
    )


def _eval_row_with_truth():
    return {"id": "e1", "input": "applicant blurb", "ground_truth_decision": "approve"}


def test_runner_accepts_jury_judge_all_three_parse(tmp_path):
    """Soft DoD #14: a JuryJudge plugs into run_variant unchanged and the
    Cell carries a ContinuousScore == mean of the 3 parseable votes."""
    topo = _toy_topology(n=1)
    judge = _stub_jury(
        tmp_path,
        responses_by_vendor={
            "anthropic": "Score: 4\nRationale: a",
            "openai": "Score: 6\nRationale: o",
            "gemini": "Score: 8\nRationale: g",
        },
    )
    cell = _run(
        run_variant(
            topo,
            _eval_row_with_truth(),
            llm_callable=FakeLLM(),
            judge=judge,
            judge_prompt_renderer=judge.render_prompt,
            rubric_hash=judge.rubric_hash,
            judge_model_version=judge.model,
        )
    )
    assert isinstance(cell.judge_result, ContinuousScore)
    assert cell.judge_result.value == pytest.approx(0.6)
    assert cell.judge_failure is False
    # DoD #14: vendor_overlap reflects all three vendors that produced a vote.
    assert judge.judge_independence == {
        "vendor_overlap": ["anthropic", "gemini", "openai"],
    }
    assert judge.judge_failure_rate == 0.0


def test_runner_accepts_jury_judge_one_unparseable(tmp_path):
    """1 of 3 vendors unparseable → Cell still gets a ContinuousScore
    (mean of the surviving 2), judge_failure stays False, JuryJudge
    accounting reflects the partial failure."""
    topo = _toy_topology(n=1)
    judge = _stub_jury(
        tmp_path,
        responses_by_vendor={
            "anthropic": "no score line here",
            "openai": "Score: 6\nRationale: o",
            "gemini": "Score: 8\nRationale: g",
        },
    )
    cell = _run(
        run_variant(
            topo,
            _eval_row_with_truth(),
            llm_callable=FakeLLM(),
            judge=judge,
            judge_prompt_renderer=judge.render_prompt,
            rubric_hash=judge.rubric_hash,
            judge_model_version=judge.model,
        )
    )
    assert isinstance(cell.judge_result, ContinuousScore)
    assert cell.judge_result.value == pytest.approx(0.7)
    assert cell.judge_failure is False
    assert judge.judge_failure_rate == pytest.approx(1 / 3)
    assert judge.judge_independence == {
        "vendor_overlap": ["gemini", "openai"],
    }


def test_runner_marks_judge_failure_when_all_three_unparseable(tmp_path):
    """All 3 vendors unparseable → JuryJudge.score raises
    JudgeUnparseableError → runner's _run_judge catches it and produces
    a judge_failure=True Cell. This is what §8's degraded-status logic
    consumes."""
    topo = _toy_topology(n=1)
    judge = _stub_jury(
        tmp_path,
        responses_by_vendor={
            "anthropic": "no score 1",
            "openai": "no score 2",
            "gemini": "no score 3",
        },
    )
    cell = _run(
        run_variant(
            topo,
            _eval_row_with_truth(),
            llm_callable=FakeLLM(),
            judge=judge,
            judge_prompt_renderer=judge.render_prompt,
            rubric_hash=judge.rubric_hash,
            judge_model_version=judge.model,
        )
    )
    assert cell.judge_result is None
    assert cell.judge_failure is True
    assert "JudgeUnparseableError" in (cell.judge_failure_reason or "")
    assert judge.judge_failure_rate == 1.0


def test_runner_jury_judge_dry_run_returns_half(tmp_path):
    """Dry-run jury never calls any LLM — every cell gets ContinuousScore(0.5)."""
    topo = _toy_topology(n=1)

    def boom(**_kwargs):
        raise AssertionError("dry-run must not call any LLM")

    from fuzzeval.judge import JuryJudge

    rubric = tmp_path / "rubric.txt"
    rubric.write_text(
        "Variant: {variant_id}\nApplicant: {applicant_blurb}\n"
        "Truth: {ground_truth_decision}\nOutput: {agent_output}\n"
        "Score: <int 0-10>\nRationale: <text>\n",
        encoding="utf-8",
    )
    judge = JuryJudge(
        rubric_path=rubric,
        llm_callables={"anthropic": boom, "openai": boom, "gemini": boom},
        dry_run=True,
    )
    cell = _run(
        run_variant(
            topo,
            _eval_row_with_truth(),
            llm_callable=FakeLLM(),
            judge=judge,
            judge_prompt_renderer=judge.render_prompt,
            rubric_hash=judge.rubric_hash,
            judge_model_version=judge.model,
        )
    )
    assert isinstance(cell.judge_result, ContinuousScore)
    assert cell.judge_result.value == 0.5
    assert cell.judge_failure is False


def test_runner_jury_judge_n10_accounting_accumulates(tmp_path):
    """BLUEPRINT line 445 soft-DoD analogue: ``topofuzz run --jury-of-3
    --n 10`` (BLUEPRINT spells the legacy ``tsf`` CLI; real binary is
    ``topofuzz``) completes with a coherent judge_failure_rate. Here we
    drive 10 cells against a stub jury where one vendor flakes on every
    call — failure_rate must be 10/30 = 1/3."""
    topo = _toy_topology(n=1)

    from fuzzeval.judge import JuryJudge

    class _CycleStub:
        """Returns the same response on every call (vs the single-shot
        stub in _stub_jury which exhausts after one use)."""

        def __init__(self, text):
            self._text = text
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            return {
                "choices": [{"message": {"content": self._text}}],
                "model": kwargs["model"],
            }

    rubric = tmp_path / "rubric.txt"
    rubric.write_text(
        "Variant: {variant_id}\nApplicant: {applicant_blurb}\n"
        "Truth: {ground_truth_decision}\nOutput: {agent_output}\n"
        "Score: <int 0-10>\nRationale: <text>\n",
        encoding="utf-8",
    )
    judge = JuryJudge(
        rubric_path=rubric,
        llm_callables={
            "anthropic": _CycleStub("no score here"),
            "openai": _CycleStub("Score: 6\nRationale: o"),
            "gemini": _CycleStub("Score: 8\nRationale: g"),
        },
    )
    for i in range(10):
        cell = _run(
            run_variant(
                topo,
                {"id": f"e{i}", "input": "x", "ground_truth_decision": "approve"},
                llm_callable=FakeLLM(),
                judge=judge,
                judge_prompt_renderer=judge.render_prompt,
                rubric_hash=judge.rubric_hash,
                judge_model_version=judge.model,
            )
        )
        assert isinstance(cell.judge_result, ContinuousScore)
        assert cell.judge_result.value == pytest.approx(0.7)
    assert judge.total_vendor_calls == 30
    assert judge.failed_vendor_calls == 10
    assert judge.judge_failure_rate == pytest.approx(1 / 3)
    assert judge.judge_independence == {
        "vendor_overlap": ["gemini", "openai"],
    }


# ---------------------------------------------------------------------------
# LiteLLM model-id alias at the call boundary
# ---------------------------------------------------------------------------


def test_litellm_model_id_translates_google_prefix() -> None:
    """``google/gemini-X`` is the project-canonical name; LiteLLM needs
    ``gemini/gemini-X``. The alias fires at the call boundary so the
    internal cache key, manifest, attestation, and §8 vendor overlap
    all keep the canonical form."""
    from topofuzz.runner import _litellm_model_id

    assert _litellm_model_id("google/gemini-2.5-flash") == (
        "gemini/gemini-2.5-flash"
    )
    assert _litellm_model_id("google/gemini-2.5-pro") == (
        "gemini/gemini-2.5-pro"
    )


def test_litellm_model_id_passes_through_other_providers() -> None:
    """Anthropic, OpenAI, and bare model ids that LiteLLM already
    recognises pass through unchanged."""
    from topofuzz.runner import _litellm_model_id

    assert (
        _litellm_model_id("anthropic/claude-opus-4-7")
        == "anthropic/claude-opus-4-7"
    )
    assert _litellm_model_id("openai/gpt-4o") == "openai/gpt-4o"
    # Bare model id (no slash) — passthrough.
    assert _litellm_model_id("gpt-4o") == "gpt-4o"
    # Unknown prefix — passthrough; lets a forward-compat fixture
    # opt into a LiteLLM-native id (e.g. vertex_ai/) without our
    # alias getting in the way.
    assert (
        _litellm_model_id("vertex_ai/gemini-2.5-pro")
        == "vertex_ai/gemini-2.5-pro"
    )


def test_litellm_model_id_non_string_passes_through() -> None:
    """Defensive: a fake ``llm_callable`` injected by a test can use
    any sentinel for ``model``. The alias must not crash on non-str."""
    from topofuzz.runner import _litellm_model_id

    sentinel = object()
    assert _litellm_model_id(sentinel) is sentinel  # type: ignore[arg-type]
    assert _litellm_model_id(None) is None  # type: ignore[arg-type]
