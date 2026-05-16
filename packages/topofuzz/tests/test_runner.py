# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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
    _ANTHROPIC_SERVED_MAP,
    _run_judge,
    Cell,
    NodeOutput,
    RunWriter,
    anthropic_resolver,
    default_resolver,
    gemini_resolver,
    openai_resolver,
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


# ---------------------------------------------------------------------------
# Vendor canonical → served resolvers (gemini/openai/anthropic/default).
# Closes BACKLOG: model_served_version_resolver canonical/served
# mismatch. Without these resolvers the §7 cache keyed on canonical
# `google/...` / `openai/...` / `anthropic/...` IDs while providers
# echoed bare/snapshot IDs in response.model, mismatch-warning every
# call and burning the cache.
# ---------------------------------------------------------------------------


def test_gemini_resolver_strips_canonical_prefix():
    """``google/gemini-X`` → ``gemini-X`` (the actual Google AI Studio
    served ID). Non-google inputs pass through unchanged so downstream
    composition is safe."""
    assert gemini_resolver("google/gemini-2.5-flash") == "gemini-2.5-flash"
    assert gemini_resolver("google/gemini-2.5-pro") == "gemini-2.5-pro"
    assert gemini_resolver("google/gemini-1.5-flash") == "gemini-1.5-flash"
    # Non-prefixed pass-through (the served form itself round-trips).
    assert gemini_resolver("gemini-2.5-flash") == "gemini-2.5-flash"
    # Other-vendor inputs pass through (composability).
    assert gemini_resolver("openai/gpt-4o") == "openai/gpt-4o"


def test_openai_resolver_strips_canonical_prefix():
    assert openai_resolver("openai/gpt-4o") == "gpt-4o"
    assert openai_resolver("openai/gpt-4o-mini") == "gpt-4o-mini"
    assert openai_resolver("gpt-4o") == "gpt-4o"
    assert openai_resolver("google/gemini-2.5-pro") == "google/gemini-2.5-pro"


def test_anthropic_resolver_maps_latest_aliases_to_dated_snapshots():
    """The Anthropic API resolves ``*-latest`` aliases to date-stamped
    snapshots; the resolver mirrors that mapping so cache keys match
    what ``response.model`` will carry."""
    # Every known mapping round-trips.
    for alias, snapshot in _ANTHROPIC_SERVED_MAP.items():
        assert anthropic_resolver(f"anthropic/{alias}") == snapshot
        # Bare alias (caller already stripped) also resolves.
        assert anthropic_resolver(alias) == snapshot

    # Unknown alias: prefix-stripped, no snapshot lookup (best effort).
    # The runner's resolver-mismatch dedup filter will surface the
    # mismatch as a single WARNING per process, signalling that the
    # snapshot table needs updating.
    assert anthropic_resolver(
        "anthropic/claude-9-future-latest"
    ) == "claude-9-future-latest"


def test_default_resolver_dispatches_by_vendor_prefix():
    """``default_resolver`` is what the CLI wires by default. It must
    dispatch by canonical prefix to the per-vendor resolver — Gemini,
    OpenAI, Anthropic — and fall back to identity for unknown vendors
    so the historical no-resolver default is preserved."""
    # Each known vendor routes to its own resolver.
    assert default_resolver("google/gemini-2.5-flash") == "gemini-2.5-flash"
    assert default_resolver("openai/gpt-4o") == "gpt-4o"
    assert default_resolver(
        "anthropic/claude-3-5-sonnet-latest"
    ) == "claude-3-5-sonnet-20241022"
    # Unknown vendors fall through to identity.
    assert default_resolver("xai/grok-2") == "xai/grok-2"
    assert default_resolver("mistral/mistral-large") == "mistral/mistral-large"
    # Already-bare ids round-trip.
    assert default_resolver("gemini-2.5-flash") == "gemini-2.5-flash"


def test_runner_with_default_resolver_emits_no_mismatch_warning_on_gemini(
    tmp_path, caplog,
):
    """Integration test: with ``default_resolver`` wired up, a 1-cell
    run against a Gemini-canonical model that the provider serves as
    its bare ID emits NO ``model_served_version_resolver returned ...
    but provider served ...`` warning. This is the regression test
    for the canonical/served mismatch BACKLOG item.

    Empirically reproduces the observed v0.5 gallery behavior (every
    cell printed the warning once with ``_identity_resolver``);
    asserts the warning disappears when ``default_resolver`` is used.
    A second call is then verified to hit the §7 cache — the whole
    point of getting the served-version slot right is cache effectiveness
    (RT audit MINOR 5b).
    """
    topo = _toy_topology(n=1, model="google/gemini-2.5-flash")
    # Simulate Google AI Studio's actual served behavior: response.model
    # is the bare ID without the canonical ``google/`` prefix.
    fake = FakeLLM(served_override="gemini-2.5-flash")
    llm_cache = LLMCache(tmp_path)

    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        cell = _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=fake,
                llm_cache=llm_cache,
                model_served_version_resolver=default_resolver,
            )
        )

    # Sanity: cell completed cleanly.
    assert cell.node_outputs[0].model_served_version == "gemini-2.5-flash"

    # The exact warning text from runner.py line ~454. Asserting on the
    # full template fragment guards against partial-match false negatives
    # if a future log message reuses the word "resolver".
    mismatch_warnings = [
        rec for rec in caplog.records
        if (
            "but provider served" in rec.getMessage()
            and "resolver" in rec.getMessage().lower()
        )
    ]
    assert mismatch_warnings == [], (
        "default_resolver must produce zero canonical/served mismatch "
        "warnings on the Gemini happy path — saw:\n"
        + "\n".join(rec.getMessage() for rec in mismatch_warnings)
    )

    # Cache-effectiveness regression. Without the resolver fix the §7
    # cache key carried the canonical prefix while the response served
    # the bare id, so warm runs cache-missed every time. A correctly
    # resolved key MUST round-trip on the next lookup — the entire
    # business case for closing this MAJOR.
    cell2 = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=fake,
            llm_cache=llm_cache,
            model_served_version_resolver=default_resolver,
        )
    )
    assert cell2.node_outputs[0].cache_hit is True
    assert len(fake.calls) == 1, (
        "Second run with default_resolver must hit cache — saw "
        f"{len(fake.calls)} LLM calls (expected 1)"
    )

    llm_cache.close()


def test_anthropic_stale_snapshot_map_surfaces_resolver_mismatch_warning(
    tmp_path, caplog,
):
    """When ``_ANTHROPIC_SERVED_MAP`` becomes stale (Anthropic ships a
    new snapshot but the in-repo table hasn't been refreshed), the
    canonical/served mismatch warning MUST fire so operators can see
    the table needs updating. This pins the staleness-canary contract
    that audit-pass-1 RT MAJOR called out as untested.

    We simulate staleness by passing a `*-latest` alias the map DOES
    know but pretending the provider serves a newer date stamp. The
    resolver's prediction is the old snapshot; the provider's served
    id is the (hypothetical) new one. The warning must fire.
    """
    new_anthropic_snapshot = "claude-3-5-sonnet-99999999"  # future date
    topo = _toy_topology(n=1, model="anthropic/claude-3-5-sonnet-latest")
    fake = FakeLLM(served_override=new_anthropic_snapshot)
    llm_cache = LLMCache(tmp_path)

    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=fake,
                llm_cache=llm_cache,
                model_served_version_resolver=default_resolver,
            )
        )

    mismatch_warnings = [
        rec for rec in caplog.records
        if (
            "but provider served" in rec.getMessage()
            and new_anthropic_snapshot in rec.getMessage()
        )
    ]
    assert mismatch_warnings, (
        "Stale _ANTHROPIC_SERVED_MAP MUST surface the resolver-mismatch "
        "warning — without this canary, the snapshot table can drift "
        "silently for months and operators only notice via cost reports"
    )

    llm_cache.close()


def test_runner_with_identity_resolver_still_warns_on_gemini(tmp_path, caplog):
    """Positive control: the original ``_identity_resolver`` default
    DOES produce the mismatch warning that ``default_resolver``
    eliminates. Pins both sides of the contract — if a future change
    silently disables the warning at the runner layer, this test will
    fail rather than letting :func:`test_runner_with_default_resolver_emits_no_mismatch_warning_on_gemini`
    succeed for the wrong reason.
    """
    topo = _toy_topology(n=1, model="google/gemini-2.5-flash")
    fake = FakeLLM(served_override="gemini-2.5-flash")
    llm_cache = LLMCache(tmp_path)

    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        _run(
            run_variant(
                topo,
                {"id": "e1", "input": "x"},
                llm_callable=fake,
                llm_cache=llm_cache,
                # No resolver kwarg → default `_identity_resolver`.
            )
        )

    mismatch_warnings = [
        rec for rec in caplog.records
        if (
            "but provider served" in rec.getMessage()
            and "resolver" in rec.getMessage().lower()
        )
    ]
    assert len(mismatch_warnings) >= 1, (
        "identity-resolver MUST emit the canonical/served mismatch "
        "warning so the default_resolver regression test is a real "
        "test of behavior change, not a passthrough"
    )

    llm_cache.close()


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
    # The judge sees the DECLARED terminal node's output. For this linear
    # toy topology the terminal happens to also be the last walked node;
    # assert against ``topo.terminal`` explicitly rather than ``[-1]`` so
    # the test pins the *contract* (judge the verdict node), not the
    # coincidence. See test_runner_judges_declared_terminal_not_last_walked_node.
    terminal_out = next(
        o for o in cell.node_outputs if o.node_id == topo.terminal[0]
    )
    assert pld["agent_output"] == terminal_out.response_text


def test_runner_judges_declared_terminal_not_last_walked_node():
    """Regression guard: the judge must score the topology's DECLARED
    terminal node, NOT whatever node was walked last.

    ``order_swap`` can land a non-verdict agent (e.g. ``intake``) last;
    judging ``outputs[-1]`` there fabricates verdict-flips and corrupts
    every score-matrix metric (see .planning/verdict-flip-rootcause/).
    This is the structural check that would have caught that bug — it
    builds a topology whose terminal is NOT the last walked node and
    asserts the judge still scores the terminal."""
    cfg = LLMConfig(model="google/gemini-2.5-pro", temperature=0.0)
    # Nodes walk in order n0, n1, n2 — but n0 is the declared terminal,
    # so outputs[-1] (n2) is deliberately NOT the verdict node.
    topo = Topology(
        name="terminal-not-last",
        nodes=tuple(
            Node(id=f"n{i}", role=f"r{i}", llm_config=cfg) for i in range(3)
        ),
        edges=(
            Edge(from_id="n0", to_id="n1"),
            Edge(from_id="n1", to_id="n2"),
        ),
        entrypoint="n0",
        terminal=("n0",),
    )
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.7))
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=judge,
        )
    )
    pld = judge.seen_payloads[0]
    terminal_out = next(o for o in cell.node_outputs if o.node_id == "n0")
    assert pld["agent_output"] == terminal_out.response_text
    # And explicitly NOT the last walked node.
    assert pld["agent_output"] != cell.node_outputs[-1].response_text


def test_runner_judges_lex_first_terminal_when_multi_terminal():
    """Multi-terminal topology: the judge scores the lex-first declared
    terminal node, never the last walked node.

    Locks the contract the step-20a audit's B1 exposed — the supervisor
    gallery fixture was briefly multi-terminal (``critic`` + ``supervisor``
    via a stray ``FINISH`` -> END edge) and judged ``critic`` only by
    lexical luck. If a future fixture regresses to multi-terminal, this
    test pins exactly which node gets judged.

    Note: this topology is a synthetic shape (node ``c`` is a dangling
    non-terminal) built to exercise ``_run_judge``'s terminal selection
    directly — the LangGraph adapter would not normally emit it."""
    cfg = LLMConfig(model="google/gemini-2.5-pro", temperature=0.0)
    # Nodes walk a, b, c. terminal is passed UNSORTED as ("b", "a") to
    # prove the runner judges the lex-first terminal ("a"), not the
    # tuple-first one ("b"); c is last-walked but is not a terminal.
    topo = Topology(
        name="multi-terminal",
        nodes=tuple(
            Node(id=nid, role=f"r_{nid}", llm_config=cfg)
            for nid in ("a", "b", "c")
        ),
        edges=(
            Edge(from_id="a", to_id="b"),
            Edge(from_id="b", to_id="c"),
        ),
        entrypoint="a",
        terminal=("b", "a"),
    )
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.7))
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=judge,
        )
    )
    pld = judge.seen_payloads[0]
    out = {o.node_id: o.response_text for o in cell.node_outputs}
    assert pld["agent_output"] == out["a"]  # lex-first terminal
    assert pld["agent_output"] != out["b"]  # not the other terminal
    assert pld["agent_output"] != out["c"]  # not the last walked node


def test_run_judge_marks_failure_when_terminal_absent_from_outputs():
    """Defence-in-depth: if the declared terminal node is absent from the
    walked outputs, the judge call marks the cell judge_failure — it does
    NOT fall back to ``outputs[-1]``. That fallback is the exact
    wrong-node-laundering bug the terminal-aware judge exists to kill.

    This guard is CURRENTLY UNREACHABLE via ``run_variant`` (the IR
    validator guarantees ``terminal`` entries exist in ``nodes`` and the
    runner walks every node) — the test calls ``_run_judge`` directly to
    pin the contract so a future generator that drops the terminal node
    fails loud instead of silently re-introducing the bug. Do not delete
    it as 'dead code'."""
    out = NodeOutput(
        node_id="n0",
        role="r0",
        response_text="some text",
        model_served_version="google/gemini-2.5-pro",
        cache_hit=False,
        latency_ms=0,
    )
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.9))
    result, judge_failure, reason = _run(
        _run_judge(
            judge=judge,
            judge_cache=None,
            outputs=[out],
            terminal=("does_not_exist",),
            eval_row={"id": "e1", "input": "x"},
            variant_id="v1",
            rubric_hash="h",
            judge_model_version="m",
        )
    )
    assert result is None
    assert judge_failure is True
    assert "does_not_exist" in reason
    # The judge must NOT have been consulted — no fallback scoring.
    assert judge.seen_payloads == []


# ---------------------------------------------------------------------------
# Step 20a — first-class verdict_node IR field
# ---------------------------------------------------------------------------


def test_runner_judges_verdict_node_when_set_overrides_terminal():
    """Step 20a: when ``Topology.verdict_node`` is set, the judge scores
    THAT node's output, not the lex-first terminal.

    This is the contract that lets a supervisor or multi-terminal topology
    name its judged node explicitly, removing the lexical-luck fragility
    of "terminal = anything wired to END."
    """
    cfg = LLMConfig(model="google/gemini-2.5-pro", temperature=0.0)
    # terminal = ("a", "b") — lex-first is "a". verdict_node = "b" should
    # override and the judge should score b, not a.
    topo = Topology(
        name="verdict-node-override",
        nodes=tuple(
            Node(id=nid, role=f"r_{nid}", llm_config=cfg)
            for nid in ("a", "b", "c")
        ),
        edges=(
            Edge(from_id="a", to_id="b"),
            Edge(from_id="b", to_id="c"),
        ),
        entrypoint="a",
        terminal=("a", "b"),
        verdict_node="b",
    )
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.7))
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=judge,
        )
    )
    pld = judge.seen_payloads[0]
    out = {o.node_id: o.response_text for o in cell.node_outputs}
    assert pld["agent_output"] == out["b"]  # verdict_node, not terminal[0]
    assert pld["agent_output"] != out["a"]  # not the lex-first terminal
    assert pld["agent_output"] != out["c"]  # not the last walked node


def test_runner_falls_back_to_terminal_when_verdict_node_none():
    """Step 20a v0.5-compat: when verdict_node is None (default), the
    runner falls back to the lex-first terminal — the pre-20a behaviour,
    preserved bit-for-bit so v0.5 fixtures keep working unchanged."""
    cfg = LLMConfig(model="google/gemini-2.5-pro", temperature=0.0)
    topo = Topology(
        name="v05-shape",
        nodes=tuple(
            Node(id=nid, role=f"r_{nid}", llm_config=cfg)
            for nid in ("a", "b", "c")
        ),
        edges=(
            Edge(from_id="a", to_id="b"),
            Edge(from_id="b", to_id="c"),
        ),
        entrypoint="a",
        terminal=("b", "a"),  # unsorted — lex-first is "a"
        # verdict_node intentionally NOT set
    )
    assert topo.verdict_node is None
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.7))
    cell = _run(
        run_variant(
            topo,
            {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=judge,
        )
    )
    pld = judge.seen_payloads[0]
    out = {o.node_id: o.response_text for o in cell.node_outputs}
    assert pld["agent_output"] == out["a"]  # lex-first terminal


def test_run_judge_marks_failure_when_verdict_node_absent_from_outputs():
    """Step 20a defence-in-depth: if verdict_node is set but the declared
    node is absent from walked outputs, the judge call marks the cell
    judge_failure with verdict_node in the reason — NOT a fallback to
    terminal[0] (which would silently launder a wrong-node score)."""
    out = NodeOutput(
        node_id="n0",
        role="r0",
        response_text="some text",
        model_served_version="google/gemini-2.5-pro",
        cache_hit=False,
        latency_ms=0,
    )
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.9))
    result, judge_failure, reason = _run(
        _run_judge(
            judge=judge,
            judge_cache=None,
            outputs=[out],
            terminal=("n0",),  # would succeed on its own
            verdict_node="missing_node",  # but verdict_node overrides
            eval_row={"id": "e1", "input": "x"},
            variant_id="v1",
            rubric_hash="h",
            judge_model_version="m",
        )
    )
    assert result is None
    assert judge_failure is True
    assert "missing_node" in reason
    assert "verdict_node" in reason
    # The judge must NOT have been consulted — no fallback to terminal.
    assert judge.seen_payloads == []


def test_run_judge_suppresses_multi_terminal_warning_when_verdict_node_set(caplog):
    """Step 20a: the multi-terminal "ambiguous which one is judged" warning
    only fires when verdict_node is unset. Setting verdict_node is the
    user's explicit disambiguation — log noise would falsely imply
    ambiguity remains."""
    import logging
    out_a = NodeOutput(
        node_id="a", role="r", response_text="A",
        model_served_version="m", cache_hit=False, latency_ms=0,
    )
    out_b = NodeOutput(
        node_id="b", role="r", response_text="B",
        model_served_version="m", cache_hit=False, latency_ms=0,
    )
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.7))
    with caplog.at_level(logging.WARNING, logger="topofuzz.runner"):
        _run(
            _run_judge(
                judge=judge, judge_cache=None,
                outputs=[out_a, out_b],
                terminal=("a", "b"),
                verdict_node="b",
                eval_row={"id": "e1"}, variant_id="v1",
                rubric_hash="h", judge_model_version="m",
            )
        )
    assert not any(
        "multi-terminal" in r.message.lower()
        or "terminal nodes" in r.message.lower()
        for r in caplog.records
    )
    assert judge.seen_payloads[0]["agent_output"] == "B"


def test_run_judge_failure_reason_uses_unified_terminal_node_prefix():
    """L861: BOTH the explicit-verdict_node branch and the v0.5 terminal
    fallback emit `failure_reason` with the same `terminal node(s) `
    prefix. An operator's saved
    ``WHERE failure_reason LIKE 'terminal node%'`` SQL filter
    (established under v0.5 before verdict_node shipped in v0.6) must
    match BOTH failure shapes; the v0.6 verdict_node-absence rows used
    to silently miss that filter."""
    out = NodeOutput(
        node_id="actual_node",
        role="r0",
        response_text="walked here",
        model_served_version="m",
        cache_hit=False,
        latency_ms=0,
    )
    judge = FakeJudge(score_to_return=ContinuousScore(value=0.9))

    # Branch 1: explicit verdict_node is set but absent from outputs.
    _, jf_v, reason_v = _run(
        _run_judge(
            judge=judge,
            judge_cache=None,
            outputs=[out],
            terminal=("actual_node",),
            verdict_node="missing_v",
            eval_row={"id": "e1"},
            variant_id="v1",
            rubric_hash="h",
            judge_model_version="m",
        )
    )
    assert jf_v is True
    assert reason_v.startswith("terminal node(s) "), (
        f"verdict_node-absence reason must use unified prefix, got: {reason_v!r}"
    )
    # The `verdict_node=` literal stays in the suffix so cause is
    # grep-distinguishable from a v0.5 multi-terminal absence.
    assert "verdict_node=" in reason_v
    assert "missing_v" in reason_v

    # Branch 2: no verdict_node, lex-first terminal fallback absent.
    _, jf_t, reason_t = _run(
        _run_judge(
            judge=judge,
            judge_cache=None,
            outputs=[out],
            terminal=("missing_a", "missing_b"),
            verdict_node=None,
            eval_row={"id": "e1"},
            variant_id="v1",
            rubric_hash="h",
            judge_model_version="m",
        )
    )
    assert jf_t is True
    assert reason_t.startswith("terminal node(s) "), (
        f"v0.5 terminal-fallback reason must use unified prefix, got: {reason_t!r}"
    )
    # The v0.5 branch does NOT carry `verdict_node=` so operators can
    # still distinguish the two failure shapes if they want.
    assert "verdict_node=" not in reason_t


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
    assert ver == 6  # v6: judge_results.votes_json (per-vendor jury breakdown)


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
            judge=FakeJudge(
                score_to_return=ContinuousScore(
                    value=0.4, rationale="continuous judge said so"
                )
            ),
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
    # Continuous cells now persist the judge's rationale (was always NULL).
    assert by_eval["e1"][4] == "continuous judge said so"
    assert by_eval["e2"][1] == "binary"
    assert by_eval["e2"][3] == 0
    assert by_eval["e2"][4] == "r"
    assert by_eval["e3"][1] == "failure"
    assert "boom" in by_eval["e3"][5]
    assert by_eval["e4"][1] == "absent"


def test_run_writer_continuous_empty_rationale_persists_as_empty_string(tmp_path):
    """A ContinuousScore with no rationale persists "" — the same sentinel
    the BinaryScore path uses for an empty rationale. The value is written
    verbatim; it is NOT collapsed to SQL NULL (that would split the
    "no rationale" sentinel by judge_kind)."""
    topo = _toy_topology(n=1)
    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(score_to_return=ContinuousScore(value=0.6)),
        )
    )
    with RunWriter(tmp_path, "run-1", topo) as w:
        w.write(cell)
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        rationale = db.execute(
            "SELECT rationale FROM judge_results WHERE eval_id = 'e1'"
        ).fetchone()[0]
    finally:
        db.close()
    assert rationale == ""


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
        # Schema bumped v1 → v6 cascade: cost telemetry (v2) + §A runs
        # cols (v3) + rubric_hash on runs (v4 / step 9) + flip_confirmations
        # table (v5 / noise-floor `topofuzz confirm`) + judge_results
        # votes_json column (v6 / F3 jury vote breakdown).
        assert db.execute("PRAGMA user_version").fetchone()[0] == 6
        # v5: the flip_confirmations table exists post-migrate (created
        # by CREATE TABLE IF NOT EXISTS — no ALTER needed for a new table).
        legacy_tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "flip_confirmations" in legacy_tables
        # v6: votes_json column landed on judge_results — NULL on legacy
        # rows so single-judge cells from v1-v5 round-trip unchanged.
        judge_cols_v6 = {
            row[1]
            for row in db.execute(
                "PRAGMA table_info(judge_results)"
            ).fetchall()
        }
        assert "votes_json" in judge_cols_v6
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
    Cell carries a ContinuousScore. F3 majority-vote contract: votes
    [0.4, 0.6, 0.8] binarise to [0, 1, 1] → 2-of-3 approve → 1.0."""
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
    assert cell.judge_result.value == pytest.approx(1.0)
    assert cell.judge_failure is False
    # DoD #14: vendor_overlap reflects all three vendors that produced a vote.
    assert judge.judge_independence == {
        "vendor_overlap": ["anthropic", "gemini", "openai"],
    }
    assert judge.judge_failure_rate == 0.0


def test_runner_accepts_jury_judge_one_unparseable(tmp_path):
    """1 of 3 vendors unparseable → Cell still gets a ContinuousScore
    (majority of the surviving 2 — both vote approve so unanimous
    2-of-2 majority = 1.0), judge_failure stays False, JuryJudge
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
    assert cell.judge_result.value == pytest.approx(1.0)
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


def test_runner_jury_judge_dry_run_returns_majority_approve(tmp_path):
    """Dry-run jury never calls any LLM. Each vendor returns
    ContinuousScore(0.5); inclusive-threshold binarisation maps each
    to APPROVE; 3-of-3 unanimous → majority scalar 1.0."""
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
    assert cell.judge_result.value == 1.0
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
        # Surviving 2/2 vendors both approve (0.6 + 0.8 each ≥ 0.5) →
        # 2-of-2 unanimous → majority 1.0.
        assert cell.judge_result.value == pytest.approx(1.0)
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


# ---------------------------------------------------------------------------
# F3 — jury votes persistence (votes_json column + traces.jsonl mirror)
# ---------------------------------------------------------------------------


def test_judge_row_packs_votes_json_when_continuous_score_carries_votes():
    """``_judge_row`` returns 6 columns; the last is JSON when votes are
    present, NULL otherwise."""
    from topofuzz.runner import _judge_row

    cell = _run(
        run_variant(
            _toy_topology(n=1), {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(
                score_to_return=ContinuousScore(
                    value=0.7,
                    rationale="combined",
                    votes=(
                        ("gemini", 0.7, "approve"),
                        ("deepseek", 0.6, "marginal"),
                    ),
                ),
            ),
        )
    )
    kind, score, passed, rationale, failure, votes_json = _judge_row(cell)
    assert kind == "continuous"
    assert score == pytest.approx(0.7)
    assert passed is None
    assert rationale == "combined"
    assert failure is None
    assert votes_json is not None
    parsed = json.loads(votes_json)
    assert parsed == [
        {"vendor": "gemini", "score": 0.7, "rationale": "approve"},
        {"vendor": "deepseek", "score": 0.6, "rationale": "marginal"},
    ]


def test_judge_row_returns_null_votes_json_for_single_judge_continuous():
    """A ContinuousScore with the default ``votes=()`` (e.g. SingleJudge)
    persists votes_json as NULL — readers use that as the "no jury
    breakdown" sentinel."""
    from topofuzz.runner import _judge_row

    cell = _run(
        run_variant(
            _toy_topology(n=1), {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(
                score_to_return=ContinuousScore(value=0.4, rationale="single"),
            ),
        )
    )
    _, _, _, _, _, votes_json = _judge_row(cell)
    assert votes_json is None


def test_run_writer_persists_votes_json_round_trip(tmp_path):
    """End-to-end: write a Cell whose ContinuousScore carries votes, read
    back the votes_json column, confirm the persisted JSON survives."""
    topo = _toy_topology(n=1)
    cell = _run(
        run_variant(
            topo, {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(
                score_to_return=ContinuousScore(
                    value=0.55,
                    rationale="jury",
                    votes=(
                        ("gemini", 0.7, "ok"),
                        ("gemini_2", 0.4, "concerns"),
                        ("gemini_3", 0.55, "neutral"),
                    ),
                ),
            ),
        )
    )
    with RunWriter(tmp_path, "run-jury-1", topo) as w:
        w.write(cell)
    db = sqlite3.connect(str(tmp_path / "run.sqlite"))
    try:
        row = db.execute(
            "SELECT score_value, votes_json FROM judge_results "
            "WHERE run_id='run-jury-1'"
        ).fetchone()
    finally:
        db.close()
    assert row is not None
    assert row[0] == pytest.approx(0.55)
    parsed = json.loads(row[1])
    assert [entry["vendor"] for entry in parsed] == [
        "gemini", "gemini_2", "gemini_3",
    ]
    assert parsed[0]["score"] == pytest.approx(0.7)
    assert parsed[2]["rationale"] == "neutral"

    # And the traces.jsonl mirror carries the same data so off-line
    # replay tooling can rebuild the breakdown without re-opening SQLite.
    line = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(line)
    assert record["judge_result"]["kind"] == "continuous"
    assert record["judge_result"]["votes"] == parsed


def test_judge_row_caps_votes_json_size():
    """F3 Fuzzer m4 audit fix: when the per-vendor rationale × N pushes
    the JSON column past _VOTES_JSON_MAX_BYTES, the packer drops the
    rationale strings but keeps vendor + score intact — the
    integrity-critical data."""
    from topofuzz.runner import _VOTES_JSON_MAX_BYTES, _judge_row

    huge = "x" * 20000  # well over the cap
    cell = _run(
        run_variant(
            _toy_topology(n=1), {"id": "e1", "input": "x"},
            llm_callable=FakeLLM(),
            judge=FakeJudge(
                score_to_return=ContinuousScore(
                    value=1.0,
                    rationale="joined",
                    votes=(
                        ("gemini", 0.7, huge),
                        ("deepseek", 0.6, huge),
                        ("openai", 0.55, huge),
                    ),
                ),
            ),
        )
    )
    _, _, _, _, _, votes_json = _judge_row(cell)
    assert votes_json is not None
    assert len(votes_json) <= _VOTES_JSON_MAX_BYTES
    parsed = json.loads(votes_json)
    # Vendor names + scores survive the trim; rationales are dropped to "".
    assert [e["vendor"] for e in parsed] == ["gemini", "deepseek", "openai"]
    assert [e["score"] for e in parsed] == [0.7, 0.6, 0.55]
    for e in parsed:
        assert e["rationale"] == ""
