# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for the §5a static risk scanner (Weekend 5 step 14).

Five synthetic topologies per BLUEPRINT_v5plus.md line 406:

* HIGH — 4-agent linear chain, no aggregator
* MEDIUM — 2-agent linear chain, no aggregator
* LOW — chain with terminal aggregator
* all-same-model — promotion from LOW/MEDIUM → MEDIUM/HIGH
* with-aggregator — chain that terminates in aggregator absorbs HIGH

Plus targeted promotion + edge-case tests so every row of the §5a table
exercises both fire and non-fire paths. The scanner is a pure function
of the IR, so we construct ``Topology`` directly — no LangGraph
required — keeping the test verify-path-safe (matches the import-guard
contract for the scanner module itself).
"""

from __future__ import annotations

from topofuzz.ir import Edge, LLMConfig, Node, Topology
from topofuzz.scan import ScanResult, scan


def _make_chain(
    n_agents: int,
    *,
    kinds: tuple[str, ...] | None = None,
    models: tuple[str, ...] | None = None,
    temperatures: tuple[float, ...] | None = None,
) -> Topology:
    """Construct an ``a0 -> a1 -> ... -> a{n-1}`` linear chain Topology.

    Each per-node attribute can be overridden by passing a tuple of
    length ``n_agents``. Defaults: kind=agent, model=``"m"``, temp=0.0.
    The chain has exactly one terminal (``a{n-1}``) and one entrypoint
    (``a0``).
    """
    ids = tuple(f"a{i}" for i in range(n_agents))
    kinds_t = kinds if kinds is not None else ("agent",) * n_agents
    models_t = models if models is not None else ("m",) * n_agents
    temps_t = temperatures if temperatures is not None else (0.0,) * n_agents
    nodes = tuple(
        Node(
            id=ids[i],
            role=ids[i],
            kind=kinds_t[i],  # type: ignore[arg-type]
            llm_config=LLMConfig(model=models_t[i], temperature=temps_t[i]),
        )
        for i in range(n_agents)
    )
    edges = tuple(Edge(from_id=ids[i], to_id=ids[i + 1]) for i in range(n_agents - 1))
    return Topology(
        name="t",
        nodes=nodes,
        edges=edges,
        entrypoint=ids[0],
        terminal=(ids[-1],),
    )


# --- 5 canonical synthetic topologies per DoD #11 ---------------------


def test_scan_high_risk_linear_chain_4_agents() -> None:
    """4-agent linear chain, distinct models, temp=0 → HIGH base, no promo."""
    topology = _make_chain(
        4, models=("m1", "m2", "m3", "m4"), temperatures=(0.0, 0.0, 0.0, 0.0)
    )
    result = scan(topology)
    assert result.badge == "HIGH"
    assert "linear_chain_3plus_no_aggregator" in result.matched_patterns
    assert "all_agents_share_model" not in result.matched_patterns
    assert "temperature_gt_0" not in result.matched_patterns
    assert "HIGH risk" in result.rationale


def test_scan_medium_risk_linear_chain_2_agents() -> None:
    """2-agent linear chain → MEDIUM base."""
    topology = _make_chain(2, models=("m1", "m2"))
    result = scan(topology)
    assert result.badge == "MEDIUM"
    assert "linear_chain_2_no_aggregator" in result.matched_patterns
    assert "linear_chain_3plus_no_aggregator" not in result.matched_patterns


def test_scan_low_risk_chain_with_terminal_aggregator() -> None:
    """``a0 (agent) -> agg (aggregator)`` chain → LOW (terminal absorbs)."""
    nodes = (
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m1")),
        Node(id="agg", role="agg", kind="aggregator", llm_config=LLMConfig(model="m2")),
    )
    edges = (Edge(from_id="a0", to_id="agg"),)
    topology = Topology(
        name="t", nodes=nodes, edges=edges, entrypoint="a0", terminal=("agg",)
    )
    result = scan(topology)
    assert result.badge == "LOW"
    assert "chain_with_terminal_aggregator" in result.matched_patterns
    # The 2-agent base rule must NOT fire when an aggregator is present —
    # the aggregator changes which row of the §5a table applies.
    assert "linear_chain_2_no_aggregator" not in result.matched_patterns


def test_scan_all_same_model_promotes_medium_to_high() -> None:
    """2-agent chain (MEDIUM base) + all share model (+1 step) → HIGH."""
    topology = _make_chain(2, models=("shared", "shared"))
    result = scan(topology)
    assert result.badge == "HIGH"
    assert "linear_chain_2_no_aggregator" in result.matched_patterns
    assert "all_agents_share_model" in result.matched_patterns
    assert "Promoted by" in result.rationale


def test_scan_with_aggregator_4_chain_stays_low() -> None:
    """4-node chain where the terminal is an aggregator → LOW base.

    Even though the chain length matches the HIGH rule on count, the
    HIGH rule's predicate requires "no aggregator anywhere", so HIGH
    does not fire. LOW (terminal aggregator) wins.
    """
    nodes = (
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m1")),
        Node(id="a1", role="a1", kind="agent", llm_config=LLMConfig(model="m2")),
        Node(id="a2", role="a2", kind="agent", llm_config=LLMConfig(model="m3")),
        Node(id="agg", role="agg", kind="aggregator", llm_config=LLMConfig(model="m4")),
    )
    edges = (
        Edge(from_id="a0", to_id="a1"),
        Edge(from_id="a1", to_id="a2"),
        Edge(from_id="a2", to_id="agg"),
    )
    topology = Topology(
        name="t", nodes=nodes, edges=edges, entrypoint="a0", terminal=("agg",)
    )
    result = scan(topology)
    assert result.badge == "LOW"
    assert "chain_with_terminal_aggregator" in result.matched_patterns
    assert "linear_chain_3plus_no_aggregator" not in result.matched_patterns


# --- promotion + edge-case coverage -----------------------------------


def test_scan_temperature_gt_0_promotes_high_chain_stays_high() -> None:
    """HIGH base + temp promotion → stays HIGH (capped at HIGH)."""
    topology = _make_chain(
        3, models=("m1", "m2", "m3"), temperatures=(0.0, 0.3, 0.0)
    )
    result = scan(topology)
    assert result.badge == "HIGH"
    assert "temperature_gt_0" in result.matched_patterns


def test_scan_double_promotion_caps_at_high() -> None:
    """MEDIUM base + 2 promotions (all-share + temp>0) → HIGH (capped)."""
    topology = _make_chain(
        2, models=("shared", "shared"), temperatures=(0.0, 0.5)
    )
    result = scan(topology)
    assert result.badge == "HIGH"
    assert "all_agents_share_model" in result.matched_patterns
    assert "temperature_gt_0" in result.matched_patterns


def test_scan_router_midflow_medium() -> None:
    """``agent -> router -> agent -> agent`` with router NOT at boundary."""
    nodes = (
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m1")),
        Node(id="r0", role="r0", kind="router", llm_config=LLMConfig(model="m2")),
        Node(id="a1", role="a1", kind="agent", llm_config=LLMConfig(model="m3")),
        Node(id="a2", role="a2", kind="agent", llm_config=LLMConfig(model="m4")),
    )
    edges = (
        Edge(from_id="a0", to_id="r0"),
        Edge(from_id="r0", to_id="a1"),
        Edge(from_id="a1", to_id="a2"),
    )
    topology = Topology(
        name="t", nodes=nodes, edges=edges, entrypoint="a0", terminal=("a2",)
    )
    result = scan(topology)
    # 3 agents + 1 router; linear chain holds → linear_3plus_no_aggregator
    # fires (HIGH). Mid-flow router also fires (MEDIUM). max → HIGH.
    assert result.badge == "HIGH"
    assert "chain_with_router_midflow" in result.matched_patterns
    assert "linear_chain_3plus_no_aggregator" in result.matched_patterns


def test_scan_router_at_entrypoint_does_not_count_as_midflow() -> None:
    """Router AS entrypoint is the entry shape, not 'mid-flow'."""
    nodes = (
        Node(id="r0", role="r0", kind="router", llm_config=LLMConfig(model="m1")),
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m2")),
    )
    edges = (Edge(from_id="r0", to_id="a0"),)
    topology = Topology(
        name="t", nodes=nodes, edges=edges, entrypoint="r0", terminal=("a0",)
    )
    result = scan(topology)
    assert "chain_with_router_midflow" not in result.matched_patterns


def test_scan_single_agent_low_no_match() -> None:
    """1-node topology: no linear-chain rule fires; the all-share rule
    is gated to n >= 2 because the heterogeneity rationale is vacuous
    on a 1-agent topology. Net result: LOW with no matched patterns.
    """
    topology = _make_chain(1)
    result = scan(topology)
    assert result.badge == "LOW"
    assert result.matched_patterns == ()
    assert "no §5a structural patterns matched" in result.rationale


def test_scan_single_agent_distinct_implicit_no_promotion_low() -> None:
    """1-agent topology with no llm_config → no promotions, LOW baseline."""
    node = Node(id="a0", role="a0", kind="agent", llm_config=None)
    topology = Topology(
        name="t", nodes=(node,), edges=(), entrypoint="a0", terminal=("a0",)
    )
    result = scan(topology)
    assert result.badge == "LOW"
    assert result.matched_patterns == ()
    assert "no §5a structural patterns matched" in result.rationale


def test_scan_all_share_promotion_skipped_when_any_config_is_none() -> None:
    """If any agent has llm_config=None, all-share rule cannot evaluate
    and is conservatively skipped (no promo)."""
    nodes = (
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m")),
        Node(id="a1", role="a1", kind="agent", llm_config=None),
        Node(id="a2", role="a2", kind="agent", llm_config=LLMConfig(model="m")),
    )
    edges = (
        Edge(from_id="a0", to_id="a1"),
        Edge(from_id="a1", to_id="a2"),
    )
    topology = Topology(
        name="t", nodes=nodes, edges=edges, entrypoint="a0", terminal=("a2",)
    )
    result = scan(topology)
    # Linear 3+ no aggregator → HIGH base. No promo (config-None blocked).
    assert result.badge == "HIGH"
    assert "all_agents_share_model" not in result.matched_patterns


def test_scan_result_to_json_dict_is_stable_shape() -> None:
    """JSON dict has exactly three keys with the documented types."""
    result = scan(_make_chain(2, models=("m1", "m2")))
    d = result.to_json_dict()
    assert set(d.keys()) == {"badge", "matched_patterns", "rationale"}
    assert d["badge"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(d["matched_patterns"], list)
    assert all(isinstance(p, str) for p in d["matched_patterns"])
    assert isinstance(d["rationale"], str)


def test_scan_result_is_frozen() -> None:
    """ScanResult is a frozen dataclass — mutation must raise."""
    import dataclasses

    result = scan(_make_chain(2))
    assert isinstance(result, ScanResult)
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.badge = "LOW"  # type: ignore[misc]


def test_scan_runs_well_under_one_second() -> None:
    """BLUEPRINT DoD #11: ``topofuzz scan`` must complete in <1s. The
    pure-function scanner is the dominant cost path; loaders + JSON
    encode are negligible. A 50-agent synthetic chain is a generous
    upper bound on real-world topology size for v0.5.

    Budget bumped from 1s → 5s per audit fix NIT-1/m4 (pass-1): cold
    Windows CI under Defender real-time scanning has been observed
    inflating microbenchmarks; 5s × 100 iterations on a 50-node chain
    still catches accidental O(n²) regressions without flaking under
    load.
    """
    import time

    big = _make_chain(50, models=tuple(f"m{i}" for i in range(50)))
    t0 = time.perf_counter()
    for _ in range(100):
        scan(big)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"100 scans of 50-node chain took {elapsed:.3f}s"


def test_scan_rejects_cycle_topologies() -> None:
    """RT-M2 audit fix (pass-1): an explicit ``entrypoint == terminal``
    early-out in ``_is_linear_chain`` makes cycle rejection intentional.

    Test path: construct an IR that bypasses the Topology validator's
    cycle hygiene (it allows entrypoint==terminal as long as the id
    exists) and confirm the scanner treats it as "not a chain". We do
    this by building a 2-node graph where both nodes form a cycle:
    ``a0 -> a1 -> a0``, with entrypoint=a0 and terminal=(a0,). Topology
    accepts this shape; ``_is_linear_chain`` must reject it.
    """
    from topofuzz.scan import _is_linear_chain

    nodes = (
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m")),
        Node(id="a1", role="a1", kind="agent", llm_config=LLMConfig(model="m")),
    )
    edges = (
        Edge(from_id="a0", to_id="a1"),
        Edge(from_id="a1", to_id="a0"),
    )
    topology = Topology(
        name="cycle", nodes=nodes, edges=edges, entrypoint="a0", terminal=("a0",)
    )
    assert _is_linear_chain(topology) is False
    # The full scan should not award MEDIUM/HIGH from the chain rules
    # on this shape; the only thing that could fire is the all-share
    # promotion, capping at MEDIUM.
    result = scan(topology)
    assert result.badge == "MEDIUM"
    assert "linear_chain_3plus_no_aggregator" not in result.matched_patterns
    assert "linear_chain_2_no_aggregator" not in result.matched_patterns


def test_scan_rejects_self_loop_on_terminal() -> None:
    """RT-M1 audit fix (pass-2): a self-loop on the terminal node is a
    cycle shape that the entrypoint==terminal early-out does NOT catch
    (entry=a0, terminal=a1, a1→a1). The edge-count + degree-walk in
    ``_is_linear_chain`` must still reject it: terminal a1 has
    out_deg=1 (the self-loop) while expected_out=0 → False.
    """
    from topofuzz.scan import _is_linear_chain

    nodes = (
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m1")),
        Node(id="a1", role="a1", kind="agent", llm_config=LLMConfig(model="m2")),
    )
    # 2 edges on 2 nodes — the edge-count gate (n_edges == n_nodes - 1
    # → 1) already rejects this shape, so the degree-walk never runs.
    # The assertion is still meaningful: it pins the rejection
    # regardless of which guard catches it.
    edges = (
        Edge(from_id="a0", to_id="a1"),
        Edge(from_id="a1", to_id="a1"),
    )
    topology = Topology(
        name="self_loop", nodes=nodes, edges=edges, entrypoint="a0", terminal=("a1",)
    )
    assert _is_linear_chain(topology) is False
    result = scan(topology)
    assert "linear_chain_2_no_aggregator" not in result.matched_patterns
    assert "linear_chain_3plus_no_aggregator" not in result.matched_patterns


def test_scan_terminal_aggregator_requires_chain_shape() -> None:
    """CR-MAJOR-2 / RT-M1 audit fix (pass-1): the terminal-aggregator
    rule fires only on a chain shape with at least one non-aggregator
    upstream. A fan-in into a single aggregator terminal is NOT a
    chain — the §5a rationale ("aggregator absorbs upstream variance")
    presumes a single linear upstream path.
    """
    # Two entrypoint-candidates fan into one aggregator. Topology
    # construction rejects multi-entrypoint setups indirectly: there is
    # exactly one ``entrypoint`` field, so we model fan-in as two
    # nodes with no inbound edges but one declared entrypoint. The IR
    # validator does not block this — it only checks id existence.
    nodes = (
        Node(id="a0", role="a0", kind="agent", llm_config=LLMConfig(model="m1")),
        Node(id="a1", role="a1", kind="agent", llm_config=LLMConfig(model="m2")),
        Node(id="agg", role="agg", kind="aggregator", llm_config=LLMConfig(model="m3")),
    )
    edges = (
        Edge(from_id="a0", to_id="agg"),
        Edge(from_id="a1", to_id="agg"),
    )
    topology = Topology(
        name="fan_in", nodes=nodes, edges=edges, entrypoint="a0", terminal=("agg",)
    )
    result = scan(topology)
    # ``_is_linear_chain`` rejects (a1 has in_deg=0 but is not entry).
    # Without the chain-shape guard, the LOW terminal-aggregator rule
    # would still fire. The guard prevents that wrong-by-rationale
    # badge.
    assert "chain_with_terminal_aggregator" not in result.matched_patterns


def test_scan_terminal_aggregator_requires_non_aggregator_upstream() -> None:
    """CR-MAJOR-2 / RT-M1 audit fix (pass-1): a 1-node topology where
    the lone node is an aggregator (and is both entry + terminal) is
    not a "chain with a terminal aggregator" — there is no upstream.
    The rule must NOT fire.
    """
    node = Node(
        id="agg", role="agg", kind="aggregator", llm_config=LLMConfig(model="m")
    )
    topology = Topology(
        name="lone_agg",
        nodes=(node,),
        edges=(),
        entrypoint="agg",
        terminal=("agg",),
    )
    result = scan(topology)
    assert "chain_with_terminal_aggregator" not in result.matched_patterns
    # No agents → no promotions either; no base rule applies → LOW.
    assert result.badge == "LOW"
    assert result.matched_patterns == ()
