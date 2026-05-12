"""Tests for the LangGraph -> Topology IR adapter."""

from __future__ import annotations

import sys
from typing import TypedDict

import pytest

# Skip the whole file if langgraph somehow isn't installed (e.g. running
# against the bare tsf[verify] dep set). The adapter itself doesn't need
# langgraph at import time, but every test builds a real StateGraph.
pytest.importorskip("langgraph")

from topofuzz.adapter import from_langgraph  # noqa: E402
from topofuzz.fixtures.lending_4role import (  # noqa: E402
    build_graph as build_lending_4role,
)


class _S(TypedDict, total=False):
    x: int


def _noop(state):
    return state


def test_adapter_module_does_not_eagerly_import_langgraph():
    # `tsf[verify]` users install topofuzz without langgraph; importing
    # `topofuzz.adapter` (e.g. for the from_langgraph re-export from a
    # higher-level entry point) must not pull langgraph into sys.modules.
    # Verified in a fresh subprocess so the test runner's preloaded
    # langgraph doesn't pollute the check.
    import subprocess

    script = (
        "import sys\n"
        "before = {m for m in sys.modules if m == 'langgraph' "
        "or m.startswith('langgraph.')}\n"
        "import topofuzz.adapter\n"
        "after = {m for m in sys.modules if m == 'langgraph' "
        "or m.startswith('langgraph.')}\n"
        "leaked = after - before\n"
        "if leaked:\n"
        "    raise SystemExit(f'leaked langgraph modules: {sorted(leaked)}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"adapter eager-imports langgraph (or subprocess failed): "
        f"stderr={result.stderr!r}"
    )


class TestLendingFixture:
    def test_nodes_in_declared_order(self):
        topo = from_langgraph(build_lending_4role(), name="lending_4role")
        assert [n.id for n in topo.nodes] == [
            "intake",
            "risk",
            "policy",
            "decision",
        ]

    def test_node_defaults_propagate(self):
        topo = from_langgraph(build_lending_4role(), name="lending_4role")
        for node in topo.nodes:
            assert node.kind == "agent"
            assert node.llm_config is None
            assert node.pinned is False
            assert node.role == node.id

    def test_entrypoint_and_terminal(self):
        topo = from_langgraph(build_lending_4role(), name="lending_4role")
        assert topo.entrypoint == "intake"
        assert topo.terminal == ("decision",)

    def test_edges(self):
        topo = from_langgraph(build_lending_4role(), name="lending_4role")
        assert {(e.from_id, e.to_id) for e in topo.edges} == {
            ("intake", "risk"),
            ("risk", "policy"),
            ("policy", "decision"),
        }

    def test_name_passes_through(self):
        topo = from_langgraph(build_lending_4role(), name="lending_4role")
        assert topo.name == "lending_4role"

    def test_canonical_hash_stable_across_walks(self):
        # Round-trip property: walking the same graph twice produces IRs
        # with the same JCS canonical hash.
        a = from_langgraph(build_lending_4role(), name="lending_4role")
        b = from_langgraph(build_lending_4role(), name="lending_4role")
        assert a.canonical_hash() == b.canonical_hash()


class TestConditionalEdges:
    def test_with_path_map_is_in_scope(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(_S)
        g.add_node("router", _noop)
        g.add_node("yes", _noop)
        g.add_node("no", _noop)
        g.add_edge(START, "router")
        g.add_conditional_edges(
            "router", lambda s: "yes", {"yes": "yes", "no": "no"}
        )
        g.add_edge("yes", END)
        g.add_edge("no", END)

        topo = from_langgraph(g.compile(), name="branchy")
        assert {n.id for n in topo.nodes} == {"router", "yes", "no"}
        assert topo.entrypoint == "router"
        assert set(topo.terminal) == {"yes", "no"}
        # Pin the full edge set — an unexpected extra edge would slip
        # past a `in` check otherwise.
        assert {(e.from_id, e.to_id) for e in topo.edges} == {
            ("router", "yes"),
            ("router", "no"),
        }

    def test_without_path_map_raises(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(_S)
        g.add_node("router", _noop)
        g.add_node("a", _noop)
        g.add_edge(START, "router")
        g.add_conditional_edges("router", lambda s: "a")  # no path_map
        g.add_edge("a", END)

        with pytest.raises(NotImplementedError, match="path_map"):
            from_langgraph(g.compile())


class TestSubgraphs:
    def test_depth_one_subgraph_in_scope(self):
        # BLUEPRINT §5 says depth-1 subgraphs are in scope. xray=1 in
        # LangGraph 1.x flattens the wrapper: inner nodes appear with
        # namespaced ids (`sub:inner_a`), the wrapper node disappears,
        # and edges adjacent to the wrapper land on the namespaced
        # boundary nodes.
        from langgraph.graph import END, START, StateGraph

        inner = StateGraph(_S)
        inner.add_node("inner_a", _noop)
        inner.add_node("inner_b", _noop)
        inner.add_edge(START, "inner_a")
        inner.add_edge("inner_a", "inner_b")
        inner.add_edge("inner_b", END)

        outer = StateGraph(_S)
        outer.add_node("pre", _noop)
        outer.add_node("sub", inner.compile())
        outer.add_node("post", _noop)
        outer.add_edge(START, "pre")
        outer.add_edge("pre", "sub")
        outer.add_edge("sub", "post")
        outer.add_edge("post", END)

        topo = from_langgraph(outer.compile(), name="with_subgraph")
        ids = {n.id for n in topo.nodes}
        assert ids == {"pre", "post", "sub:inner_a", "sub:inner_b"}
        assert topo.entrypoint == "pre"
        assert topo.terminal == ("post",)
        # Inner edges plus the boundary edges to/from the namespaced
        # subgraph nodes must all be present.
        edge_pairs = {(e.from_id, e.to_id) for e in topo.edges}
        assert edge_pairs == {
            ("pre", "sub:inner_a"),
            ("sub:inner_a", "sub:inner_b"),
            ("sub:inner_b", "post"),
        }

    def test_depth_two_raises(self):
        from langgraph.graph import END, START, StateGraph

        inner = StateGraph(_S)
        inner.add_node("ix", _noop)
        inner.add_edge(START, "ix")
        inner.add_edge("ix", END)

        middle = StateGraph(_S)
        middle.add_node("mid_inner", inner.compile())
        middle.add_edge(START, "mid_inner")
        middle.add_edge("mid_inner", END)

        outer = StateGraph(_S)
        outer.add_node("wrapped_middle", middle.compile())
        outer.add_edge(START, "wrapped_middle")
        outer.add_edge("wrapped_middle", END)

        with pytest.raises(NotImplementedError, match="depth"):
            from_langgraph(outer.compile())


class _FakeBuilder:
    """Stand-in for `StateGraph._builder` shape that the adapter consumes.

    We construct corner-case graphs that LangGraph's own input validation
    would refuse (sentinel-named nodes, malformed entrypoints) by stubbing
    the exact attributes the adapter touches: `nodes`, `branches`, `edges`.
    """

    def __init__(self, nodes=None, branches=None):
        from collections import defaultdict

        self.nodes = nodes or {}
        # `branches` shape: dict[source_node, dict[branch_name, BranchSpec]].
        self.branches = branches or defaultdict(dict)


class _FakeDrawableEdge:
    __slots__ = ("source", "target", "conditional")

    def __init__(self, source, target, conditional=False):
        self.source = source
        self.target = target
        self.conditional = conditional


class _FakeDrawable:
    def __init__(self, node_ids, edges):
        # Adapter only iterates `nodes` and `edges`; a dict-of-None mimics
        # LangGraph's DrawableGraph.nodes (a name -> Node mapping).
        self.nodes = {nid: None for nid in node_ids}
        self.edges = edges


class _FakeCompiled:
    def __init__(self, builder, drawable):
        self.builder = builder
        self._drawable = drawable

    def get_graph(self, xray=0):
        return self._drawable


class TestEntrypointErrors:
    def test_no_start_edge_raises_clear_message(self):
        # Graph has nodes and an END edge but no START edge. We can't get
        # LangGraph itself to compile this (it validates entrypoints), so
        # we hand the adapter a duck-typed compiled-graph stub.
        compiled = _FakeCompiled(
            builder=_FakeBuilder(nodes={"a": None}),
            drawable=_FakeDrawable(
                node_ids=["__start__", "a", "__end__"],
                edges=[_FakeDrawableEdge("a", "__end__")],
            ),
        )
        with pytest.raises(NotImplementedError, match="no entrypoint"):
            from_langgraph(compiled)

    def test_multi_start_fanout_raises(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(_S)
        g.add_node("a", _noop)
        g.add_node("b", _noop)
        # `add_conditional_edges` from START is the realistic way users
        # construct multi-entrypoint graphs.
        g.add_conditional_edges(
            START, lambda s: "a", {"a": "a", "b": "b"}
        )
        g.add_edge("a", END)
        g.add_edge("b", END)
        # Strict regex on the multi-START-specific phrase so a future
        # refactor that merges this into the no-START branch fails this
        # test (no-START emits "no entrypoint" — see test above).
        with pytest.raises(NotImplementedError, match="Multi-START fan-out"):
            from_langgraph(g.compile())


class TestSentinelCollision:
    def test_user_node_named_start_raises_via_fake_builder(self):
        # LangGraph 1.x blocks `add_node("__start__", ...)` at construction
        # with `ValueError: Node '__start__' is reserved.`, so the only way
        # for a sentinel-collision builder to reach the adapter is via a
        # bypass (direct dict mutation, custom Pregel subclass, future API
        # change). The adapter's defensive check guards against that —
        # we exercise it with a fake builder.
        compiled = _FakeCompiled(
            builder=_FakeBuilder(nodes={"__start__": None, "a": None}),
            drawable=_FakeDrawable(
                node_ids=["__start__", "a", "__end__"],
                edges=[
                    _FakeDrawableEdge("__start__", "a"),
                    _FakeDrawableEdge("a", "__end__"),
                ],
            ),
        )
        with pytest.raises(NotImplementedError, match="sentinel"):
            from_langgraph(compiled)


class TestCompiledInputContract:
    def test_non_compiled_input_raises_clearly(self):
        class _NotCompiled:
            pass

        with pytest.raises(NotImplementedError, match="StateGraph.compile"):
            from_langgraph(_NotCompiled())


class TestLlmConfigs:
    def test_llm_configs_kwarg_attaches_per_node(self):
        # Without populating llm_config, BLUEPRINT §6 `role_swap` and
        # `decoding_param_swap` generators are dead on arrival. The
        # adapter must give callers a way to attach configs at walk time.
        from topofuzz.ir import LLMConfig

        cfg_a = LLMConfig(model="anthropic/claude-opus-4-7", seed=1)
        cfg_b = LLMConfig(model="openai/gpt-5", temperature=0.3)
        topo = from_langgraph(
            build_lending_4role(),
            name="lending_4role",
            llm_configs={"intake": cfg_a, "risk": cfg_b},
        )
        by_id = {n.id: n for n in topo.nodes}
        assert by_id["intake"].llm_config == cfg_a
        assert by_id["risk"].llm_config == cfg_b
        # Unmapped nodes still get None — no spurious default config.
        assert by_id["policy"].llm_config is None
        assert by_id["decision"].llm_config is None

    def test_unknown_llm_configs_key_raises(self):
        # Typos are the realistic failure mode here. We surface them
        # immediately rather than silently dropping the entry.
        from topofuzz.ir import LLMConfig

        with pytest.raises(NotImplementedError, match="not in the walked graph"):
            from_langgraph(
                build_lending_4role(),
                llm_configs={
                    "intake": LLMConfig(model="m"),
                    "typo_id": LLMConfig(model="m"),
                },
            )

    def test_llm_configs_changes_canonical_hash(self):
        # Attaching configs MUST change the canonical hash — the LLM
        # cache key in §7 depends on this varying with model identity.
        from topofuzz.ir import LLMConfig

        bare = from_langgraph(build_lending_4role())
        enriched = from_langgraph(
            build_lending_4role(),
            llm_configs={"intake": LLMConfig(model="m")},
        )
        assert bare.canonical_hash() != enriched.canonical_hash()


class TestReservedNodeIdChars:
    def test_colon_in_node_id_rejected_via_fake_builder(self):
        # `":"` is reserved because xray=1 flattens subgraph nodes as
        # `f"{wrapper}:{inner}"` — a literal "foo:bar" user id would be
        # byte-indistinguishable from a depth-1 subgraph flatten and
        # corrupt the IR-keyed LLM cache. LangGraph 1.x also blocks this
        # at `add_node` time (same NS_SEP guard); the only way for a
        # colon-bearing builder to reach the adapter is a bypass, so we
        # exercise the defensive check with a fake builder.
        compiled = _FakeCompiled(
            builder=_FakeBuilder(nodes={"foo:bar": None}),
            drawable=_FakeDrawable(
                node_ids=["__start__", "foo:bar", "__end__"],
                edges=[
                    _FakeDrawableEdge("__start__", "foo:bar"),
                    _FakeDrawableEdge("foo:bar", "__end__"),
                ],
            ),
        )
        with pytest.raises(NotImplementedError, match="reserved character"):
            from_langgraph(compiled)


class TestTerminalDedup:
    def test_duplicate_terminal_edges_collapse(self):
        # Multiple `add_edge(node, END)` from the same source must
        # produce `terminal == ("node",)`, not `("node", "node")`.
        # LangGraph itself dedupes `add_edge` calls, but we use a fake
        # drawable here so this is a true regression pin on the adapter's
        # `sorted(set(...))` collapse.
        compiled = _FakeCompiled(
            builder=_FakeBuilder(nodes={"a": None}),
            drawable=_FakeDrawable(
                node_ids=["__start__", "a", "__end__"],
                edges=[
                    _FakeDrawableEdge("__start__", "a"),
                    _FakeDrawableEdge("a", "__end__"),
                    _FakeDrawableEdge("a", "__end__"),
                ],
            ),
        )
        topo = from_langgraph(compiled)
        assert topo.terminal == ("a",)
