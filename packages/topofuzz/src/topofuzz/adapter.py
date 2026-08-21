# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Walk a compiled LangGraph into a Topology IR (BLUEPRINT_v5plus.md §5).

In-scope LangGraph constructs:

- Linear chains of ``add_node`` + ``add_edge``.
- ``add_conditional_edges`` with an explicit ``path_map`` (static branches).
- Subgraphs at depth 1, introspected via ``get_graph(xray=1)``. In
  LangGraph 1.x, ``xray=1`` flattens the subgraph: internal nodes appear
  with namespaced ids (``subgraph_name:inner_id``), and the wrapper node
  is replaced by direct edges to/from the inner boundary.

Out-of-scope constructs raise ``NotImplementedError`` with a link to the
v1.0 GitHub tracker:

- ``add_conditional_edges`` without ``path_map`` (AST-resolved branches).
- Subgraphs at depth >= 2 *when added directly via*
  ``add_node(name, compiled_subgraph)``. Subgraphs wrapped in a custom
  ``Runnable`` (e.g. ``RunnableLambda``) cannot be statically inspected
  and slip past this check; that pattern is rare in practice and not
  supported.
- ``Command``/interrupt-based human-in-loop nodes (``kind=human`` is
  recorded in the IR but not exercised by variant generators in v0.5).
- Multi-START fan-out (e.g. ``add_conditional_edges(START, ...)``) and
  graphs with zero ``add_edge(START, ...)``.

**Known limitation — Send API.** The ``Send`` primitive returns dynamic
fan-out targets from a path function. When ``Send`` is used *with* a
``path_map``, ``BranchSpec.ends`` is non-None and this adapter cannot
distinguish it from a static branch — the IR records the declared
destinations but runtime may invoke them N times with different payloads.
v0.5 documents this gap; the Weekend 2 runner is expected to detect
``Send`` returns at call time and error out. Users who know they need
``Send`` should track v1.0 issue (see _OOS_TRACKER).

``langgraph`` is imported lazily so ``topofuzz[verify]`` users do not pull it.
"""

from __future__ import annotations

from typing import Any

from topofuzz.ir import Edge, LLMConfig, Node, Topology

# Sentinel node names LangGraph uses for the implicit start/end of a graph.
_START_NODE = "__start__"
_END_NODE = "__end__"

# Reserved in user-supplied node ids because LangGraph's xray=1 flatten
# synthesizes subgraph nodes as `f"{wrapper}:{inner}"`. A user node named
# `foo:bar` would be byte-indistinguishable from a depth-1 subgraph node,
# producing canonical_hash collisions and cache poisoning.
_RESERVED_NODE_ID_CHARS = (":",)

_OOS_TRACKER = "https://github.com/fourwheels2512/fuzzlyzer/issues"


def from_langgraph(
    compiled: Any,
    *,
    name: str = "topology",
    llm_configs: dict[str, LLMConfig] | None = None,
    verdict_node: str | None = None,
    verdict_label_true: str | None = None,
    verdict_label_false: str | None = None,
) -> Topology:
    """Convert a compiled LangGraph (``StateGraph.compile()``) into a Topology.

    Args:
        compiled: A ``CompiledStateGraph`` from ``StateGraph.compile()``.
        name: Human-readable topology name (stored in the IR for the report).
        llm_configs: Optional mapping from node id to ``LLMConfig``. Required
            for the BLUEPRINT §6 ``role_swap`` and ``decoding_param_swap``
            generators to fire — they key on ``llm_config.model``. Unmapped
            nodes get ``llm_config=None``. Unknown keys raise
            ``NotImplementedError`` so typos surface immediately.
        verdict_node: Optional v0.6 first-class declaration of "the node
            whose output IS the verdict the judge scores," distinct from
            ``terminal`` ("any node wired to END"). LangGraph cannot
            introspect this from the compiled graph (nothing in
            ``StateGraph`` names a verdict node), so the user passes it
            explicitly. Unknown ids raise ``NotImplementedError``. When
            ``None`` the runner falls back to the lex-first ``terminal``
            entry — the v0.5 behaviour, preserved bit-for-bit.
        verdict_label_true: Optional v0.6 producer-side declaration of the
            human-readable label rendered next to the judge's
            ``passes_eval=True`` decision in the CEO card and inspect
            view. Pass-through to the IR which strips whitespace,
            upper-cases, and rejects strings >12 chars or that collapse
            against ``verdict_label_false`` post-normalize. ``None``
            preserves the default ``"APPROVE"`` (lending semantics).
        verdict_label_false: Same shape as ``verdict_label_true``;
            ``None`` defaults to ``"DENY"``. Examples: an adversarial-
            relay fixture passes ``"ESCALATE"`` / ``"DEFER"``; a
            healthcare triage fixture passes ``"URGENT"`` / ``"ROUTINE"``.

    Raises:
        NotImplementedError: If the graph uses an out-of-scope construct
            (see module docstring), its compiled object does not expose
            the LangGraph 1.x StateGraph builder API, ``llm_configs``
            references a node id that is not present in the walked graph,
            or ``verdict_node`` is not in the walked graph.
        pydantic.ValidationError: If either ``verdict_label_*`` violates
            the IR normalisation rules (non-string, empty after strip,
            >12 chars, or pair collision post-normalize).
    """
    builder = getattr(compiled, "builder", None)
    if builder is None:
        raise NotImplementedError(
            f"from_langgraph requires the result of `StateGraph.compile()` "
            f"(LangGraph 1.x); got {type(compiled).__name__} which does not "
            f"expose `.builder`. See {_OOS_TRACKER}."
        )

    _reject_sentinel_collisions(builder)
    _reject_reserved_id_chars(builder)
    _reject_unmapped_conditional_edges(builder)
    _reject_nested_subgraphs(builder)

    drawable = compiled.get_graph(xray=1)
    walked_ids = {
        node_id
        for node_id in drawable.nodes
        if node_id not in (_START_NODE, _END_NODE)
    }

    configs = llm_configs or {}
    unknown = set(configs) - walked_ids
    if unknown:
        raise NotImplementedError(
            f"llm_configs references node id(s) not in the walked graph: "
            f"{sorted(unknown)!r}. Known node ids: {sorted(walked_ids)!r}. "
            f"See {_OOS_TRACKER}."
        )
    if verdict_node is not None and verdict_node not in walked_ids:
        raise NotImplementedError(
            f"verdict_node {verdict_node!r} is not in the walked graph. "
            f"Known node ids: {sorted(walked_ids)!r}. See {_OOS_TRACKER}."
        )

    nodes: list[Node] = [
        Node(
            id=node_id,
            role=node_id,
            kind="agent",
            llm_config=configs.get(node_id),
        )
        for node_id in drawable.nodes
        if node_id not in (_START_NODE, _END_NODE)
    ]

    edges: list[Edge] = []
    entrypoints: list[str] = []
    terminals: list[str] = []
    for edge in drawable.edges:
        if edge.source == _START_NODE:
            entrypoints.append(edge.target)
            continue
        if edge.target == _END_NODE:
            terminals.append(edge.source)
            continue
        edges.append(Edge(from_id=edge.source, to_id=edge.target))

    if not entrypoints:
        raise NotImplementedError(
            f"Topology has no entrypoint — no `add_edge(START, ...)` declared. "
            f"v0.5 requires exactly one START edge. See {_OOS_TRACKER}."
        )
    if len(set(entrypoints)) != 1:
        raise NotImplementedError(
            f"Topology must have exactly one entrypoint; found "
            f"{sorted(set(entrypoints))!r}. Multi-START fan-out (including "
            f"`add_conditional_edges(START, ...)`) is out of scope for v0.5. "
            f"See {_OOS_TRACKER}."
        )

    # Build kwargs dict so unset (None) label kwargs preserve the IR's
    # native defaults — passing ``verdict_label_true=None`` would trip the
    # field-validator's "must be str" branch instead of using "APPROVE".
    topo_kwargs: dict[str, Any] = dict(
        name=name,
        nodes=nodes,
        edges=edges,
        entrypoint=entrypoints[0],
        terminal=sorted(set(terminals)),
        verdict_node=verdict_node,
    )
    if verdict_label_true is not None:
        topo_kwargs["verdict_label_true"] = verdict_label_true
    if verdict_label_false is not None:
        topo_kwargs["verdict_label_false"] = verdict_label_false
    return Topology(**topo_kwargs)


def _reject_sentinel_collisions(builder: Any) -> None:
    # LangGraph 1.x normally blocks `add_node("__start__"|"__end__", ...)`
    # at construction time, but we still defend against malformed builders
    # (custom Pregel subclasses, direct dict mutation, future API changes).
    # Silently filtering a sentinel-named node later would leave the IR
    # with dangling edge references. We only check `builder.nodes`, not
    # `builder.branches`, because routing *from* START is legitimate
    # LangGraph syntax — that case is caught by the multi-entrypoint
    # check downstream.
    for node_name in builder.nodes:
        if node_name in (_START_NODE, _END_NODE):
            raise NotImplementedError(
                f"Node name {node_name!r} collides with a LangGraph sentinel. "
                f"Rename the node. See {_OOS_TRACKER}."
            )


def _reject_reserved_id_chars(builder: Any) -> None:
    # See `_RESERVED_NODE_ID_CHARS` rationale: a literal ":" in a user
    # node id is byte-indistinguishable from a depth-1-subgraph flatten,
    # so the canonical hash collides between structurally different
    # graphs and the LLM-response cache (keyed on IR hash) gets poisoned.
    for node_name in builder.nodes:
        for ch in _RESERVED_NODE_ID_CHARS:
            if ch in node_name:
                raise NotImplementedError(
                    f"Node id {node_name!r} contains reserved character "
                    f"{ch!r}. LangGraph's xray=1 flatten synthesizes "
                    f"subgraph nodes with this character; allowing it in "
                    f"user ids would corrupt the canonical_hash and the "
                    f"§7 LLM cache key. See {_OOS_TRACKER}."
                )


def _reject_unmapped_conditional_edges(builder: Any) -> None:
    # LangGraph 1.x exposes the path_map dict as `BranchSpec.ends`, set to
    # None when the user called `add_conditional_edges` without one. v0.5
    # refuses to guess destinations; callers must declare them up front so
    # the variant generator can enumerate them.
    for source_node, branches in builder.branches.items():
        for branch_name, branch in branches.items():
            if getattr(branch, "ends", None) is None:
                raise NotImplementedError(
                    f"Conditional edge from {source_node!r} (branch "
                    f"{branch_name!r}) was declared without a path_map. "
                    f"v0.5 requires an explicit path_map (the `path_map` "
                    f"kwarg on `add_conditional_edges`) so the variant "
                    f"generator can enumerate destinations. "
                    f"See {_OOS_TRACKER}."
                )


def _reject_nested_subgraphs(builder: Any) -> None:
    # A node whose runnable carries its own `.builder` is a depth-1 subgraph
    # (in scope). If that builder *also* contains a node with `.builder`,
    # that is depth 2 (out of scope).
    for node_name, node_spec in builder.nodes.items():
        inner = _builder_of(node_spec)
        if inner is None:
            continue
        for sub_name, sub_spec in inner.nodes.items():
            if _builder_of(sub_spec) is not None:
                raise NotImplementedError(
                    f"Subgraph node {node_name!r} contains another subgraph "
                    f"node {sub_name!r} (depth >= 2). v0.5 supports depth-1 "
                    f"subgraphs only. See {_OOS_TRACKER}."
                )


def _builder_of(node_spec: Any) -> Any | None:
    runnable = getattr(node_spec, "runnable", None)
    return getattr(runnable, "builder", None)
