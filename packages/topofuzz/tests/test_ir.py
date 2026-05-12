"""Tests for topofuzz.ir Pydantic models and JCS canonical hash."""

from __future__ import annotations

import hashlib

import pytest
import rfc8785
from pydantic import ValidationError

from topofuzz.ir import Edge, LLMConfig, Node, Topology


def _sample_topology() -> Topology:
    return Topology(
        name="sample",
        nodes=[
            Node(
                id="a",
                role="intake",
                llm_config=LLMConfig(model="anthropic/claude-opus-4-7", seed=42),
            ),
            Node(id="b", role="risk"),
        ],
        edges=[Edge(from_id="a", to_id="b")],
        entrypoint="a",
        terminal=["b"],
    )


class TestFrozen:
    def test_topology_field_assignment_rejected(self):
        t = _sample_topology()
        with pytest.raises(ValidationError):
            t.name = "other"  # type: ignore[misc]

    def test_node_field_assignment_rejected(self):
        t = _sample_topology()
        with pytest.raises(ValidationError):
            t.nodes[0].role = "other"  # type: ignore[misc]

    def test_llm_config_field_assignment_rejected(self):
        c = LLMConfig(model="m")
        with pytest.raises(ValidationError):
            c.temperature = 0.5  # type: ignore[misc]

    def test_edge_field_assignment_rejected(self):
        e = Edge(from_id="a", to_id="b")
        with pytest.raises(ValidationError):
            e.to_id = "c"  # type: ignore[misc]

    def test_topology_collections_are_tuples(self):
        # Lists in passed at construction get coerced to tuples by Pydantic.
        # This is the property that makes the IR deeply immutable.
        t = _sample_topology()
        assert isinstance(t.nodes, tuple)
        assert isinstance(t.edges, tuple)
        assert isinstance(t.terminal, tuple)

    def test_nodes_tuple_cannot_be_mutated(self):
        t = _sample_topology()
        with pytest.raises(AttributeError):
            t.nodes.append(Node(id="x", role="x"))  # type: ignore[attr-defined]

    def test_edges_tuple_cannot_be_mutated(self):
        t = _sample_topology()
        with pytest.raises(AttributeError):
            t.edges.append(Edge(from_id="a", to_id="b"))  # type: ignore[attr-defined]

    def test_terminal_tuple_cannot_be_mutated(self):
        t = _sample_topology()
        with pytest.raises(AttributeError):
            t.terminal.append("c")  # type: ignore[attr-defined]


class TestDefaults:
    def test_node_defaults(self):
        n = Node(id="x", role="r")
        assert n.kind == "agent"
        assert n.pinned is False
        assert n.llm_config is None

    def test_llm_config_defaults(self):
        c = LLMConfig(model="m")
        assert c.temperature == 0.0
        assert c.max_tokens == 4096
        assert c.seed is None

    def test_tds_version_pinned(self):
        t = _sample_topology()
        assert t.tds_version == "0.5.0"


class TestCanonicalHash:
    def test_is_deterministic_64char_hex(self):
        a = _sample_topology()
        b = _sample_topology()
        h = a.canonical_hash()
        assert h == b.canonical_hash()
        assert len(h) == 64
        int(h, 16)  # raises ValueError if not hex

    def test_round_trip_through_json(self):
        a = _sample_topology()
        payload = a.model_dump_json()
        b = Topology.model_validate_json(payload)
        assert a.canonical_hash() == b.canonical_hash()

    def test_invariant_under_dict_key_order(self):
        # Build two equivalent dicts with intentionally opposite insertion
        # orders at every level. Bypass json.dumps -> json.loads (which
        # would canonicalize key order on its own) so the test actually
        # exercises rfc8785's key-sorting behaviour.
        ordered = _sample_topology().model_dump(mode="json")
        reversed_payload = _deep_reverse(ordered)
        assert list(reversed_payload.keys()) != list(ordered.keys())
        h_ordered = hashlib.sha256(rfc8785.dumps(ordered)).hexdigest()
        h_reversed = hashlib.sha256(rfc8785.dumps(reversed_payload)).hexdigest()
        assert h_ordered == h_reversed

    def test_changes_on_node_role_change(self):
        a = _sample_topology()
        b = Topology(
            name="sample",
            nodes=[
                Node(
                    id="a",
                    role="intake_different",
                    llm_config=LLMConfig(
                        model="anthropic/claude-opus-4-7", seed=42
                    ),
                ),
                Node(id="b", role="risk"),
            ],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["b"],
        )
        assert a.canonical_hash() != b.canonical_hash()

    def test_changes_on_edge_reorder(self):
        # Edges are an ordered tuple; reversing insertion order must
        # change the hash. (Variant generation depends on edge order
        # being meaningful.)
        base_nodes = [Node(id=n, role=n) for n in ("a", "b", "c")]
        t1 = Topology(
            name="t",
            nodes=base_nodes,
            edges=[Edge(from_id="a", to_id="b"), Edge(from_id="b", to_id="c")],
            entrypoint="a",
            terminal=["c"],
        )
        t2 = Topology(
            name="t",
            nodes=base_nodes,
            edges=[Edge(from_id="b", to_id="c"), Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["c"],
        )
        assert t1.canonical_hash() != t2.canonical_hash()

    def test_negative_zero_temperature_hashes_identically(self):
        # RFC 8785 normalises negative zero to positive zero, so a user
        # who writes `temperature=-0.0` (e.g. from a subtraction in a
        # config layer) gets the same canonical hash as `temperature=0.0`.
        # This pins that contract so a future rfc8785 release that
        # regresses it shows up here.
        pos = Topology(
            name="t",
            nodes=[Node(id="a", role="a", llm_config=LLMConfig(model="m", temperature=0.0))],
            edges=[],
            entrypoint="a",
            terminal=["a"],
        )
        neg = Topology(
            name="t",
            nodes=[Node(id="a", role="a", llm_config=LLMConfig(model="m", temperature=-0.0))],
            edges=[],
            entrypoint="a",
            terminal=["a"],
        )
        assert pos.canonical_hash() == neg.canonical_hash()

    def test_golden_hash_for_sample_topology(self):
        # Regression test: if Pydantic ever changes field emission order
        # or rfc8785 ever changes its number canonicalisation, this fails
        # and we get a heads-up before signatures from old runs go stale.
        # `pydantic<3` and `langgraph<2` are pinned in pyproject.toml to
        # contain this risk. To re-bless intentionally: print
        # `_sample_topology().canonical_hash()` and paste it here.
        assert (
            _sample_topology().canonical_hash()
            == "546baf9cbf0bd9f2888791bbdf34359e7cb5f721eb495dc529d39a0d0bdcac0c"
        )


class TestStructuralIntegrity:
    def test_empty_nodes_rejected(self):
        with pytest.raises(ValidationError, match="at least one node"):
            Topology(name="t", nodes=[], edges=[], entrypoint="x", terminal=["x"])

    def test_duplicate_node_ids_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate node ids"):
            Topology(
                name="t",
                nodes=[Node(id="a", role="a"), Node(id="a", role="b")],
                edges=[],
                entrypoint="a",
                terminal=["a"],
            )

    def test_entrypoint_not_in_nodes_rejected(self):
        with pytest.raises(ValidationError, match="entrypoint .* not in node ids"):
            Topology(
                name="t",
                nodes=[Node(id="a", role="a")],
                edges=[],
                entrypoint="ghost",
                terminal=["a"],
            )

    def test_empty_terminal_rejected(self):
        with pytest.raises(ValidationError, match="at least one terminal"):
            Topology(
                name="t",
                nodes=[Node(id="a", role="a")],
                edges=[],
                entrypoint="a",
                terminal=[],
            )

    def test_terminal_not_in_nodes_rejected(self):
        with pytest.raises(ValidationError, match="terminal .* not in node ids"):
            Topology(
                name="t",
                nodes=[Node(id="a", role="a")],
                edges=[],
                entrypoint="a",
                terminal=["ghost"],
            )

    def test_edge_from_id_not_in_nodes_rejected(self):
        with pytest.raises(ValidationError, match="edge.from_id .* not in node ids"):
            Topology(
                name="t",
                nodes=[Node(id="a", role="a"), Node(id="b", role="b")],
                edges=[Edge(from_id="ghost", to_id="b")],
                entrypoint="a",
                terminal=["b"],
            )

    def test_edge_to_id_not_in_nodes_rejected(self):
        with pytest.raises(ValidationError, match="edge.to_id .* not in node ids"):
            Topology(
                name="t",
                nodes=[Node(id="a", role="a"), Node(id="b", role="b")],
                edges=[Edge(from_id="a", to_id="ghost")],
                entrypoint="a",
                terminal=["b"],
            )


def _deep_reverse(obj):
    if isinstance(obj, dict):
        return {k: _deep_reverse(v) for k, v in reversed(list(obj.items()))}
    if isinstance(obj, list):
        return [_deep_reverse(x) for x in obj]
    return obj
