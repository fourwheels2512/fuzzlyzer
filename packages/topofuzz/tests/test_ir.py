# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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


class TestLLMConfigCanonicalHash:
    def test_is_deterministic_64char_hex(self):
        a = LLMConfig(model="anthropic/claude-opus-4-7", temperature=0.0, seed=1)
        b = LLMConfig(model="anthropic/claude-opus-4-7", temperature=0.0, seed=1)
        h = a.canonical_hash()
        assert h == b.canonical_hash()
        assert len(h) == 64
        int(h, 16)  # raises ValueError if not hex

    def test_changes_on_model_change(self):
        a = LLMConfig(model="m1", temperature=0.0)
        b = LLMConfig(model="m2", temperature=0.0)
        assert a.canonical_hash() != b.canonical_hash()

    def test_changes_on_temperature_change(self):
        a = LLMConfig(model="m", temperature=0.0)
        b = LLMConfig(model="m", temperature=0.3)
        assert a.canonical_hash() != b.canonical_hash()

    def test_changes_on_seed_change(self):
        a = LLMConfig(model="m", seed=1)
        b = LLMConfig(model="m", seed=2)
        assert a.canonical_hash() != b.canonical_hash()

    def test_negative_zero_hashes_identically_to_zero(self):
        # rfc8785 normalises -0.0 → 0; this must hold here too so a
        # config-layer subtraction producing -0.0 doesn't tank cache hits.
        pos = LLMConfig(model="m", temperature=0.0)
        neg = LLMConfig(model="m", temperature=-0.0)
        assert pos.canonical_hash() == neg.canonical_hash()

    def test_golden_hash_for_known_config(self):
        # Regression pin: if pydantic emits fields in a new order, or
        # rfc8785 changes number canonicalisation, this fails before
        # signatures from older runs go stale. To re-bless intentionally:
        # print LLMConfig(...).canonical_hash() and paste the new value.
        c = LLMConfig(model="anthropic/claude-opus-4-7", temperature=0.0, seed=42)
        assert (
            c.canonical_hash()
            == "a35e2625e0cffe5a35064eb4d7bba3ce709e056a3d154525a04c58b4964f3364"
        )


class TestLLMConfigFiniteValidator:
    def test_nan_temperature_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            LLMConfig(model="m", temperature=float("nan"))

    def test_pos_inf_temperature_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            LLMConfig(model="m", temperature=float("inf"))

    def test_neg_inf_temperature_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            LLMConfig(model="m", temperature=float("-inf"))

    def test_negative_zero_accepted(self):
        # -0.0 is finite. Pinning this contract so a stricter validator
        # tweak doesn't accidentally exclude users hitting -0.0 via
        # config-layer arithmetic.
        c = LLMConfig(model="m", temperature=-0.0)
        assert c.temperature == 0.0

    def test_bool_temperature_rejected(self):
        # RT polish-audit MAJ-3: bool is a subclass of int in Python and
        # Pydantic v2 silently coerces it. `temperature=True` would
        # JCS-encode as `true` while `temperature=1.0` encodes as `1.0`,
        # producing two distinct canonical hashes for logically-equivalent
        # configs and silently missing the §7 cache.
        with pytest.raises(ValidationError, match="bool"):
            LLMConfig(model="m", temperature=True)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="bool"):
            LLMConfig(model="m", temperature=False)  # type: ignore[arg-type]


class TestCanonicalHashChainConsistency:
    """Single canonical-hash recipe across LLMConfig, Topology, and the
    LLMCache key producer (RT audit M4: divergence would silently
    re-bucket cache keys and corrupt the §7 invariant)."""

    def test_topology_hash_matches_shared_helper(self):
        # If Topology.canonical_hash ever diverges from the shared
        # fuzzeval.canonical helper, the §7 cache identity gets two
        # different hash universes overnight. This test pins that they
        # produce byte-identical output for the same input.
        #
        # The recipe strips ``verdict_node`` when None and BOTH
        # ``verdict_label_true``/``verdict_label_false`` when BOTH are at
        # the default pair ``("APPROVE","DENY")`` so v0.5 fixtures hash
        # byte-identically to their hash before either field existed —
        # see Topology.canonical_hash docstring.
        from fuzzeval.canonical import canonical_hash as shared

        topo = _sample_topology()
        expected_view = topo.model_dump(mode="json")
        if expected_view.get("verdict_node") is None:
            expected_view.pop("verdict_node", None)
        if (
            expected_view.get("verdict_label_true") == "APPROVE"
            and expected_view.get("verdict_label_false") == "DENY"
        ):
            expected_view.pop("verdict_label_true", None)
            expected_view.pop("verdict_label_false", None)
        assert topo.canonical_hash() == shared(expected_view)

    def test_topology_hash_with_verdict_node_set_includes_field(self):
        # When verdict_node is set, it IS part of the topology identity
        # (judging a different node is a different measurement) and
        # participates in the canonical hash. This pins the dispatch
        # branch where the field is preserved through to JCS.
        from fuzzeval.canonical import canonical_hash as shared

        topo = Topology(
            name="sample",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["a", "b"],
            verdict_node="b",
        )
        expected_view = topo.model_dump(mode="json")
        assert expected_view["verdict_node"] == "b"
        # Labels are at default ("APPROVE","DENY") here — strip them
        # under the same pair-level recipe as canonical_hash so the
        # shared-helper check stays apples-to-apples.
        if (
            expected_view.get("verdict_label_true") == "APPROVE"
            and expected_view.get("verdict_label_false") == "DENY"
        ):
            expected_view.pop("verdict_label_true", None)
            expected_view.pop("verdict_label_false", None)
        assert topo.canonical_hash() == shared(expected_view)

    def test_llm_config_hash_matches_shared_helper(self):
        from fuzzeval.canonical import canonical_hash as shared

        cfg = LLMConfig(model="m", temperature=0.0, seed=1)
        assert cfg.canonical_hash() == shared(cfg.model_dump(mode="json"))


class TestVerdictNode:
    """Step 20a: optional first-class ``verdict_node`` field, distinct from
    ``terminal``."""

    def test_default_is_none(self):
        topo = _sample_topology()
        assert topo.verdict_node is None

    def test_set_to_existing_node_id_accepted(self):
        topo = Topology(
            name="t",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["b"],
            verdict_node="a",
        )
        assert topo.verdict_node == "a"

    def test_set_to_unknown_node_id_rejected(self):
        with pytest.raises(ValidationError, match="verdict_node"):
            Topology(
                name="t",
                nodes=[Node(id="a", role="r")],
                edges=[],
                entrypoint="a",
                terminal=["a"],
                verdict_node="ghost",
            )

    def test_v05_hash_unchanged_after_field_added(self):
        # Regression guard: a v0.5-shape topology (no verdict_node) must
        # canonical-hash byte-identically to its hash before this field
        # existed. The hash recipe strips verdict_node when None for
        # exactly this reason — protecting persisted run.sqlite
        # ir_canonical_hash columns, signed bundles, and JudgeCache
        # rebucketing from invalidation. The pinned hexes below were
        # produced under the pre-20a IR — if either changes, every
        # v0.5 fixture's hash on disk diverges.
        #
        # Pin (1) — synthetic toy topology.
        toy = Topology(
            name="v05-frozen",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["b"],
        )
        # Hex pinned from the strip-when-None reconstruction (see the
        # independent `shared(v05_view)` check 10 lines below — that's
        # the load-bearing recipe guard; this literal is the future-
        # regression sentinel that catches rfc8785 / pydantic / JCS
        # churn before it silently re-buckets every persisted
        # ir_canonical_hash. NOT independently verified against a
        # pre-20a build — the recipe equivalence is what proves v0.5
        # wire shape, not this literal.
        assert (
            toy.canonical_hash()
            == "94a458671a4cb056c00f0d5247576e4dc1ccbaaa802fb8724736b29aba1f0327"  # noqa: E501
        ), (
            "Strip-when-None recipe broken — v0.5 fixture hash diverged. "
            "Persisted run.sqlite ir_canonical_hash columns will no "
            "longer match topology regeneration. Investigate before "
            "shipping. Got: " + toy.canonical_hash()
        )
        # Independently recompute under the explicit "strip verdict_node
        # when None + strip BOTH verdict_label_* when BOTH at default"
        # recipe — proves the canonical_hash method matches the
        # documented v0.5 wire shape.
        from fuzzeval.canonical import canonical_hash as shared

        v05_view = toy.model_dump(mode="json")
        v05_view.pop("verdict_node", None)
        # v0.5 toy has both labels at default → both stripped under the
        # pair-level rule documented in canonical_hash.
        v05_view.pop("verdict_label_true", None)
        v05_view.pop("verdict_label_false", None)
        assert "verdict_node" not in v05_view
        assert "verdict_label_true" not in v05_view
        assert "verdict_label_false" not in v05_view
        assert toy.canonical_hash() == shared(v05_view)
        # Pin (2) — Documents that ``model_dump_json()`` is NOT
        # equivalent to the canonical_hash recipe. The strip lives only
        # in ``canonical_hash``; a caller who hashes
        # ``json.loads(t.model_dump_json())`` thinking it's equivalent
        # gets a DIFFERENT bucket. Pinning this gap surfaces any future
        # attempt to "simplify" canonical_hash to call model_dump_json
        # directly (which would silently re-bucket every persisted
        # variant_id under the v0.5-compat strip).
        import json
        dump_view = json.loads(toy.model_dump_json())
        assert dump_view.get("verdict_node") is None
        assert "verdict_node" in dump_view
        # model_dump_json emits the labels at default too — proves the
        # strip is canonical_hash-only.
        assert dump_view.get("verdict_label_true") == "APPROVE"
        assert dump_view.get("verdict_label_false") == "DENY"
        assert shared(dump_view) != toy.canonical_hash()

    def test_setting_verdict_node_changes_hash(self):
        a = Topology(
            name="t",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["a", "b"],
        )
        b = Topology(
            name="t",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["a", "b"],
            verdict_node="b",
        )
        # Same topology shape; different judged node IS a different
        # measurement identity.
        assert a.canonical_hash() != b.canonical_hash()

    def test_round_trip_through_json_preserves_verdict_node(self):
        a = Topology(
            name="t",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["a", "b"],
            verdict_node="b",
        )
        b = Topology.model_validate_json(a.model_dump_json())
        assert b.verdict_node == "b"
        assert a.canonical_hash() == b.canonical_hash()


class TestVerdictLabels:
    """Step 20a/F6 follow-up: producer-side ``verdict_label_true`` and
    ``verdict_label_false`` IR fields, default pair ``("APPROVE","DENY")``,
    stripped from canonical_hash as a pair when both at default."""

    def _two_node_topo(self, **labels) -> Topology:
        return Topology(
            name="t",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["b"],
            **labels,
        )

    def test_defaults_are_approve_deny(self):
        t = self._two_node_topo()
        assert t.verdict_label_true == "APPROVE"
        assert t.verdict_label_false == "DENY"

    def test_lowercase_input_normalized_to_uppercase(self):
        t = self._two_node_topo(
            verdict_label_true="escalate", verdict_label_false="defer"
        )
        assert t.verdict_label_true == "ESCALATE"
        assert t.verdict_label_false == "DEFER"

    def test_whitespace_stripped(self):
        t = self._two_node_topo(
            verdict_label_true="  escalate  ",
            verdict_label_false="\tdefer\n",
        )
        assert t.verdict_label_true == "ESCALATE"
        assert t.verdict_label_false == "DEFER"

    def test_empty_after_strip_rejected(self):
        with pytest.raises(ValidationError, match="non-empty after"):
            self._two_node_topo(verdict_label_true="   ")

    def test_non_string_rejected(self):
        with pytest.raises(ValidationError, match="must be str"):
            self._two_node_topo(verdict_label_true=123)  # type: ignore[arg-type]

    def test_over_twelve_chars_rejected(self):
        # 13 chars after strip+upper.
        with pytest.raises(ValidationError, match="≤12 chars"):
            self._two_node_topo(verdict_label_true="thirteen_char")

    def test_exactly_twelve_chars_accepted(self):
        t = self._two_node_topo(
            verdict_label_true="twelve_chars", verdict_label_false="DENY"
        )
        assert t.verdict_label_true == "TWELVE_CHARS"
        assert len(t.verdict_label_true) == 12

    def test_collision_after_normalize_rejected(self):
        # "approve" and "APPROVE" both normalize to "APPROVE" — the
        # pair collapses and would render two excerpts under the same
        # label. Same logic as the publish-side defense in
        # ``publish._build_ceo_card_context``.
        with pytest.raises(ValidationError, match="must differ"):
            self._two_node_topo(
                verdict_label_true="approve", verdict_label_false="APPROVE"
            )

    def test_v05_hash_unchanged_after_label_fields_added(self):
        # Regression guard: a v0.5-shape topology (no explicit labels,
        # default pair) must canonical-hash byte-identically to its hash
        # before the label fields existed. Without the pair-level strip,
        # JCS encodes the explicit defaults and silently rebuckets every
        # persisted ir_canonical_hash and signed bundle.
        toy = Topology(
            name="v05-frozen",
            nodes=[Node(id="a", role="r"), Node(id="b", role="r")],
            edges=[Edge(from_id="a", to_id="b")],
            entrypoint="a",
            terminal=["b"],
        )
        # Same sentinel as test_v05_hash_unchanged_after_field_added —
        # if EITHER the verdict_node strip OR the verdict_label pair
        # strip breaks, every persisted hash diverges.
        assert (
            toy.canonical_hash()
            == "94a458671a4cb056c00f0d5247576e4dc1ccbaaa802fb8724736b29aba1f0327"  # noqa: E501
        ), (
            "v0.5 fixture hash diverged after adding verdict_label_* "
            "fields. The strip-when-default recipe is broken. Persisted "
            "run.sqlite ir_canonical_hash columns will no longer match "
            "topology regeneration. Got: " + toy.canonical_hash()
        )

    def test_setting_non_default_labels_changes_hash(self):
        a = self._two_node_topo()
        b = self._two_node_topo(
            verdict_label_true="escalate", verdict_label_false="defer"
        )
        # Same topology shape; different verdict-label pair IS a
        # different measurement contract identity.
        assert a.canonical_hash() != b.canonical_hash()

    def test_partial_opt_in_recipe_equivalence(self):
        # Recipe-equivalence pin: when only one label differs from
        # default, the label-strip rule does NOT fire (it's pair-level),
        # so canonical_hash MUST equal the shared-helper hash on the
        # full view (with both labels present). This proves the
        # canonical_hash implementation matches the documented recipe
        # for the partial-opt-in case. Companion test
        # ``test_partial_opt_in_distinct_from_full_default`` covers the
        # "shares a bucket with v0.5 default" anti-property.
        from fuzzeval.canonical import canonical_hash as shared

        partial = self._two_node_topo(
            verdict_label_true="escalate", verdict_label_false="DENY"
        )
        view = partial.model_dump(mode="json")
        # No label strip fires here — both labels are still in the view.
        assert view.get("verdict_label_true") == "ESCALATE"
        assert view.get("verdict_label_false") == "DENY"
        # The verdict_node=None strip still fires under canonical_hash;
        # mirror it on the view so the shared-helper comparison is
        # apples-to-apples (this test isolates the label-strip rule).
        view.pop("verdict_node", None)
        assert partial.canonical_hash() == shared(view)

    def test_partial_opt_in_distinct_from_full_default(self):
        # Explicit "partial opt-in" (verdict_label_true changed, _false
        # at default value "DENY") must produce a DIFFERENT hash than
        # a pure-default v0.5 topology — proves the pair-level strip
        # rule doesn't accidentally swallow the half-set case.
        full_default = self._two_node_topo()
        partial = self._two_node_topo(
            verdict_label_true="escalate", verdict_label_false="DENY"
        )
        assert full_default.canonical_hash() != partial.canonical_hash()

    def test_round_trip_through_json_preserves_labels(self):
        a = self._two_node_topo(
            verdict_label_true="escalate", verdict_label_false="defer"
        )
        b = Topology.model_validate_json(a.model_dump_json())
        assert b.verdict_label_true == "ESCALATE"
        assert b.verdict_label_false == "DEFER"
        assert a.canonical_hash() == b.canonical_hash()

    def test_frozen_field_assignment_rejected(self):
        t = self._two_node_topo()
        with pytest.raises(ValidationError):
            t.verdict_label_true = "ESCALATE"  # type: ignore[misc]


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
