"""Tests for BLUEPRINT_v5plus.md §6 variant generators."""

from __future__ import annotations

import math
from itertools import permutations

import pytest
from pydantic import ValidationError

from topofuzz.ir import Edge, LLMConfig, Node, Topology
from topofuzz.variants import (
    decoding_param_swap,
    generate_variants,
    order_swap,
    role_swap,
)


def _chain(
    n: int,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    pinned: set[str] | None = None,
) -> Topology:
    """Build a linear-chain Topology of ``n`` agents (ids a0..a{n-1})."""
    pinned = pinned or set()
    cfg = (
        LLMConfig(model=model, temperature=temperature) if model is not None else None
    )
    nodes = tuple(
        Node(id=f"a{i}", role=f"a{i}", llm_config=cfg, pinned=(f"a{i}" in pinned))
        for i in range(n)
    )
    edges = tuple(Edge(from_id=f"a{i}", to_id=f"a{i+1}") for i in range(n - 1))
    return Topology(
        name="chain",
        nodes=nodes,
        edges=edges,
        entrypoint="a0",
        terminal=[f"a{n-1}"],
    )


def _lending_enriched() -> Topology:
    """Build the lending_4role IR with realistic LLM configs."""
    pytest.importorskip("langgraph")
    from topofuzz.adapter import from_langgraph
    from topofuzz.fixtures.lending_4role import LLM_CONFIGS, build_graph

    return from_langgraph(
        build_graph(), name="lending_4role", llm_configs=LLM_CONFIGS
    )


# ---------------------------------------------------------------------------
# order_swap
# ---------------------------------------------------------------------------


class TestOrderSwap:
    def test_returns_original_as_variant_zero(self):
        # The §6 contract — original is variant 0 — is what lets the
        # report show a baseline alongside permutations. Identity check
        # (not equality) so a future "always rebuild" refactor fails here.
        topo = _chain(4)
        variants = order_swap(topo)
        assert variants[0] is topo

    def test_default_cap_is_eight(self):
        topo = _chain(4)  # 4! = 24 perms available
        assert len(order_swap(topo)) == 8

    def test_respects_custom_max_variants(self):
        topo = _chain(4)
        assert len(order_swap(topo, max_variants=3)) == 3

    def test_max_variants_one_returns_only_original(self):
        topo = _chain(4)
        variants = order_swap(topo, max_variants=1)
        assert len(variants) == 1
        assert variants[0] is topo

    def test_max_variants_below_one_raises(self):
        topo = _chain(4)
        with pytest.raises(ValueError, match="max_variants"):
            order_swap(topo, max_variants=0)

    def test_max_variants_negative_raises(self):
        topo = _chain(4)
        with pytest.raises(ValueError, match="max_variants"):
            order_swap(topo, max_variants=-1)

    def test_max_variants_nan_rejected(self):
        # NaN bypasses naive `< 1` because NaN compares False to anything.
        # Without explicit type guarding, max_variants=NaN crashes deep
        # in slice coercion with a confusing TypeError (RT audit m-RT-2).
        topo = _chain(4)
        with pytest.raises(ValueError, match="positive int"):
            order_swap(topo, max_variants=float("nan"))  # type: ignore[arg-type]

    def test_max_variants_float_rejected(self):
        topo = _chain(4)
        with pytest.raises(ValueError, match="positive int"):
            order_swap(topo, max_variants=2.5)  # type: ignore[arg-type]

    def test_max_variants_bool_rejected(self):
        # bool is a subclass of int in Python. `True == 1` would silently
        # work as "give me the original only," which is almost certainly
        # not what a caller meant. Reject for clarity.
        topo = _chain(4)
        with pytest.raises(ValueError, match="positive int"):
            order_swap(topo, max_variants=True)  # type: ignore[arg-type]

    def test_single_node_yields_only_original(self):
        topo = _chain(1)
        variants = order_swap(topo)
        assert len(variants) == 1
        assert variants[0] is topo

    def test_two_nodes_yields_at_most_two_variants(self):
        # 2! = 2, so even with max_variants=8 we get original + 1 swap = 2.
        topo = _chain(2)
        variants = order_swap(topo, max_variants=8)
        assert len(variants) == 2
        assert variants[0] is topo
        assert [n.id for n in variants[1].nodes] == ["a1", "a0"]

    def test_all_pinned_topology_yields_only_original(self):
        # When everything is pinned there is no permutation degree of
        # freedom — order_swap should not stall in an empty-shuffle loop.
        topo = _chain(4, pinned={"a0", "a1", "a2", "a3"})
        variants = order_swap(topo)
        assert len(variants) == 1
        assert variants[0] is topo

    def test_pinned_node_keeps_slot_index(self):
        topo = _chain(4, pinned={"a2"})
        variants = order_swap(topo)
        for v in variants:
            assert v.nodes[2].id == "a2", (
                f"pinned a2 moved to a different slot in variant: "
                f"{[n.id for n in v.nodes]}"
            )

    def test_pinned_node_in_middle_with_three_unpinned(self):
        # Stronger pin: 3 non-pinned nodes around slot-1 pinned. 3! = 6
        # perms available; only slots 0, 2, 3 should ever shuffle.
        topo = _chain(4, pinned={"a1"})
        variants = order_swap(topo, max_variants=10)
        assert len(variants) == math.factorial(3)
        for v in variants:
            assert v.nodes[1].id == "a1"
            assert {v.nodes[0].id, v.nodes[2].id, v.nodes[3].id} == {
                "a0",
                "a2",
                "a3",
            }

    def test_variants_preserve_node_ids_as_a_set(self):
        topo = _chain(4)
        original_ids = {n.id for n in topo.nodes}
        for v in order_swap(topo):
            assert {n.id for n in v.nodes} == original_ids

    def test_variants_preserve_edges_entrypoint_terminal(self):
        # §6: order_swap reorders the nodes tuple only; the runner (§7)
        # executes in tuple order. The graph metadata is untouched so the
        # cache key & report can still reference the original graph shape.
        topo = _chain(4)
        for v in order_swap(topo):
            assert v.edges == topo.edges
            assert v.entrypoint == topo.entrypoint
            assert v.terminal == topo.terminal

    def test_determinism_with_same_seed(self):
        topo = _chain(4)
        a = order_swap(topo, rng_seed=42)
        b = order_swap(topo, rng_seed=42)
        assert [v.canonical_hash() for v in a] == [v.canonical_hash() for v in b]

    def test_seed_actually_threads_into_sampling(self):
        # Across a handful of seeds, the variant sequences should not all
        # be identical. Tests the "rng_seed isn't being ignored" contract
        # without depending on a specific Python MT output (which can
        # drift across versions). 5 seeds × astronomically-low collision
        # prob makes a flake essentially impossible.
        topo = _chain(4)
        sequences = {
            tuple(v.canonical_hash() for v in order_swap(topo, rng_seed=s))
            for s in (0, 1, 7, 42, 999)
        }
        assert len(sequences) >= 2

    def test_variants_are_valid_topology_instances(self):
        topo = _chain(4)
        for v in order_swap(topo):
            assert isinstance(v, Topology)

    def test_variants_have_distinct_canonical_hashes(self):
        # The §7 cache keys variants by canonical_hash; if any two
        # variants here collide the cache will silently dedupe them,
        # wasting one of the §6 budget slots.
        topo = _chain(4)
        hashes = [v.canonical_hash() for v in order_swap(topo)]
        assert len(set(hashes)) == len(hashes)

    def test_does_not_mutate_input_topology(self):
        # The IR is frozen so mutation would itself raise, but this also
        # pins that no shared mutable state (e.g. the nodes tuple) leaks
        # back into the input.
        topo = _chain(4)
        original_hash = topo.canonical_hash()
        order_swap(topo)
        assert topo.canonical_hash() == original_hash

    def test_perm_count_never_exceeds_factorial(self):
        # 3! = 6 perms total → max 6 variants even when asked for 8.
        topo = _chain(3)
        variants = order_swap(topo, max_variants=8)
        assert len(variants) == math.factorial(3)

    def test_three_node_chain_enumerates_all_six_orderings(self):
        # Tiny-chain exhaustive coverage — catches off-by-one in the
        # subset-index → topology-slot mapping.
        topo = _chain(3)
        variants = order_swap(topo, max_variants=10)
        observed = {tuple(n.id for n in v.nodes) for v in variants}
        expected = set(permutations(("a0", "a1", "a2")))
        assert observed == expected

    def test_four_node_chain_enumerates_all_24_orderings_with_high_cap(self):
        # 4! = 24; with max_variants=100 we should see every ordering.
        topo = _chain(4)
        variants = order_swap(topo, max_variants=100)
        observed = {tuple(n.id for n in v.nodes) for v in variants}
        expected = set(permutations(("a0", "a1", "a2", "a3")))
        assert observed == expected
        assert len(variants) == math.factorial(4)

    def test_seven_node_chain_yields_eight_distinct_variants(self):
        # 7! = 5040 perms; default cap=8 means we only need 7 non-identity
        # samples plus the original. N=7 > _ORDER_SWAP_ENUMERATE_MAX_N=6
        # so this path also smoke-tests the rejection sampler (it's the
        # only way 8 distinct variants come back at this N within the
        # default attempt budget). Memory-bound is NOT asserted here —
        # the audit instrumented it (≈16 KB peak) before this test was
        # introduced; if a future change blows that up the symptom will
        # be a hung test rather than a failed assertion.
        topo = _chain(7)
        variants = order_swap(topo)
        assert len(variants) == 8
        observed = {tuple(n.id for n in v.nodes) for v in variants}
        # Original + 7 distinct samples.
        assert len(observed) == 8

    def test_ten_node_chain_finishes_fast(self):
        # Regression pin for the step-4 RT M-RT-1 finding: the original
        # O(N!) enumeration consumed 7 s / 472 MB at N=10. The rejection
        # sampler should finish a default-cap order_swap in well under
        # a second. We use a deliberately generous time budget so this
        # doesn't flake on a loaded CI node; the real signal is "doesn't
        # hang for minutes."
        import time

        topo = _chain(10)
        start = time.perf_counter()
        variants = order_swap(topo)
        elapsed = time.perf_counter() - start
        assert len(variants) == 8
        assert elapsed < 5.0, f"order_swap(N=10) took {elapsed:.2f}s — regression?"


# ---------------------------------------------------------------------------
# role_swap
# ---------------------------------------------------------------------------


class TestRoleSwap:
    def test_skipped_when_no_llm_configs(self):
        # Bare topology — all llm_config=None. §6 contract: empty list.
        topo = _chain(4)
        assert role_swap(topo) == []

    def test_skipped_when_no_two_nodes_share_a_model(self):
        # Distinct model per node — no matching pair.
        nodes = tuple(
            Node(
                id=f"a{i}",
                role=f"a{i}",
                llm_config=LLMConfig(model=f"vendor/model-{i}"),
            )
            for i in range(4)
        )
        topo = Topology(
            name="chain",
            nodes=nodes,
            edges=(Edge(from_id="a0", to_id="a1"),),
            entrypoint="a0",
            terminal=["a3"],
        )
        assert role_swap(topo) == []

    def test_emits_one_variant_per_pair_up_to_cap(self):
        # All four nodes share the same model; all have distinct role
        # strings (a0..a3) → C(4,2) = 6 pairs, cap to 4.
        topo = _chain(4, model="m")
        variants = role_swap(topo)
        assert len(variants) == 4

    def test_respects_custom_max_variants(self):
        topo = _chain(4, model="m")
        assert len(role_swap(topo, max_variants=2)) == 2

    def test_max_variants_below_one_raises(self):
        topo = _chain(4, model="m")
        with pytest.raises(ValueError, match="max_variants"):
            role_swap(topo, max_variants=0)

    def test_max_variants_nan_rejected(self):
        topo = _chain(4, model="m")
        with pytest.raises(ValueError, match="positive int"):
            role_swap(topo, max_variants=float("nan"))  # type: ignore[arg-type]

    def test_swap_exchanges_role_between_matching_pair(self):
        # Single matching pair (a0, a1); others have a different model.
        nodes = (
            Node(id="a0", role="r0", llm_config=LLMConfig(model="shared")),
            Node(id="a1", role="r1", llm_config=LLMConfig(model="shared")),
            Node(id="a2", role="r2", llm_config=LLMConfig(model="other")),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(),
            entrypoint="a0",
            terminal=["a2"],
        )
        variants = role_swap(topo)
        assert len(variants) == 1
        v = variants[0]
        by_id = {n.id: n for n in v.nodes}
        # Roles swapped.
        assert by_id["a0"].role == "r1"
        assert by_id["a1"].role == "r0"
        # Untouched node keeps its role.
        assert by_id["a2"].role == "r2"

    def test_preserves_ids_edges_entrypoint_terminal(self):
        topo = _chain(4, model="m")
        for v in role_swap(topo):
            assert [n.id for n in v.nodes] == [n.id for n in topo.nodes]
            assert v.edges == topo.edges
            assert v.entrypoint == topo.entrypoint
            assert v.terminal == topo.terminal

    def test_does_not_swap_when_only_one_side_has_llm_config(self):
        # Mixed: one node has a model, the other has llm_config=None.
        # That is not a "pair sharing a model" — should produce no variant.
        nodes = (
            Node(id="a0", role="r0", llm_config=LLMConfig(model="m")),
            Node(id="a1", role="r1", llm_config=None),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(),
            entrypoint="a0",
            terminal=["a1"],
        )
        assert role_swap(topo) == []

    def test_pair_order_is_deterministic_by_declared_index(self):
        # All nodes share the same model and have distinct roles: pairs
        # are (0,1),(0,2),(0,3),(1,2) in declared order.
        topo = _chain(4, model="m")
        variants = role_swap(topo, max_variants=4)
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2)]
        for v, (i, j) in zip(variants, expected_pairs):
            assert v.nodes[i].role == topo.nodes[j].role
            assert v.nodes[j].role == topo.nodes[i].role

    def test_preserves_llm_config_on_swap(self):
        # role_swap exchanges role only — temperature/seed/etc must
        # stay on the same node so decoding_param_swap stays composable.
        nodes = (
            Node(
                id="a0",
                role="r0",
                llm_config=LLMConfig(model="m", temperature=0.5, seed=1),
            ),
            Node(
                id="a1",
                role="r1",
                llm_config=LLMConfig(model="m", temperature=0.0, seed=2),
            ),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(),
            entrypoint="a0",
            terminal=["a1"],
        )
        v = role_swap(topo)[0]
        by_id = {n.id: n for n in v.nodes}
        assert by_id["a0"].llm_config.temperature == 0.5
        assert by_id["a0"].llm_config.seed == 1
        assert by_id["a1"].llm_config.temperature == 0.0
        assert by_id["a1"].llm_config.seed == 2

    def test_variants_have_distinct_hashes(self):
        topo = _chain(4, model="m")
        hashes = [v.canonical_hash() for v in role_swap(topo)]
        assert len(set(hashes)) == len(hashes)

    def test_skips_pairs_with_identical_role_strings(self):
        # RT audit M-RT-2: swapping role X with role X yields a no-op
        # variant whose canonical_hash matches the original — that would
        # silently consume a §6 budget slot. role_swap must skip those.
        nodes = tuple(
            Node(id=f"a{i}", role="SAME_ROLE", llm_config=LLMConfig(model="m"))
            for i in range(4)
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(Edge(from_id="a0", to_id="a1"),),
            entrypoint="a0",
            terminal=["a3"],
        )
        assert role_swap(topo) == []

    def test_role_swap_with_mixed_identical_and_distinct_roles(self):
        # 4 nodes, model "m". Roles: a0=R, a1=R, a2=R, a3=Q.
        # Eligible pairs (after skipping same-role pairs): (0,3),(1,3),(2,3).
        nodes = (
            Node(id="a0", role="R", llm_config=LLMConfig(model="m")),
            Node(id="a1", role="R", llm_config=LLMConfig(model="m")),
            Node(id="a2", role="R", llm_config=LLMConfig(model="m")),
            Node(id="a3", role="Q", llm_config=LLMConfig(model="m")),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(Edge(from_id="a0", to_id="a3"),),
            entrypoint="a0",
            terminal=["a3"],
        )
        variants = role_swap(topo, max_variants=10)
        assert len(variants) == 3
        # Each variant should have the chosen left index now carrying "Q"
        # and a3 now carrying "R".
        for v, src_idx in zip(variants, (0, 1, 2)):
            assert v.nodes[src_idx].role == "Q"
            assert v.nodes[3].role == "R"

    def test_skips_pinned_nodes(self):
        # RT audit M-RT-4: pinned must freeze the agent across ALL
        # generators, not just order_swap. A pinned node should never be
        # one half of a role_swap pair.
        nodes = (
            Node(id="a0", role="r0", llm_config=LLMConfig(model="m"), pinned=True),
            Node(id="a1", role="r1", llm_config=LLMConfig(model="m")),
            Node(id="a2", role="r2", llm_config=LLMConfig(model="m")),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(Edge(from_id="a0", to_id="a1"),),
            entrypoint="a0",
            terminal=["a2"],
        )
        variants = role_swap(topo)
        # a0 is pinned, so only the (a1, a2) pair qualifies.
        assert len(variants) == 1
        # a0's role must be untouched in every variant.
        for v in variants:
            assert v.nodes[0].role == "r0"


# ---------------------------------------------------------------------------
# decoding_param_swap
# ---------------------------------------------------------------------------


class TestDecodingParamSwap:
    def test_one_variant_per_qualifying_agent(self):
        # 4 agents, all at temp=0.0 → 4 variants.
        topo = _chain(4, model="m", temperature=0.0)
        assert len(decoding_param_swap(topo)) == 4

    def test_skips_node_with_no_llm_config(self):
        # Mixed: two nodes have configs at temp=0.0, two are None.
        nodes = (
            Node(id="a0", role="r0", llm_config=LLMConfig(model="m", temperature=0.0)),
            Node(id="a1", role="r1", llm_config=None),
            Node(id="a2", role="r2", llm_config=LLMConfig(model="m", temperature=0.0)),
            Node(id="a3", role="r3", llm_config=None),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(),
            entrypoint="a0",
            terminal=["a3"],
        )
        variants = decoding_param_swap(topo)
        assert len(variants) == 2

    def test_skips_nodes_already_at_nonzero_temperature(self):
        # §6: only nodes pinned at exactly 0.0 are lifted; an already-warm
        # agent is left alone (otherwise we'd be running the variant matrix
        # over a temperature gradient the user did not pick).
        topo = _chain(3, model="m", temperature=0.5)
        assert decoding_param_swap(topo) == []

    def test_returns_empty_when_no_llm_configs(self):
        topo = _chain(4)
        assert decoding_param_swap(topo) == []

    def test_variant_perturbs_exactly_one_agent(self):
        topo = _chain(4, model="m", temperature=0.0)
        variants = decoding_param_swap(topo)
        for i, v in enumerate(variants):
            perturbed = [
                idx
                for idx, n in enumerate(v.nodes)
                if n.llm_config is not None and n.llm_config.temperature != 0.0
            ]
            assert perturbed == [i]
            assert v.nodes[i].llm_config.temperature == 0.3

    def test_does_not_change_model_seed_or_max_tokens(self):
        # The single-field delta is the contract: only temperature moves.
        # Sensitivity attribution depends on this — if max_tokens flipped
        # too, we couldn't tell which dimension drove a score change.
        nodes = (
            Node(
                id="a0",
                role="r0",
                llm_config=LLMConfig(
                    model="m", temperature=0.0, max_tokens=2048, seed=99
                ),
            ),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(),
            entrypoint="a0",
            terminal=["a0"],
        )
        v = decoding_param_swap(topo)[0]
        cfg = v.nodes[0].llm_config
        assert cfg.model == "m"
        assert cfg.max_tokens == 2048
        assert cfg.seed == 99
        assert cfg.temperature == 0.3

    def test_respects_max_variants_cap(self):
        topo = _chain(4, model="m", temperature=0.0)
        assert len(decoding_param_swap(topo, max_variants=2)) == 2

    def test_max_variants_below_one_raises(self):
        topo = _chain(4, model="m", temperature=0.0)
        with pytest.raises(ValueError, match="max_variants"):
            decoding_param_swap(topo, max_variants=0)

    def test_max_variants_nan_rejected(self):
        topo = _chain(4, model="m", temperature=0.0)
        with pytest.raises(ValueError, match="positive int"):
            decoding_param_swap(topo, max_variants=float("nan"))  # type: ignore[arg-type]

    def test_max_variants_none_means_no_extra_cap(self):
        topo = _chain(4, model="m", temperature=0.0)
        # No-cap should still respect the §6 N-variants-per-N-nodes contract.
        assert len(decoding_param_swap(topo, max_variants=None)) == 4

    def test_negative_zero_treated_as_low(self):
        # rfc8785 + Python float equality both treat -0.0 == 0.0, so a
        # user who hits -0.0 via a config subtraction still gets a lift.
        # This pins that contract so a future float-normalisation tweak
        # doesn't accidentally exclude them.
        nodes = (
            Node(
                id="a0",
                role="r0",
                llm_config=LLMConfig(model="m", temperature=-0.0),
            ),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(),
            entrypoint="a0",
            terminal=["a0"],
        )
        assert len(decoding_param_swap(topo)) == 1

    def test_variants_have_distinct_hashes(self):
        topo = _chain(4, model="m", temperature=0.0)
        hashes = [v.canonical_hash() for v in decoding_param_swap(topo)]
        assert len(set(hashes)) == len(hashes)

    def test_skips_pinned_nodes(self):
        # RT audit M-RT-4: pinned freezes the agent across all generators.
        # A pinned agent at temp=0.0 must NOT get a lift to 0.3.
        nodes = (
            Node(
                id="a0",
                role="r0",
                llm_config=LLMConfig(model="m", temperature=0.0),
                pinned=True,
            ),
            Node(
                id="a1",
                role="r1",
                llm_config=LLMConfig(model="m", temperature=0.0),
            ),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(Edge(from_id="a0", to_id="a1"),),
            entrypoint="a0",
            terminal=["a1"],
        )
        variants = decoding_param_swap(topo)
        # Only a1 qualifies.
        assert len(variants) == 1
        for v in variants:
            # a0 stays at 0.0 in every variant.
            assert v.nodes[0].llm_config.temperature == 0.0


# ---------------------------------------------------------------------------
# generate_variants (combined)
# ---------------------------------------------------------------------------


class TestGenerateVariants:
    def test_lending_4role_yields_8_to_16_variants(self):
        # The §6 envelope check for the canonical demo fixture: 8
        # order_swap + 2 role_swap + 4 decoding_param_swap = 14.
        topo = _lending_enriched()
        variants = generate_variants(topo)
        assert 8 <= len(variants) <= 16
        assert len(variants) == 14, (
            f"expected 14 variants for lending_4role "
            f"(8 order + 2 role + 4 decoding); got {len(variants)}"
        )

    def test_lending_4role_original_is_at_index_zero(self):
        topo = _lending_enriched()
        variants = generate_variants(topo)
        assert variants[0] is topo

    def test_lending_4role_all_hashes_distinct(self):
        topo = _lending_enriched()
        variants = generate_variants(topo)
        hashes = [v.canonical_hash() for v in variants]
        assert len(set(hashes)) == len(hashes)

    def test_bare_topology_falls_back_to_order_swap_only(self):
        # No llm_configs → role_swap and decoding_param_swap both return
        # empty. The user still gets order_swap variants. (Documents the
        # M-NEW-2 contract: without the adapter's llm_configs kwarg,
        # role/decoding swaps are no-ops by construction.)
        topo = _chain(4)
        variants = generate_variants(topo)
        # 8 order_swap variants (incl. original) + 0 + 0.
        assert len(variants) == 8
        assert all(n.llm_config is None for v in variants for n in v.nodes)

    def test_rng_seed_threaded_through_to_order_swap(self):
        # generate_variants must let callers reproduce by passing rng_seed.
        topo = _chain(4)
        a = [v.canonical_hash() for v in generate_variants(topo, rng_seed=7)]
        b = [v.canonical_hash() for v in generate_variants(topo, rng_seed=7)]
        c_seeds = {
            tuple(v.canonical_hash() for v in generate_variants(topo, rng_seed=s))
            for s in (0, 1, 7, 42, 999)
        }
        assert a == b
        # Across 5 distinct seeds we should see at least 2 distinct
        # sequences. Collision probability is effectively zero; this
        # avoids the flake risk of a single-pair "different seeds differ"
        # assertion while still pinning that the seed is actually used.
        assert len(c_seeds) >= 2

    def test_input_topology_unchanged_after_generation(self):
        # End-to-end "no mutation" pin, cheap fixture (no langgraph req'd).
        topo = _chain(4, model="m", temperature=0.0)
        before = topo.canonical_hash()
        generate_variants(topo)
        assert topo.canonical_hash() == before

    def test_pinned_node_frozen_across_all_generators(self):
        # RT audit M-RT-4: end-to-end pin contract. A pinned agent should
        # appear identical (same slot, same role, same llm_config) in
        # EVERY variant produced by generate_variants.
        nodes = (
            Node(
                id="a0",
                role="r0",
                llm_config=LLMConfig(model="m", temperature=0.0),
                pinned=True,
            ),
            Node(
                id="a1",
                role="r1",
                llm_config=LLMConfig(model="m", temperature=0.0),
            ),
            Node(
                id="a2",
                role="r2",
                llm_config=LLMConfig(model="m", temperature=0.0),
            ),
            Node(
                id="a3",
                role="r3",
                llm_config=LLMConfig(model="m", temperature=0.0),
            ),
        )
        topo = Topology(
            name="t",
            nodes=nodes,
            edges=(Edge(from_id="a0", to_id="a1"),),
            entrypoint="a0",
            terminal=["a3"],
        )
        pinned = topo.nodes[0]
        for v in generate_variants(topo):
            assert v.nodes[0] == pinned, (
                f"pinned a0 changed in a variant: was {pinned!r}, "
                f"got {v.nodes[0]!r}"
            )


# ---------------------------------------------------------------------------
# Structural invariants (frozen IR + validators run on every variant)
# ---------------------------------------------------------------------------


class TestVariantStructuralIntegrity:
    def test_variants_are_frozen(self):
        # Use the specific Pydantic exception so a future IR change that
        # raises a different exception (e.g. AttributeError from `__slots__`)
        # surfaces here instead of being swallowed by a broad `Exception`.
        topo = _chain(4, model="m", temperature=0.0)
        for v in generate_variants(topo):
            with pytest.raises(ValidationError):
                v.name = "mutated"  # type: ignore[misc]

    def test_variants_pass_structural_validation_on_construction(self):
        # We rebuild every variant through Topology(...) so the
        # @model_validator runs — this confirms it actually does, by
        # asserting variants can round-trip through JSON without losing
        # their hash (model_validate_json re-runs the validator too).
        topo = _chain(4, model="m", temperature=0.0)
        for v in generate_variants(topo):
            roundtripped = Topology.model_validate_json(v.model_dump_json())
            assert roundtripped.canonical_hash() == v.canonical_hash()
