# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Three variant generators per BLUEPRINT_v5plus.md §6.

Design notes
============

**Freshly constructed, not mutated.** The Topology IR is frozen (§5), so
every variant is a new :class:`Topology` built via its public constructor.
We deliberately avoid :meth:`pydantic.BaseModel.model_copy` at the
Topology level because Pydantic v2's ``model_copy`` does *not* invoke
``@model_validator(mode='after')``, and step 3's audit explicitly added
those validators to catch dangling-reference IRs. Going through
``Topology(...)`` keeps that guarantee on every variant. (Leaf models
``Node`` / ``LLMConfig`` carry no structural validators, so
``model_copy`` for field-level deltas there is safe.)

**Caps from §6.**

- ``order_swap``: at most 8 variants (including the original at index 0).
- ``role_swap``: at most 4 variants (no original).
- ``decoding_param_swap``: at most N variants for N qualifying agents
  (no original).

For a 4-agent topology with two model groups and all-zero temperatures
that yields up-to 8 + 2 + 4 = 14 variants — inside the §6 8–16 envelope.
The actual ``order_swap`` count is bounded by the number of *valid
topological orderings* of the DAG (see below), so a strict linear chain
returns just the original.

**Pinned nodes.** ``Node.pinned=True`` excludes a node from *all* three
generators (it keeps its slot index in order_swap, is skipped during
role_swap pairing, and is not perturbed by decoding_param_swap). This is
what makes the §6a "pin its position" influence-remediation text actually
freeze the agent — see ``ir.py`` Node.pinned for the contract.

**Determinism.** ``order_swap`` enumerates topological orderings in a
deterministic backtracking order and then shuffles the result via a
seeded :class:`random.Random` (default ``rng_seed=0``), so the same
topology produces the same variant sequence across runs. ``role_swap``
and ``decoding_param_swap`` iterate the input ``nodes`` tuple in
declared order, which is itself stable (the adapter sorts terminal ids
but preserves user node order).

**Edge-aware ``order_swap`` (v0.5.1).** ``order_swap`` was edge-blind in
v0.5 — it permuted the node tuple without consulting ``topology.edges``,
so on a near-linear-chain topology most "permutations" executed agents
in an order the graph's own data dependencies forbid. The §7 runner
bypasses LangGraph's edge-driven scheduler and runs ``nodes`` in tuple
order, so an edge-invalid variant did not crash — it measured a
scrambled-information-flow effect instead of pure topology fragility,
muddling the verdict-flip headline.

v0.5.1 replaces that with a topological-sort-respecting sampler: a
backtracking enumerator emits permutations that are valid linear
extensions of the DAG induced by ``topology.edges`` (pinned nodes still
keep their slots and must remain available at the appropriate
backtracking step, or that branch is pruned). The enumeration is capped
at ``max(max_variants * 8, 64)`` to bound work on highly-branchy DAGs
where the number of linear extensions can be huge; the cap is well above
the §6 default of 7 non-identity variants.

Net effect on the headline:

- **Linear chain** — only the original is a valid topological ordering,
  so ``order_swap`` returns ``[original]``. The verdict-flip headline
  loses the order_swap contribution entirely (correct: there is no
  ordering fragility to measure on a strict chain).
- **Branchy DAG** — multiple linear extensions exist, the sampler emits
  them, and the headline now reflects "fragility under valid orderings"
  rather than a mixture with edge-invalid scrambled flows.

**Cycles.** A topology with a directed cycle has zero valid topological
orderings; ``_enumerate_topological_orderings`` returns an empty list
and ``order_swap`` falls back to ``[original]``. The original itself
may still be edge-invalid (the adapter does not topologically sort
declared nodes); that condition is surfaced separately by the CLI's
``_variant_violates_edges`` warning on the loaded topology.

**Determinism contract scope.** ``rng_seed`` produces a reproducible
permutation sequence within a single Python *minor* version. Python
upgrades may change ``random.Random``'s internal output for the same
seed (CPython documents Mersenne Twister stability only within minor
releases). Callers depending on hash-stable variant lists across Python
upgrades should pin their Python version or stage a re-bless.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from topofuzz.ir import Node, Topology

if TYPE_CHECKING:
    from topofuzz.ir import Edge

# §6 caps. order_swap includes the original at index 0; role_swap and
# decoding_param_swap do not include the original.
_ORDER_SWAP_MAX = 8
_ROLE_SWAP_MAX = 4

# Temperatures §6 spec. Decoding param swap flips low → high for each
# agent currently at ``low``. Kept as module constants (rather than
# defaults on every callsite) so a future calibration change touches one
# place.
_DECODING_LOW = 0.0
_DECODING_HIGH = 0.3

# Hard upper bound on the topological-ordering enumeration pool, before
# we trim to ``max_variants``. A DAG's number of linear extensions can be
# huge (10! = 3.6M on a fully-disconnected graph), so we cap to bound
# backtracking work. When the number of linear extensions exceeds the
# cap, the enumerator returns a random subset rather than a deterministic
# prefix — see the per-step shuffle in ``_enumerate_topological_orderings``.
# Without that shuffle, the deterministic ``range(n)`` candidate iteration
# would pin low-indexed nodes to leading slots on wide / sparse DAGs
# (e.g. a 20-node free graph: every emitted ordering shared the prefix
# ``a0..a14`` because the cap of 64 < 6! = 720). The seeded per-step
# shuffle breaks that collapse so every node has a chance at every slot,
# while preserving the ``rng_seed`` determinism contract.
_ENUMERATION_FLOOR = 64


def _validate_max_variants(max_variants: int, *, allow_none: bool = False) -> None:
    """Reject NaN / inf / bool / floats / negatives.

    Pydantic does not guard these (max_variants is a function arg, not a
    model field). Without this, ``max_variants=float('nan')`` slips past
    ``< 1`` (NaN compares False to anything) and crashes deep in slice
    coercion with a confusing TypeError. RT audit m-RT-2.
    """
    if allow_none and max_variants is None:
        return
    if isinstance(max_variants, bool) or not isinstance(max_variants, int):
        raise ValueError(
            f"max_variants must be a positive int, got {type(max_variants).__name__} "
            f"{max_variants!r}"
        )
    if max_variants < 1:
        raise ValueError(f"max_variants must be >= 1, got {max_variants}")


def _rebuild(topology: Topology, new_nodes: tuple[Node, ...]) -> Topology:
    """Construct a new Topology with ``new_nodes``, preserving all other fields.

    Goes through ``Topology(...)`` rather than ``model_copy`` so the
    structural ``@model_validator(mode='after')`` runs on every variant
    — that is the contract step 3's audit pinned. If a future generator
    introduces a bug that produces dangling node ids, this constructor
    raises immediately instead of shipping a corrupt IR downstream.
    """
    return Topology(
        tds_version=topology.tds_version,
        name=topology.name,
        nodes=new_nodes,
        edges=topology.edges,
        entrypoint=topology.entrypoint,
        terminal=topology.terminal,
        verdict_node=topology.verdict_node,
        # Step 20a/F6 follow-up — propagate the producer-side label pair
        # to every variant. Without this, a non-lending fixture
        # (verdict_label_true="ESCALATE") would have its identity-variant
        # (variant 0) carry the custom pair while permuted variants
        # silently fall back to the default APPROVE/DENY pair. The
        # pair-level strip rule in canonical_hash would then OMIT the
        # pair on variants 1..N (defaulted) but RETAIN it on variant 0
        # (custom), producing asymmetric bucketing — variant 0's cache
        # row would NOT match the same node-shape topology declared
        # without verdict_labels, while variants 1..N WOULD. Caught by
        # red-team pass-1 B1.
        verdict_label_true=topology.verdict_label_true,
        verdict_label_false=topology.verdict_label_false,
    )


def _enumerate_topological_orderings(
    nodes: tuple[Node, ...],
    edges: "tuple[Edge, ...]",
    *,
    limit: int,
    rng: random.Random | None = None,
) -> list[tuple[int, ...]]:
    """Up to ``limit`` topological orderings of ``nodes`` respecting ``edges``.

    Each ordering is a tuple of length ``len(nodes)`` whose ``k``-th entry
    is the *original* index of the node placed at slot ``k`` in the output
    variant. So the identity ordering ``(0, 1, ..., n-1)`` means "the
    variant's node tuple equals the input's node tuple."

    The enumeration is a Kahn-style backtracking sampler:

    - At each slot ``k``, choose any node whose remaining in-degree is 0
      (all its predecessors have already been placed) and which has not
      yet been placed. If the slot is *pinned* (the original node at that
      slot has ``pinned=True``), the only legal choice is that pinned
      node — if it is not currently available (in-degree > 0), this
      branch of the search is pruned.
    - Recurse with the chosen node placed, in-degrees of its children
      decremented, and the in-use flag set. Undo on return.
    - Stop once ``limit`` orderings have been collected.

    Sampling. When ``rng`` is None, the candidate loop iterates
    ``range(n)`` deterministically — orderings starting with low-indexed
    nodes come first. When ``rng`` is a seeded :class:`random.Random`,
    each non-pinned slot's candidate list is shuffled in place before
    being iterated. This breaks the structural prefix bias that would
    otherwise collapse the sample on wide / sparse DAGs (RT audit F2-1:
    on a 20-node free graph with cap=64, every emitted ordering shared
    the prefix ``a0..a14`` because 64 < 6! = 720 — the cap was reached
    deep inside the first low-index branch before the search ever
    backtracked to a different slot-0 choice). The shuffle is per-step,
    so the resulting distribution is not uniform over linear extensions
    (uniform sampling is NP-hard), but every node has a non-zero
    probability at every slot it is reachable in.

    Determinism. Same ``rng`` state → same enumeration sequence. Callers
    threading ``random.Random(rng_seed)`` get reproducible output. The
    ``rng=None`` path is also deterministic (the bias is the cost of
    that determinism) and is useful for ordering tests that need a
    stable prefix.

    Self-loops (``from_id == to_id``) and edges referencing nodes outside
    ``nodes`` contribute no constraint. The Topology validator rejects
    dangling edges at construction, so the latter is defensive; the
    former matches the existing :func:`topofuzz.cli._variant_violates_edges`
    contract. Duplicate edge declarations are deduplicated internally so
    they do not inflate in-degree past what the in-degree decrements can
    drain.

    Cycles. A topology with a directed cycle has no valid topological
    ordering; ``indegree`` never reaches zero for some node and
    backtracking returns an empty list. Callers are responsible for
    handling the empty result (``order_swap`` falls back to
    ``[original]``).

    Pinned-slot infeasibility. If the original topology was declared out
    of dependency order — e.g. ``nodes=(b, a)`` with edge ``a -> b`` —
    *and* some nodes are pinned in a way that no topological extension
    can satisfy, the enumerator simply returns an empty list. The
    "original out-of-order" condition is itself surfaced separately by
    the CLI's ``_variant_violates_edges`` warning on the loaded topology.
    """
    n = len(nodes)
    if n == 0 or limit <= 0:
        return []

    # Pinned slot s requires the original node at index s to remain at
    # slot s in every variant. Build a slot→required-original-index map
    # and a set of pinned original indices for fast lookup.
    pinned_slots: dict[int, int] = {
        i: i for i, node in enumerate(nodes) if node.pinned
    }
    pinned_set: set[int] = set(pinned_slots.values())

    # Build the DAG over original indices. Self-loops and dangling edges
    # contribute no constraint (see docstring). Duplicate edges inflate
    # in-degree by design — the constraint is still "u before v," counted
    # once per declaration — so we deduplicate at the (u, v) pair level
    # to avoid the rare topology that declares the same edge twice from
    # leaving a child permanently unreachable.
    id_to_idx = {node.id: i for i, node in enumerate(nodes)}
    seen_edges: set[tuple[int, int]] = set()
    children: list[list[int]] = [[] for _ in range(n)]
    indegree: list[int] = [0] * n
    for edge in edges:
        u = id_to_idx.get(edge.from_id)
        v = id_to_idx.get(edge.to_id)
        if u is None or v is None:
            # Dangling edge — defensive; Topology validator catches this
            # at construction, so we should never reach here in practice.
            continue
        if u == v:
            # Self-loop — equal positions are not "backwards" (see the
            # ``_variant_violates_edges`` contract for the same rule).
            continue
        if (u, v) in seen_edges:
            continue
        seen_edges.add((u, v))
        children[u].append(v)
        indegree[v] += 1

    results: list[tuple[int, ...]] = []
    in_use: list[bool] = [False] * n
    indeg: list[int] = list(indegree)
    current: list[int] = []

    def backtrack(slot: int) -> None:
        if len(results) >= limit:
            return
        if slot == n:
            results.append(tuple(current))
            return

        if slot in pinned_slots:
            req = pinned_slots[slot]
            # Pinned slot must take its required node. If that node is
            # already placed, or some predecessor hasn't been placed yet,
            # there is no extension consistent with this pin from here.
            if in_use[req] or indeg[req] != 0:
                return
            candidates: tuple[int, ...] = (req,)
        else:
            candidate_list = [
                i
                for i in range(n)
                if not in_use[i] and indeg[i] == 0 and i not in pinned_set
            ]
            # Shuffle each non-pinned slot's candidates per the audit
            # finding: without this, ``range(n)``'s deterministic order
            # pins low-indexed nodes to leading slots on wide DAGs.
            # The seeded rng keeps the contract reproducible per
            # ``rng_seed`` while breaking the prefix collapse.
            if rng is not None:
                rng.shuffle(candidate_list)
            candidates = tuple(candidate_list)

        for i in candidates:
            in_use[i] = True
            current.append(i)
            for c in children[i]:
                indeg[c] -= 1
            backtrack(slot + 1)
            for c in children[i]:
                indeg[c] += 1
            current.pop()
            in_use[i] = False
            if len(results) >= limit:
                return

    backtrack(0)
    return results


def order_swap(
    topology: Topology,
    *,
    rng_seed: int = 0,
    max_variants: int = _ORDER_SWAP_MAX,
) -> list[Topology]:
    """Up to ``max_variants`` valid topological orderings of the agents.

    Variant 0 is always the original topology (same identity, not a
    rebuild). The remaining variants permute only the slots occupied by
    non-pinned nodes; pinned nodes (``Node.pinned=True``) keep their
    slot indices. Every emitted variant is a *valid linear extension*
    of the DAG induced by ``topology.edges`` — that is, for every edge
    ``from_id -> to_id``, the position of ``from_id`` in the variant
    comes strictly before the position of ``to_id``. ``edges``,
    ``entrypoint``, and ``terminal`` are unchanged — the §7 runner
    executes agents in ``nodes`` order, so reordering the tuple *is*
    the execution-order change, but every permutation respects the
    graph's own data dependencies.

    Sampling is deterministic for a given ``rng_seed`` (default ``0``).
    The enumerator emits orderings in a deterministic backtracking
    order; the shuffle that follows is seeded; same seed → same
    sequence. Per §6, there is no isomorphism dedup — duplicates
    (if any) are absorbed by the §7 LLM-response cache downstream.

    On a strict linear chain — where exactly one topological ordering
    exists — the only valid extension is the original, so this
    function returns ``[original]``. That is the correct measurement:
    a linear chain has no ordering fragility to fuzz. For a fragility
    headline, run against a branchy fixture.

    On a topology with a directed cycle — where no topological
    ordering exists — this function also returns ``[original]``. The
    original itself may then be edge-invalid; the CLI surfaces that
    separately via ``_variant_violates_edges``.

    Args:
        topology: The base topology to permute.
        rng_seed: Seed for permutation sampling. Same seed → same sequence.
        max_variants: Upper bound; the original counts as 1.

    Returns:
        A list of distinct Topology instances. Length is at most
        ``max_variants`` and at most ``1 + (#linear extensions - 1)``
        for the DAG, with the original always at index 0.
    """
    _validate_max_variants(max_variants)

    nodes = topology.nodes
    variants: list[Topology] = [topology]
    if max_variants < 2:
        return variants

    # Enumerate a generous pool, then trim. The pool size is well above
    # the §6 default (7 variants) without enumerating astronomically-many
    # linear extensions on a near-disconnected graph. We thread a single
    # seeded RNG through the per-step candidate shuffle (inside the
    # enumerator) AND the final pool reorder, so the same ``rng_seed``
    # reproduces the same variant sequence across runs.
    enumeration_cap = max(max_variants * 8, _ENUMERATION_FLOOR)
    rng = random.Random(rng_seed)
    orderings = _enumerate_topological_orderings(
        nodes, topology.edges, limit=enumeration_cap, rng=rng,
    )

    identity = tuple(range(len(nodes)))
    # Drop the identity ordering — it equals the original, which we
    # already include as variant 0. A duplicate would silently consume
    # one of the §6 budget slots with a hash-twin variant.
    non_identity = [perm for perm in orderings if perm != identity]
    if not non_identity:
        return variants

    # Final reorder of the pool before trimming. Per-step shuffle inside
    # the enumerator already randomises which extensions are reached;
    # this trailing shuffle randomises which subset of the reached pool
    # survives the ``[: max_variants - 1]`` slice. Both share the same
    # rng instance so the determinism contract holds.
    rng.shuffle(non_identity)

    for perm in non_identity[: max_variants - 1]:
        new_nodes = tuple(nodes[i] for i in perm)
        variants.append(_rebuild(topology, new_nodes))

    return variants


def role_swap(
    topology: Topology,
    *,
    max_variants: int = _ROLE_SWAP_MAX,
) -> list[Topology]:
    """Up to ``max_variants`` variants swapping ``role`` between model-matching pairs.

    A pair ``(i, j)`` qualifies when:

    - both ``nodes[i].llm_config`` and ``nodes[j].llm_config`` are not None,
    - their ``model`` strings match,
    - both nodes are unpinned (RT audit M-RT-4: pinned freezes ALL generators),
    - **their ``role`` strings differ** (a swap of identical roles is a
      no-op variant whose canonical hash equals the original — it would
      silently consume a §6 4-variant slot; RT audit M-RT-2).

    Pairs are emitted in declared-index order ``(i < j)`` for determinism.
    The original topology is *not* included — callers that want it should
    pair this with :func:`order_swap`.

    Returns an empty list when no two nodes form a qualifying pair — this
    is the §6 "skipped if no two nodes share a model" contract. The
    adapter's ``llm_configs`` kwarg matters here: without it, every node
    has ``llm_config=None`` and role_swap is a no-op (the M-NEW-2
    step-3 finding).
    """
    _validate_max_variants(max_variants)

    nodes = topology.nodes
    pairs: list[tuple[int, int]] = []
    for i in range(len(nodes)):
        ni = nodes[i]
        if ni.pinned or ni.llm_config is None:
            continue
        for j in range(i + 1, len(nodes)):
            nj = nodes[j]
            if nj.pinned or nj.llm_config is None:
                continue
            if ni.llm_config.model != nj.llm_config.model:
                continue
            if ni.role == nj.role:
                # Swap is a no-op; skip rather than mint a hash-duplicate variant.
                continue
            pairs.append((i, j))

    variants: list[Topology] = []
    for i, j in pairs[:max_variants]:
        new_nodes_list = list(nodes)
        new_nodes_list[i] = nodes[i].model_copy(update={"role": nodes[j].role})
        new_nodes_list[j] = nodes[j].model_copy(update={"role": nodes[i].role})
        variants.append(_rebuild(topology, tuple(new_nodes_list)))
    return variants


def decoding_param_swap(
    topology: Topology,
    *,
    max_variants: int | None = None,
) -> list[Topology]:
    """One variant per agent currently at ``temperature=0.0``, lifted to ``0.3``.

    Skips nodes with ``llm_config=None`` (temperature is undefined there),
    pinned nodes (RT audit M-RT-4), and nodes whose temperature is anything
    other than exactly ``0.0``. §6 caps at N variants for N qualifying
    agents; ``max_variants`` further trims that list (``None`` — the
    default — means "no extra cap, one variant per qualifying node").

    Order is the declared node order so callers get a stable mapping
    between variant index and which agent was perturbed.
    """
    _validate_max_variants(max_variants, allow_none=True)

    nodes = topology.nodes
    variants: list[Topology] = []
    for i, node in enumerate(nodes):
        if node.pinned:
            continue
        cfg = node.llm_config
        if cfg is None:
            continue
        # Exact equality is the §6 contract: only nodes pinned at the low
        # anchor get a lift. Negative zero compares equal to 0.0 under ==
        # in Python, and rfc8785 normalises both to "0" in JCS (see
        # ir.py canonical_hash), so a user who hits -0.0 still qualifies.
        if cfg.temperature != _DECODING_LOW:
            continue
        new_cfg = cfg.model_copy(update={"temperature": _DECODING_HIGH})
        new_node = node.model_copy(update={"llm_config": new_cfg})
        new_nodes_list = list(nodes)
        new_nodes_list[i] = new_node
        variants.append(_rebuild(topology, tuple(new_nodes_list)))
        if max_variants is not None and len(variants) >= max_variants:
            break
    return variants


def generate_variants(
    topology: Topology,
    *,
    rng_seed: int = 0,
) -> list[Topology]:
    """Run all three §6 generators and concatenate.

    Output layout::

        index 0           — original (from order_swap)
        index 1..K        — additional order_swap permutations (valid
                            topological extensions of the DAG; may be 0
                            for a linear chain)
        index K+1..K+R    — role_swap variants
        index K+R+1..end  — decoding_param_swap variants

    Per §6 no isomorphism dedup is applied — the §7 LLM-response cache
    absorbs duplicate canonical hashes at runtime. Callers computing
    variance / influence on top of these variants (§9, §6a) should
    dedupe by ``canonical_hash`` before dividing by count, otherwise the
    denominator is inflated by hash-twin variants and the CI tightens
    artificially. Each individual generator avoids producing trivially
    hash-duplicate output (role_swap skips same-role pairs; order_swap
    emits distinct non-identity topological extensions;
    decoding_param_swap perturbs a distinct node per variant), but
    cross-generator collisions are still possible for pathological
    topologies and are the caller's problem to dedupe.
    """
    out = order_swap(topology, rng_seed=rng_seed)
    out.extend(role_swap(topology))
    out.extend(decoding_param_swap(topology))
    return out
