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
that yields 8 + 2 + 4 = 14 variants — inside the §6 8–16 envelope.

**Pinned nodes.** ``Node.pinned=True`` excludes a node from *all* three
generators (it keeps its slot index in order_swap, is skipped during
role_swap pairing, and is not perturbed by decoding_param_swap). This is
what makes the §6a "pin its position" influence-remediation text actually
freeze the agent — see ``ir.py`` Node.pinned for the contract.

**Determinism.** ``order_swap`` samples permutations using a seeded
:class:`random.Random` (default ``rng_seed=0``) so the same topology
produces the same variant sequence across runs. ``role_swap`` and
``decoding_param_swap`` iterate the input ``nodes`` tuple in declared
order, which is itself stable (the adapter sorts terminal ids but
preserves user node order).

**Order-swap memory bound.** For ``N <= 6`` non-pinned nodes we
enumerate all N! permutations (max 720 at N=6) and shuffle. For ``N > 6``
we rejection-sample random permutations, which keeps memory at
O(max_variants) instead of O(N!). The cutoff catches the edge case
noted in the step-4 RT audit (N=10 produces 3.6M perms ≈ 472 MB before
the cap took effect).

**Determinism contract scope.** ``rng_seed`` produces a reproducible
permutation sequence within a single Python *minor* version. Python
upgrades may change ``random.Random``'s internal output for the same
seed (CPython documents Mersenne Twister stability only within minor
releases). Callers depending on hash-stable variant lists across Python
upgrades should pin their Python version or stage a re-bless.
"""

from __future__ import annotations

import random
from itertools import permutations
from math import factorial

from topofuzz.ir import Node, Topology

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

# At or below this non-pinned-node count we enumerate all N! perms and
# shuffle (max 720 at N=6). Above it we rejection-sample. 6 was picked
# because 7! = 5040 starts to look wasteful when only ~7 variants are
# kept; the exact threshold is not load-bearing.
_ORDER_SWAP_ENUMERATE_MAX_N = 6


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
    )


def _sample_perms(
    n: int, *, target: int, rng: random.Random
) -> list[tuple[int, ...]]:
    """Return up to ``target`` distinct non-identity permutations of ``range(n)``.

    For ``N <= _ORDER_SWAP_ENUMERATE_MAX_N`` we enumerate all N! perms —
    max 720 at N=6 — and shuffle, which is unbiased. For larger N we
    rejection-sample so memory stays at O(target) instead of O(N!).

    The rejection sampler may underfill by 1–2 perms when ``target``
    approaches ``N! - 1`` (coupon-collector variance against the
    attempt budget). The §6 contract is an upper bound — at the v0.5
    default cap of 7 non-identity perms this never bites — so we
    return whatever distinct perms we got rather than retrying forever.
    """
    if target <= 0:
        return []
    identity = tuple(range(n))
    if n <= _ORDER_SWAP_ENUMERATE_MAX_N:
        all_perms = [p for p in permutations(identity) if p != identity]
        rng.shuffle(all_perms)
        return all_perms[:target]

    # Rejection-sample. The attempt budget is sized for the typical case
    # (target << N!); near-full-coverage requests may underfill — see
    # the docstring note above.
    seen: set[tuple[int, ...]] = {identity}
    chosen: list[tuple[int, ...]] = []
    attempt_budget = max(target * 10, 64)
    indices = list(identity)
    while len(chosen) < target and attempt_budget > 0:
        attempt_budget -= 1
        rng.shuffle(indices)
        perm = tuple(indices)
        if perm in seen:
            continue
        seen.add(perm)
        chosen.append(perm)
    return chosen


def order_swap(
    topology: Topology,
    *,
    rng_seed: int = 0,
    max_variants: int = _ORDER_SWAP_MAX,
) -> list[Topology]:
    """Up to ``max_variants`` orderings of the non-pinned agents.

    Variant 0 is always the original topology (same identity, not a
    rebuild). The remaining variants permute only the slots occupied by
    non-pinned nodes; pinned nodes (``Node.pinned=True``) keep their
    slot indices in the ``nodes`` tuple. ``edges``, ``entrypoint``, and
    ``terminal`` are unchanged — the §7 runner executes agents in
    ``nodes`` order, so reordering the tuple *is* the execution-order
    change.

    Sampling is deterministic for a given ``rng_seed`` (default ``0``).
    Per §6, there is no isomorphism dedup — duplicates (if any) are
    absorbed by the §7 LLM-response cache downstream.

    Args:
        topology: The base topology to permute.
        rng_seed: Seed for permutation sampling. Same seed → same sequence.
        max_variants: Upper bound; the original counts as 1.

    Returns:
        A list of distinct Topology instances. Length is at most
        ``max_variants`` and at most ``factorial(non_pinned_count)``,
        with the original always at index 0.
    """
    _validate_max_variants(max_variants)

    nodes = topology.nodes
    non_pinned_indices = tuple(i for i, n in enumerate(nodes) if not n.pinned)

    variants: list[Topology] = [topology]
    if max_variants < 2 or len(non_pinned_indices) < 2:
        # Either the caller wants the original only, or there is no
        # permutation degree of freedom (0 or 1 non-pinned node).
        return variants

    non_pinned_nodes = tuple(nodes[i] for i in non_pinned_indices)
    n = len(non_pinned_nodes)
    max_distinct = factorial(n) - 1
    target = min(max_variants - 1, max_distinct)

    rng = random.Random(rng_seed)
    chosen = _sample_perms(n, target=target, rng=rng)

    for perm in chosen:
        new_nodes_list = list(nodes)
        for subset_slot, src_subset_idx in enumerate(perm):
            target_slot = non_pinned_indices[subset_slot]
            new_nodes_list[target_slot] = non_pinned_nodes[src_subset_idx]
        variants.append(_rebuild(topology, tuple(new_nodes_list)))

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
        index 1..K        — additional order_swap permutations
        index K+1..K+R    — role_swap variants
        index K+R+1..end  — decoding_param_swap variants

    Per §6 no isomorphism dedup is applied — the §7 LLM-response cache
    absorbs duplicate canonical hashes at runtime. Callers computing
    variance / influence on top of these variants (§9, §6a) should
    dedupe by ``canonical_hash`` before dividing by count, otherwise the
    denominator is inflated by hash-twin variants and the CI tightens
    artificially. Each individual generator avoids producing trivially
    hash-duplicate output (role_swap skips same-role pairs; order_swap
    samples distinct non-identity perms; decoding_param_swap perturbs a
    distinct node per variant), but cross-generator collisions are still
    possible for pathological topologies and are the caller's problem
    to dedupe.
    """
    out = order_swap(topology, rng_seed=rng_seed)
    out.extend(role_swap(topology))
    out.extend(decoding_param_swap(topology))
    return out
