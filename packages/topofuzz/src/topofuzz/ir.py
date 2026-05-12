"""Pydantic Topology IR for TSF v5+ (BLUEPRINT_v5plus.md §5).

A canonical, JCS-hashable representation of a multi-agent LLM topology.
Collections are `tuple[...]` rather than `list[...]` so a constructed
`Topology` is *deeply* immutable — once built, every field and container
is hashable and signature-stable. Pydantic v2 coerces list literals to
tuples at validation time, so callers can still pass lists for ergonomics.

Structural integrity is enforced at construction (entrypoint and terminal
ids must exist in `nodes`, edges must reference existing nodes, node ids
must be unique). Semantic invariants (e.g. "temperature>0 requires seed
to be set") are out of scope per BLUEPRINT §5; revisit if Pramana needs
them.
"""

from __future__ import annotations

import math
from typing import Literal

from fuzzeval.canonical import canonical_hash as _canonical_hash
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    # None where the provider has no `seed` parameter (e.g. Anthropic).
    seed: int | None = None

    @field_validator("temperature", mode="before")
    @classmethod
    def _reject_bool_temperature(cls, v: object) -> object:
        # `bool` is a subclass of `int` in Python; without this guard,
        # `LLMConfig(temperature=True)` is accepted and JCS-encodes as
        # `true` while `temperature=1.0` JCS-encodes as `1.0` — two
        # logically-equivalent configs producing different canonical
        # hashes, which silently misses the §7 cache. Reject at the
        # boundary so the bug surfaces here, not as a downstream cache
        # miss. (RT polish-audit MAJ-3.)
        if isinstance(v, bool):
            raise ValueError(
                f"LLMConfig.temperature must be a float; got bool {v!r}. "
                "bool is silently coerced by Pydantic and produces "
                "ambiguous JCS encoding."
            )
        return v

    @field_validator("temperature")
    @classmethod
    def _reject_non_finite_temperature(cls, v: float) -> float:
        # NaN / +/-Inf construct fine (they're floats) but `canonical_hash`
        # crashes on them because rfc8785 cannot represent them in JCS.
        # Raise at construction so the bug surfaces here, not several
        # layers later in the §7 cache-key build. Negative-zero is fine —
        # rfc8785 normalises it to "0" and Python `==` treats it as 0.0,
        # so `decoding_param_swap` and the IR hash are consistent.
        if not math.isfinite(v):
            raise ValueError(
                f"LLMConfig.temperature must be finite; got {v!r}. "
                "NaN / Inf are not representable in JCS canonical hash."
            )
        return v

    def canonical_hash(self) -> str:
        """SHA-256 of the RFC 8785 (JCS) canonicalization of this config.

        Used by the §7 cache key (``llm_config_hash`` component). Goes
        through the shared :func:`fuzzeval.canonical.canonical_hash` so
        the recipe stays in lockstep with :meth:`Topology.canonical_hash`.
        """
        return _canonical_hash(self.model_dump(mode="json"))


class Node(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    role: str
    kind: Literal["agent", "router", "aggregator", "human"] = "agent"
    llm_config: LLMConfig | None = None
    # Excluded from ALL variant generators (§6). A pinned node keeps its
    # slot index in order_swap, is not eligible for role_swap pairing,
    # and does not get its temperature perturbed by decoding_param_swap.
    #
    # NOTE: BLUEPRINT_v5plus.md §6 text mentions `pinned` only for
    # `order_swap`. We deliberately extend it to all three generators
    # because the §6a influence-remediation flow tells users to "pin its
    # position" to freeze the agent — if pinning only stopped order_swap,
    # a user following that remediation would still see variance from the
    # role/decoding generators and the recommendation would self-falsify.
    # See step-4 audit M-RT-4 for the reasoning trace.
    pinned: bool = False


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_id: str
    to_id: str


class Topology(BaseModel):
    model_config = ConfigDict(frozen=True)

    tds_version: Literal["0.5.0"] = "0.5.0"
    name: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    entrypoint: str
    # Sorted by the adapter for hash stability — user declaration order
    # is not preserved. Variant generators must not assume `terminal[0]`
    # carries any priority.
    terminal: tuple[str, ...]

    @model_validator(mode="after")
    def _check_structural_integrity(self) -> "Topology":
        ids = [n.id for n in self.nodes]
        if not ids:
            raise ValueError("Topology must have at least one node")
        if len(ids) != len(set(ids)):
            dupes = sorted({nid for nid in ids if ids.count(nid) > 1})
            raise ValueError(f"Duplicate node ids: {dupes}")
        id_set = set(ids)
        if self.entrypoint not in id_set:
            raise ValueError(
                f"entrypoint {self.entrypoint!r} is not in node ids {sorted(id_set)}"
            )
        if not self.terminal:
            raise ValueError("Topology must have at least one terminal node")
        for t in self.terminal:
            if t not in id_set:
                raise ValueError(
                    f"terminal {t!r} is not in node ids {sorted(id_set)}"
                )
        for e in self.edges:
            if e.from_id not in id_set:
                raise ValueError(
                    f"edge.from_id {e.from_id!r} is not in node ids {sorted(id_set)}"
                )
            if e.to_id not in id_set:
                raise ValueError(
                    f"edge.to_id {e.to_id!r} is not in node ids {sorted(id_set)}"
                )
        return self

    def canonical_hash(self) -> str:
        """SHA-256 of the RFC 8785 (JCS) canonicalization of this topology.

        Deterministic regardless of dict insertion order or platform.
        The basis for signatures (§11) and cache keys (§7). Goes through
        :func:`fuzzeval.canonical.canonical_hash` so the hashing recipe
        is shared with :meth:`LLMConfig.canonical_hash` and the cache
        key producer in :mod:`fuzzeval.cache` — a divergence here would
        silently re-bucket cache identity and corrupt the §7 invariant.
        """
        return _canonical_hash(self.model_dump(mode="json"))
