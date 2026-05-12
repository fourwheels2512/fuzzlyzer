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

import hashlib
from typing import Literal

import rfc8785
from pydantic import BaseModel, ConfigDict, model_validator


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    # None where the provider has no `seed` parameter (e.g. Anthropic).
    seed: int | None = None


class Node(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    role: str
    kind: Literal["agent", "router", "aggregator", "human"] = "agent"
    llm_config: LLMConfig | None = None
    # Excluded from order_swap variant generation (§6).
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

        Deterministic regardless of dict insertion order or platform. The
        basis for signatures (§11) and cache keys (§7).
        """
        return hashlib.sha256(
            rfc8785.dumps(self.model_dump(mode="json"))
        ).hexdigest()
