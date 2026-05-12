"""Variant generators for Topology IR (BLUEPRINT_v5plus.md §6).

Three generators turn a base Topology into a set of execution variants:

- :func:`order_swap` — permute non-pinned agents (up to 8; original first).
- :func:`role_swap` — swap ``role`` between agents that share a model
  (up to 4; empty list if no model-matching pair exists).
- :func:`decoding_param_swap` — flip ``temperature=0.0`` to
  ``temperature=0.3`` for each qualifying agent.

:func:`generate_variants` runs all three and concatenates the output. Per
§6 there is no isomorphism dedup at generation time — the §7 cache
absorbs duplicates downstream.

All variants are freshly-constructed :class:`Topology` instances so the
structural ``@model_validator(mode="after")`` runs on each one; the input
topology is never mutated (and could not be — the IR is frozen per §5).
"""

from __future__ import annotations

from topofuzz.variants.generators import (
    decoding_param_swap,
    generate_variants,
    order_swap,
    role_swap,
)

__all__ = [
    "decoding_param_swap",
    "generate_variants",
    "order_swap",
    "role_swap",
]
