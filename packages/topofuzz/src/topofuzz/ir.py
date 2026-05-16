# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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
import unicodedata
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
    # Optional first-class declaration of "the node whose output IS the
    # verdict the judge scores," distinct from `terminal` ("any node wired
    # to END"). `terminal` conflates "produces the verdict" with
    # "supervisor short-circuits to END," producing lexical-luck fragility
    # on multi-terminal or supervisor topologies. Set this to remove that
    # ambiguity; leave `None` to fall back to the lex-first `terminal`
    # entry (the v0.5 behaviour, preserved bit-for-bit).
    #
    # The canonical_hash recipe OMITS this field when `None`, so a v0.5
    # fixture (no verdict_node) hashes byte-identically to its hash before
    # this field existed — JCS would otherwise encode an explicit
    # `"verdict_node": null` and silently rebucket cache identity. A
    # v0.6 fixture (verdict_node set) gets a legitimately new hash
    # distinct from the same topology declared without verdict_node,
    # because the judged node IS part of the topology's identity.
    verdict_node: str | None = None

    # Step 20a/F6 follow-up (v0.6): producer-side declaration of the
    # two human-readable verdict labels the CEO card and inspect view
    # render next to the judge's binary decision. The default pair
    # ``("APPROVE", "DENY")`` matches the lending fixture's semantics
    # so a v0.5 topology stays byte-identical on the canonical_hash and
    # on the persisted manifest.json. Non-lending fixtures (adversarial
    # relay → ESCALATE/DEFER; healthcare triage → URGENT/ROUTINE) opt
    # in by passing both kwargs to ``from_langgraph``.
    #
    # Field-validator normalisation (mode="before"):
    #   - reject non-string at the boundary;
    #   - strip leading/trailing whitespace;
    #   - upper-case;
    #   - reject empty-after-strip (would render as blank in the PDF);
    #   - cap at 12 chars (the side-by-side flip card in
    #     ``templates/publish_ceo_card.html.j2`` overflows above ~14
    #     chars; 12 is the safe maximum that also matches the
    #     defense-in-depth slice in ``publish._build_ceo_card_context``).
    #
    # Model-validator collision check (mode="after"):
    #   - reject ``verdict_label_true == verdict_label_false`` post-
    #     normalize. Two labels that collapse to the same string destroy
    #     the side-by-side story (a buyer would see "APPROVE / APPROVE"
    #     with two different excerpts, which is worse than no flip card).
    #
    # The canonical_hash recipe OMITS BOTH fields when BOTH equal the
    # default pair, so a v0.5 fixture hashes byte-identically. As soon
    # as either label differs from default, both stay in the payload —
    # the label pair is part of the topology's identity (judging the
    # same node but labelling its outputs differently IS a different
    # measurement contract).
    verdict_label_true: str = "APPROVE"
    verdict_label_false: str = "DENY"

    @field_validator(
        "verdict_label_true", "verdict_label_false", mode="before"
    )
    @classmethod
    def _normalize_verdict_label(cls, v: object) -> object:
        # Strict-type guard against ``str`` subclasses that override
        # ``__str__``/``.upper()``/``.strip()``/``__eq__`` (red-team
        # pass-1 M5): an attacker subclass that returns non-str from
        # ``.upper()`` could bypass the collision check post-normalize.
        # Buyer-facing signed bundle — lock to the canonical class.
        if type(v) is not str:
            raise ValueError(
                f"verdict label must be str (not subclass); got "
                f"{type(v).__name__}."
            )
        # NFKC normalize FIRST (red-team pass-1 B2): without this, a
        # manifest hand-edit or supply-chain tamper can slip a label
        # that visually renders as "APPROVE" past the collision check
        # via zero-width-space, fullwidth Latin, RTL-override, or
        # ligature tricks. NFKC collapses fullwidth → ASCII and combines
        # compatibility forms. We additionally strip:
        #   - Cc (control chars, e.g. TAB, NUL),
        #   - Cf (format chars, e.g. ZWSP U+200B, BOM U+FEFF, BIDI
        #     overrides U+202A-E),
        #   - M  (combining marks Mn/Mc/Me, e.g. U+FE0F variation
        #     selectors, U+0323 combining-dot-below) — red-team pass-2
        #     MAJ-1: a label like "APPROVE️" survives NFKC + Cc/Cf
        #     strip and renders visually identical to "APPROVE" but
        #     bytewise differs, breaking the "v0.5 byte-identical hash"
        #     invariant and bypassing the "differs from default → persist
        #     labels in manifest" gate.
        # AFTER the filter the string is restricted to L/N/P/S/Z
        # (letters/numbers/punctuation/symbols/separators) — printable
        # only, no homoglyph or zero-width tricks.
        normalized_nfkc = unicodedata.normalize("NFKC", v)
        normalized = "".join(
            ch
            for ch in normalized_nfkc
            if unicodedata.category(ch)[0] not in ("C", "M")
        ).strip().upper()
        if not normalized:
            raise ValueError(
                "verdict label must be non-empty after NFKC + strip + "
                f"control-char filter; got {v!r}."
            )
        if len(normalized) > 12:
            raise ValueError(
                f"verdict label must be ≤12 chars after normalize; got "
                f"{len(normalized)} chars in {normalized!r}. Side-by-side "
                "flip card overflows above this limit."
            )
        return normalized

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
        if self.verdict_node is not None and self.verdict_node not in id_set:
            raise ValueError(
                f"verdict_node {self.verdict_node!r} is not in node ids "
                f"{sorted(id_set)}"
            )
        if self.verdict_label_true == self.verdict_label_false:
            # Two labels that collapse to the same string post-normalize
            # destroy the side-by-side story (the CEO card would show
            # "APPROVE / APPROVE" with two excerpts). Fail at IR
            # construction time so a bad fixture never reaches the
            # orchestrator. Mirrors the manifest-side check in
            # ``publish._build_ceo_card_context`` which already rejects
            # the same collision for hand-edited manifests.
            raise ValueError(
                f"verdict_label_true and verdict_label_false must differ "
                f"after normalize; both collapsed to "
                f"{self.verdict_label_true!r}. (If you passed e.g. "
                f"'approve' / 'APPROVE', NFKC + .upper() collapses them.)"
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

        ``verdict_node=None`` is OMITTED from the canonicalised dict so a
        v0.5 topology (no verdict_node field on the wire) hashes
        byte-identically to its hash before this field existed. Without
        the omission, JCS would encode an explicit ``"verdict_node":null``
        and silently re-bucket every cached cell and signed bundle. When
        the field is set, it IS part of the topology's identity (judging
        a different node is a different measurement) and participates in
        the hash.

        Step 20a/F6 follow-up: ``verdict_label_true``/
        ``verdict_label_false`` are BOTH OMITTED from the canonicalised
        dict when BOTH equal the default pair ``("APPROVE","DENY")``.
        Strip is at the pair level — if either label differs from default,
        BOTH stay in the payload (the label pair is part of the topology's
        identity; judging the same node but labelling its outputs
        differently IS a different measurement contract). Pinned by
        ``test_ir.test_v05_hash_unchanged_after_label_fields_added``.

        NOTE: ``self.model_dump_json()`` does NOT apply either strip — it
        emits ``"verdict_node":null`` literally and emits the labels even
        at default. A caller who hashes ``json.loads(t.model_dump_json())``
        directly gets a DIFFERENT bucket than ``t.canonical_hash()`` on a
        v0.5-shape topology. Always go through ``canonical_hash()`` (or
        the analogous ``fuzzeval.canonical.canonical_hash`` on the
        stripped view) for cache / signing identity.
        """
        payload = self.model_dump(mode="json")
        if payload.get("verdict_node") is None:
            payload.pop("verdict_node", None)
        if (
            payload.get("verdict_label_true") == "APPROVE"
            and payload.get("verdict_label_false") == "DENY"
        ):
            payload.pop("verdict_label_true", None)
            payload.pop("verdict_label_false", None)
        return _canonical_hash(payload)
