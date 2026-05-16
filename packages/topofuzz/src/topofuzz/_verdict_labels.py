# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Shared verdict-label normalize for consumer paths.

Producer-side IR validation (``topofuzz.ir.Topology._normalize_verdict_label``)
applies strict NFKC + Cc/Cf/M strip + ``.strip().upper()`` + 12-char cap
before a manifest is ever written. Consumer paths
(``topofuzz.inspect._resolve_verdict_labels``,
``topofuzz.publish._build_ceo_card_context``) must apply the SAME
normalize step so that:

1. A well-formed manifest written by ``topofuzz run`` round-trips
   byte-identically through consumer rendering — no spurious diff
   between producer-side label policy and consumer rendering.
2. A hand-edited or supply-chain-tampered ``manifest.json`` cannot
   smuggle homoglyph tampers (combining marks, ZWSP, fullwidth Latin,
   NBSP padding) past the consumer normalize. The IR validator catches
   these at construction time, but the manifest is operator-readable
   post-hoc — a tampered manifest is the only thing standing between
   the buyer's eye and the CEO PDF / inspect HTML, so the consumer
   path is the last line of defense.

This module exposes the byte-faithful subset of the producer normalize
that is safe to apply at consumer read time. Type-strictness,
emptiness, and >12-char policy decisions remain in each consumer:
producer raises at IR construction; consumers fall back to defaults
(defense-in-depth for buyer-facing artifacts).

Imported by both ``topofuzz.inspect`` and ``topofuzz.publish``. Do not
inline — every caller must reach the same bytes.
"""

from __future__ import annotations

import unicodedata

__all__ = ["_normalize_verdict_label_for_consumer"]


def _normalize_verdict_label_for_consumer(raw: str) -> str:
    """Return ``raw`` after producer-faithful normalize.

    Mirrors ``topofuzz.ir.Topology._normalize_verdict_label`` byte-for-byte
    for the normalize step:

    1. NFKC normalize (collapses fullwidth → ASCII, composes compatibility
       forms; defeats fullwidth-Latin and ligature homoglyph tampers).
    2. Filter Unicode categories ``C`` (Cc/Cf/Cn/Co/Cs — control, format
       incl. ZWSP/BOM/BIDI overrides, unassigned, private-use, surrogate)
       and ``M`` (Mn/Mc/Me — combining marks incl. variation selectors,
       combining diacritics). After this filter the string is restricted
       to ``L`` / ``N`` / ``P`` / ``S`` / ``Z`` (letters, numbers,
       punctuation, symbols, separators) — printable only, no zero-width
       or homoglyph tricks.
    3. ``.strip().upper()`` — final whitespace trim + case fold.

    Does NOT enforce length cap, emptiness, or type strictness. Callers
    are responsible:

    * **Producer** (``ir.py``) raises ``ValueError`` at IR construction
      for empty / >12-char / non-``str`` inputs.
    * **Consumers** (``inspect.py``, ``publish.py``) log a WARNING and
      fall back to the default pair (``APPROVE`` / ``DENY``). The buyer
      should see the default labels, not a half-custom pair or a
      truncated string.

    Args:
        raw: A ``str`` instance. The strict-class check (``type(v) is str``
            rather than ``isinstance``) lives in the IR producer because
            buyer-facing signed bundles need the canonical class; consumer
            inputs always come from ``json.loads`` and are guaranteed plain
            ``str``, so ``isinstance`` is sufficient here.

    Returns:
        The normalized string. May be empty when the input is all
        whitespace, all combining marks, or all control / format
        characters — callers must handle the empty-result case (typically
        by falling back to the default).

    Raises:
        TypeError: If ``raw`` is not a ``str`` instance. Callers should
            guard with ``isinstance(raw, str)`` before invoking.
    """
    # Producer uses ``type(v) is str`` (strict class) at IR construction
    # because a buyer-facing signed bundle locks the canonical class to
    # defeat str-subclass __eq__/__upper__ overrides. Consumer inputs
    # always come from ``json.loads`` which is guaranteed plain ``str``,
    # so ``isinstance`` is sufficient here and lets in-memory test
    # fixtures pass non-attack subclasses (e.g. ``UserString``-style)
    # without spurious failures. See ``ir.py:185`` for the producer side.
    if not isinstance(raw, str):
        raise TypeError(
            f"_normalize_verdict_label_for_consumer requires str input, "
            f"got {type(raw).__name__}"
        )
    normalized_nfkc = unicodedata.normalize("NFKC", raw)
    return "".join(
        ch
        for ch in normalized_nfkc
        if unicodedata.category(ch)[0] not in ("C", "M")
    ).strip().upper()
