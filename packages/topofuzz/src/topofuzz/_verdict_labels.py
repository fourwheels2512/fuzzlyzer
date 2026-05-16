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

__all__ = [
    "_normalize_verdict_label_for_consumer",
    "LABEL_ASCII_ALLOWLIST",
]


# Positive ASCII-printable allowlist for verdict labels (L949 closed
# 2026-05-16). After NFKC + C/M filter + ``.strip().upper()``, the
# producer and every consumer require the remaining string to consist
# ONLY of these bytes:
#
#   * ``A``-``Z`` (capital ASCII letters — the case fold is always
#     applied before the allowlist check, so lowercase input is
#     accepted and upper-cased first)
#   * ``0``-``9`` (digits — for labels like "ACTION-1" / "REASON-2A")
#   * ASCII space, hyphen, underscore, forward slash — for multi-word
#     labels like "PEND REVIEW" / "ESCALATE-L2" (all ≤12 chars, the
#     producer cap; "MANUAL REVIEW" at 13 chars exceeds the cap)
#
# Why this exists: prior versions allowed Unicode categories ``L``,
# ``N``, ``P``, ``S``, ``Z`` to pass (everything except control and
# combining marks). That let two homoglyph classes survive:
#
#   * **Invisible Lo letters** — rendered as zero-width in most fonts,
#     so a tampered manifest carrying ``"APㅤPROVE"`` would
#     pixel-render as ``"APPROVE"`` on the CEO PDF but mismatch the
#     buyer's ASCII ``grep`` for ``"APPROVE"``. The canonical surviving
#     codepoint is **U+1160 HANGUL JUNGSEONG FILLER** (Lo, NFKC-stable);
#     **U+3164 HANGUL FILLER** is the more common input form but is
#     NFKC-UNSTABLE — it decomposes to U+1160 in step 1 above, which
#     then survives the C/M filter. Either way the L949 allowlist
#     catches the post-NFKC byte.
#   * **Latin-lookalike Lu/Ll letters** (e.g. U+0410 CYRILLIC CAPITAL
#     LETTER A, U+0391 GREEK CAPITAL LETTER ALPHA, U+13AA CHEROKEE
#     LETTER GO) — all NFKC-self-stable, all render pixel-identical to
#     ASCII Latin A on every font with full Unicode support, but
#     mismatch ``grep "APPROVE"`` byte-wise.
#
# Both attacks are defeated by a positive ASCII allowlist. The cost
# is that any label carrying a non-ASCII letter is rejected — Spanish
# ``"APROBÉ"`` (with U+00C9 LATIN CAPITAL LETTER E WITH ACUTE, also
# category Lu), German ``"GENEHMIGT"`` (ASCII, accepted), French
# ``"ÉCHEC"`` (with U+00C9, rejected), etc. Per
# [[project_fuzzlyzer_gtm_cold_email_ceos]] every realistic buyer
# (US regulated-enterprise CROs/CAIOs) expects ASCII labels for grep
# auditing; i18n labels via an opt-in profile is a v0.7 BACKLOG.
LABEL_ASCII_ALLOWLIST: frozenset[str] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_/"
)


def _normalize_verdict_label_for_consumer(raw: str) -> str:
    """Return ``raw`` after producer-faithful normalize, or ``""`` on tamper.

    Mirrors ``topofuzz.ir.Topology._normalize_verdict_label`` byte-for-byte
    for the normalize step:

    1. NFKC normalize (collapses fullwidth → ASCII, composes compatibility
       forms; defeats fullwidth-Latin and ligature homoglyph tampers).
    2. Filter Unicode categories ``C`` (Cc/Cf/Cn/Co/Cs — control, format
       incl. ZWSP/BOM/BIDI overrides, unassigned, private-use, surrogate)
       and ``M`` (Mn/Mc/Me — combining marks incl. variation selectors,
       combining diacritics).
    3. ``.strip().upper()`` — final whitespace trim + case fold.
    4. **Positive ASCII allowlist** (L949 hardening, 2026-05-16): every
       remaining character must be in :data:`LABEL_ASCII_ALLOWLIST`. A
       single non-allowlist character causes the function to return
       ``""`` — signalling the caller to fall back to the default pair.
       Without this gate, a tampered manifest with a Cyrillic А or
       HANGUL FILLER would pixel-render as ``"APPROVE"`` on the CEO PDF
       while mismatching the buyer's grep on the manifest bytes.

    Producer (``ir.py``) RAISES on a non-allowlist character (strict
    fail at IR construction); this consumer-side helper DROPS the whole
    label (lenient fall-back to default at render time). The asymmetry
    is by design: producer-side bad input is an authoring bug
    (developer fixes it now); consumer-side bad input is a
    supply-chain tamper (buyer renders the safer default label rather
    than the attacker's payload).

    Args:
        raw: A ``str`` instance. The strict-class check (``type(v) is str``
            rather than ``isinstance``) lives in the IR producer because
            buyer-facing signed bundles need the canonical class; consumer
            inputs always come from ``json.loads`` and are guaranteed plain
            ``str``, so ``isinstance`` is sufficient here.

    Returns:
        The normalized string. ``""`` when the input is all whitespace,
        all combining marks, all control / format characters, OR
        contains any character outside :data:`LABEL_ASCII_ALLOWLIST`.
        Callers must handle the empty-result case (typically by falling
        back to the default pair).

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
    cleaned = "".join(
        ch
        for ch in normalized_nfkc
        if unicodedata.category(ch)[0] not in ("C", "M")
    ).strip().upper()
    if not cleaned:
        return ""
    # L949: refuse on a single non-allowlist character. Returning
    # the partial filtered string (e.g. "PPROVE" after dropping the
    # leading Cyrillic А) would render a label the buyer never
    # authorized — strictly WORSE than rendering the default. Empty
    # signals "tamper detected; consumer falls back to default."
    if any(ch not in LABEL_ASCII_ALLOWLIST for ch in cleaned):
        return ""
    return cleaned
