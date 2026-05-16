# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Operator-facing pre-flight diagnostic: which vendor API keys does
topofuzz see in the environment?

BACKLOG entry "MINOR — `topofuzz keys` CLI subcommand" (2026-05-16).
Without a pre-flight check, a misconfigured key surfaces only after
several seconds of orchestrator warmup at the first vendor call.
``topofuzz keys`` runs in <100ms, no vendor calls, masks every value
shown so it is safe to paste a copy of the output into a support
ticket.

Procurement-readiness angle: enterprise security teams can verify
exactly which keys topofuzz reads, with the actual values masked, by
running this command in front of them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# Minimum length for a key to be considered "real" rather than a
# placeholder / typo.
#
# Spec deviation (pass-1 RT M1): BACKLOG line 1376 said 8 chars with
# a 4+4 mask and "NO middle ellipsis" as the boundary case. That would
# print the ENTIRE 8-char value verbatim (4 prefix + 4 suffix == the
# whole key). The whole point of `topofuzz keys` is procurement-safe
# masking, so we bump the floor to 12: a 12-char value renders as
# ``AAAA..DDDD`` (4+2+4) — strictly fewer chars than the original.
#
# Empirical safety check (pass-2 RT M2): every currently-supported
# vendor's minimum real key length is far above 12:
#   * Gemini ``AIza...`` keys: 39 chars
#   * OpenAI legacy ``sk-...`` keys: 51 chars; ``sk-proj-...`` 164+
#   * Anthropic ``sk-ant-api03-...`` keys: ~108 chars
#   * DeepSeek ``sk-...`` keys: ~35 chars
# 12 is well below any plausible single-char truncation typo of these.
#
# Values below this threshold render with status ``too_short`` and
# empty ``masked_value`` — we refuse to print any prefix/suffix
# because the whole short string might BE the key.
_MIN_REAL_KEY_LENGTH = 12

# Vendor list — order matters for stable CLI output. Public (no
# underscore) so the test suite + any downstream tooling can import
# it directly; pass-2 RT m2 promoted it to part of the stable API.
# Mirrors BACKLOG entry 4 spec lines 1352-1357.
#
# If a future agent adds a new vendor, they MUST also update
# SECURITY.md "Network Egress" in the same PR. The
# ``test_vendor_list_matches_security_md_network_egress`` drift
# guard enforces the link both ways.
VENDORS: tuple[tuple[str, str], ...] = (
    ("google", "GEMINI_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
)


# ASCII ellipsis. The Unicode `…` (U+2026) would mangle on legacy
# Windows cp1252 consoles (procurement demos on stock cmd.exe); two
# ASCII dots are universally renderable. Two not three, so the
# masked form stays visually compact next to the 4-char prefix/suffix.
_MASK_ELLIPSIS = ".."


@dataclass(frozen=True, slots=True)
class KeyInfo:
    """A single vendor's key inspection result.

    Frozen so callers can rely on identity-stable rows for display.

    Stability: this dataclass is part of the topofuzz public-ish API
    (``from topofuzz.key_inspector import KeyInfo``). Field order +
    names are stable within a topofuzz major version. New fields may
    be added at the end without breaking existing callers; renames /
    reorders are reserved for major version bumps.
    """

    vendor: str
    env_var: str
    status: str  # "present" | "missing" | "too_short"
    masked_value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "vendor": self.vendor,
            "env_var": self.env_var,
            "status": self.status,
            "masked_value": self.masked_value,
        }


def _mask(value: str) -> str:
    """Mask a key for display.

    Shows the first 4 + last 4 chars with an ASCII ellipsis between
    (e.g. ``AIza..hgQ4``). The middle is ALWAYS hidden — even for
    values at the ``_MIN_REAL_KEY_LENGTH = 12`` floor, the mask
    leaves at least 4 chars unrevealed in the middle.

    Raises:
        ValueError: ``len(value) < _MIN_REAL_KEY_LENGTH``. Uses
            ``ValueError`` (not ``assert``) so ``python -O`` cannot
            strip the guard and silently leak short values.
    """
    if len(value) < _MIN_REAL_KEY_LENGTH:
        raise ValueError(
            "internal: _mask requires length >= _MIN_REAL_KEY_LENGTH; "
            "caller must filter short values first"
        )
    return value[:4] + _MASK_ELLIPSIS + value[-4:]


def inspect_keys(env: Mapping[str, str] | None = None) -> list[KeyInfo]:
    """Inspect the environment for known vendor API keys.

    Args:
        env: optional mapping override (for tests). When ``None``,
            reads ``os.environ`` at call time so a process that sets a
            key after import still sees it.

    Returns one :class:`KeyInfo` per registered vendor in declaration
    order. Status taxonomy (threshold = ``_MIN_REAL_KEY_LENGTH`` = 12):

    * ``"present"`` — variable is set, stripped value length ≥ 12.
    * ``"missing"`` — variable is unset, empty, or whitespace-only.
    * ``"too_short"`` — variable is set, stripped value length < 12
      (likely a placeholder or typo). ``masked_value`` is empty for
      this state — we refuse to leak any prefix of a short value
      because the entire short string might BE the key.
    """
    src: Mapping[str, str] = env if env is not None else os.environ
    rows: list[KeyInfo] = []
    for vendor, env_var in VENDORS:
        # Strip whitespace BEFORE the missing-check. A whitespace-only
        # value (``export GEMINI_API_KEY="   "``) is operator-intent
        # "unset" plus a copy/paste accident, NOT a 3-char key. Pass-1
        # RT M3 fix.
        raw = src.get(env_var, "").strip()
        if not raw:
            rows.append(
                KeyInfo(
                    vendor=vendor,
                    env_var=env_var,
                    status="missing",
                    masked_value="",
                )
            )
        elif len(raw) < _MIN_REAL_KEY_LENGTH:
            rows.append(
                KeyInfo(
                    vendor=vendor,
                    env_var=env_var,
                    status="too_short",
                    masked_value="",
                )
            )
        else:
            rows.append(
                KeyInfo(
                    vendor=vendor,
                    env_var=env_var,
                    status="present",
                    masked_value=_mask(raw),
                )
            )
    return rows


def format_table(rows: list[KeyInfo]) -> str:
    """Render rows as a 3-column human-readable table.

    Layout: vendor / env var / status+masked. Column widths are
    static and sized to the widest expected values so CI / docs
    snapshots stay byte-stable across runs and operating systems.
    """
    # Static widths chosen so the longest vendor ("anthropic" = 9) and
    # the longest env var ("ANTHROPIC_API_KEY" = 17) fit with a
    # 2-space gutter.
    vendor_w = 12
    env_w = 22
    header = (
        f"{'vendor':<{vendor_w}}{'env var':<{env_w}}status"
    )
    sep = "-" * (vendor_w + env_w + len("status"))
    lines = [header, sep]
    for row in rows:
        if row.status == "present":
            status_field = f"present  {row.masked_value}"
        elif row.status == "too_short":
            status_field = (
                f"too_short  (value < {_MIN_REAL_KEY_LENGTH} chars; "
                "not displayed)"
            )
        else:
            status_field = "missing"
        lines.append(
            f"{row.vendor:<{vendor_w}}{row.env_var:<{env_w}}{status_field}"
        )
    return "\n".join(lines)


def at_least_one_present(rows: list[KeyInfo]) -> bool:
    """Return True if any row has status ``"present"`` — used by the
    CLI to choose exit code 0 vs 1."""
    return any(row.status == "present" for row in rows)
