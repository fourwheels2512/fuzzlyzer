# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Cost helpers — the UI ALWAYS shows real dollars, never raw topofuzz units.

The topofuzz cost meter uses placeholder static rates that run ~2.75x hot
for Gemini (per the 2026-05-15 session finding in project_session_2026
_05_15_handoff.md). User-facing dollars in the UI are computed by
dividing the raw topofuzz cost cap (and the live raw spend during a run)
by COST_DIVISOR. The CLI still receives the raw cap on --max-cost-usd —
this divisor is purely a display correction.

If the topofuzz internal cost meter is ever fixed (replaces the placeholder
rates with real ones), set COST_DIVISOR = 1.0. The UI continues working
unchanged; the only difference is "raw" and "real" become the same number.
"""

from __future__ import annotations

COST_DIVISOR: float = 2.75
"""Divide raw topofuzz cost meter by this for real-dollar display."""


def real_dollars(raw_topofuzz_usd: float) -> float:
    """Convert a raw topofuzz cost meter value to real dollars."""
    if raw_topofuzz_usd < 0:
        raise ValueError("cost cannot be negative")
    return raw_topofuzz_usd / COST_DIVISOR


def raw_cap_for_real_budget(real_usd: float) -> float:
    """Inverse of :func:`real_dollars` — used to pick --max-cost-usd."""
    if real_usd < 0:
        raise ValueError("cost cannot be negative")
    return real_usd * COST_DIVISOR


def format_dollars(usd: float) -> str:
    """Render a USD amount the way a non-coder reads money: ``$1.80``."""
    return f"${usd:,.2f}"
