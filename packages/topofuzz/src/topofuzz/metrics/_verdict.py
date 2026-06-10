# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Shared 3-state verdict resolution for ``flip_confirmations`` rows.

:mod:`topofuzz.metrics.verdict_stability` (the noise floor) and
:mod:`topofuzz.metrics.confirmed_verdict_flip` (the corrected rate) must
agree on what verdict a confirm-pass repeat produced — the corrected rate
is only meaningful against the floor if both metrics classify the same
row the same way. This module is the single definition both import, so
the rule cannot drift between them.

The rule (2026-05-30, 3-state):

* A row carrying an explicit ``verdict_class`` of ``"approve"`` /
  ``"deny"`` / ``"conditional"`` votes as that class. This is what stops
  a genuinely conditional decision (e.g. "approve only via FHA, pending
  appraisal") from being binarised at 0.5 and coin-flipped across
  re-runs — the fake-flip inflation the 3-state work removed.
* Any other row (legacy pre-v7 rows, ``""`` numeric scores, the
  ``"dryrun"`` sentinel) falls back to the legacy
  ``score_value >= threshold`` rule, NORMALISED to the same
  ``"approve"`` / ``"deny"`` strings — never a bare bool. A bool ``True``
  never equals the string ``"approve"``, so a mixed block (class-carrying
  rows alongside legacy rows) compared with bools would read a genuinely
  unanimous block as non-unanimous (RT 2026-05-30).
* A row with no usable verdict at all (judge failure / missing score)
  resolves to ``None`` — missing data, which callers exclude.
"""

from __future__ import annotations

#: The canonical explicit class strings the v7 ``verdict_class`` column
#: may carry that resolve as-is (everything else threshold-falls-back).
_EXPLICIT_CLASSES = ("approve", "deny", "conditional")


def verdict_of(row: dict, threshold: float) -> str | None:
    """3-state verdict for a ``flip_confirmations`` row, as a canonical
    class string (``"approve"`` / ``"deny"`` / ``"conditional"``), or
    ``None`` when the row produced no usable verdict.

    See the module docstring for the full rule and its rationale.
    """
    vclass = row.get("verdict_class")
    if isinstance(vclass, str) and vclass in _EXPLICIT_CLASSES:
        return vclass
    score = row.get("score_value")
    if score is None:
        return None
    return "approve" if score >= threshold else "deny"
