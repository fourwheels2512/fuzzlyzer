# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""``topofuzz inspect``: side-by-side flip inspector.

Answers the second skeptical question after a non-zero flip rate:
*"where in the reasoning chain did the AIs diverge?"*. ``topofuzz repro``
proves the flip is real; ``topofuzz inspect`` shows the reader the exact
agent whose response changed when the ordering changed.

Design contract:

* Pure module — the public entrypoints (:func:`build_inspect_html`,
  :func:`write_inspect_html`) are byte-deterministic for a fixed
  ``(run_dir, eval_id, threshold)``. No ``datetime.now()``, no
  process-salted hashes.
* Never imports the user's fixture. All inputs come from
  ``manifest.json`` and ``run.sqlite``.
* LLM-generated text is autoescaped by Jinja2 before risk-phrase
  highlighting runs; the highlighter only wraps already-escaped text
  in ``<span>`` tags, so a hostile agent response cannot inject HTML.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from topofuzz.metrics.verdict_flip import (
    DEFAULT_FLIP_THRESHOLD as DEFAULT_VERDICT_THRESHOLD,
)
from topofuzz._verdict_labels import _normalize_verdict_label_for_consumer

_log = logging.getLogger(__name__)

# Same filesystem-safe charset repro uses for the default output filename
# (run_dir / inspect_<eid>.html). Refuses path-traversal/symlink-bait.
_SAFE_EID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Hard cap on per-agent ``response_text`` and judge rationale length, in
# bytes of UTF-8. Above the cap, the text is truncated and a sentinel
# appended. The inspector is a visual reading aid — anyone needing the
# untruncated text reads `run.sqlite` directly. Without a cap, a single
# pathological agent response can: (a) blow the gzip payload past
# reasonable limits, (b) make the three-pass regex highlighter spin for
# seconds on the UI route's event loop (audit RT-pass-1 MINOR-1).
_MAX_RESPONSE_BYTES = 64 * 1024
_TRUNCATE_SENTINEL = (
    "\n\n[truncated by topofuzz inspect — read run.sqlite for full text]"
)


class InspectError(ValueError):
    """Raised when an inspector view cannot be built (no flip, bad input)."""


# Risk-phrase highlighting. Three colour groups: DENY (red), APPROVE
# (green), CONCERN (amber). The patterns target inflected forms a real
# underwriter prompt elicits ("denial", "approving", "high risk") rather
# than naive stems, so we don't accidentally highlight neutral words
# like "low" or "approximate". Order matters: longer multi-word phrases
# first so they aren't shadowed by their single-word constituents.
#
# These are matched against the **already-HTML-escaped** response text,
# so `<` `>` and `&` are never inside a token — word-boundary semantics
# match Python's normal `\b`. The patterns are compiled with
# ``re.IGNORECASE`` so "Deny" and "deny" both highlight.
_HIGHLIGHT_GROUPS: tuple[tuple[str, str], ...] = (
    (
        "phrase-concern",
        r"\b(high[\s-]+risk|low[\s-]+risk|concerning|concerned|concerns?|"
        r"exceptions?|caveats?|caution(?:ary|s)?)\b",
    ),
    (
        "phrase-deny",
        r"\b(deny|denied|denying|denial|denials|"
        r"decline|declined|declines|declining|declination|"
        r"reject|rejected|rejects|rejecting|rejection)\b",
    ),
    (
        "phrase-approve",
        r"\b(approve|approved|approves|approving|approval|approvals|"
        r"grant|granted|grants|granting)\b",
    ),
)
_COMPILED_HIGHLIGHTS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (cls, re.compile(pat, re.IGNORECASE)) for cls, pat in _HIGHLIGHT_GROUPS
)


def find_inspect_pair(
    run_dir: Path,
    *,
    eval_id: str | None = None,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> tuple[str, int, int]:
    """Return ``(eval_id, canonical_idx, flipping_idx)`` for one flip.

    * ``canonical_idx`` is always ``0`` — ``manifest.variant_canonical_hashes[0]``
      is the §6 kickoff-insertion-order original ordering and the natural
      "left side" of the comparison.
    * ``flipping_idx`` is the smallest variant index (>0) whose binarised
      verdict differs from the canonical's on this eval.
    * Eval selection: when ``eval_id`` is None, scan eval_ids in
      lexicographic order and pick the first one with a flip.

    Both selections are deterministic so successive ``topofuzz inspect``
    calls on the same run produce the same view.

    Cells where the original variant has no scored verdict (judge
    failure) are skipped — there's no reference frame to flip against.

    Raises:
        InspectError: no flip found, ``eval_id`` doesn't exist, the
            original variant has no verdict for ``eval_id``, or
            ``manifest.json`` is missing/malformed.
    """
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    hashes = manifest["variant_canonical_hashes"]
    if not hashes:
        raise InspectError(
            f"manifest.variant_canonical_hashes is empty in run "
            f"{run_dir.name} — cannot identify the original variant."
        )
    original_hash = hashes[0]
    hash_to_idx = {h: i for i, h in enumerate(hashes)}

    con = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        try:
            rows = list(
                con.execute(
                    "SELECT variant_id, eval_id, score_value, passed "
                    "FROM judge_results "
                    "ORDER BY eval_id, variant_id"
                )
            )
        except sqlite3.DatabaseError as e:
            raise InspectError(
                f"run.sqlite in {run_dir.name} is missing the expected "
                f"judge_results schema or is corrupt: {e}"
            ) from e
    finally:
        con.close()

    by_eval: dict[str, dict[int, bool]] = defaultdict(dict)
    for vid, eid, score, passed in rows:
        idx = hash_to_idx.get(vid)
        if idx is None:
            continue
        if passed is not None:
            verdict = bool(passed)
        elif score is not None:
            verdict = bool(score >= threshold)
        else:
            continue
        by_eval[eid][idx] = verdict

    if eval_id is not None:
        if eval_id not in by_eval:
            raise InspectError(
                f"eval_id={eval_id!r} has no scored judge results in run "
                f"{run_dir.name} (all rows may be judge-failures)"
            )
        candidates: list[str] = [eval_id]
    else:
        candidates = sorted(by_eval.keys())

    for eid in candidates:
        verdicts = by_eval[eid]
        original_verdict = verdicts.get(0)
        if original_verdict is None:
            # No reference frame on this eval; skip silently when
            # auto-scanning, but raise an explicit error when the caller
            # named the eval — the caller deserves to know.
            if eval_id is not None:
                raise InspectError(
                    f"eval_id={eval_id!r}: the original variant "
                    f"({original_hash[:8]}) has no scored verdict — "
                    "judge failure on the reference. There's no left-hand "
                    "side to compare against."
                )
            continue
        for other_idx in sorted(i for i in verdicts if i != 0):
            if verdicts[other_idx] != original_verdict:
                _log.debug(
                    "inspect: pair selected eid=%s canonical=0 flipping=%d",
                    eid, other_idx,
                )
                return eid, 0, other_idx

    suffix = f" for eval {eval_id!r}" if eval_id else ""
    raise InspectError(
        f"No verdict flip found in run {run_dir.name}{suffix}. "
        "Topofuzz inspect requires at least one eval row where the "
        "original variant disagrees with another variant on the "
        "binarised verdict."
    )


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    """Read+parse ``manifest.json`` once. Centralises error messages."""
    try:
        return json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as e:
        raise InspectError(
            f"manifest.json missing in run_dir {run_dir.name!r}"
        ) from e
    except json.JSONDecodeError as e:
        raise InspectError(
            f"manifest.json in {run_dir.name!r} is not valid JSON: {e}"
        ) from e


def _resolve_verdict_labels(manifest: dict[str, Any]) -> tuple[str, str]:
    """Return ``(label_true, label_false)`` resolved from manifest with
    defense-in-depth.

    Mirrors ``topofuzz.publish._build_ceo_card_context``'s manifest-label
    handling so the side-by-side flip HTML renders identical labels to
    the CEO PDF for the same run (Step 20a/F6 follow-up):

    1. **XOR half-set:** the orchestrator persists labels as a pair
       (both-or-neither, gated on differs-from-default). A manifest with
       exactly one key non-empty is an obvious tamper or hand-edit —
       log + fall back to BOTH defaults rather than silently defaulting
       the unset half against an explicit other half.
    2. **Producer-faithful normalize:** apply NFKC + Cc/Cf/M strip +
       ``.strip().upper()`` via :func:`_normalize_verdict_label_for_consumer`.
       Without this, a hand-edited manifest with ZWSP / combining marks /
       fullwidth Latin / NBSP padding survives the consumer normalize
       and renders visually-as-default with bytewise-tampered content.
       The shared helper is byte-faithful to the producer IR validator
       so a well-formed manifest round-trips identically and a tampered
       one collapses to the same bytes the IR would have produced.
    3. **>12-char cap (POST-normalize):** the producer IR caps labels at
       12 chars after the full normalize chain. A tampered manifest with
       a label that normalizes to >12 chars would render truncated
       nonsense — log + fall back to BOTH defaults so the buyer sees the
       same default pair rather than a half-custom pair. Measured on the
       normalized string (not the raw) to mirror producer policy: a
       ZWSP-padded 15-char raw that collapses to 7 chars post-normalize
       is fine, a 14-char fullwidth raw that collapses to 14 ASCII chars
       is not.
    4. **Collision after normalize:** two labels that collapse to the
       same string destroy the side-by-side story (a buyer would see
       "APPROVE / APPROVE" with two different excerpts, worse than no
       inspect view). Raise :class:`InspectError`. Producer rejects this
       at IR construction; consumer is last line of defense.
    5. **Defaults fallback:** absent / empty / wrong-type / empty-after-
       normalize values resolve to ``("APPROVE", "DENY")``.
    """
    label_true_raw = manifest.get("verdict_label_true")
    label_false_raw = manifest.get("verdict_label_false")
    # Normalize first so the XOR check below operates on POST-normalize
    # results. Without this, a label like `"​"` (ZWSP only) is
    # truthy under `.strip()` (which does NOT strip Cf characters),
    # passes the XOR as "set", then collapses to empty in the helper
    # and falls back to the default partner — yielding a half-custom
    # pair (custom-true + default-false) that defeats the pair-level
    # fallback the consumer claims to enforce. Pass-1 dual-audit BLOCKER
    # equivalent (CR M-1 + RT M1, 2026-05-16).
    norm_true = (
        _normalize_verdict_label_for_consumer(label_true_raw)
        if isinstance(label_true_raw, str) else ""
    )
    norm_false = (
        _normalize_verdict_label_for_consumer(label_false_raw)
        if isinstance(label_false_raw, str) else ""
    )
    if bool(norm_true) != bool(norm_false):
        _log.warning(
            "inspect: manifest has exactly one of verdict_label_true/false "
            "non-empty after NFKC + Cc/Cf/M strip (true=%r, false=%r) — "
            "both must resolve to a non-empty pair or neither (orchestrator "
            "persists as a pair). Falling back to APPROVE/DENY defaults.",
            label_true_raw, label_false_raw,
        )
        norm_true = ""
        norm_false = ""
    for raw_label, norm, key in (
        (label_true_raw, norm_true, "verdict_label_true"),
        (label_false_raw, norm_false, "verdict_label_false"),
    ):
        if len(norm) > 12:
            _log.warning(
                "inspect: manifest %s=%r normalizes to >12 chars after "
                "NFKC + Cc/Cf/M strip — exceeds the side-by-side flip-"
                "card cap. Falling back to APPROVE/DENY for BOTH labels "
                "(pair-level fallback; the partner label is discarded "
                "too so the reader sees the same default pair rather "
                "than a half-custom pair).",
                key, raw_label,
            )
            norm_true = ""
            norm_false = ""
            break
    label_true = norm_true if norm_true else "APPROVE"
    label_false = norm_false if norm_false else "DENY"
    if label_true == label_false:
        raise InspectError(
            f"manifest verdict_label_true/false collide after "
            f"normalization (raw: true={label_true_raw!r}, "
            f"false={label_false_raw!r}; both → {label_true!r}); "
            f"refuse to render."
        )
    return label_true, label_false


def _load_cell(
    run_dir: Path, variant_id: str, eval_id: str
) -> dict[str, Any]:
    """Return the cell + judge data for one (variant, eval) pair.

    Shape::

        {
            "node_outputs": [{"node_id", "role", "response_text",
                              "model_served_version", ...}, ...],
            "judge_passed": bool | None,
            "judge_score": float | None,
            "judge_rationale": str | None,
            "jury_votes": list[dict] | None,
            "error": str | None,
        }

    ``jury_votes`` is ``None`` for single-judge cells, or a list of
    ``{"vendor": str, "score": float, "rationale": str}`` rows when the
    judge was a :class:`JuryJudge`. The list mirrors the JuryJudge's
    construction order; a vendor that failed to parse contributes no
    row (the JuryJudge already filtered the unparseable ones).

    Raises:
        InspectError: cell missing — should never happen because the
            cell is paired with a judge_results row that was already
            confirmed in :func:`find_inspect_pair`, but defended for
            forensic clarity if it does.
    """
    con = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        cell_row = con.execute(
            "SELECT node_outputs_json, error "
            "FROM cells "
            "WHERE variant_id=? AND eval_id=?",
            (variant_id, eval_id),
        ).fetchone()
        # votes_json was added in schema v6 (F3 jury vote breakdown).
        # `_load_cell` reads run.sqlite read-only — it never opens the
        # write path that runs ``_migrate``, so a forward-compat
        # legacy-v5 DB still has no votes_json column. Probe the column
        # list and fall back to NULL when absent so this loader works
        # against ANY supported schema version.
        judge_cols = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(judge_results)"
            ).fetchall()
        }
        has_votes_json = "votes_json" in judge_cols
        select_cols = (
            "passed, score_value, rationale, votes_json"
            if has_votes_json
            else "passed, score_value, rationale, NULL"
        )
        judge_row = con.execute(
            f"SELECT {select_cols} "
            "FROM judge_results "
            "WHERE variant_id=? AND eval_id=?",
            (variant_id, eval_id),
        ).fetchone()
    finally:
        con.close()

    if cell_row is None:
        raise InspectError(
            f"Cell missing for variant={variant_id[:8]} eval={eval_id} "
            f"in run {run_dir.name} — judge_results pointed to a row "
            "the cells table doesn't have."
        )
    try:
        node_outputs = json.loads(cell_row[0])
    except (TypeError, json.JSONDecodeError) as e:
        raise InspectError(
            f"node_outputs_json malformed for variant={variant_id[:8]} "
            f"eval={eval_id}: {e}"
        ) from e
    if not isinstance(node_outputs, list):
        raise InspectError(
            f"node_outputs_json is not a JSON array for "
            f"variant={variant_id[:8]} eval={eval_id}"
        )
    # Audit RT-pass-1 MINOR-2: each element must be a dict so `_agent_view`
    # can call `.get(...)` on it. A list like `[42, "string", null]` would
    # otherwise reach `_agent_view` and raise `AttributeError` deep inside
    # the rendering path, surfaced as a 500 from the UI route.
    for i, item in enumerate(node_outputs):
        if not isinstance(item, dict):
            raise InspectError(
                f"node_outputs_json[{i}] is a {type(item).__name__}, "
                f"expected object; variant={variant_id[:8]} eval={eval_id}"
            )

    judge_passed: bool | None = None
    judge_score: float | None = None
    judge_rationale: str | None = None
    jury_votes: list[dict[str, Any]] | None = None
    if judge_row is not None:
        passed_raw, score_raw, rationale_raw, votes_json_raw = judge_row
        judge_passed = (
            bool(passed_raw) if passed_raw is not None else None
        )
        judge_score = (
            float(score_raw) if score_raw is not None else None
        )
        judge_rationale = (
            rationale_raw if isinstance(rationale_raw, str) else None
        )
        if isinstance(votes_json_raw, str) and votes_json_raw:
            try:
                parsed = json.loads(votes_json_raw)
            except json.JSONDecodeError:
                # Persisted column malformed (shouldn't happen — the
                # writer always JSON-dumps). Treat as a missing breakdown
                # rather than failing the whole cell render — single-
                # judge UX still works without the panel.
                parsed = None
            if isinstance(parsed, list):
                # Defensive shape filter: keep only well-formed entries
                # so a future schema drift doesn't crash the template.
                jury_votes = [
                    {
                        "vendor": str(entry.get("vendor", "")),
                        "score": float(entry.get("score", 0.0)),
                        "rationale": str(entry.get("rationale", "")),
                    }
                    for entry in parsed
                    if isinstance(entry, dict)
                    and "vendor" in entry
                    and "score" in entry
                ] or None

    return {
        "node_outputs": node_outputs,
        "judge_passed": judge_passed,
        "judge_score": judge_score,
        "judge_rationale": judge_rationale,
        "jury_votes": jury_votes,
        "error": cell_row[1],
    }


def _highlight_risk_phrases(escaped_text: str) -> str:
    """Wrap risk phrases in `<span class="phrase-*">`.

    Operates on **already-HTML-escaped** text — Jinja2's autoescape ran
    upstream, so this function never sees raw `<`, `>`, or `&`. Returns
    a plain str that is safe to mark as ``Markup`` (the caller does that).

    Overlapping patterns: we apply groups in declared order, and each
    pass refuses to descend into spans we already wrote (by skipping any
    match whose run overlaps an existing span). Overlap detection uses
    a simple "already-wrapped position set" computed after each pass —
    fine for our short response texts (< a few KB).
    """
    if not escaped_text:
        return escaped_text
    # `wrapped[i]` == True means character `i` of the current `text` is
    # already inside a wrap from an earlier pass. We rebuild this between
    # passes because the indices shift as we insert `<span>` tags.
    text = escaped_text
    for css_class, pattern in _COMPILED_HIGHLIGHTS:
        wrapped_intervals = _existing_wrap_intervals(text)
        out: list[str] = []
        cursor = 0
        for m in pattern.finditer(text):
            start, end = m.span()
            if _overlaps(start, end, wrapped_intervals):
                continue
            out.append(text[cursor:start])
            out.append(f'<span class="{css_class}">')
            out.append(text[start:end])
            out.append("</span>")
            cursor = end
        out.append(text[cursor:])
        text = "".join(out)
    return text


_EXISTING_SPAN_RE = re.compile(
    r'<span class="phrase-[a-z]+">.*?</span>', re.DOTALL
)


def _existing_wrap_intervals(text: str) -> list[tuple[int, int]]:
    """Return (start, end) intervals already occupied by phrase spans."""
    return [m.span() for m in _EXISTING_SPAN_RE.finditer(text)]


def _overlaps(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for a, b in intervals:
        if start < b and a < end:
            return True
    return False


def _render_rationale(rationale: str | None, escape: Any, Markup: Any) -> Any:
    """Build the rationale `Markup` for one panel.

    Returns ``Markup("")`` when the judge wrote no rationale — the template
    treats that as falsy and hides the block. Truncates very long
    rationales to keep payload bounded (same cap as agent responses).
    """
    if rationale is None or rationale == "":
        return Markup("")
    capped = _truncate_for_render(rationale)
    return Markup(_highlight_risk_phrases(str(escape(capped))))


def _truncate_for_render(text: str, *, max_bytes: int = _MAX_RESPONSE_BYTES) -> str:
    """Cap ``text`` at ``max_bytes`` UTF-8 bytes, appending the sentinel.

    Truncates on a UTF-8 boundary so the returned string is always valid
    UTF-8 — important because the rendered HTML is emitted as UTF-8 and
    a mid-codepoint cut would produce a mojibake byte the browser would
    render as ``U+FFFD``.
    """
    if not text:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # Back off until we land on a clean codepoint boundary.
    cut = max_bytes
    while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
        cut -= 1
    head = encoded[:cut].decode("utf-8", errors="ignore")
    return head + _TRUNCATE_SENTINEL


def _agent_view(
    node_output: dict[str, Any], escape: Any
) -> dict[str, Any]:
    """Build one agent card's template-ready dict.

    ``escape`` is the Jinja2 escape callable passed in by the caller; we
    use it directly so the highlighter operates on the same escaped form
    Jinja would otherwise render.
    """
    response_raw = node_output.get("response_text", "")
    if not isinstance(response_raw, str):
        response_raw = json.dumps(response_raw, ensure_ascii=False)
    response_raw = _truncate_for_render(response_raw)
    escaped = str(escape(response_raw))
    highlighted = _highlight_risk_phrases(escaped)
    return {
        "node_id": str(node_output.get("node_id", "")),
        "role": str(node_output.get("role", "")),
        "model": str(node_output.get("model_served_version", "")),
        "cache_hit": bool(node_output.get("cache_hit", False)),
        "response_html": highlighted,
    }


def _verdict_label(
    passed: bool | None,
    score: float | None,
    threshold: float,
    label_true: str = "APPROVE",
    label_false: str = "DENY",
) -> str:
    """Plain-English verdict label for a cell.

    When ``passed`` is recorded (binary judge), use it directly. Otherwise
    derive from ``score`` vs ``threshold`` — the same §6 binarisation the
    verdict-flip metric uses — so a continuous-judge cell still reads as
    APPROVE/DENY visually instead of a neutral "score N.NN" pill.

    ``label_true`` / ``label_false`` default to the lending-domain
    APPROVE/DENY pair but accept manifest-driven overrides resolved by
    :func:`_resolve_verdict_labels` so a non-lending fixture (e.g.
    adversarial-relay's ESCALATE/DEFER) renders consistent labels in the
    inspect HTML and the CEO PDF for the same run.
    """
    if passed is True:
        return label_true
    if passed is False:
        return label_false
    if score is not None:
        verdict = label_true if score >= threshold else label_false
        return f"{verdict} ({score:.2f})"
    return "(no verdict)"


def _jury_votes_view(
    votes: list[dict[str, Any]] | None,
    threshold: float,
    escape: Any,
    Markup: Any,
    label_true: str = "APPROVE",
    label_false: str = "DENY",
) -> dict[str, Any] | None:
    """Template-ready breakdown of a JuryJudge's per-vendor votes.

    Returns ``None`` for single-judge cells (the template's ``{% if %}``
    guard then hides the panel). For jury cells, returns::

        {
            "approve_count": int,
            "deny_count": int,
            "total": int,                 # surviving vendor count
            "majority_label": label_true | label_false | "TIE → <false>",
            "majority_class": str,         # CSS class for the badge
            "rows": [{"vendor", "score_pct", "verdict_label",
                      "verdict_class", "rationale_html"}, ...],
        }

    Per-vendor verdicts use the same ``score >= threshold`` binarisation
    that drives the overall verdict pill, so the count of approves in
    the rows matches the count the user sees in the panel header.

    ``label_true`` / ``label_false`` default to APPROVE/DENY but accept
    manifest-driven overrides resolved by
    :func:`_resolve_verdict_labels` so a non-lending fixture (e.g.
    adversarial-relay's ESCALATE/DEFER) renders consistent labels in the
    inspect HTML and the CEO PDF for the same run. The
    ``approve_count``/``deny_count`` field names stay stable (they're
    template contracts) — the labels are user-facing, the field names
    are internal.

    Fuzzer M2 audit fix: a deadlocked jury (``approve_count * 2 ==
    total``, e.g. one vendor failed and the surviving 1/1 split) gets
    a dedicated TIE label so the panel doesn't silently look like a
    consensus. The headline `value` already resolved the tie to 0.0
    (the false-side label) per the compliance-conservative convention;
    the panel just spells out that the jury was deadlocked.

    Rationales are escaped + risk-phrase-highlighted exactly like the
    aggregate rationale block above so the same legend covers both.
    """
    if not votes:
        return None
    approve_count = 0
    deny_count = 0
    rows: list[dict[str, Any]] = []
    for v in votes:
        score = float(v.get("score", 0.0))
        approved = score >= threshold
        if approved:
            approve_count += 1
            verdict_class = "verdict-approve"
            verdict_label = label_true
        else:
            deny_count += 1
            verdict_class = "verdict-deny"
            verdict_label = label_false
        rationale_text = str(v.get("rationale", ""))
        rationale_html = _render_rationale(
            rationale_text or None, escape, Markup
        )
        rows.append({
            "vendor": str(v.get("vendor", "")),
            "score_pct": f"{score * 100:.0f}%",
            "score_value": f"{score:.2f}",
            "verdict_label": verdict_label,
            "verdict_class": verdict_class,
            "rationale_html": rationale_html,
        })
    total = len(rows)
    if approve_count * 2 > total:
        majority_label = label_true
        majority_class = "verdict-approve"
    elif approve_count * 2 == total:
        # Deadlocked jury: 1/1 (one vendor failed), 1/2 (one failed),
        # any even-total split. Resolves to the false-side label at the
        # score level but the panel labels it explicitly so the user
        # understands the headline pill came from the conservative-tie
        # default.
        majority_label = f"TIE → {label_false}"
        majority_class = "verdict-unknown"
    else:
        majority_label = label_false
        majority_class = "verdict-deny"
    return {
        "approve_count": approve_count,
        "deny_count": deny_count,
        "total": total,
        "majority_label": majority_label,
        "majority_class": majority_class,
        "rows": rows,
    }


def _ablation_view(
    sidecar: dict[str, Any] | None,
    *,
    canonical_hash: str,
    flipping_hash: str,
    eval_id: str,
    escape: Any,
    Markup: Any,
) -> dict[str, Any] | None:
    """Template-ready per-flip attribution panel from ``ablation.json``.

    Returns ``None`` when:

    * The sidecar is absent (``topofuzz ablate`` has not been run on
      this run dir yet) — the template's ``{% if %}`` guard then hides
      the panel.
    * The sidecar has no ``results`` entries matching this specific
      (canonical, flipping, eval) triple — every ablation point was
      for a different flip. Hiding the panel here means a single
      F4 sidecar can land alongside an `inspect` view chosen for a
      *different* flipping pair without producing a misleading
      empty-panel render.

    Per-flip filter shape (Fuzzer-style measurement-integrity check):
    the attribution rendered MUST be the attribution computed against
    THIS exact ``(canonical_hash, flipping_hash, eval_id)`` triple —
    not the run-level aggregate. The run-level ``by_node`` lives in
    the sidecar for the CLI summary, but the per-flip panel re-computes
    locally so the buyer cannot read the wrong number off the wrong
    flip.
    """
    if not sidecar:
        return None
    results = sidecar.get("results")
    if not isinstance(results, list) or not results:
        return None
    relevant = [
        r for r in results
        if isinstance(r, dict)
        and r.get("canonical_variant_id") == canonical_hash
        and r.get("flipping_variant_id") == flipping_hash
        and r.get("eval_id") == eval_id
    ]
    if not relevant:
        return None
    # Aggregate by node_id (the panel iterates V's nodes — ordering
    # comes from the order the entries appear in the sidecar, which
    # mirrors V's execution order).
    by_node: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for r in relevant:
        nid = str(r.get("ablated_node_id", ""))
        if not nid:
            continue
        if nid not in by_node:
            order.append(nid)
            by_node[nid] = {
                "flips_closed": 0,
                "total_meaningful": 0,
                "is_terminal": False,
                "kind": str(r.get("kind", "")),
                "new_terminal_response_html": Markup(""),
            }
        slot = by_node[nid]
        kind = str(r.get("kind", ""))
        if kind == "substitute_terminal":
            slot["is_terminal"] = True
        if kind in ("substitute_terminal", "reexecute_terminal"):
            slot["total_meaningful"] += 1
            if r.get("flip_closed") is True:
                slot["flips_closed"] += 1
        slot["kind"] = kind
        # Pass-1 Fuzzer MINOR m3 fix: a `failed` row's
        # ``new_terminal_response_text`` is whatever was set before the
        # error (often empty, but never a measured output). Render
        # explicit fail text so the buyer doesn't infer the leftover
        # is "the post-ablation output".
        if kind == "failed":
            slot["new_terminal_response_html"] = Markup(
                "[re-execution failed — no terminal output recorded]"
            )
        elif r.get("new_terminal_response_text"):
            slot["new_terminal_response_html"] = Markup(
                _highlight_risk_phrases(
                    str(escape(_truncate_for_render(
                        str(r["new_terminal_response_text"])
                    )))
                )
            )
    rows: list[dict[str, Any]] = []
    # Pass-1 Fuzzer BLOCKER B2 fix: skip kinds produce
    # ``total_meaningful=0`` and ``attribution_pct=0.0``. The template
    # MUST distinguish "0% (measured, no flip closed)" from
    # "(skipped, no measurement)" so a buyer doesn't read the 0% pill
    # as "this agent contributes nothing". The `has_measurement` flag
    # gates the template's bar+pct vs em-dash render.
    SKIP_KINDS = {
        "skip_downstream",
        "skip_no_canonical_output",
        "skip_no_v_output",
        "failed",
    }
    for nid in order:
        slot = by_node[nid]
        total = slot["total_meaningful"]
        pct = (
            round(100.0 * slot["flips_closed"] / total, 2) if total > 0 else 0.0
        )
        kind = slot["kind"]
        has_measurement = kind not in SKIP_KINDS and total > 0
        rows.append({
            "node_id": nid,
            "flips_closed": slot["flips_closed"],
            "total_meaningful": total,
            "attribution_pct": pct,
            "is_terminal": slot["is_terminal"],
            "kind": kind,
            "has_measurement": has_measurement,
            "new_terminal_response_html": slot["new_terminal_response_html"],
        })
    # Headline summary: the strongest non-terminal attribution AMONG
    # MEASURED rows only. The buyer-actionable answer is "the upstream
    # agent that, when set to canonical, closes this flip". The terminal
    # substitution is the always-100% baseline (excluded). Pass-1 Fuzzer
    # BLOCKER B1 fix: when every upstream agent was structurally
    # skip_downstream / skip_no_canonical_output, surface the structural
    # cause rather than a bland "no upstream" — the buyer needs to know
    # the verdict node ran first in this permutation.
    measured_non_terminal = [
        r for r in rows if not r["is_terminal"] and r["has_measurement"]
    ]
    skipped_non_terminal = [
        r for r in rows if not r["is_terminal"] and not r["has_measurement"]
    ]
    if measured_non_terminal:
        top = max(measured_non_terminal, key=lambda r: r["attribution_pct"])
        headline = (
            f"{top['node_id']} — {top['attribution_pct']:.0f}% of this flip "
            f"closes when it matches canonical"
        )
    elif skipped_non_terminal:
        # All upstream nodes were structurally excluded — most often
        # because the terminal ran early in V's permutation.
        headline = (
            "No upstream attribution measurable for this flip: every "
            "upstream agent ran AFTER the terminal in V's execution "
            "order (or had no canonical output to swap in). The verdict "
            "node was decided before any upstream context arrived."
        )
    else:
        headline = "(no upstream agents in V's prior slice for this flip)"
    return {
        "headline": headline,
        "rows": rows,
        "scope_note": (
            "Attribution shown for THIS flip only "
            "(canonical vs. flipping variant, single eval row). The "
            "CLI's `topofuzz ablate` summary aggregates across every "
            "flipping triple in the run; numbers may differ."
        ),
    }


def build_inspect_html(
    run_dir: Path,
    *,
    eval_id: str | None = None,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> tuple[str, str]:
    """Return ``(html_text, chosen_eval_id)``.

    Pure function: byte-identical output for identical
    ``(run_dir, eval_id, threshold)``. The template only renders
    manifest-frozen values (``run_id``, ``topology_name``, ``eval_id``,
    ordering arrays, cell contents) — never ``datetime.now()`` — so
    re-rendering on a different date doesn't perturb the bytes.
    """
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    # Pass-2 RT-specialist MAJ-3 fix: find_inspect_pair MUST run before
    # _resolve_verdict_labels. A degenerate run (no flip) is the dominant
    # real-world failure; manifest tampering is a corner case. Without
    # this ordering, a colliding-labels manifest on a no-flip run would
    # raise the less-useful "collide after normalization" error instead
    # of "No verdict flip" — operator-confusing. Labels are only used
    # after pair resolution, so the swap is free.
    eid, canonical_idx, flipping_idx = find_inspect_pair(
        run_dir, eval_id=eval_id, threshold=threshold
    )
    # Step 20a/F6 follow-up: resolve manifest-driven verdict labels with
    # the same defense-in-depth posture as ``publish._build_ceo_card_context``
    # so the side-by-side flip HTML renders the same label pair as the
    # CEO PDF for the same run (BACKLOG 2026-05-16 — inspect consumer
    # wiring). Defaults to ("APPROVE", "DENY") when manifest is absent
    # / opted-out / tampered.
    label_true, label_false = _resolve_verdict_labels(manifest)
    hashes = manifest["variant_canonical_hashes"]
    orderings = manifest.get("variant_orderings", [])
    # Audit RT-pass-1 MAJOR-2: a manifest with fewer variant_orderings than
    # variant_canonical_hashes (or a non-list) would IndexError below and
    # surface as a 500 from the UI route. Validate up front and raise a
    # caller-catchable InspectError instead.
    if not isinstance(orderings, list):
        raise InspectError(
            "manifest.variant_orderings is not a list — manifest is "
            "internally inconsistent."
        )
    required = max(canonical_idx, flipping_idx) + 1
    if len(orderings) < required:
        raise InspectError(
            f"manifest.variant_orderings has {len(orderings)} entries but "
            f"the verdict pair references index {required - 1} — manifest "
            "is internally inconsistent."
        )
    # Audit RT-pass-2 MINOR-B: per-element shape check. Without this, a
    # string element renders as a char-list of node ids in the template
    # ("a", "b", "c") and a dict renders its keys — both nonsensical
    # but they slip past Jinja's escape (which is happy with any iterable).
    for idx in (canonical_idx, flipping_idx):
        if not isinstance(orderings[idx], list):
            raise InspectError(
                f"manifest.variant_orderings[{idx}] is a "
                f"{type(orderings[idx]).__name__}, expected list."
            )
    canonical_hash = hashes[canonical_idx]
    flipping_hash = hashes[flipping_idx]
    canonical_ordering = list(orderings[canonical_idx])
    flipping_ordering = list(orderings[flipping_idx])

    canonical_cell = _load_cell(run_dir, canonical_hash, eid)
    flipping_cell = _load_cell(run_dir, flipping_hash, eid)

    try:
        from jinja2 import Environment, PackageLoader, StrictUndefined
        from markupsafe import Markup, escape
    except ImportError as e:
        raise InspectError(
            "topofuzz inspect requires the 'full' install extra "
            "(jinja2 / markupsafe not found). "
            "Install with `pip install topofuzz[full]`."
        ) from e

    canonical_agents = [
        _agent_view(o, escape) for o in canonical_cell["node_outputs"]
    ]
    flipping_agents = [
        _agent_view(o, escape) for o in flipping_cell["node_outputs"]
    ]

    canonical_panel = {
        "label": "Original ordering",
        "short_hash": canonical_hash[:8],
        "ordering": canonical_ordering,
        "verdict": _verdict_label(
            canonical_cell["judge_passed"],
            canonical_cell["judge_score"],
            threshold,
            label_true,
            label_false,
        ),
        "verdict_class": _verdict_css_class(
            canonical_cell["judge_passed"],
            canonical_cell["judge_score"],
            threshold,
        ),
        "rationale_html": _render_rationale(
            canonical_cell["judge_rationale"], escape, Markup
        ),
        "jury_votes": _jury_votes_view(
            canonical_cell["jury_votes"],
            threshold,
            escape,
            Markup,
            label_true,
            label_false,
        ),
        "agents": [
            {**a, "response_html": Markup(a["response_html"])}
            for a in canonical_agents
        ],
        "error": canonical_cell["error"],
    }
    flipping_panel = {
        "label": "Reordered (flipping)",
        "short_hash": flipping_hash[:8],
        "ordering": flipping_ordering,
        "verdict": _verdict_label(
            flipping_cell["judge_passed"],
            flipping_cell["judge_score"],
            threshold,
            label_true,
            label_false,
        ),
        "verdict_class": _verdict_css_class(
            flipping_cell["judge_passed"],
            flipping_cell["judge_score"],
            threshold,
        ),
        "rationale_html": _render_rationale(
            flipping_cell["judge_rationale"], escape, Markup
        ),
        "jury_votes": _jury_votes_view(
            flipping_cell["jury_votes"],
            threshold,
            escape,
            Markup,
            label_true,
            label_false,
        ),
        "agents": [
            {**a, "response_html": Markup(a["response_html"])}
            for a in flipping_agents
        ],
        "error": flipping_cell["error"],
    }

    # Optional F4 attribution panel. Pass-1 RT MAJOR M6 fix: distinguish
    # "absent" (sidecar never written — render nothing, no log) from
    # "corrupt" (sidecar present but malformed — log at WARNING AND
    # surface a visible banner inside the rendered HTML so the buyer
    # knows the ablation IS computed somewhere, the renderer just
    # couldn't read it). The previous code silently swallowed the
    # AblationError at DEBUG level — the buyer would think no ablation
    # pass had been run and pay twice.
    ablation_panel = None
    ablation_corrupt_msg: str | None = None
    try:
        from topofuzz.ablation import AblationError as _AblationError
        from topofuzz.ablation import read_ablation_sidecar

        sidecar = read_ablation_sidecar(run_dir)
        ablation_panel = _ablation_view(
            sidecar,
            canonical_hash=canonical_hash,
            flipping_hash=flipping_hash,
            eval_id=eid,
            escape=escape,
            Markup=Markup,
        )
    except _AblationError as e:
        _log.warning(
            "inspect: ablation.json in %s is unreadable: %s. The panel "
            "will render a visible warning rather than render-silent.",
            run_dir.name, e,
        )
        ablation_corrupt_msg = str(e)
        ablation_panel = None
    except (ImportError, OSError) as e:
        # ImportError = packaging bug; OSError = filesystem; both are
        # rare and bubble below; but treat them as "no panel" with a
        # WARNING log so the operator sees it.
        _log.warning(
            "inspect: ablation panel unavailable (%s: %s).",
            type(e).__name__, e,
        )
        ablation_panel = None

    env = Environment(
        loader=PackageLoader("topofuzz", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tpl = env.get_template("inspect.html.j2")
    text = tpl.render(
        run_id=manifest.get("run_id", "(unknown)"),
        topology_name=manifest.get("topology_name", "(unknown)"),
        eval_id=eid,
        threshold=threshold,
        canonical=canonical_panel,
        flipping=flipping_panel,
        ablation=ablation_panel,
        ablation_corrupt_msg=ablation_corrupt_msg,
    )
    return text, eid


def _verdict_css_class(
    passed: bool | None, score: float | None, threshold: float
) -> str:
    """CSS class for the verdict pill — approve / deny / unknown.

    Falls back to score-vs-threshold when ``passed`` isn't recorded so
    a continuous-judge pill still renders in the right colour.
    """
    if passed is True:
        return "verdict-approve"
    if passed is False:
        return "verdict-deny"
    if score is not None:
        return "verdict-approve" if score >= threshold else "verdict-deny"
    return "verdict-unknown"


def write_inspect_html(
    run_dir: Path,
    *,
    eval_id: str | None = None,
    out_path: Path | None = None,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> Path:
    """Build + write the side-by-side inspect HTML. Returns the path written.

    Default destination: ``<run_dir>/inspect_<eval_id>.html``. When the
    caller doesn't override ``out_path``, ``eval_id`` is validated against
    ``_SAFE_EID_RE`` to refuse path-traversal substrings — same posture
    as :func:`topofuzz.repro.write_repro_script`.
    """
    run_dir = Path(run_dir)
    text, eid = build_inspect_html(
        run_dir, eval_id=eval_id, threshold=threshold
    )
    if out_path is None:
        if not _SAFE_EID_RE.match(eid):
            raise InspectError(
                f"eval_id {eid!r} contains characters not permitted in the "
                "default output filename "
                "(allowed: A-Z, a-z, 0-9, '.', '_', '-'). "
                "Pass --out <path> to override."
            )
        target = run_dir / f"inspect_{eid}.html"
    else:
        target = Path(out_path)
    _atomic_write_text(target, text)
    return target


def _atomic_write_text(target: Path, text: str) -> None:
    """Atomically replace ``target`` with ``text``.

    Two concurrent ``GET /runs/{id}/inspect`` requests for the same eval
    can race on the destination file: without atomicity, request A's
    ``FileResponse`` may stream the file while request B is mid-write,
    producing a half-truncated read. ``os.replace`` is atomic on every
    OS we support (including Windows, unlike ``rename``) so the reader
    always sees either the prior complete file or the new complete one,
    never a torn write (audit RT-pass-1 MAJOR-1).

    ``os.replace`` can still fail on Windows when an external process
    (antivirus, indexer) has the destination open with non-shared mode.
    We wrap that into :class:`InspectError` so callers see one canonical
    failure type with a useful hint (audit RT-pass-2 MINOR-C).
    """
    target = Path(target)
    # ``delete=False`` so ``os.replace`` is the only step that can fail;
    # we manage cleanup explicitly on the error path.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        dir=str(target.parent),
        suffix="",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        try:
            os.replace(tmp_path, target)
        except PermissionError as e:
            raise InspectError(
                f"Could not write {target.name}: {e}. On Windows this "
                "usually means another process (antivirus, file indexer) "
                "has the file open. Retry, or pass --out to a different "
                "directory."
            ) from e
    except BaseException:
        # Best-effort cleanup; suppress secondary errors so we re-raise
        # the original.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
