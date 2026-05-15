"""Step 17 (Weekend 5): natural-language summary paragraph.

Pure stats → English. No LLM call. Emits a one-paragraph plain-text
verdict that opens the report HTML, derived from two already-persisted
fields:

* ``headline.verdict_flip_rate`` — the user-readable pull-quote number
  shipped by step 15a (% of (variant, eval) pairs whose binary verdict
  differs from the original variant on the same eval). Drives the
  ``stable`` / ``unstable`` verdict word.
* ``findings.per_agent_influence`` — the §6a per-agent influence list
  shipped by step 15. Names the primary driver.

Per :mod:`topofuzz.report`'s verify-path discipline, this module imports
nothing heavier than the stdlib so it can be re-used in a verify-only
install if a future caller ever wants the same paragraph generator on
the verify side. It is currently called only from :mod:`topofuzz.report`
(which is already gated behind the ``[full]`` extras).

Per BLUEPRINT_v5plus.md line 256, the target paragraph reads like:

    Your topology is **unstable**: 50% of answers flipped when agent
    order changed (verdict_flip_rate = 0.50 [0.00, 1.00] 95% CI). The
    primary driver is ``risk_analyzer`` (influence = 0.67). Recommended
    next step: pin its position or add a verification step downstream.

Step-17 kickoff decisions (locked, do not re-litigate):

1. **Verdict source = verdict_flip_rate**, not variance_across_orderings.
   The user-facing pull-quote number is the same one the paragraph
   reports, so the verdict tier is reconstructable from the text.
2. **Two-tier verdict**: ``stable`` (< 5% flips) / ``unstable`` (≥ 5%).
   "Degraded" is a *run-health* term (judge_failure_rate), not a
   *topology-behaviour* term — using it here would conflict with the
   §A run-status taxonomy.
3. **Primary driver = agent with highest influence point estimate.**
   Degenerate rows (``influence is None``) are skipped. Negative
   influence rows are skipped too — a "primary driver" is something
   that *causes* flips, not a stabiliser. When no usable row exists
   the driver / recommendation sentence is omitted entirely.
4. **Recommended-next-step copy is hardcoded** for v0.5
   (``"pin its position or add a verification step downstream"``).
   Role-aware advice deferred to v0.6.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# Defensive ceiling on legitimate influence magnitudes. Influence is a
# probability difference per BLUEPRINT_v5plus.md line 187 — theoretically
# bounded in ``[-1, 1]``. Bootstrap noise and degenerate compute paths
# can push a few percent outside that range. We cap at 10× to leave
# generous slack while bounding the rendered numeric format width
# (audit pass-1 RT-MAJOR-3: ``{1e308:.2f}`` formats to ~310 chars and
# would bloat the HTML). Anything above this is rejected as
# numerically pathological.
_INFLUENCE_MAGNITUDE_CEILING: float = 10.0


# >= this binary-verdict-flip rate triggers the "unstable" tier.
# 5% = "one in twenty answers changed when the topology was permuted"
# — a defensible bar for "real instability detected" that's an order
# of magnitude tighter than the 31% headline from Bajaj 2026 §3.2.
FLIP_RATE_UNSTABLE_THRESHOLD: float = 0.05

# Hardcoded v0.5 recommendation. Role-aware advice deferred to v0.6
# (see module docstring decision #4).
DEFAULT_RECOMMENDATION: str = (
    "pin its position or add a verification step downstream"
)


@dataclass(frozen=True)
class SummaryContext:
    """Inputs + rendered text for the natural-language summary partial.

    Frozen so it round-trips cleanly into the Jinja context and any
    test assertion that compares two renders for equality. Mirrors the
    :class:`topofuzz.metrics.worstcase.WorstCasePanel` pattern.

    ``verdict_word`` is one of ``"stable"``, ``"unstable"``,
    ``"unavailable"`` (no flip-rate metric in the source). The HTML
    template branches on this directly.

    ``summary_text`` carries the plain-text paragraph (used for unit
    tests and the JSON-mirror surface). ``summary_html`` carries the
    same content with ``**word**`` -> ``<strong>word</strong>`` and
    `` `word` `` -> ``<code>word</code>`` AND every other character
    HTML-escaped via :func:`html.escape` — so it is safe to mark
    ``| safe`` in the Jinja template.
    """

    verdict_word: str
    verdict_flip_rate: float | None
    flip_rate_ci_lower: float | None
    flip_rate_ci_upper: float | None
    wilson_ci_lower: float | None
    wilson_ci_upper: float | None
    wilson_conf_level: float | None
    primary_driver: str | None
    primary_influence: float | None
    summary_text: str
    summary_html: str


def render_summary(
    verdict_flip_rate: Mapping[str, Any] | None,
    per_agent_influence: Sequence[Mapping[str, Any]] | None,
) -> SummaryContext:
    """Build the natural-language summary from headline + findings dicts.

    Args:
        verdict_flip_rate: ``headline.verdict_flip_rate`` from
            ``report.json``. Shape: ``{point, ci_lower, ci_upper,
            degenerate_flag, numerator, denominator, wilson_ci_lower,
            wilson_ci_upper}`` — but any missing key, ``None`` value, or
            non-numeric value degrades gracefully to "unavailable".
            ``None`` (the whole dict missing) is the pre-step-15a run
            case and is also handled. The wilson_* keys are pre-step-F1
            null on older runs; the rendered paragraph drops the Wilson
            sentence cleanly when missing.
        per_agent_influence: ``findings.per_agent_influence`` list. Each
            row carries ``{node_id, influence, ci_lower, ci_upper,
            degenerate, degenerate_reason}``. Degenerate rows and rows
            with ``influence is None`` or ≤ 0 are filtered out.

    Returns:
        :class:`SummaryContext` carrying both structured fields (for
        downstream JSON / testability) and the fully rendered
        ``summary_text`` paragraph. Never raises.
    """
    flip_point = _coerce_float(verdict_flip_rate, "point")
    flip_lower = _coerce_float(verdict_flip_rate, "ci_lower")
    flip_upper = _coerce_float(verdict_flip_rate, "ci_upper")
    wilson_lower = _coerce_float(verdict_flip_rate, "wilson_ci_lower")
    wilson_upper = _coerce_float(verdict_flip_rate, "wilson_ci_upper")
    wilson_conf = _coerce_float(verdict_flip_rate, "wilson_conf_level")
    primary_node, primary_inf = _pick_primary_driver(per_agent_influence)

    if flip_point is None:
        verdict = "unavailable"
    elif flip_point < FLIP_RATE_UNSTABLE_THRESHOLD:
        verdict = "stable"
    else:
        verdict = "unstable"

    text = _render_paragraph(
        verdict=verdict,
        flip_point=flip_point,
        flip_lower=flip_lower,
        flip_upper=flip_upper,
        wilson_lower=wilson_lower,
        wilson_upper=wilson_upper,
        wilson_conf=wilson_conf,
        primary_node=primary_node,
        primary_inf=primary_inf,
    )

    return SummaryContext(
        verdict_word=verdict,
        verdict_flip_rate=flip_point,
        flip_rate_ci_lower=flip_lower,
        flip_rate_ci_upper=flip_upper,
        wilson_ci_lower=wilson_lower,
        wilson_ci_upper=wilson_upper,
        wilson_conf_level=wilson_conf,
        primary_driver=primary_node,
        primary_influence=primary_inf,
        summary_text=text,
        summary_html=_text_to_html(text),
    )


def _coerce_float(d: Mapping[str, Any] | None, key: str) -> float | None:
    """Return ``d[key]`` as a *finite* ``float`` or ``None`` on any failure.

    Treats missing dict, non-Mapping, missing key, ``None`` value,
    non-numeric value, AND non-finite floats (NaN / +inf / -inf) all as
    ``None``. The non-finite check is audit pass-1 RT-BLOCKER-1:
    ``numpy`` and ``json.dumps(allow_nan=True)`` (the stdlib default)
    happily round-trip ``NaN`` through ``report.json``, and a downstream
    ``round(NaN * 100)`` would raise ``ValueError`` mid-render.

    F1 audit pass-1: explicitly reject ``bool`` (``True``/``False`` are
    ``isinstance(int)`` and ``float(True) == 1.0`` silently), and reject
    ``str`` so a hand-edited / mis-typed ``"0.20"`` does NOT silently
    coerce — it must match the UI helper's strict-type contract so the
    same ``report.json`` does not render two different ways across the
    §17 paragraph and the UI subtitle.
    """
    if not isinstance(d, Mapping):
        return None
    v = d.get(key)
    if v is None or isinstance(v, (bool, str)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _pick_primary_driver(
    rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[str | None, float | None]:
    """Return ``(node_id, influence)`` of the strongest *driver* agent.

    "Driver" = strictly positive *finite* influence (flips the answer
    when its slot changes). Negative-influence rows (stabilisers),
    degenerate rows, and non-finite values are all excluded:

    * Negative is excluded by design — a "primary driver" sentence
      would be nonsensical for a stabilising agent.
    * Non-finite (NaN / Inf) is excluded because ``NaN <= 0.0`` is
      ``False`` in IEEE-754, so a NaN would slip through a plain
      ``> 0`` filter (audit pass-1 RT-MAJOR-2). numpy can legitimately
      emit NaN here when the bootstrap-CI compute path degenerates,
      and step 15a's ``report.json`` mirror persists NaN as JSON
      ``NaN`` (numpy default).
    * Values whose magnitude exceeds
      :data:`_INFLUENCE_MAGNITUDE_CEILING` are rejected as numerically
      pathological (audit pass-1 RT-MAJOR-3 — bounds the rendered
      numeric format width).

    Tiebreak on equal influence is lex-asc on ``node_id`` for
    determinism (so two runs with the same matrix render the same
    text). Returns ``(None, None)`` when no usable row exists.
    """
    if not rows:
        return None, None

    usable: list[tuple[str, float]] = []
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        if r.get("degenerate"):
            continue
        node = r.get("node_id")
        inf = r.get("influence")
        if not isinstance(node, str) or inf is None:
            continue
        try:
            inf_f = float(inf)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(inf_f):
            continue
        if inf_f <= 0.0:
            continue
        if abs(inf_f) > _INFLUENCE_MAGNITUDE_CEILING:
            continue
        usable.append((node, inf_f))

    if not usable:
        return None, None
    # Sort by influence DESC, then node_id ASC, take the top.
    usable.sort(key=lambda t: (-t[1], t[0]))
    return usable[0]


def _render_paragraph(
    *,
    verdict: str,
    flip_point: float | None,
    flip_lower: float | None,
    flip_upper: float | None,
    wilson_lower: float | None,
    wilson_upper: float | None,
    wilson_conf: float | None,
    primary_node: str | None,
    primary_inf: float | None,
) -> str:
    """Format the one-paragraph plaintext summary.

    Output is plain text (not HTML); the template wraps it in a styled
    ``<div>`` and renders Markdown-style ``**word**`` emphasis via a
    minimal pass. We keep this pure text so unit tests can assert on
    the strings directly without an HTML parser.
    """
    if verdict == "unavailable":
        return (
            "Topology stability could not be assessed for this run "
            "(verdict_flip_rate unavailable — typically a single-variant "
            "fixture, no continuous-judge cells, or a pre-step-15a run)."
        )

    flip_pct = _format_percent(flip_point)
    flip_quote = _format_flip_quote(flip_point, flip_lower, flip_upper)
    wilson_clause = _format_wilson_clause(
        wilson_lower, wilson_upper, conf=wilson_conf,
    )

    if verdict == "stable":
        head = (
            f"Your topology is **stable**: only {flip_pct} of answers "
            f"flipped when agent order changed ({flip_quote})."
        )
        if primary_node is not None and primary_inf is not None:
            tail = (
                f" The highest-influence agent was `{primary_node}` "
                f"(influence = {primary_inf:.2f}), but no agent dominated "
                f"the outcome. No action recommended."
            )
        else:
            tail = " No action recommended."
        return head + wilson_clause + tail

    # verdict == "unstable"
    head = (
        f"Your topology is **unstable**: {flip_pct} of answers flipped "
        f"when agent order changed ({flip_quote})."
    )
    if primary_node is not None and primary_inf is not None:
        tail = (
            f" The primary driver is `{primary_node}` "
            f"(influence = {primary_inf:.2f}). Recommended next step: "
            f"{DEFAULT_RECOMMENDATION}."
        )
    else:
        tail = (
            " No single agent could be identified as the primary driver "
            "(per-agent influence is unavailable or degenerate). "
            "Investigate variance per eval row to localise the instability."
        )
    return head + wilson_clause + tail


def _format_percent(p: float | None) -> str:
    """Format a [0, 1] rate as an integer percent like ``"31%"``.

    None-safe: returns ``"?%"`` rather than crashing. Caller guarantees
    non-None for the stable/unstable paths, so the ``"?%"`` branch is
    defence in depth only.
    """
    if p is None:
        return "?%"
    return f"{round(p * 100)}%"


def _format_flip_quote(
    point: float | None,
    lower: float | None,
    upper: float | None,
) -> str:
    """Format ``verdict_flip_rate = 0.50 [0.00, 1.00] 95% CI``.

    Collapses to ``verdict_flip_rate = 0.50`` if either CI bound is
    missing (degenerate-but-point-known case).
    """
    if point is None:
        return "verdict_flip_rate unavailable"
    if lower is None or upper is None:
        return f"verdict_flip_rate = {point:.2f}"
    return (
        f"verdict_flip_rate = {point:.2f} "
        f"[{lower:.2f}, {upper:.2f}] 95% CI"
    )


def _format_wilson_clause(
    lower: float | None,
    upper: float | None,
    *,
    conf: float | None = None,
) -> str:
    """Render the Wilson CI clause appended to the head sentence.

    Returns ``" Statistical confidence: 95% confident the true rate is
    between L% and H% (Wilson interval on the paired-cell count)."``
    when both bounds are present, valid, and ordered, else ``""``. The
    leading space is intentional — the caller concatenates without an
    extra separator.

    Defensive gates (audit pass-1 RT-MAJOR-2 / Fuzzer-MAJOR-1 /
    Fuzzer-MINOR-1/2):

    * Two-bound gate — showing only one bound would imply a one-sided
      CI we did not compute.
    * Range gate — both bounds must be in ``[0, 1]``; pre-F1 producers
      that emit log-odds or unclamped raw arithmetic must NOT render as
      "between -10% and 170%".
    * Monotonicity gate — ``lower <= upper``; a corrupted dict that
      swaps the keys must NOT render "between 50% and 30%".
    * Finite gate — NaN/Inf already strip via ``_coerce_float`` upstream,
      but defending here too costs one line.

    ``conf`` is the confidence level actually used to compute the
    bounds (persisted as ``headline.verdict_flip_rate.wilson_conf_level``).
    When ``None`` (pre-F1 runs) we default the rendered percent to 95%
    — the only value the legacy code path could have produced.
    """
    if lower is None or upper is None:
        return ""
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return ""
    if not (0.0 <= lower <= upper <= 1.0):
        return ""
    conf_pct = _format_conf_pct(conf)
    return (
        f" Statistical confidence: {conf_pct} confident the true rate "
        f"is between {_format_percent(lower)} and {_format_percent(upper)} "
        f"(Wilson interval on the paired-cell count)."
    )


def _format_conf_pct(conf: float | None) -> str:
    """Render the conf level as ``"95%"``. Falls back to ``"95%"`` when
    ``conf`` is missing or out of ``(0, 1)`` — every in-tree caller of
    :func:`wilson_score_interval` uses the 0.95 default, so a missing
    ``wilson_conf_level`` (pre-F1 dict / corrupted JSON) is the only
    realistic miss-path."""
    if conf is None or not isinstance(conf, (int, float)) or isinstance(conf, bool):
        return "95%"
    cf = float(conf)
    if not math.isfinite(cf) or not (0.0 < cf < 1.0):
        return "95%"
    return f"{round(cf * 100)}%"


# Strict pair-matched alternation: must be **non-greedy-non-empty** and
# `non-greedy-non-empty` ; greedy `.*` would let `**a** b **c**` collapse
# into one giant strong span. The pattern explicitly forbids embedded
# `**` / `` ` `` inside the captured group so adjacent runs stay separate.
_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+?)`")


def _text_to_html(text: str) -> str:
    """Render the plain-text paragraph as escaped HTML with bold + code.

    The contract: every byte not inside a matched ``**word**`` or
    `` `word` `` pair is HTML-escaped via :func:`html.escape`. The
    bold / code spans are themselves emitted with escaped inner
    content (``<strong>`` and ``<code>`` are the only structural tags).
    No other HTML is permitted. This is what lets the Jinja template
    use ``| safe`` without bypassing autoescape on user-derived strings
    (``primary_node`` flows in from manifest data).

    Implementation walks the input left-to-right alternating between
    "literal" stretches (escaped) and matched pair regions, so the
    output is XSS-safe even on adversarial inputs like
    ``"</div><script>"`` in a node_id (defence in depth — node_ids
    are restricted by the IR but the boundary is loadbearing).
    """
    out: list[str] = []
    pos = 0
    # Combine the two patterns into a single alternation so left-to-
    # right ordering is preserved.
    combined = re.compile(r"\*\*([^*]+?)\*\*|`([^`]+?)`")
    for m in combined.finditer(text):
        if m.start() > pos:
            out.append(html.escape(text[pos:m.start()]))
        bold_g, code_g = m.group(1), m.group(2)
        if bold_g is not None:
            out.append(f"<strong>{html.escape(bold_g)}</strong>")
        else:
            out.append(f"<code>{html.escape(code_g)}</code>")
        pos = m.end()
    if pos < len(text):
        out.append(html.escape(text[pos:]))
    return "".join(out)
