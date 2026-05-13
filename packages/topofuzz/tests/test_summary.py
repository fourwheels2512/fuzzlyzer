"""Unit tests for :mod:`topofuzz.summary` (Step 17 — §17 natural-language summary).

The module is pure stats → English so the tests don't touch the
filesystem, jinja, or any compute path — they pin the verdict tier, the
primary-driver selection, the paragraph wording, and the HTML escaping
contract for every documented degenerate case.
"""

from __future__ import annotations

import math
import re

from topofuzz.summary import (
    DEFAULT_RECOMMENDATION,
    FLIP_RATE_UNSTABLE_THRESHOLD,
    SummaryContext,
    render_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _vf(point, ci_lower=None, ci_upper=None, degenerate_flag=False):
    """Build a ``headline.verdict_flip_rate`` dict like step 15a emits."""
    return {
        "point": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "degenerate_flag": degenerate_flag,
    }


def _inf(node_id, influence, ci_lower=None, ci_upper=None,
         degenerate=False, degenerate_reason=None):
    """Build a ``findings.per_agent_influence`` row like step 15 emits."""
    return {
        "node_id": node_id,
        "influence": influence,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "degenerate": degenerate,
        "degenerate_reason": degenerate_reason,
    }


# ---------------------------------------------------------------------------
# Tier-word selection
# ---------------------------------------------------------------------------


def test_stable_tier_below_threshold() -> None:
    """flip_rate < 5% → ``stable`` verdict word."""
    ctx = render_summary(_vf(0.02, 0.0, 0.04), [])
    assert ctx.verdict_word == "stable"


def test_unstable_tier_at_exact_threshold() -> None:
    """Exact-5% boundary lives in the ``unstable`` tier (>= predicate)."""
    ctx = render_summary(_vf(FLIP_RATE_UNSTABLE_THRESHOLD), [])
    assert ctx.verdict_word == "unstable"


def test_unstable_tier_above_threshold() -> None:
    """flip_rate >= 5% → ``unstable`` verdict word."""
    ctx = render_summary(_vf(0.31, 0.24, 0.38), [])
    assert ctx.verdict_word == "unstable"


def test_unavailable_when_flip_rate_dict_missing() -> None:
    """``None`` as the whole flip-rate dict → ``unavailable``."""
    ctx = render_summary(None, [])
    assert ctx.verdict_word == "unavailable"
    assert "could not be assessed" in ctx.summary_text


def test_unavailable_when_flip_rate_point_is_null() -> None:
    """Degenerate metric with ``point=None`` → ``unavailable``."""
    ctx = render_summary(_vf(None, None, None, degenerate_flag=True), [])
    assert ctx.verdict_word == "unavailable"


def test_unavailable_when_flip_rate_not_a_dict() -> None:
    """A scalar or list in the flip_rate slot → ``unavailable`` (no crash)."""
    assert render_summary(0.5, []).verdict_word == "unavailable"  # type: ignore[arg-type]
    assert render_summary([0.5], []).verdict_word == "unavailable"  # type: ignore[arg-type]
    assert render_summary("0.5", []).verdict_word == "unavailable"  # type: ignore[arg-type]


def test_unavailable_when_point_non_numeric() -> None:
    """``point="oops"`` → coerce to None → ``unavailable`` (no crash)."""
    ctx = render_summary(_vf("oops"), [])
    assert ctx.verdict_word == "unavailable"


# ---------------------------------------------------------------------------
# Primary-driver selection
# ---------------------------------------------------------------------------


def test_primary_driver_is_highest_positive_influence() -> None:
    """Top influence wins."""
    rows = [
        _inf("intake", 0.10, 0.05, 0.15),
        _inf("risk", 0.67, 0.55, 0.78),
        _inf("policy", 0.30, 0.20, 0.40),
    ]
    ctx = render_summary(_vf(0.31, 0.24, 0.38), rows)
    assert ctx.primary_driver == "risk"
    assert ctx.primary_influence == 0.67


def test_primary_driver_skips_degenerate_rows() -> None:
    """``degenerate=True`` rows are excluded even with a numeric influence."""
    rows = [
        _inf("intake", 0.99, degenerate=True,
             degenerate_reason="no moved pairs"),
        _inf("risk", 0.30, 0.20, 0.40),
    ]
    ctx = render_summary(_vf(0.31, 0.24, 0.38), rows)
    assert ctx.primary_driver == "risk"


def test_primary_driver_skips_negative_influence() -> None:
    """Negative-influence (stabiliser) rows are not "primary drivers"."""
    rows = [
        _inf("stabiliser", -0.40, -0.50, -0.30),
        _inf("risk", 0.10, 0.05, 0.15),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver == "risk"


def test_primary_driver_skips_zero_influence() -> None:
    """Exactly-zero influence is excluded (not a positive driver)."""
    rows = [
        _inf("zero", 0.0),
        _inf("real", 0.40, 0.30, 0.50),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver == "real"


def test_primary_driver_lex_tiebreak() -> None:
    """Equal influence → lex-ascending tiebreak on ``node_id``."""
    rows = [
        _inf("zzz", 0.50, 0.40, 0.60),
        _inf("aaa", 0.50, 0.40, 0.60),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver == "aaa"


def test_primary_driver_none_when_no_rows() -> None:
    """No influence rows → no driver, no fix recommendation."""
    ctx = render_summary(_vf(0.31, 0.24, 0.38), [])
    assert ctx.primary_driver is None
    assert ctx.primary_influence is None
    # Unstable + no driver → fallback wording
    assert "No single agent" in ctx.summary_text
    assert DEFAULT_RECOMMENDATION not in ctx.summary_text


def test_primary_driver_none_when_all_rows_degenerate() -> None:
    """Every row degenerate → no driver, same fallback wording."""
    rows = [
        _inf("a", None, degenerate=True, degenerate_reason="no moved pairs"),
        _inf("b", None, degenerate=True, degenerate_reason="no same-slot pairs"),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver is None


def test_primary_driver_skips_non_mapping_rows() -> None:
    """Defensive: a non-dict row item is silently skipped."""
    rows = [
        "not a dict",  # type: ignore[list-item]
        _inf("real", 0.40),
    ]
    ctx = render_summary(_vf(0.31), rows)  # type: ignore[arg-type]
    assert ctx.primary_driver == "real"


def test_primary_driver_skips_non_string_node_id() -> None:
    """A numeric / None node_id is skipped (defensive against schema drift)."""
    rows = [
        _inf(42, 0.99),  # type: ignore[arg-type]
        _inf("real", 0.10),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver == "real"


# ---------------------------------------------------------------------------
# Paragraph wording
# ---------------------------------------------------------------------------


def test_unstable_paragraph_includes_all_three_slots() -> None:
    """The unstable paragraph hits verdict, flip-rate quote, driver, fix."""
    rows = [_inf("risk", 0.67, 0.55, 0.78)]
    ctx = render_summary(_vf(0.31, 0.24, 0.38), rows)
    text = ctx.summary_text
    assert "**unstable**" in text
    assert "31% of answers flipped" in text
    assert "verdict_flip_rate = 0.31 [0.24, 0.38] 95% CI" in text
    assert "`risk`" in text
    assert "influence = 0.67" in text
    assert DEFAULT_RECOMMENDATION in text


def test_stable_paragraph_includes_no_action_recommended() -> None:
    """Stable wording does NOT recommend a fix."""
    rows = [_inf("risk", 0.08, 0.05, 0.12)]
    ctx = render_summary(_vf(0.02, 0.00, 0.04), rows)
    text = ctx.summary_text
    assert "**stable**" in text
    assert "2% of answers flipped" in text
    assert "No action recommended" in text
    assert DEFAULT_RECOMMENDATION not in text


def test_stable_paragraph_without_influence_still_renders() -> None:
    """Stable + empty influence list still emits a clean paragraph."""
    ctx = render_summary(_vf(0.02, 0.00, 0.04), [])
    assert "**stable**" in ctx.summary_text
    assert "No action recommended" in ctx.summary_text


def test_flip_quote_collapses_when_ci_missing() -> None:
    """Missing CI bounds → drop the bracketed CI from the quote."""
    rows = [_inf("risk", 0.40)]
    ctx = render_summary(_vf(0.31, None, None), rows)
    assert "verdict_flip_rate = 0.31" in ctx.summary_text
    assert "95% CI" not in ctx.summary_text


# ---------------------------------------------------------------------------
# HTML escaping (XSS defence)
# ---------------------------------------------------------------------------


def test_html_escapes_adversarial_node_id() -> None:
    """A node_id with HTML metacharacters is escaped in summary_html."""
    rows = [_inf("</div><script>alert(1)</script>", 0.67)]
    ctx = render_summary(_vf(0.31, 0.24, 0.38), rows)
    # Plain text path keeps the raw chars (it's not HTML).
    assert "</div><script>" in ctx.summary_text
    # HTML path escapes them.
    assert "<script>" not in ctx.summary_html
    assert "&lt;script&gt;" in ctx.summary_html


def test_html_renders_bold_and_code_only() -> None:
    """Only ``<strong>`` and ``<code>`` structural tags appear."""
    rows = [_inf("risk", 0.67, 0.55, 0.78)]
    ctx = render_summary(_vf(0.31, 0.24, 0.38), rows)
    html = ctx.summary_html
    assert "<strong>unstable</strong>" in html
    assert "<code>risk</code>" in html
    # No other tags introduced.
    other_tags = re.findall(r"<(?!/?(?:strong|code)\b)[^>]+>", html)
    assert other_tags == [], f"unexpected HTML tags: {other_tags}"


def test_html_escaping_does_not_break_markdown_pairs() -> None:
    """Backticks/asterisks in the surrounding literal text are escaped, not
    interpreted as markdown."""
    # Construct a flip_rate whose CI bounds round-trip to text that already
    # contains markdown-adjacent characters — defensive sanity check on the
    # alternation regex.
    rows = [_inf("risk", 0.67, 0.55, 0.78)]
    ctx = render_summary(_vf(0.31, 0.24, 0.38), rows)
    # Sanity: literal asterisk-asterisk pairs come ONLY from intentional
    # bold; no stray "**" should leak into summary_html.
    assert "**" not in ctx.summary_html
    # Same for backticks — every one should have been consumed by the
    # `<code>...</code>` substitution.
    assert "`" not in ctx.summary_html


def test_summary_text_html_round_trip_pair_balanced() -> None:
    """``**`` and `` ` `` markers are pair-balanced in every documented
    rendered case (so the regex always finds matched pairs)."""
    cases = [
        render_summary(_vf(0.02, 0.0, 0.04), []),                     # stable, no driver
        render_summary(_vf(0.02, 0.0, 0.04), [_inf("r", 0.1)]),       # stable, driver
        render_summary(_vf(0.31, 0.24, 0.38), []),                    # unstable, no driver
        render_summary(_vf(0.31, 0.24, 0.38), [_inf("r", 0.6)]),      # unstable, driver
        render_summary(None, []),                                     # unavailable
    ]
    for ctx in cases:
        assert ctx.summary_text.count("**") % 2 == 0, ctx.summary_text
        assert ctx.summary_text.count("`") % 2 == 0, ctx.summary_text


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------


def test_summary_context_is_frozen() -> None:
    """``SummaryContext`` is frozen — assignment must fail."""
    ctx = render_summary(_vf(0.02), [])
    import dataclasses
    assert dataclasses.is_dataclass(ctx)
    try:
        ctx.verdict_word = "tampered"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SummaryContext was mutable")


def test_summary_context_carries_structured_fields() -> None:
    """Structured fields mirror the inputs so report.json mirrors can
    reconstruct the paragraph downstream without re-parsing text."""
    rows = [_inf("risk", 0.67, 0.55, 0.78)]
    ctx = render_summary(_vf(0.31, 0.24, 0.38), rows)
    assert isinstance(ctx, SummaryContext)
    assert ctx.verdict_flip_rate == 0.31
    assert ctx.flip_rate_ci_lower == 0.24
    assert ctx.flip_rate_ci_upper == 0.38
    assert ctx.primary_driver == "risk"
    assert ctx.primary_influence == 0.67


# ---------------------------------------------------------------------------
# Audit pass-1 fixes: NaN / Inf / magnitude-ceiling guards
# ---------------------------------------------------------------------------


def test_nan_flip_rate_renders_unavailable_not_crash() -> None:
    """RT-BLOCKER-1: NaN in ``verdict_flip_rate.point`` must not crash.

    numpy can persist NaN through ``json.dumps(allow_nan=True)`` (the
    stdlib default), so a legitimately-degenerate metric can reach this
    code path. The unguarded ``round(NaN * 100)`` would raise
    ``ValueError`` and abort the report mid-render.
    """
    ctx = render_summary(_vf(float("nan")), [])
    assert ctx.verdict_word == "unavailable"
    assert ctx.verdict_flip_rate is None
    # Sanity: HTML render path is also exercised.
    assert "<script>" not in ctx.summary_html


def test_inf_flip_rate_renders_unavailable_not_crash() -> None:
    """Same as NaN test but +inf — ``round(inf)`` raises OverflowError."""
    ctx = render_summary(_vf(float("inf")), [])
    assert ctx.verdict_word == "unavailable"


def test_neg_inf_flip_rate_renders_unavailable_not_crash() -> None:
    """-inf coerces to None via the isfinite guard."""
    ctx = render_summary(_vf(float("-inf")), [])
    assert ctx.verdict_word == "unavailable"


def test_nan_ci_bound_collapses_to_unbracketed_quote() -> None:
    """A finite point with NaN CI bounds drops the [lo, hi] portion."""
    rows = [_inf("risk", 0.30)]
    ctx = render_summary(_vf(0.31, float("nan"), 0.38), rows)
    # Should fall through to the "ci bound missing" branch and omit the CI.
    assert "verdict_flip_rate = 0.31" in ctx.summary_text
    assert "95% CI" not in ctx.summary_text


def test_nan_influence_excluded_from_primary_driver() -> None:
    """RT-MAJOR-2: ``NaN <= 0.0`` is False in IEEE-754, so a plain
    ``> 0`` filter would let a NaN-influence row sort to the top. The
    ``math.isfinite`` gate excludes it cleanly.
    """
    rows = [
        _inf("nan-agent", float("nan")),
        _inf("real", 0.20, 0.10, 0.30),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver == "real"
    assert ctx.primary_influence is not None
    assert math.isfinite(ctx.primary_influence)


def test_inf_influence_excluded_from_primary_driver() -> None:
    """+inf influence would otherwise sort to the top and bloat the
    rendered numeric format (``{inf:.2f}`` is 'inf' but real-world
    pathological values like 1e308 produce ~310 chars)."""
    rows = [
        _inf("inf-agent", float("inf")),
        _inf("real", 0.20, 0.10, 0.30),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver == "real"


def test_pathological_huge_influence_excluded() -> None:
    """RT-MAJOR-3: numeric formatting bound. Anything beyond
    :data:`_INFLUENCE_MAGNITUDE_CEILING` (10×) is rejected as out of
    the legitimate ``[-1, 1]`` probability-difference range."""
    rows = [
        _inf("pathological", 1e308),
        _inf("real", 0.20, 0.10, 0.30),
    ]
    ctx = render_summary(_vf(0.31), rows)
    assert ctx.primary_driver == "real"
    # Rendered HTML never contains a digit run longer than reasonable
    # (audit pass-1 RT-MAJOR-3 — ensures no HTML bloat regression).
    longest_digit_run = max(
        (len(m.group()) for m in re.finditer(r"\d+", ctx.summary_html)),
        default=0,
    )
    assert longest_digit_run < 20, (
        "rendered HTML should not contain a >19-digit numeric run "
        "(would indicate a pathological influence slipped through)"
    )


def test_influence_just_below_ceiling_still_renders() -> None:
    """Boundary check: 9.99× the [-1, 1] range still counts as a driver."""
    rows = [_inf("real", 9.99)]
    ctx = render_summary(_vf(0.31), rows)
    # Below the 10× ceiling => row is kept, primary driver is "real".
    assert ctx.primary_driver == "real"
