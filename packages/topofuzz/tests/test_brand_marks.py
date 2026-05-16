# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for common-law trademark `™` notices on user-facing surfaces.

BACKLOG entry "MINOR — Common-law trademark `™` notice in user-facing
strings" (2026-05-16) — regression-protects the `™` mark on the
FIRST mention per page, per spec lines 1577-1578.

Surfaces:
- README.md H1 (per-package; root README is pramana-eval, skipped)
- Typer app `help=` text (CLI app description)
- report.html.j2 H1 (visible page header)
- inspect.html.j2 `<title>` (first brand mention; H1 is non-brand)
- ceo_card.html.j2 eyebrow div (first brand mention; H1 is non-brand)
- pyproject.toml description fields

Boundary the test enforces:
- ™ appears on the first mention per page.
- Code identifiers + repeated mentions on the same page are NOT
  required to have ™ (and shouldn't, per dilution concerns).
- Files are UTF-8 (™ is U+2122).
"""

from __future__ import annotations

import io
import re
import tokenize
import tomllib
from pathlib import Path

import pytest

from topofuzz.cli import app


# U+2122 TRADEMARK SIGN. Centralised so tests fail with a useful diff
# rather than mysterious whitespace mismatches.
TM = "™"


def _repo_root() -> Path:
    """Find the repo root by walking up looking for the ``.git``
    directory. Plain ``LICENSE`` exists at every package level, so
    it's a poor sentinel."""
    here = Path(__file__).resolve()
    candidate = here
    for _ in range(8):
        if (candidate / ".git").exists():
            return candidate
        candidate = candidate.parent
    pytest.fail(f"could not find .git root from {here}")


def _read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# README brand marks
# ---------------------------------------------------------------------------


def test_topofuzz_readme_h1_has_trademark() -> None:
    """`packages/topofuzz/README.md` H1 must read `# topofuzz™` —
    the brand's first user-visible mention."""
    readme = _read_utf8(_repo_root() / "packages" / "topofuzz" / "README.md")
    first_line = readme.split("\n", 1)[0]
    assert first_line == f"# topofuzz{TM}", (
        f"topofuzz README H1 missing ™ mark — got {first_line!r}"
    )


def test_fuzzeval_readme_h1_has_trademark() -> None:
    readme = _read_utf8(_repo_root() / "packages" / "fuzzeval" / "README.md")
    first_line = readme.split("\n", 1)[0]
    assert first_line == f"# fuzzeval{TM}", (
        f"fuzzeval README H1 missing ™ mark — got {first_line!r}"
    )


def test_root_readme_is_pramana_eval_not_topofuzz() -> None:
    """Root README is the pramana-eval surface, NOT topofuzz —
    intentionally skipped from the trademark sweep. Locking this in
    so a future contributor doesn't 'helpfully' add a topofuzz ™ to
    the root README.md."""
    root_readme = _read_utf8(_repo_root() / "README.md")
    first_line = root_readme.split("\n", 1)[0]
    assert first_line == "# pramana-eval", (
        f"root README is no longer pramana-eval (now {first_line!r}) — "
        "if topofuzz/fuzzeval moved to the root, add a ™-on-H1 test for "
        "the new root H1 and update this assertion."
    )


# ---------------------------------------------------------------------------
# CLI app help text
# ---------------------------------------------------------------------------


def test_typer_app_help_contains_trademark() -> None:
    """The Typer app's `help=` string surfaces in `topofuzz --help`.
    First mention of the brand must carry ™."""
    help_text = app.info.help or ""
    assert TM in help_text, (
        f"Typer app help text missing ™ — got {help_text!r}"
    )
    assert f"topofuzz{TM}" in help_text, (
        f"expected 'topofuzz™' adjacent in help text, got {help_text!r}"
    )


# ---------------------------------------------------------------------------
# Template surfaces
# ---------------------------------------------------------------------------


def _templates_dir() -> Path:
    return (
        _repo_root()
        / "packages"
        / "topofuzz"
        / "src"
        / "topofuzz"
        / "templates"
    )


# All HTML-equivalent forms of U+2122. ``&trade;`` is the named
# entity; ``&#8482;`` / ``&#x2122;`` are the decimal / hex numeric
# entities. Pass-2 Mi1: future content pasted from an HTML editor
# may use the numeric forms; the test must count them too or the
# dilution invariant silently weakens. Literal first so equality
# checks match the canonical (pass-1) form.
_TM_FORMS: tuple[str, ...] = (TM, "&trade;", "&#8482;", "&#x2122;")


def _has_brand_mark(src: str, brand: str = "topofuzz") -> bool:
    """Return True if ``src`` contains a ``<brand>`` immediately
    followed by any U+2122 encoding (literal, named entity, decimal
    or hex numeric entity). Pass-1 standardized templates to literal
    U+2122; the entity branches are forward-compat for
    third-party-edited templates and a future entity-encoding
    convention."""
    return any(f"{brand}{form}" in src for form in _TM_FORMS)


def test_report_html_first_mention_has_trademark() -> None:
    """report.html.j2's first user-visible brand mention (the H1 in
    the visible page header) must carry ™. We match via a permissive
    regex so cosmetic H1 restyles (added classes, em-dashes, wrapping)
    don't break the trademark guard, but stay anchored to the H1 so
    a stray ``topofuzz™`` elsewhere on the page can't satisfy the
    H1 contract."""
    src = _read_utf8(_templates_dir() / "report.html.j2")
    tm_alt = "(?:" + "|".join(re.escape(f) for f in _TM_FORMS) + ")"
    pattern = re.compile(
        r"<h1[^>]*>[^<]*?topofuzz" + tm_alt,
        re.IGNORECASE,
    )
    assert pattern.search(src), (
        "report.html.j2 H1 missing ™ on the first brand mention. "
        "Pattern: <h1 ...>...topofuzz™|&trade;|&#8482;|&#x2122;..."
    )


def test_inspect_html_title_has_trademark() -> None:
    """inspect.html.j2 first brand mention is the ``<title>`` (the
    H1 is the non-brand prose "Where did the AIs diverge?")."""
    src = _read_utf8(_templates_dir() / "inspect.html.j2")
    assert _has_brand_mark(src), (
        "inspect.html.j2 first brand mention missing ™"
    )


def test_ceo_card_html_first_mention_has_trademark() -> None:
    """ceo_card.html.j2 eyebrow div is the first body brand mention
    (H1 is the dynamic ``{{ headline_title }}``)."""
    src = _read_utf8(_templates_dir() / "ceo_card.html.j2")
    assert _has_brand_mark(src)


# ---------------------------------------------------------------------------
# UI templates (`topofuzz ui`) — base.html nav brand link
# ---------------------------------------------------------------------------


def _ui_templates_dir() -> Path:
    return (
        _repo_root()
        / "packages"
        / "topofuzz"
        / "src"
        / "topofuzz"
        / "ui"
        / "templates"
    )


def test_ui_base_template_brand_link_has_trademark() -> None:
    """``topofuzz ui`` renders every page through ``base.html`` —
    the nav-bar brand link is the most user-visible first-mention
    surface across the entire web UI. Marking it once here covers
    every child template (setup / live / results / history / etc.).

    Pass-2 M1 fix: regex is permissive — matches ``class="brand"``,
    ``class="brand large"``, ``class="large brand"``, single-quoted
    attribute values, and any attribute order before/after ``class``.
    """
    src = _read_utf8(_ui_templates_dir() / "base.html")
    tm_alt = "(?:" + "|".join(re.escape(f) for f in _TM_FORMS) + ")"
    pattern = re.compile(
        r'<a\b[^>]*class=["\'][^"\']*\bbrand\b[^"\']*["\']'
        r"[^>]*>\s*topofuzz" + tm_alt,
        re.IGNORECASE,
    )
    assert pattern.search(src), (
        "ui/templates/base.html nav `brand` link missing ™. This "
        "link appears on EVERY web-UI page; it's the canonical "
        "first-mention surface."
    )


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------


def test_topofuzz_pyproject_description_has_trademark() -> None:
    pyproj = tomllib.load(
        open(_repo_root() / "packages" / "topofuzz" / "pyproject.toml", "rb")
    )
    desc = pyproj["project"]["description"]
    assert TM in desc, f"topofuzz pyproject description missing ™ — {desc!r}"
    assert f"topofuzz{TM}" in desc


def test_fuzzeval_pyproject_description_has_trademark() -> None:
    pyproj = tomllib.load(
        open(_repo_root() / "packages" / "fuzzeval" / "pyproject.toml", "rb")
    )
    desc = pyproj["project"]["description"]
    assert TM in desc, f"fuzzeval pyproject description missing ™ — {desc!r}"
    assert f"fuzzeval{TM}" in desc


# ---------------------------------------------------------------------------
# Boundary conditions — what should NOT have ™
# ---------------------------------------------------------------------------


def test_no_trademark_in_python_identifiers() -> None:
    """Spec line 1574-1575: '™' must NOT appear inside code
    identifiers (function names, variable names, package names).
    Code stays ASCII.

    Tokenizer-based check (pass-1 RT m1 + CR-2 fix): tokenize each
    .py file and assert ``™`` appears only inside ``STRING`` or
    ``COMMENT`` tokens. Previous heuristic ("preceded by quote or
    '#' on the same line") had false negatives (e.g. ``x = "ok";
    bad™ = 1`` would pass).
    """
    pkg_src = (
        _repo_root() / "packages" / "topofuzz" / "src" / "topofuzz"
    )
    for py_path in pkg_src.rglob("*.py"):
        text = _read_utf8(py_path)
        if TM not in text:
            continue
        try:
            tokens = list(
                tokenize.tokenize(io.BytesIO(text.encode("utf-8")).readline)
            )
        except tokenize.TokenError as exc:
            # Pass-2 BLOCKER fix: ``tokenize.TokenizeError`` does not
            # exist on Python 3.13+ — the actual class is
            # ``TokenError`` (singular). Catching the non-existent
            # name silently masked any tokenize failure as an
            # AttributeError test crash.
            pytest.fail(f"{py_path}: tokenize failed: {exc}")
        for tok in tokens:
            if TM not in tok.string:
                continue
            assert tok.type in (tokenize.STRING, tokenize.COMMENT), (
                f"{py_path}:{tok.start[0]} has ™ in a "
                f"{tokenize.tok_name[tok.type]} token "
                f"(must be in STRING or COMMENT only): {tok.string!r}"
            )


def test_per_brand_first_mention_not_diluted_in_templates() -> None:
    """Spec line 1568: '™ on FIRST mention per page only ... per
    brand'. Pass-1 RT m3 fix: count PER BRAND, not total — a page
    that legitimately mentions all three brands (topofuzz / fuzzeval
    / fuzzlyzer) on first mention is correct, not over-marked.

    For each brand: assert ≤ 1 ™ in the template (literal + entity
    combined)."""
    brands = ("topofuzz", "fuzzeval", "fuzzlyzer")
    for template_name in (
        "report.html.j2",
        "inspect.html.j2",
        "ceo_card.html.j2",
    ):
        src = _read_utf8(_templates_dir() / template_name)
        for brand in brands:
            # Pass-2 Mi1: count across all U+2122 encodings so a
            # contributor pasting numeric entities cannot dilute the
            # mark.
            count = sum(src.count(f"{brand}{form}") for form in _TM_FORMS)
            assert count <= 1, (
                f"{template_name} has {count} '{brand}™' marks; "
                "FIRST-mention-only rule means at most 1 per brand "
                "per page (dilution risk)."
            )
