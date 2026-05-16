# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""``topofuzz publish``: package one run as a single share-by-email PDF.

Answers the buyer-pitch question: *"can I send this to a CTO who won't run
your CLI?"*. Wraps ``report.html`` + the top N flipping ``inspect_<eid>.html``
pages into one PDF whose every page carries a verify breadcrumb
(``run_id`` + ``tds_version`` + ``signed_at``) so the reader can re-attest
the source bundle without trusting the PDF itself.

Design contract:

* HTML stays canonical. PDF is a derived artifact — anyone needing
  interactivity (hover rationales, expand panels) opens the HTML.
* Pure module: identical ``(run_dir, max_flips)`` produces byte-identical
  PDF *modulo* Chromium's font hinting + Playwright's embedded creation
  timestamp. (We pin font-family at the template level and rely on
  ``--pdf=true``-style determinism arguments but do NOT promise byte
  determinism — that is a v0.6 question.)
* Lazy import for ``playwright`` and ``pypdf``: the verify-only install
  must not require either, so the module imports them inside the render
  call and raises a clean :class:`PublishError` when they are absent.
* No network. Every HTML page is rendered from ``file://`` URLs.
"""

from __future__ import annotations

import base64
import io
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

_log = logging.getLogger(__name__)

# Cap on per-call flips we will render. Above this the PDF turns into a
# small book — a buyer pitch needs the 5 most fragile evals, not the
# whole tail. Override with ``--max-flips``; the CLI also enforces a
# hard upper bound to keep Chromium memory bounded.
DEFAULT_MAX_FLIPS = 5
MAX_FLIPS_HARD_LIMIT = 50

# Same charset constraint ``topofuzz.inspect`` uses for the default
# inspect_<eid>.html filename. Locked here separately so a hostile
# eval_id surfaced from ``run.sqlite`` can be rejected BEFORE it reaches
# the ``run_dir / f"inspect_{eid}.html"`` path interpolation — without
# the regex, a "../../evil" eid would slip past the inspect re-render
# (which only validates on its OWN default-filename path) by hitting
# the publish-side ``is_file()`` short-circuit on a planted file.
# Re-defined here rather than imported from inspect to keep publish
# import-safe even if inspect's module-level deps shift in v0.6.
_SAFE_EID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Cap on Chromium per-page navigation, load-state, and PDF-render
# waits. ``networkidle`` without a timeout can hang indefinitely on a
# malformed report.html (broken inline SVG, unresolved schemeless
# ``<img src="//...">``). 15 s per phase is plenty for the
# inline-only HTML the renderer emits.
#
# Worst-case wall clock = ``(1 + max_flips) × 3 × _PER_PAGE_LOAD_TIMEOUT_MS``
# (goto + wait_for_load_state + page.pdf). At MAX_FLIPS_HARD_LIMIT=50
# that's 51 × 45 s ≈ 38 min worst case; at default max_flips=5 it's
# 6 × 45 s = 4.5 min worst case. browser launch and context creation
# fall back to Playwright's per-op default (30 s each, once per
# invocation, not in the per-page cost).
_PER_PAGE_LOAD_TIMEOUT_MS = 15_000


class PublishError(ValueError):
    """Raised when the PDF artifact cannot be produced.

    Covers missing run-dir files, missing optional deps (playwright /
    pypdf), Chromium launch failures, and downstream
    :class:`topofuzz.inspect.InspectError` /
    :class:`topofuzz.report.ReportInputError` reraises.
    """


def select_top_flipping_evals(
    run_dir: Path,
    *,
    max_flips: int = DEFAULT_MAX_FLIPS,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> list[str]:
    """Return up to ``max_flips`` eval_ids with the most variant-disagreements.

    "Most fragile" is operationalised as: how many non-canonical variants
    disagree with the canonical (variant 0) on this eval's binarised
    verdict. Ties break lex-ascending on ``eval_id`` for determinism.

    Auto-skips evals where the canonical variant has no scored verdict —
    there is no reference frame to flip against (mirrors
    ``topofuzz.inspect.find_inspect_pair`` behaviour).

    Returns ``[]`` when no flipping eval exists. The caller can still
    publish a "stable run" PDF — the report.html alone is a legitimate
    artifact.
    """
    if max_flips < 0:
        raise PublishError(
            f"max_flips must be non-negative, got {max_flips}"
        )
    if max_flips == 0:
        return []

    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    hashes = manifest.get("variant_canonical_hashes")
    if not isinstance(hashes, list) or not hashes:
        raise PublishError(
            f"manifest.variant_canonical_hashes is missing or empty in "
            f"run {run_dir.name} — cannot identify the canonical variant."
        )
    hash_to_idx = {h: i for i, h in enumerate(hashes)}

    sqlite_path = run_dir / "run.sqlite"
    if not sqlite_path.is_file():
        raise PublishError(
            f"run.sqlite is missing in run_dir {run_dir.name!r}"
        )
    # Read-only mode: publish is a packaging command and must not create
    # a -journal sibling that could race a concurrent ``topofuzz run``
    # writer (SQLITE_BUSY at minimum, journal-lock contention on
    # Windows where file locks are mandatory). Audit RT-pass-1 MAJOR-3.
    con = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro", uri=True
    )
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
            raise PublishError(
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

    # Per-eval flip count: number of non-canonical variants whose
    # verdict differs from variant 0's. Evals with no canonical verdict
    # are excluded.
    flip_counts: list[tuple[str, int]] = []
    for eid, verdicts in by_eval.items():
        canonical = verdicts.get(0)
        if canonical is None:
            continue
        n_flips = sum(
            1 for i, v in verdicts.items() if i != 0 and v != canonical
        )
        if n_flips > 0:
            flip_counts.append((eid, n_flips))

    # Sort: highest flip count first, then lex-ascending on eid for ties.
    flip_counts.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = [eid for eid, _ in flip_counts[:max_flips]]

    # BLOCKER defence (audit RT-pass-1 B1): a hostile or sloppy eval_id
    # like ``../evil`` flowing in from ``run.sqlite`` would interpolate
    # into ``run_dir / f"inspect_{eid}.html"`` and either (a) point
    # outside the run dir (path traversal) or (b) collide with a planted
    # inspect file that ``build_publish_pdf`` would skip-re-rendering and
    # feed to Chromium with the real run footer. ``topofuzz.inspect``
    # already enforces ``_SAFE_EID_RE`` for its OWN default filename;
    # publish must enforce it BEFORE the path interpolation. Refuse the
    # whole selection rather than silently dropping unsafe ids — a
    # mismatched footer is worse than a clean error.
    #
    # ``fullmatch`` not ``match``: Python's ``$`` anchor allows a single
    # trailing ``\n`` to slip past ``^...+$``, so ``"valid\n"`` would
    # bypass a `re.match` guard. ``fullmatch`` rejects any trailing
    # whitespace, including newlines (audit RT-pass-2 MINOR-N1).
    for eid in selected:
        if not _SAFE_EID_RE.fullmatch(eid):
            raise PublishError(
                f"eval_id {eid!r} in run.sqlite contains characters not "
                "permitted in a publish artifact path "
                "(allowed: A-Z, a-z, 0-9, '.', '_', '-'). Refuse to "
                "render — the run.sqlite may be tampered or the eval "
                "set must be renamed."
            )
    return selected


def load_publish_metadata(run_dir: Path) -> dict[str, str]:
    """Return ``{"run_id", "tds_version", "signed_at"}`` for the PDF footer.

    Prefers the DSSE-signed ``attestation.json`` payload (so a tampered
    ``manifest.json`` cannot forge the footer breadcrumb) and falls back
    to ``manifest.json`` when no attestation exists. When falling back,
    ``signed_at`` reads ``"(unsigned)"`` so the buyer sees the artifact
    is not bound to a signed bundle.

    Does NOT verify the signature — that is :func:`topofuzz.attest
    .load_and_verify_attestation`'s job and runs in the ``topofuzz
    verify`` path. ``publish`` is a *packaging* command; the footer is
    a pointer to the verify command, not a substitute for it.
    """
    run_dir = Path(run_dir)
    att_path = run_dir / "attestation.json"
    attestation_corrupt = False
    if att_path.is_file():
        try:
            envelope = json.loads(att_path.read_text(encoding="utf-8"))
            payload_b64 = envelope["payload"]
            payload = json.loads(base64.b64decode(payload_b64))
            return {
                "run_id": str(payload.get("run_id", "")),
                "tds_version": str(
                    payload.get("manifest", {}).get("tds_version", "")
                ),
                "signed_at": str(payload.get("signed_at", "")),
            }
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            # Distinguish "no attestation present" from "attestation
            # present but corrupt" in the footer so a buyer sees a
            # signal in the PDF itself, not just in the operator's
            # console (audit RT-pass-2 MINOR-N2). An attacker who swaps
            # a real attestation for a corrupt one to silently downgrade
            # the breadcrumb would otherwise produce a footer indistin-
            # guishable from a legitimately-unsigned bundle.
            _log.warning(
                "publish: attestation.json present but unreadable "
                "(%s) — falling back to manifest.json with "
                "corrupt-attestation footer",
                e,
            )
            attestation_corrupt = True

    manifest = _load_manifest(run_dir)
    return {
        "run_id": str(manifest.get("run_id", "")),
        "tds_version": str(manifest.get("tds_version", "")),
        "signed_at": (
            "(corrupt-attestation)"
            if attestation_corrupt
            else "(unsigned)"
        ),
    }


def build_publish_pdf(
    run_dir: Path,
    *,
    out_path: Path,
    max_flips: int = DEFAULT_MAX_FLIPS,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> Path:
    """Produce ``out_path`` containing report.pdf + top-N inspect_<eid>.pdf.

    Side effects on ``run_dir``:

    * ``report.html`` is rendered if missing (calls
      :func:`topofuzz.report.render_report`).
    * ``inspect_<eid>.html`` is rendered for each top flipping eval if
      missing (calls :func:`topofuzz.inspect.write_inspect_html`).

    These are intentional: the HTML files are useful by themselves, and
    re-using the existing renderers means the PDF visually matches what
    a buyer sees when they open the URL.

    Raises :class:`PublishError` on any failure (missing deps, missing
    files, render failure). The caller's CLI prints the message and
    exits non-zero.
    """
    if max_flips < 0:
        raise PublishError(
            f"max_flips must be non-negative, got {max_flips}"
        )
    if max_flips > MAX_FLIPS_HARD_LIMIT:
        raise PublishError(
            f"max_flips={max_flips} exceeds hard limit "
            f"{MAX_FLIPS_HARD_LIMIT} — publish would launch too many "
            "Chromium navigations; reduce or split into multiple PDFs."
        )
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise PublishError(
            f"run_dir is not a directory: {run_dir}"
        )
    out_path = Path(out_path)
    if out_path.is_dir():
        raise PublishError(
            f"out_path is an existing directory: {out_path}"
        )

    # Resolve metadata up front so a malformed attestation fails fast
    # before we burn Chromium startup time.
    metadata = load_publish_metadata(run_dir)
    footer_text = _format_footer(metadata)

    # Re-render report.html unconditionally. A stale or planted
    # report.html in the run dir would otherwise be picked up and
    # stamped with the current manifest's footer — buyer sees a footer
    # that doesn't describe the bytes (audit RT-pass-1 MAJOR-1). The
    # renderer is byte-deterministic per its docstring, so re-rendering
    # an already-correct file is idempotent and costs ~1 s.
    try:
        from topofuzz.report import (
            ReportDependencyError,
            ReportInputError,
            render_report,
        )

        report_html = render_report(run_dir)
    except ImportError as e:
        raise PublishError(
            "topofuzz publish requires the 'full' install extra to "
            "render report.html — jinja2 / matplotlib not found. "
            "Install with `pip install topofuzz[full]`."
        ) from e
    except (ReportInputError, ReportDependencyError) as e:
        raise PublishError(
            f"could not render report.html: {e}"
        ) from e

    # Compute top flipping evals and re-render each inspect_<eid>.html
    # unconditionally (same provenance argument as report.html above).
    top_eids = select_top_flipping_evals(
        run_dir, max_flips=max_flips, threshold=threshold
    )
    inspect_html_paths: list[Path] = []
    if top_eids:
        from topofuzz.inspect import InspectError, write_inspect_html

        for eid in top_eids:
            try:
                target = write_inspect_html(
                    run_dir, eval_id=eid, threshold=threshold
                )
            except InspectError as e:
                # An eval listed by ``select_top_flipping_evals`` should
                # always be inspectable, but a manifest/sqlite drift
                # could still trip the write. Surface explicitly so
                # the buyer can be told which eval was skipped rather
                # than silently dropping it.
                raise PublishError(
                    f"inspect render failed for eval_id={eid!r}: {e}"
                ) from e
            inspect_html_paths.append(target)

    # Render each HTML page to its own PDF, then merge.
    html_pages = [report_html, *inspect_html_paths]
    pdf_bytes_list = _render_html_pages_to_pdfs(html_pages, footer_text)
    merged = _merge_pdfs(pdf_bytes_list)
    _atomic_write_bytes(out_path, merged)
    return out_path


# ---------------------------------------------------------------------------
# F6 — CEO card PDF (1-page cold-email-shaped derivative of F5).
# ---------------------------------------------------------------------------


# Cap on excerpt length so a verbose agent does not blow past the
# single-page budget. Tuned visually: ~520 chars renders to ~10 lines of
# the serif body font at 10pt and still leaves room for the eval prompt.
_CEO_CARD_VERDICT_EXCERPT_LIMIT = 520
_CEO_CARD_PROMPT_EXCERPT_LIMIT = 320


def build_ceo_card_pdf(
    run_dir: Path,
    *,
    out_path: Path,
    target_company: str | None = None,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> Path:
    """Render a single-page cold-email-shaped CEO card to ``out_path``.

    A condensed derivative of the F5 publish PDF (BACKLOG F6 entry).
    Surfaces the verdict-flip headline + Wilson 95 percent CI + one
    concrete flipped verdict + the attestation breadcrumb on a single
    Letter page. Designed for an opening cold-email attachment that
    reads on a phone; the full 19-page F5 is the linked follow-up.

    ``target_company`` substitutes into the "what this means for $TARGET"
    paragraph. When ``None`` we render a generic "your stack" copy. The
    PDF metadata does NOT bake the company name (deliverability beats
    recipient-specific personalisation here per the F6 spec).

    Reuses :func:`load_publish_metadata`, :func:`select_top_flipping_evals`,
    :func:`_format_footer`, and :func:`_render_html_pages_to_pdfs` so the
    attestation footer / safe-eid guard / Chromium sandbox / atomic write
    contracts are inherited byte-identically from F5.

    Raises :class:`PublishError` on any failure. The caller's CLI prints
    the message and exits non-zero.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise PublishError(
            f"run_dir is not a directory: {run_dir}"
        )
    out_path = Path(out_path)
    if out_path.is_dir():
        raise PublishError(
            f"out_path is an existing directory: {out_path}"
        )
    if target_company is not None:
        # Length bound prevents a malicious or sloppy operator from
        # injecting a multi-paragraph "company name" that breaks the
        # single-page layout (and prevents pathological wrapping in
        # PDF metadata if we ever decide to set it later).
        if not isinstance(target_company, str):
            raise PublishError(
                f"target_company must be str, got "
                f"{type(target_company).__name__}"
            )
        if not target_company.strip():
            raise PublishError("target_company must be non-empty / non-whitespace")
        # F6 pass-1 RT M3: NFC-normalise before the length cap so a
        # decomposed-form ZWJ/combining-mark/fullwidth-digit input cannot
        # bypass the visual-width budget. Then reject bidi-override
        # codepoints (‪-‮, ⁦-⁩) and zero-width
        # joiners (​-‏) outright — they survive autoescape
        # but cosmetically poison the rendered PDF (e.g. RIGHT-TO-LEFT
        # OVERRIDE reverses subsequent text). C0/C1 controls are
        # blocked because they break Chromium text shaping in
        # undefined ways.
        import unicodedata
        target_company = unicodedata.normalize("NFC", target_company)
        for ch in target_company:
            cp = ord(ch)
            if cp < 0x20 and ch not in ("\t", "\n"):
                raise PublishError(
                    "target_company contains a C0 control character "
                    f"(U+{cp:04X}); reject."
                )
            if 0x7f <= cp <= 0x9f:
                raise PublishError(
                    "target_company contains a C1 control character "
                    f"(U+{cp:04X}); reject."
                )
            if (
                # zero-width + bidi-format
                0x200b <= cp <= 0x200f
                or 0x202a <= cp <= 0x202e
                or 0x2066 <= cp <= 0x2069
                # soft hyphen (invisible word break)
                or cp == 0x00ad
                # variation selectors (font-shaping bypass)
                or 0xfe00 <= cp <= 0xfe0f
                or 0xe0100 <= cp <= 0xe01ef
                # tag chars (Unicode tag attack payload-smuggling)
                or 0xe0000 <= cp <= 0xe007f
                # mathematical alphanumeric symbols (homoglyph
                # bypassing the visual-width budget; NFC does NOT
                # fold these — they're distinct codepoints)
                or 0x1d400 <= cp <= 0x1d7ff
            ):
                raise PublishError(
                    "target_company contains an invisible / formatting "
                    f"/ homoglyph codepoint (U+{cp:04X}); reject — "
                    "would cosmetically poison the rendered PDF or "
                    "bypass the visual-width budget."
                )
        if len(target_company) > 80:
            raise PublishError(
                f"target_company exceeds 80 chars ({len(target_company)}); "
                "trim before render."
            )

    metadata = load_publish_metadata(run_dir)
    footer_text = _format_footer(metadata)

    # Pull live numbers from the source-of-truth files: report.json for
    # the headline + Wilson CI, manifest.json for variant/eval counts +
    # cost. report.html is intentionally NOT re-rendered here (the CEO
    # card is a standalone artifact; if the operator wants the full PDF
    # too they run `topofuzz publish` separately).
    manifest = _load_manifest(run_dir)
    report_json_path = run_dir / "report.json"
    if not report_json_path.is_file():
        raise PublishError(
            "report.json is missing in run_dir — run "
            "`topofuzz publish` once (or `topofuzz report`) so the "
            "headline metric is available."
        )
    try:
        report_payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PublishError(f"could not read report.json: {e}") from e
    if not isinstance(report_payload, dict):
        raise PublishError(
            "report.json root is not an object — refuse to render."
        )

    headline = (
        report_payload.get("headline")
        if isinstance(report_payload.get("headline"), dict)
        else {}
    )
    vfr = headline.get("verdict_flip_rate") if isinstance(headline, dict) else None
    wilson = report_payload.get("wilson_addenda") if isinstance(report_payload, dict) else None

    ctx = _build_ceo_card_context(
        metadata=metadata,
        manifest=manifest,
        verdict_flip_rate=vfr if isinstance(vfr, dict) else None,
        wilson_addenda=wilson if isinstance(wilson, dict) else None,
        target_company=target_company,
        run_dir=run_dir,
        threshold=threshold,
    )

    html = _render_ceo_card_html(ctx)
    # F6 pass-1 CR N2 + pass-2 CR I1: sweep stale debris on entry. A
    # prior Chromium handle-hold (Windows) can leave the tempfile
    # unlinked; without the sweep an automation loop accumulates one
    # orphan per failed-unlink. The prefix is narrowed to
    # ``ceo_card_tmp_*`` (not ``ceo_card_*``) so a legitimate operator
    # file like ``topofuzz inspect --out ceo_card_demo.html`` doesn't
    # get silently unlinked.
    for stale in run_dir.glob("ceo_card_tmp_*.html"):
        try:
            stale.unlink()
        except OSError:
            _log.debug("publish: could not sweep stale ceo card %s", stale)
    # Render the single HTML page through the shared Chromium path so
    # the page-footer breadcrumb is identical to F5's. We write a
    # transient .html into run_dir so Playwright's file:// resolution
    # matches the F5 flow (some templates rely on relative asset paths,
    # though ceo_card.html.j2 deliberately inlines everything).
    with tempfile.NamedTemporaryFile(
        prefix="ceo_card_tmp_",
        suffix=".html",
        dir=run_dir,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as fh:
        fh.write(html)
        tmp_html_path = Path(fh.name)
    try:
        pdf_bytes_list = _render_html_pages_to_pdfs([tmp_html_path], footer_text)
    finally:
        try:
            tmp_html_path.unlink()
        except OSError:
            # The Chromium renderer may still hold an open handle for a
            # tick on Windows; leave the temp file rather than crash the
            # publish call. Next invocation's sweep will clean it up.
            _log.warning(
                "publish: could not unlink ceo_card scratch file %s",
                tmp_html_path,
            )
    if len(pdf_bytes_list) != 1:
        raise PublishError(
            f"ceo card render produced {len(pdf_bytes_list)} PDFs; "
            "expected exactly 1 page."
        )
    # F6 pass-1 CR M2: enforce the "1-2 page" claim. The HTML-file count
    # above only catches a 0/2-file split; an HTML rendered to 3 paper
    # pages still slips through. Count via pypdf before atomic-writing.
    pdf_bytes = pdf_bytes_list[0]
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        n_pages = len(reader.pages)
    except ImportError:
        # pypdf is part of the [pdf] extra; if it's missing here we
        # already failed during F5's _render_html_pages_to_pdfs import.
        # Defensive only.
        n_pages = 1
    except Exception as e:  # noqa: BLE001 — refuse the artifact, not crash
        raise PublishError(
            f"could not validate ceo card page count: {e}"
        ) from e
    if n_pages > 2:
        raise PublishError(
            f"ceo card rendered to {n_pages} pages; the F6 spec is "
            "1-2 pages. Inspect the source HTML for runaway content "
            "(usually a long topology_name or eval prompt)."
        )
    _atomic_write_bytes(out_path, pdf_bytes)
    return out_path


def _build_ceo_card_context(
    *,
    metadata: dict[str, str],
    manifest: dict[str, Any],
    verdict_flip_rate: dict[str, Any] | None,
    wilson_addenda: dict[str, Any] | None,
    target_company: str | None,
    run_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    """Assemble the Jinja context for ceo_card.html.j2.

    Pulled out for testability — the template render is byte-deterministic
    given this dict, so tests can pin headline text / CI strings without
    spinning up Chromium.
    """
    topology_name = str(manifest.get("topology_name") or "(unknown)")
    n_variants = manifest.get("n_variants")
    n_evals = manifest.get("n_evals")
    cost_actual = manifest.get("cost_actual_usd")

    flip_rate, n_pairs, n_flips = _extract_flip_counts(verdict_flip_rate)
    wilson_lo, wilson_hi = _extract_wilson_bounds(wilson_addenda, flip_rate, n_pairs, n_flips)

    if flip_rate is not None and n_pairs and n_pairs > 0 and n_flips and n_flips > 0:
        flip_pct_str = f"{flip_rate * 100:.0f}%"
        flip_label = (
            f"of variant re-orderings flipped the verdict on this "
            f"pipeline ({n_flips}/{n_pairs} pairs)"
        )
        if wilson_lo is not None and wilson_hi is not None:
            ci_str = (
                f"Wilson 95% CI [{wilson_lo * 100:.1f}%, "
                f"{wilson_hi * 100:.1f}%] — distinguishable from "
                f"zero at p<0.05."
            )
        else:
            ci_str = (
                f"Headline computed from {n_pairs} variant pairs. "
                "Wilson CI unavailable in this run."
            )
    else:
        flip_pct_str = ""
        flip_label = ""
        ci_str = ""

    target_phrase = target_company.strip() if target_company else None
    if target_phrase:
        what_para = _format_what_paragraph(
            target_phrase, topology_name, flip_rate
        )
        cta_line = (
            f"Want this run reproduced on {target_phrase}'s own "
            "agent topology? Reply to this email — repro is free."
        )
    else:
        what_para = _format_what_paragraph(
            "your stack", topology_name, flip_rate
        )
        cta_line = (
            "Want this run reproduced on your own agent topology? "
            "Reply to this email — repro is free."
        )

    headline_title, headline_lede = _format_headline(topology_name, flip_rate, n_flips)

    flip_card = _build_flip_card(run_dir, manifest, threshold)

    cost_str = (
        f"${float(cost_actual):.2f}"
        if isinstance(cost_actual, (int, float)) and not isinstance(cost_actual, bool)
        else "—"
    )
    n_pairs_str = str(n_pairs) if n_pairs else "0"

    def _safe_int(v: Any) -> int | str:
        # F6 pass-1 CR N1: bool is a subclass of int; reject so the
        # template never renders True/False as a "count."
        if isinstance(v, bool) or not isinstance(v, int):
            return "—"
        return v

    return {
        "headline_title": headline_title,
        "headline_lede": headline_lede,
        "flip_pct_str": flip_pct_str,
        "flip_label": flip_label,
        "ci_str": ci_str,
        "what_paragraph": what_para,
        "cta_line": cta_line,
        "topology_name": topology_name,
        "n_variants": _safe_int(n_variants),
        "n_evals": _safe_int(n_evals),
        "cost_str": cost_str,
        "n_pairs_str": n_pairs_str,
        "flip_card": flip_card,
        "run_id": metadata.get("run_id") or "(unknown)",
        "tds_version": metadata.get("tds_version") or "(unknown)",
        "signed_at": metadata.get("signed_at") or "(unsigned)",
    }


def _extract_flip_counts(
    verdict_flip_rate: dict[str, Any] | None,
) -> tuple[float | None, int | None, int | None]:
    """Pull (point, n_pairs, n_flips) from headline.verdict_flip_rate.

    Defensive: handles missing keys, non-numeric values, and the bool-as-
    int trap. Returns ``(None, None, None)`` on any unusable shape so
    the template's "stable run" branch fires.
    """
    if not isinstance(verdict_flip_rate, dict):
        return None, None, None
    point_raw = verdict_flip_rate.get("point")
    n_pairs_raw = verdict_flip_rate.get("n_pairs")
    n_flips_raw = verdict_flip_rate.get("n_flips")

    def _to_float(v: Any) -> float | None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if v != v:  # NaN
            return None
        return float(v)

    def _to_int(v: Any) -> int | None:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        return v

    return _to_float(point_raw), _to_int(n_pairs_raw), _to_int(n_flips_raw)


def _extract_wilson_bounds(
    wilson_addenda: dict[str, Any] | None,
    flip_rate: float | None,
    n_pairs: int | None,
    n_flips: int | None,
) -> tuple[float | None, float | None]:
    """Pull Wilson 95% CI lo/hi.

    Prefers the persisted ``wilson_addenda`` block (computed by report.py
    at render time). Falls back to recomputing from the counts when the
    block is absent (e.g. a v0.5 run.sqlite that predates Wilson).
    """
    if isinstance(wilson_addenda, dict):
        lo = wilson_addenda.get("ci_lower")
        hi = wilson_addenda.get("ci_upper")
        if (
            isinstance(lo, (int, float))
            and not isinstance(lo, bool)
            and isinstance(hi, (int, float))
            and not isinstance(hi, bool)
        ):
            return float(lo), float(hi)
    # Fallback: recompute if we have the counts.
    if (
        flip_rate is not None
        and n_pairs is not None
        and n_flips is not None
        and n_pairs > 0
        and 0 <= n_flips <= n_pairs
    ):
        try:
            from topofuzz.metrics import wilson_score_interval

            lo, hi = wilson_score_interval(n_flips, n_pairs, conf=0.95)
            if lo is not None and hi is not None:
                return float(lo), float(hi)
        except Exception:  # noqa: BLE001 — fallback path; refuse to crash
            pass
    return None, None


def _format_headline(
    topology_name: str, flip_rate: float | None, n_flips: int | None
) -> tuple[str, str]:
    """Headline + lede tuned to the run shape."""
    pretty_name = topology_name.replace("_", " ").replace("-", " ")
    if flip_rate is not None and n_flips and n_flips > 0:
        title = f"This {pretty_name} pipeline flips its verdict under re-ordering."
        lede = (
            "Same model, same prompts, same eval. Reorder the agents and "
            "the answer changes — measured under a jury-of-3, not a "
            "single judge."
        )
    else:
        title = f"This {pretty_name} pipeline held its verdict."
        lede = (
            "Re-ordering the agents did not flip the answer across the "
            "tested variants — a publishable claim, not a luck of the draw."
        )
    return title, lede


def _format_what_paragraph(
    target: str, topology_name: str, flip_rate: float | None
) -> str:
    """The "what this means for $TARGET" body paragraph.

    Template per F6 spec — handcrafted (no LLM, no per-target brief),
    references the topology shape so the reader recognises themselves.
    """
    pretty_name = topology_name.replace("_", " ").replace("-", " ")
    if flip_rate is not None and flip_rate > 0:
        return (
            f"If {target} runs a multi-agent pipeline that looks like "
            f"this {pretty_name} shape — intake then risk then policy "
            "then a decision agent, or any analogous chain — the "
            "auditable claim 'the model decided X' is conditional on an "
            "implementation detail (the agent order) that nobody "
            "documents. Compliance reviewers will not catch this. "
            "topofuzz does."
        )
    return (
        f"If {target} runs a multi-agent pipeline that looks like this "
        f"{pretty_name} shape, this run is the kind of stability claim "
        "you can put in a SOC 2 or model-risk-management report. "
        "topofuzz produced the bundle; the attestation footer below "
        "lets any reviewer re-run the test bit-for-bit."
    )


def _build_flip_card(
    run_dir: Path, manifest: dict[str, Any], threshold: float
) -> dict[str, Any] | None:
    """Pick the single most-flipped eval and assemble a side-by-side card.

    Re-uses :func:`select_top_flipping_evals` so the eid safety check
    runs. Returns ``None`` on any data-shape failure (no flips, missing
    columns, malformed JSON) — the template silently omits the section,
    matching the "stable run" branch of the headline.
    """
    try:
        top = select_top_flipping_evals(
            run_dir, max_flips=1, threshold=threshold
        )
    except PublishError:
        return None
    if not top:
        return None
    eid = top[0]
    sqlite_path = run_dir / "run.sqlite"
    if not sqlite_path.is_file():
        return None
    hashes = manifest.get("variant_canonical_hashes")
    if not isinstance(hashes, list) or not hashes:
        return None
    canonical_hash = hashes[0]
    hash_to_idx = {h: i for i, h in enumerate(hashes)}
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        # Eval prompt for the excerpt.
        try:
            row = con.execute(
                "SELECT input FROM evals WHERE id = ?", (eid,)
            ).fetchone()
            eval_prompt = row[0] if row and isinstance(row[0], str) else ""
        except sqlite3.DatabaseError:
            eval_prompt = ""
        # All rows for this eval, then pick canonical + first variant
        # that flipped against it.
        rows = list(
            con.execute(
                "SELECT variant_id, score_value, passed FROM judge_results "
                "WHERE eval_id = ? ORDER BY variant_id",
                (eid,),
            )
        )
        canonical_verdict = None
        flipping_vid = None
        for vid, score, passed in rows:
            idx = hash_to_idx.get(vid)
            if idx is None:
                continue
            if passed is not None:
                v = bool(passed)
            elif score is not None:
                v = bool(score >= threshold)
            else:
                continue
            if idx == 0:
                canonical_verdict = v
            elif canonical_verdict is not None and v != canonical_verdict:
                flipping_vid = vid
                break
        if canonical_verdict is None or flipping_vid is None:
            return None
        # Pull node_outputs_json for both variants for excerpt + ordering.
        try:
            canonical_row = con.execute(
                "SELECT node_outputs_json FROM cells "
                "WHERE eval_id = ? AND variant_id = ?",
                (eid, canonical_hash),
            ).fetchone()
            variant_row = con.execute(
                "SELECT node_outputs_json FROM cells "
                "WHERE eval_id = ? AND variant_id = ?",
                (eid, flipping_vid),
            ).fetchone()
        except sqlite3.DatabaseError:
            return None
    finally:
        con.close()

    # F6 pass-1 RT M1: mirror runner._run_judge dispatch so the
    # buyer-facing excerpt comes from the SAME node the judge scored.
    # ``verdict_node`` (v0.6) wins; ``terminal`` (v0.5) is the fallback.
    # F6 pass-2 RT m1: an empty / whitespace verdict_node in a
    # malformed manifest silently fell back to outputs[-1] — surface
    # the malformed manifest so the operator sees it instead of
    # shipping a degraded card.
    verdict_node_id = manifest.get("verdict_node")
    if isinstance(verdict_node_id, str) and not verdict_node_id.strip():
        _log.warning(
            "publish: manifest.verdict_node is empty/whitespace — "
            "treating as unset and falling back to terminal[0]."
        )
        verdict_node_id = None
    if not isinstance(verdict_node_id, str):
        verdict_node_id = None
    terminal_nodes_raw = manifest.get("terminal_nodes")
    if isinstance(terminal_nodes_raw, list) and all(
        isinstance(t, str) and t.strip() for t in terminal_nodes_raw
    ):
        terminal_nodes = tuple(terminal_nodes_raw)
    else:
        terminal_nodes = None
    canonical_excerpt, canonical_ordering = _excerpt_terminal_output(
        canonical_row[0] if canonical_row else None,
        verdict_node=verdict_node_id,
        terminal_nodes=terminal_nodes,
    )
    variant_excerpt, variant_ordering = _excerpt_terminal_output(
        variant_row[0] if variant_row else None,
        verdict_node=verdict_node_id,
        terminal_nodes=terminal_nodes,
    )
    if not canonical_excerpt or not variant_excerpt:
        return None

    # F6 pass-1 CR M1 + pass-2 RT B1: APPROVE/DENY is the lending-
    # domain default. Producer-side wiring (Step 20a/F6 follow-up,
    # 2026-05-16) now lands non-default labels in the manifest via the
    # IR field, but a hand-edited or tampered manifest can still slip
    # malformed values past the IR — this consumer path stays
    # defense-in-depth.
    label_true_raw = manifest.get("verdict_label_true")
    label_false_raw = manifest.get("verdict_label_false")
    # Step 20a/F6 follow-up dual-audit pass-1 RT M2: pair-level XOR
    # check on the manifest. The orchestrator persists labels only as
    # a pair (both-or-neither, gated on differs-from-default), so a
    # manifest with exactly one key non-empty is an obvious tamper or
    # hand-edit. Log and fall back to BOTH defaults rather than letting
    # publish silently default one half against an explicit other half.
    _has_true = isinstance(label_true_raw, str) and label_true_raw.strip()
    _has_false = isinstance(label_false_raw, str) and label_false_raw.strip()
    if bool(_has_true) != bool(_has_false):
        _log.warning(
            "publish: manifest has exactly one of verdict_label_true/false "
            "set (true=%r, false=%r) — both must be set together or "
            "neither (orchestrator persists as a pair). Falling back to "
            "APPROVE/DENY defaults.",
            label_true_raw, label_false_raw,
        )
        label_true_raw = None
        label_false_raw = None
    # Step 20a/F6 follow-up dual-audit pass-1 RT M1 + CR MAJOR-2: a
    # manifest with a >12-char label used to silently slice [:12],
    # which (a) renders truncated nonsense in the CEO PDF and (b) can
    # collide with the partner label post-slice in ways the producer
    # IR validator would reject. Hard-reject and fall back to defaults
    # so the publish path matches the producer's policy.
    for raw_label, key in (
        (label_true_raw, "verdict_label_true"),
        (label_false_raw, "verdict_label_false"),
    ):
        if isinstance(raw_label, str) and len(raw_label.strip()) > 12:
            _log.warning(
                "publish: manifest %s=%r is >12 chars after strip — "
                "exceeds the side-by-side flip-card cap. Falling back "
                "to APPROVE/DENY for BOTH labels (pair-level fallback; "
                "the partner label is discarded too so the buyer sees "
                "the same default pair rather than a half-custom pair).",
                key, raw_label,
            )
            label_true_raw = None
            label_false_raw = None
            break
    label_true = (
        label_true_raw.strip().upper()[:12]
        if isinstance(label_true_raw, str) and label_true_raw.strip()
        else "APPROVE"
    )
    label_false = (
        label_false_raw.strip().upper()[:12]
        if isinstance(label_false_raw, str) and label_false_raw.strip()
        else "DENY"
    )
    # F6 pass-2 RT M1: same-after-normalize labels destroy the
    # side-by-side story. Refuse the card if a manifest opt-in
    # collapses the pair. CR pass-2 MINOR-1: include raw values so the
    # operator can see which input pair collided rather than guessing
    # from the post-normalize string alone.
    if label_true == label_false:
        raise PublishError(
            f"manifest verdict_label_true/false collide after "
            f"normalization (raw: true={label_true_raw!r}, "
            f"false={label_false_raw!r}; both → {label_true!r}); "
            f"refuse to render."
        )
    return {
        "eval_prompt_excerpt": _truncate(
            eval_prompt or "(eval prompt unavailable)",
            _CEO_CARD_PROMPT_EXCERPT_LIMIT,
        ),
        "canonical_verdict": label_true if canonical_verdict else label_false,
        "variant_verdict": label_false if canonical_verdict else label_true,
        "canonical_class": "approve" if canonical_verdict else "deny",
        "variant_class": "deny" if canonical_verdict else "approve",
        "canonical_ordering": canonical_ordering,
        "variant_ordering": variant_ordering,
        "canonical_excerpt": _truncate(
            canonical_excerpt, _CEO_CARD_VERDICT_EXCERPT_LIMIT
        ),
        "variant_excerpt": _truncate(
            variant_excerpt, _CEO_CARD_VERDICT_EXCERPT_LIMIT
        ),
    }


def _excerpt_terminal_output(
    node_outputs_json: str | None,
    *,
    verdict_node: str | None = None,
    terminal_nodes: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Return ``(verdict_text, ordering_str)`` for a cell's node_outputs.

    The CEO-card buyer reads the excerpt next to a verdict badge
    (APPROVE / DENY) — so the excerpt MUST come from the same node the
    judge scored, otherwise a buyer who later runs ``topofuzz inspect``
    sees a different excerpt next to the same verdict and loses trust.
    Mirrors :func:`topofuzz.runner._run_judge` dispatch (Step 20a):

    1. ``verdict_node`` if supplied — the v0.6 first-class field.
    2. The lex-first entry of ``terminal_nodes`` otherwise.
    3. ``outputs[-1]`` as the last-resort fallback when neither is
       provided (e.g. a v0.5 manifest predating the verdict_node-aware
       publish path); a one-line ``_log.debug`` records the fallback so
       an operator can grep for it.
    """
    if not isinstance(node_outputs_json, str) or not node_outputs_json:
        return "", ""
    try:
        outputs = json.loads(node_outputs_json)
    except ValueError:
        return "", ""
    if not isinstance(outputs, list) or not outputs:
        return "", ""
    ordering = " → ".join(
        str(o.get("node_id") or "?")
        for o in outputs
        if isinstance(o, dict)
    )
    by_node = {
        o.get("node_id"): o for o in outputs if isinstance(o, dict)
    }
    target_output: dict | None = None
    if verdict_node is not None:
        target_output = by_node.get(verdict_node)
    elif terminal_nodes:
        for tid in sorted(terminal_nodes):
            cand = by_node.get(tid)
            if isinstance(cand, dict):
                target_output = cand
                break
    if target_output is None:
        # Last-resort fallback for v0.5 manifests with neither field.
        _log.debug(
            "ceo card: falling back to outputs[-1] excerpt (no "
            "verdict_node or terminal_nodes supplied; v0.5 manifest)"
        )
        cand = outputs[-1]
        target_output = cand if isinstance(cand, dict) else None
    if target_output is None:
        return "", ordering
    text = target_output.get("response_text")
    if not isinstance(text, str):
        return "", ordering
    return text, ordering


def _truncate(text: str, limit: int) -> str:
    """Whitespace-collapse + ellipsis-truncate for layout safety."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _render_ceo_card_html(ctx: dict[str, Any]) -> str:
    """Render ceo_card.html.j2 with the assembled context."""
    try:
        import jinja2
    except ImportError as e:
        raise PublishError(
            "topofuzz publish --ceo-card requires the 'full' install "
            "extra — jinja2 not found. Install with "
            "`pip install topofuzz[full]`."
        ) from e

    template_dir = Path(__file__).parent / "templates"
    # F6 pass-1 CR N3: pass `autoescape=True` explicitly rather than
    # rely on filename-suffix matching — a future rename or a template
    # that drops the .j2 suffix should not silently turn escaping off.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template("ceo_card.html.j2")
    return template.render(**ctx)


def _format_footer(metadata: dict[str, str]) -> str:
    """Footer string injected on every PDF page.

    Three fields separated by middle-dots so the breadcrumb scans at a
    glance: ``topofuzz run <run_id> · TDS v<tds_version> · signed <ts>``.
    A reader copy-pasting any field into ``topofuzz verify`` should land
    on the exact bundle.
    """
    run_id = metadata.get("run_id") or "(unknown)"
    tds = metadata.get("tds_version") or "(unknown)"
    signed = metadata.get("signed_at") or "(unsigned)"
    # Surface a warning when any field is unknown/unsigned — operators
    # mailing the PDF deserve to see a footer-data-missing signal once,
    # not discover it when a buyer queries why "(unknown)" appeared in
    # the breadcrumb (audit RT-pass-1 MINOR-5).
    if run_id == "(unknown)" or tds == "(unknown)":
        _log.warning(
            "publish: footer fields incomplete (run_id=%r, "
            "tds_version=%r) — manifest may be missing fields",
            run_id, tds,
        )
    # Strip subseconds from ISO-8601 to keep the footer tight without
    # losing the day-of-issue context. Tightened guard: only strip when
    # the string actually looks like an ISO-8601 timestamp ("T" present
    # and a trailing 'Z'); otherwise pass through unchanged so a
    # malformed signed_at ("garbage.txt") surfaces verbatim to the
    # reader rather than being silently truncated (audit CR-MINOR-2 +
    # RT-pass-1 MINOR-N2).
    if (
        signed
        and signed != "(unsigned)"
        and "T" in signed
        and "." in signed
        and signed.endswith("Z")
    ):
        # "2026-05-15T02:07:48.945561Z" → "2026-05-15T02:07:48Z"
        head, _, _ = signed.partition(".")
        signed = head + "Z"
    return f"topofuzz run {run_id} · TDS v{tds} · signed {signed}"


def _render_html_pages_to_pdfs(
    html_paths: list[Path],
    footer_text: str,
) -> list[bytes]:
    """Render each HTML file to a PDF byte-string.

    Lazy-imports playwright so the verify-only install path doesn't
    require it.
    """
    try:
        from playwright.sync_api import (
            Error as PlaywrightError,
            sync_playwright,
        )
    except ImportError as e:
        raise PublishError(
            "topofuzz publish requires the 'pdf' install extra. "
            "Install with `pip install topofuzz[pdf]` and then "
            "`playwright install chromium`."
        ) from e

    # The footer template Playwright accepts is HTML; escape the text
    # to defend against an attacker-controlled run_id (which is a
    # short ULID + date in practice, but the contract should not assume).
    footer_html = (
        '<div style="font-family: -apple-system, Helvetica, Arial, '
        'sans-serif; font-size: 8pt; color: #897764; '
        'width: 100%; text-align: center; padding: 0 12mm;">'
        f"{_html_escape(footer_text)}"
        "</div>"
    )
    # Empty header keeps Chromium's default off (an empty div with the
    # same wrapper still forces display_header_footer to consult the
    # template instead of injecting the default URL/title pair).
    empty_header = '<div style="display:none;"></div>'

    pdf_bytes_list: list[bytes] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                context = browser.new_context()
                try:
                    page = context.new_page()
                    # Apply the per-phase ceiling to ALL page operations
                    # (goto, wait_for_load_state, pdf, etc.). Playwright's
                    # ``page.pdf()`` does not accept an inline ``timeout``
                    # kwarg, so the per-page default is the only way to
                    # bound the PDF-render phase without wrapping the
                    # call in a thread (audit RT-pass-2 MINOR-N1).
                    page.set_default_timeout(_PER_PAGE_LOAD_TIMEOUT_MS)
                    for html_path in html_paths:
                        page.goto(
                            html_path.resolve().as_uri(),
                            timeout=_PER_PAGE_LOAD_TIMEOUT_MS,
                        )
                        # Wait for inline SVG + CSS to finish before
                        # printing — the report has matplotlib-emitted
                        # SVGs that layout asynchronously. ``networkidle``
                        # without a timeout can hang on a malformed
                        # report.html (broken inline image schemes,
                        # never-resolving requests); per-page ceiling
                        # bounds the worst-case wall clock per the
                        # ``_PER_PAGE_LOAD_TIMEOUT_MS`` docstring (audit
                        # RT-pass-1 MAJOR-2).
                        page.wait_for_load_state(
                            "networkidle",
                            timeout=_PER_PAGE_LOAD_TIMEOUT_MS,
                        )
                        pdf_bytes_list.append(
                            page.pdf(
                                format="A4",
                                print_background=True,
                                display_header_footer=True,
                                header_template=empty_header,
                                footer_template=footer_html,
                                margin={
                                    "top": "16mm",
                                    "bottom": "18mm",
                                    "left": "12mm",
                                    "right": "12mm",
                                },
                            )
                        )
                finally:
                    # Audit CR-pass-1 MAJOR — context.close MUST run even
                    # when a per-page navigation/print raises, else the
                    # BrowserContext can hold a non-shared file handle on
                    # the rendered HTML and block atomic-rename on Windows.
                    context.close()
            finally:
                browser.close()
    except PlaywrightError as e:
        # Most commonly: "Executable doesn't exist" from a missing
        # `playwright install chromium`. Translate into a PublishError
        # so the CLI prints a single coherent message.
        raise PublishError(
            f"Chromium failed to launch: {e}. Run "
            "`playwright install chromium` once to download the browser."
        ) from e
    return pdf_bytes_list


def _merge_pdfs(pdf_bytes_list: list[bytes]) -> bytes:
    """Concatenate per-page PDFs into a single bytestring.

    Lazy-imports pypdf so the verify-only install path doesn't require
    it. For a single-element list, skips the merge step entirely.
    """
    if not pdf_bytes_list:
        raise PublishError(
            "no PDF pages to merge — internal error in publish pipeline"
        )
    if len(pdf_bytes_list) == 1:
        return pdf_bytes_list[0]
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.errors import PdfReadError
    except ImportError as e:
        raise PublishError(
            "topofuzz publish requires the 'pdf' install extra "
            "(pypdf missing). Install with `pip install topofuzz[pdf]`."
        ) from e

    try:
        writer = PdfWriter()
        for blob in pdf_bytes_list:
            reader = PdfReader(io.BytesIO(blob))
            for page in reader.pages:
                writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except PdfReadError as e:
        # Translate the pypdf failure into the module's canonical error
        # type — preserves the "every external exception → PublishError"
        # contract documented at module-level (audit CR-pass-1 MAJOR).
        raise PublishError(
            f"PDF merge failed — Chromium produced a blob pypdf "
            f"refused: {e}"
        ) from e


def _atomic_write_bytes(target: Path, blob: bytes) -> None:
    """Atomically replace ``target`` with ``blob``.

    Mirrors :func:`topofuzz.inspect._atomic_write_text` semantics —
    tempfile in the destination dir, then ``os.replace``. Wraps
    ``PermissionError`` (Windows AV / indexer holds the file open) into
    :class:`PublishError`.
    """
    target = Path(target)
    parent = target.parent
    if not parent.is_dir():
        raise PublishError(
            f"out_path parent directory does not exist: {parent}"
        )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        dir=str(parent),
        suffix="",
    )
    tmp_path = Path(tmp_name)
    # Wrap ``os.fdopen`` so a failure before its ``with`` body cannot
    # leak the raw fd (audit CR-pass-1 MAJOR + RT-pass-1 NIT-N4). The
    # pattern mirrors CPython's ``tempfile.NamedTemporaryFile``.
    try:
        f = os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    try:
        try:
            f.write(blob)
        finally:
            f.close()
        try:
            os.replace(tmp_path, target)
        except PermissionError as e:
            raise PublishError(
                f"Could not write {target.name}: {e}. On Windows this "
                "usually means another process (antivirus, file indexer) "
                "has the file open."
            ) from e
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _html_escape(text: str) -> str:
    """Minimal HTML-escape for footer-template interpolation.

    Defends against an attacker-controlled ``run_id`` / ``tds_version`` /
    ``signed_at`` from a malicious manifest. The fields are
    short-ASCII in practice but the publish contract should never
    assume.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    """Read+parse ``manifest.json``. Mirrors ``inspect._load_manifest``.

    Duplicated rather than imported so this module remains import-safe
    for the verify-only install (the inspect module pulls jinja2 lazily,
    not at import time, but defence in depth).
    """
    try:
        return json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as e:
        raise PublishError(
            f"manifest.json missing in run_dir {run_dir.name!r}"
        ) from e
    except json.JSONDecodeError as e:
        raise PublishError(
            f"manifest.json in {run_dir.name!r} is not valid JSON: {e}"
        ) from e
