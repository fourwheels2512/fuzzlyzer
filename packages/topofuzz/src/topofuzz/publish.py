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
from topofuzz._verdict_labels import _normalize_verdict_label_for_consumer

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
    target_company = _validate_target_company(target_company)

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


def _validate_target_company(target_company: str | None) -> str | None:
    """NFC + invisible/control-char + length-bound validation.

    Shared by ``build_ceo_card_pdf`` and ``build_consistency_card_pdf``
    so the security posture stays byte-identical across the 1-page PDF
    derivatives. Returns the NFC-normalised string (or ``None`` if input
    was ``None``). Raises :class:`PublishError` on any rejection so the
    operator sees the codepoint that tripped the guard.

    F6 pass-1 RT M3 + F8 reuse: NFC-normalise BEFORE the length cap so a
    decomposed-form ZWJ / combining-mark / fullwidth-digit input cannot
    bypass the visual-width budget. Then reject:

    * C0 (``U+0000-U+001F`` ex. ``\\t``, ``\\n``) and C1 (``U+007F-U+009F``)
      controls — break Chromium text shaping in undefined ways.
    * Zero-width and bidi-format codepoints (``U+200B-U+200F``,
      ``U+202A-U+202E``, ``U+2066-U+2069``) and soft hyphen (``U+00AD``)
      — survive autoescape but cosmetically poison the rendered PDF.
    * Variation selectors (``U+FE00-U+FE0F``, ``U+E0100-U+E01EF``) and
      Unicode tag chars (``U+E0000-U+E007F``) — font-shaping bypass and
      Unicode-tag-attack payload-smuggling vector.
    * Mathematical alphanumeric symbols (``U+1D400-U+1D7FF``) — distinct
      codepoints that NFC does NOT fold; a homoglyph bypass of the
      visual-width budget.
    """
    if target_company is None:
        return None
    if not isinstance(target_company, str):
        raise PublishError(
            f"target_company must be str, got "
            f"{type(target_company).__name__}"
        )
    if not target_company.strip():
        raise PublishError("target_company must be non-empty / non-whitespace")
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
            0x200b <= cp <= 0x200f
            or 0x202a <= cp <= 0x202e
            or 0x2066 <= cp <= 0x2069
            or cp == 0x00ad
            or 0xfe00 <= cp <= 0xfe0f
            or 0xe0100 <= cp <= 0xe01ef
            or 0xe0000 <= cp <= 0xe007f
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
    return target_company


def build_consistency_card_pdf(
    run_dir: Path,
    *,
    out_path: Path,
    target_company: str | None = None,
) -> Path:
    """Render a single-page Cross-Vendor Consistency card to ``out_path``.

    F8 (2026-05-16). Surfaces the procurement-flavoured finding the F6
    CEO card cannot: how many distinct vendors actually backed the
    jury-of-3 audit, and what fraction of jury votes failed. Compliance
    reviewers and procurement leads have a budget line for "cross-vendor
    consistency" but not for "topology fragility under permutation," so
    this card is the cold/warm-pitch hook the F5+F6 PDFs are not.

    Pure replay over ``report.json`` aggregate fields — no LLM spend, no
    ``run.sqlite`` required. The card refuses to render when the source
    run was single-judge (no jury_independence block populated).

    ``target_company`` substitutes into the "what this means for $TARGET"
    paragraph, same as the F6 CEO card. Validation is shared via
    :func:`_validate_target_company`.

    Reuses :func:`load_publish_metadata`, :func:`_format_footer`,
    :func:`_render_html_pages_to_pdfs`, and :func:`_atomic_write_bytes`
    so the attestation footer / Chromium sandbox / atomic-write
    contracts are inherited byte-identically from F5/F6.

    Raises :class:`PublishError` on any failure.
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
    target_company = _validate_target_company(target_company)

    metadata = load_publish_metadata(run_dir)
    footer_text = _format_footer(metadata)

    manifest = _load_manifest(run_dir)
    report_json_path = run_dir / "report.json"
    if not report_json_path.is_file():
        raise PublishError(
            "report.json is missing in run_dir — run "
            "`topofuzz publish` once (or `topofuzz report`) so the "
            "judge-independence block is available."
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
    independence = headline.get("judge_independence") if isinstance(headline, dict) else None
    if not isinstance(independence, dict):
        raise PublishError(
            "report.json has no headline.judge_independence block. The "
            "consistency card requires a jury-of-N source run — single-"
            "judge runs do not produce cross-vendor data."
        )
    # F8 RT pass-1 M1+M2: filter out source runs that do NOT look like
    # jury-of-N. The check raises when ALL three signals are negative
    # (judge_protocol != "jury_of_3" AND jury_of_3 flag not True AND
    # < 2 judges). A producer that wrote ``judge_protocol="jury_of_3"``
    # on a single-judge run will slip past this gate (F8 RT pass-2 R2,
    # safe-degenerate: the downstream ``have_juror_evidence`` predicate
    # in ``_format_consistency_headline`` falls back to ``unknown``
    # severity rather than fabricating a jury-of-N claim). A future
    # pass should tighten to AND-of-both for defense-in-depth.
    judge_protocol = independence.get("judge_protocol")
    judges_list = independence.get("judges")
    n_judges = len(judges_list) if isinstance(judges_list, list) else 0
    cli_args = manifest.get("cli_args") if isinstance(manifest, dict) else None
    cli_jury_flag = (
        cli_args.get("jury_of_3") if isinstance(cli_args, dict) else None
    )
    if judge_protocol != "jury_of_3" and not (
        cli_jury_flag is True or n_judges >= 2
    ):
        raise PublishError(
            "report.json has a judge_independence block but no signal "
            "indicates the source was a jury-of-N — judge_protocol is "
            f"{judge_protocol!r}, manifest.cli_args.jury_of_3 is "
            f"{cli_jury_flag!r}, judges array has {n_judges} entry(s). "
            "The consistency card refuses to render this as "
            "cross-vendor evidence."
        )

    # F8 Stage A: auto-resolve ablation.json so the F4 per-agent
    # attribution headline wins when the source run has it. We do NOT
    # require the file; runs without ablation fall back to the
    # vendor-concentration narrative (regression-safe). Defensive
    # parse: malformed JSON or non-dict root logs a warning and
    # behaves as if the file was absent, rather than refusing the
    # whole card.
    ablation: dict[str, Any] | None = None
    ablation_path = run_dir / "ablation.json"
    if ablation_path.is_file():
        try:
            ablation_payload = json.loads(
                ablation_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as e:
            _log.warning(
                "publish: ablation.json present but unreadable (%s); "
                "falling back to vendor-concentration headline.",
                e,
            )
        else:
            if isinstance(ablation_payload, dict):
                ablation = ablation_payload
            else:
                _log.warning(
                    "publish: ablation.json root is not an object "
                    "(got %s); falling back to vendor-concentration "
                    "headline.",
                    type(ablation_payload).__name__,
                )

    ctx = _build_consistency_card_context(
        metadata=metadata,
        manifest=manifest,
        independence=independence,
        target_company=target_company,
        ablation=ablation,
    )

    html = _render_consistency_card_html(ctx)
    # Mirror the F6 stale-debris sweep so an interrupted Chromium handle
    # on Windows can't accumulate orphans. Narrow prefix so a legitimate
    # operator file `consistency_card_demo.html` isn't swept.
    for stale in run_dir.glob("consistency_card_tmp_*.html"):
        try:
            stale.unlink()
        except OSError:
            _log.debug("publish: could not sweep stale consistency card %s", stale)
    with tempfile.NamedTemporaryFile(
        prefix="consistency_card_tmp_",
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
            _log.warning(
                "publish: could not unlink consistency card scratch file %s",
                tmp_html_path,
            )
    if len(pdf_bytes_list) != 1:
        raise PublishError(
            f"consistency card render produced {len(pdf_bytes_list)} PDFs; "
            "expected exactly 1 page."
        )
    pdf_bytes = pdf_bytes_list[0]
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        n_pages = len(reader.pages)
    except ImportError:
        n_pages = 1
    except Exception as e:  # noqa: BLE001 — refuse the artifact, not crash
        raise PublishError(
            f"could not validate consistency card page count: {e}"
        ) from e
    if n_pages > 2:
        raise PublishError(
            f"consistency card rendered to {n_pages} pages; the F8 spec is "
            "1-2 pages. Inspect the source HTML for runaway content "
            "(usually a long topology_name or jury vendor list)."
        )
    _atomic_write_bytes(out_path, pdf_bytes)
    return out_path


# ---------------------------------------------------------------------------
# F7 — Counterfactual-card PDF (1-2 page derivative of `topofuzz audit`).
# ---------------------------------------------------------------------------


_COUNTERFACTUAL_CARD_NODE_ROW_LIMIT = 8
"""How many rows the by-agent table shows. A 4-role pipeline has ~3
non-terminal roles plus the terminal control; 8 covers the realistic
upper bound while keeping the card to 1-2 paper pages."""

_COUNTERFACTUAL_CARD_SWAP_ROW_LIMIT = 6
"""How many swap-pair rows the drilldown table shows. Six covers a
typical 4-dim audit with 1-2 substitutions per dim; an audit that
generated more will surface its by-dim rates in the dim section."""

_COUNTERFACTUAL_CARD_DIM_DISPLAY_LIMIT = 30
"""Hard cap on rendered dim / swap-token strings. Matches the role-name
cap so a hostile audit.json can't push the card past its page budget."""


def build_counterfactual_card_pdf(
    run_dir: Path,
    *,
    out_path: Path,
    target: str | None = None,
) -> Path:
    """Render a 1-2 page counterfactual-audit card to ``out_path``.

    F7 (2026-05-17). The cold-email enclosure for ``topofuzz audit``:
    surfaces the by-agent attribution headline (which non-terminal node
    contributed most to demographic-counterfactual verdict flips) plus
    the by-swap-pair drilldown and the by-dim flip-rate summary.

    Reads ``<run_dir>/audit.json`` via
    :func:`topofuzz.counterfactual_audit.read_audit_sidecar`. Refuses to
    render when the sidecar is absent (the operator must run
    ``topofuzz audit`` first).

    Pitch-copy guardrail (non-negotiable per F7 spec): the rendered PDF
    says "candidate disparate-impact signal worth a follow-up audit",
    NOT "your model is biased". "Disparate impact" is a legal term of
    art (ECOA Reg-B, FHA, ADEA) that this engine does NOT compute — we
    measure a flip-rate sample and attribute it to upstream agents.

    Reuses :func:`load_publish_metadata`, :func:`_format_footer`,
    :func:`_render_html_pages_to_pdfs`, and :func:`_atomic_write_bytes`
    so the attestation footer / Chromium sandbox / atomic-write
    contracts are inherited byte-identically from F5/F6/F8.

    Raises :class:`PublishError` on any failure.
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
    target = _validate_target_company(target)

    metadata = load_publish_metadata(run_dir)
    footer_text = _format_footer(metadata)

    manifest = _load_manifest(run_dir)

    # Read the audit sidecar. ``read_audit_sidecar`` returns ``None``
    # when the file is absent; raises ``AuditError`` when it is present
    # but unreadable / wrong-typed. Translate both surfaces into
    # ``PublishError`` so the CLI prints a single coherent message.
    #
    # Pass-1 MAJOR M5 fix: ``counterfactual_audit`` is shipped as part of
    # the topofuzz package itself — there is no install extra that gates
    # it. The prior error message ("install the full topofuzz package,
    # not just the verify-only extras") misrouted operators toward a
    # non-existent install fix. Mirror inspect.py's pattern of importing
    # an adjacent topofuzz module inline at the call site (inspect.py
    # imports ablation.py the same way), and rewrite the error to the
    # actually-reachable failure surface: the topofuzz package is
    # corrupt / partially uninstalled.
    try:
        from topofuzz.counterfactual_audit import (
            AUDIT_SIDECAR_FILENAME,
            AuditError,
            read_audit_sidecar,
        )
    except ImportError as e:
        raise PublishError(
            "topofuzz publish --counterfactual-card could not import "
            "topofuzz.counterfactual_audit. The module is shipped with "
            "the topofuzz package itself, so an ImportError here "
            "indicates a corrupt or partial install. Reinstall topofuzz "
            "(`pip install --force-reinstall topofuzz`) before retrying."
        ) from e

    try:
        audit = read_audit_sidecar(run_dir)
    except AuditError as e:
        raise PublishError(f"could not read audit.json: {e}") from e
    if audit is None:
        raise PublishError(
            "audit.json is missing in run_dir — run "
            "`topofuzz audit <run_dir>` once so the counterfactual "
            "card has data to render."
        )

    # Pass-2 BLOCKER B1 fix + Pass-2 MAJOR M3 fix: defensive
    # second-line schema check at the consumer site. ``read_audit_sidecar``
    # is the primary gate (see counterfactual_audit.py); this is
    # belt-and-suspenders so a future code path that bypasses the
    # reader (e.g. an in-memory cached sidecar dict) cannot render a
    # v1-shape sidecar with v2-code semantics.
    from topofuzz.counterfactual_audit import (
        _AUDIT_SIDECAR_SCHEMA_VERSION,
    )
    observed_version = audit.get("schema_version")
    if observed_version != _AUDIT_SIDECAR_SCHEMA_VERSION:
        raise PublishError(
            f"audit.json schema_version is {observed_version!r}, "
            f"expected {_AUDIT_SIDECAR_SCHEMA_VERSION}. Older sidecars "
            "carry different denominator semantics (v1 quoted Phase-2 "
            "flipping triples; v2 quotes Phase-1 attempts) — rerun "
            "`topofuzz audit` against the source run to regenerate."
        )

    # Pass-RT-B1 fix (dual-audit, 2026-05-17): refuse to render a
    # procurement-grade PDF off a dry-run sidecar. Under ``topofuzz
    # audit --dry-run`` every Phase-1 agent call + jury vote is
    # stubbed, jury verdicts pin to 0.5, and per-dim flip rates
    # become an artifact of the source run's canonical scores rather
    # than measured disparate-impact signal. A buyer who screenshots
    # the resulting PDF and emails it to compliance is shipping
    # fiction. Refuse loudly with a clear remediation path.
    if audit.get("dry_run", False):
        raise PublishError(
            f"{AUDIT_SIDECAR_FILENAME} was produced by `topofuzz "
            "audit --dry-run` (every LLM call was stubbed; per-dim "
            "flip rates and per-node attribution are NOT measured "
            "signal). Refusing to render a procurement-grade PDF "
            "off stub data. Re-run `topofuzz audit <run_dir>` "
            "without --dry-run to regenerate the sidecar before "
            "publishing."
        )

    ctx = _build_counterfactual_card_context(
        metadata=metadata,
        manifest=manifest,
        audit=audit,
        target=target,
    )

    html = _render_counterfactual_card_html(ctx)
    # Mirror the F6/F8 stale-debris sweep so an interrupted Chromium
    # handle on Windows can't accumulate orphans. Narrow prefix so a
    # legitimate operator file `counterfactual_card_demo.html` isn't
    # swept.
    for stale in run_dir.glob("counterfactual_card_tmp_*.html"):
        try:
            stale.unlink()
        except OSError:
            _log.debug(
                "publish: could not sweep stale counterfactual card %s",
                stale,
            )
    # Pass-1 BLOCKER B5 fix: pypdf MUST be importable so the page-cap
    # check actually runs. The prior code silently treated an
    # ImportError as ``n_pages = 1`` — a runaway template on a slim
    # install would ship a 10-page PDF to a buyer without tripping
    # the F7 1-2-page hard cap. Fail eagerly with the same wording
    # the consistency-card sibling uses for the same dependency
    # surface (mirrors line 2280 of publish.py for the [full] extra).
    try:
        import pypdf  # noqa: F401 — checked before render to fail fast.
    except ImportError as e:
        raise PublishError(
            "topofuzz publish --counterfactual-card requires the "
            "'full' install (pip install topofuzz[full]) to enforce "
            "the F7 spec's 1-2 page hard cap. Install the full extras "
            "or use the consistency-card PDF, which has the same "
            "dependency."
        ) from e

    # Pass-1 MINOR Mi8 fix: assign tmp_html_path BEFORE fh.write so a
    # mid-write OSError / disk-full surfaces with a clean unlink path
    # in the finally block, rather than leaving a half-written scratch
    # file in the run_dir.
    with tempfile.NamedTemporaryFile(
        prefix="counterfactual_card_tmp_",
        suffix=".html",
        dir=run_dir,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as fh:
        tmp_html_path = Path(fh.name)
        fh.write(html)
    try:
        pdf_bytes_list = _render_html_pages_to_pdfs(
            [tmp_html_path], footer_text
        )
    finally:
        try:
            tmp_html_path.unlink()
        except OSError:
            _log.warning(
                "publish: could not unlink counterfactual card scratch "
                "file %s",
                tmp_html_path,
            )
    if len(pdf_bytes_list) != 1:
        raise PublishError(
            f"counterfactual card render produced {len(pdf_bytes_list)} "
            "PDFs; expected exactly 1."
        )
    pdf_bytes = pdf_bytes_list[0]
    # Pass-1 BLOCKER B5 fix: pypdf is now mandatory (imported above);
    # no fallback ``n_pages = 1`` branch.
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        n_pages = len(reader.pages)
    except Exception as e:  # noqa: BLE001 — refuse the artifact, not crash
        raise PublishError(
            f"could not validate counterfactual card page count: {e}"
        ) from e
    if n_pages > 2:
        raise PublishError(
            f"counterfactual card rendered to {n_pages} pages; the F7 "
            "spec is 1-2 pages. Inspect the source HTML for runaway "
            "content (usually a long topology_name or oversized "
            "by_swap_pair table)."
        )
    _atomic_write_bytes(out_path, pdf_bytes)
    return out_path


def _safe_dim_token(name: str) -> str:
    """NFC-normalize, strip invisible/control codepoints, length-bound.

    Reused for dim names, from_token, and to_token cells so a hostile
    audit.json can't inflate the rendered table past the 1-2 page
    budget. Autoescape catches XSS; this cap catches overflow.

    Pass-1 MINOR Mi2 fix: also NFC-normalize the input and strip the
    same family of invisible/control/homoglyph codepoints that
    :func:`_validate_target_company` rejects. The dim-token cell is
    surfaced inside a PDF table; a hand-edited ``audit.json`` carrying
    U+202E (RLO bidi override) in a ``from_token`` cosmetically poisons
    every row to its right. Unlike the target-company path (which
    raises), this helper sanitizes silently because the dim-token cell
    is rendered as a non-fatal display fragment — refusing the whole
    render over a stray ZWSP is over-correction. The stripped form
    matches what the autoescaped HTML would have rendered if the
    codepoints had been "harmless" — a buyer-readable token without
    the bidi/zero-width payload.
    """
    if not isinstance(name, str):
        return "—"
    import unicodedata
    original_input = name
    name = unicodedata.normalize("NFC", name)
    out_chars: list[str] = []
    for ch in name:
        cp = ord(ch)
        if cp < 0x20 and ch not in ("\t", "\n"):
            # C0 control char — drop silently. Tab/newline tolerated
            # because Jinja autoescape collapses them in table cells.
            continue
        if 0x7f <= cp <= 0x9f:
            # C1 control char — drop silently.
            continue
        if (
            0x200b <= cp <= 0x200f
            or 0x202a <= cp <= 0x202e
            or 0x2066 <= cp <= 0x2069
            or cp == 0x00ad
            or cp == 0x2060  # Pass-2 RT MINOR Mi3 fix: word joiner
            or cp == 0xfeff  # Pass-2 RT MINOR Mi3 fix: BOM / ZWNBSP
            or 0xfe00 <= cp <= 0xfe0f
            or 0xe0100 <= cp <= 0xe01ef
            or 0xe0000 <= cp <= 0xe007f
            or 0x1d400 <= cp <= 0x1d7ff
        ):
            # Invisible / bidi-format / variation-selector / Unicode-
            # tag / mathematical-alphanumeric homoglyph — drop silently.
            continue
        out_chars.append(ch)
    sanitized = "".join(out_chars)
    # Pass-2 RT MAJOR M2 fix: emit a forensics warning whenever
    # sanitization actually MODIFIED the input — defensible silent
    # stripping for stray ZWSPs becomes irresponsible silence for
    # U+202E (RLO bidi override) and the rest of the homoglyph family,
    # which are malicious-input signals. One warning per call only
    # (not per token codepoint) — sufficient for forensics, doesn't
    # spam the log. Inputs[:50] truncation protects the log line
    # itself from a hostile multi-kB token.
    if sanitized != original_input:
        n_stripped = len(original_input) - len(sanitized)
        _log.warning(
            "publish: stripped %d invisible/bidi codepoints from dim "
            "token %r → %r",
            n_stripped,
            original_input[:50],
            sanitized[:50],
        )
    if not sanitized:
        # Token was 100% invisible/control payload — surface a sentinel
        # so the rendered cell isn't blank.
        return "—"
    if len(sanitized) > _COUNTERFACTUAL_CARD_DIM_DISPLAY_LIMIT:
        return sanitized[: _COUNTERFACTUAL_CARD_DIM_DISPLAY_LIMIT - 1] + "…"
    return sanitized


def _extract_by_node_rows(
    audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Pull the audit-wide ``by_node`` table → list of dicts for the template.

    Returns ``(non_terminal_rows_plus_optional_terminal, terminal_row)``.
    Non-terminal rows are sorted by ``attribution_pct`` descending;
    ties on name ascending for byte-determinism. The terminal row (if
    present) is appended LAST and styled as the experimental control —
    its attribution_pct is always 100% by construction
    (``substitute_terminal`` ablation kind), so labelling it as a
    control prevents a buyer from mis-reading it as "the terminal node
    is the bias source".
    """
    by_node = audit.get("by_node")
    if not isinstance(by_node, dict):
        return [], None
    non_terminal: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    for name, payload in by_node.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        pct = payload.get("attribution_pct")
        if isinstance(pct, bool) or not isinstance(pct, (int, float)):
            continue
        if pct != pct:  # NaN
            continue
        if pct > 100:
            continue
        # Pass-2 RT MAJOR-3 fold (folded from 2026-05-18 dual-audit
        # pass 2): the swap-pair extractor now warns on negative
        # attribution_pct; mirror that posture here so a producer
        # regression in the audit-wide by_node table surfaces in
        # stderr / CI logs rather than disappearing into a silent
        # drop. ``_extract_by_node_rows`` feeds the F7 headline picker,
        # so a negative-pct producer bug that should have surfaced as
        # a row could otherwise cause ``any_nonterminal_attribution``
        # to silently flip to False and route the headline to the
        # inconclusive branch when it shouldn't have.
        if pct < 0:
            _log.warning(
                "publish: by_node[%s] has negative attribution_pct=%r; "
                "dropping (producers should never emit negatives).",
                name, pct,
            )
            continue
        flips_closed = payload.get("flips_closed")
        total_meaningful = payload.get("total_meaningful")
        if (
            isinstance(total_meaningful, int)
            and not isinstance(total_meaningful, bool)
            and total_meaningful == 0
        ):
            continue
        is_terminal = payload.get("is_terminal") is True
        row = {
            "name": _safe_dim_token(name),
            "attribution_pct": float(pct),
            "attribution_pct_str": f"{float(pct):.1f}%",
            "flips_closed": (
                flips_closed
                if isinstance(flips_closed, int)
                and not isinstance(flips_closed, bool)
                else None
            ),
            "total_meaningful": (
                total_meaningful
                if isinstance(total_meaningful, int)
                and not isinstance(total_meaningful, bool)
                else None
            ),
            "is_terminal": is_terminal,
        }
        if is_terminal:
            # Only keep ONE terminal row even if a malformed sidecar
            # contains multiple — the buyer-facing card has a single
            # control slot.
            if terminal is None:
                terminal = row
        else:
            non_terminal.append(row)
    non_terminal.sort(key=lambda r: (-r["attribution_pct"], r["name"]))
    # Cap the non-terminal list, then append the terminal as a labelled
    # control row at the bottom. Truncation count tracked separately so
    # the template can surface "+N more" if needed in a later iteration.
    rows = non_terminal[: _COUNTERFACTUAL_CARD_NODE_ROW_LIMIT]
    if terminal is not None:
        rows.append(terminal)
    return rows, terminal


def _extract_swap_pair_rows(
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pull the ``by_swap_pair`` table → flat list of rows for the template.

    Sorted by ``flipping_pairs`` descending so the most-impactful
    substitution leads. Top-2 contributing nodes per row are rendered
    inline as ``"role 64% · role 55%"``.

    Pass-1 MINOR Mi5 fix: the audit-wide ``by_node`` table is the
    source of truth for ``is_terminal`` (set on the per-node slot at
    aggregation time). We pre-extract the terminal node IDs from the
    audit-wide table and filter them out of the per-swap-pair
    contributors. The prior comment claimed the filter caught
    ``substitute_terminal`` rows via ``tm == 0``, but the terminal
    control has ``tm > 0`` (it's the experimental baseline) and
    ``pct == 100``, so without this lookup it leaked into the "Top
    contributors" cell and crowded out actionable upstream agents.
    """
    by_sp = audit.get("by_swap_pair")
    if not isinstance(by_sp, dict):
        return []
    # Pass-1 MINOR Mi5 fix: collect terminal node ids from the audit-
    # wide by_node table (single authoritative source).
    terminal_node_ids: set[str] = set()
    audit_by_node = audit.get("by_node")
    if isinstance(audit_by_node, dict):
        for nid, slot in audit_by_node.items():
            if (
                isinstance(nid, str)
                and isinstance(slot, dict)
                and slot.get("is_terminal") is True
            ):
                terminal_node_ids.add(nid)
    rows: list[dict[str, Any]] = []
    for flat_key, payload in by_sp.items():
        if not isinstance(payload, dict):
            continue
        dim = payload.get("dim") or ""
        from_token = payload.get("from_token") or ""
        to_token = payload.get("to_token") or ""
        flipping_pairs = payload.get("flipping_pairs")
        total_pairs = payload.get("total_pairs")
        if (
            isinstance(flipping_pairs, bool)
            or not isinstance(flipping_pairs, int)
        ):
            flipping_pairs = 0
        if (
            isinstance(total_pairs, bool)
            or not isinstance(total_pairs, int)
        ):
            total_pairs = 0
        # Top-2 contributing non-terminal nodes for this swap-pair.
        per_node = payload.get("by_node")
        contrib_rows: list[tuple[str, float]] = []
        if isinstance(per_node, dict):
            for nid, pn in per_node.items():
                if not isinstance(nid, str) or not isinstance(pn, dict):
                    continue
                pct = pn.get("attribution_pct")
                if isinstance(pct, bool) or not isinstance(pct, (int, float)):
                    continue
                if pct != pct or pct > 100:  # NaN or out-of-range high
                    continue
                # Pass-1 MINOR Mi3 fix (folded from 2026-05-18 dual-
                # audit): emit a WARNING on negative attribution_pct
                # before dropping, matching the existing
                # ``negative flipping_pairs`` warning pattern at
                # publish.py ~line 1572. Producers should never emit
                # negative attribution_pct; logging at the dropped-row
                # boundary lets a future producer regression surface
                # in stderr / CI logs rather than disappearing into a
                # silent drop.
                if pct < 0:
                    _log.warning(
                        "publish: by_swap_pair[%s].by_node[%s] has "
                        "negative attribution_pct=%r; dropping from "
                        "drilldown (producers should never emit "
                        "negatives).",
                        flat_key, nid, pct,
                    )
                    continue
                # Pass-1 MINOR Mi5 fix: cross-reference audit-wide
                # by_node to filter the experimental control out of
                # the drilldown. tm==0 alone was insufficient.
                if nid in terminal_node_ids:
                    continue
                tm = pn.get("total_meaningful")
                if (
                    isinstance(tm, int)
                    and not isinstance(tm, bool)
                    and tm == 0
                ):
                    continue
                contrib_rows.append((nid, float(pct)))
        contrib_rows.sort(key=lambda x: (-x[1], x[0]))
        # 2026-05-18 fix: drop 0% rows from the "top contributors"
        # cell. A drilldown row of "intake 0% · risk 0%" reads to a
        # buyer as "these agents contributed" when the truth is "no
        # agent's removal closed any flip on this swap-pair". The
        # by-agent attribution table on the same page still shows
        # the same 0/0 numbers; this surface is for the buyer's "who
        # do I look at first" eye, so it should be empty (the row's
        # "—" sentinel) when no agent has a positive signal.
        contrib_rows = [(n, p) for n, p in contrib_rows if p > 0.0]
        top_strs = [
            f"{_safe_dim_token(n)} {pct:.0f}%"
            for n, pct in contrib_rows[:2]
        ]
        top_contributors = " · ".join(top_strs)
        rows.append({
            "dim": _safe_dim_token(str(dim)),
            "from_token": _safe_dim_token(str(from_token)),
            "to_token": _safe_dim_token(str(to_token)),
            "flipping_pairs": flipping_pairs,
            "total_pairs": total_pairs,
            "top_contributors": top_contributors,
        })
    rows.sort(
        key=lambda r: (-r["flipping_pairs"], r["dim"], r["from_token"])
    )
    return rows[:_COUNTERFACTUAL_CARD_SWAP_ROW_LIMIT]


def _extract_by_dim_rows(
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pull the ``by_dim`` table → one row per audited dim."""
    by_dim = audit.get("by_dim")
    if not isinstance(by_dim, dict):
        return []
    rows: list[dict[str, Any]] = []
    for dim, payload in by_dim.items():
        if not isinstance(dim, str) or not isinstance(payload, dict):
            continue
        flipping_pairs = payload.get("flipping_pairs")
        total_pairs = payload.get("total_pairs")
        flip_rate = payload.get("flip_rate_pct")
        if (
            isinstance(flipping_pairs, bool)
            or not isinstance(flipping_pairs, int)
        ):
            flipping_pairs = 0
        if (
            isinstance(total_pairs, bool)
            or not isinstance(total_pairs, int)
        ):
            total_pairs = 0
        if (
            isinstance(flip_rate, bool)
            or not isinstance(flip_rate, (int, float))
        ):
            flip_rate_str = "—"
        elif flip_rate != flip_rate:  # NaN
            flip_rate_str = "—"
        else:
            flip_rate_str = f"{float(flip_rate):.1f}%"
        rows.append({
            "dim": _safe_dim_token(dim),
            "flipping_pairs": flipping_pairs,
            "total_pairs": total_pairs,
            "flip_rate_pct_str": flip_rate_str,
        })
    # Stable order: by flipping_pairs descending, ties on dim ascending.
    rows.sort(key=lambda r: (-r["flipping_pairs"], r["dim"]))
    return rows


def _format_counterfactual_headline(
    *,
    by_node_rows: list[dict[str, Any]],
    swap_pair_rows: list[dict[str, Any]],
    n_flipping_pairs: int,
    n_canonical_pairs: int,
    topology_name: str,
    partial: bool,
    n_ablation_points: int = 0,
    n_ablation_points_meaningful: int = 0,
) -> tuple[str, str, str, str, str]:
    """Pick the headline tuple (title, lede, stat, stat_label, sub_line).

    Leads with the BY-AGENT attribution finding (the F7 wedge versus
    LangFair / MALIBU) — NOT with the raw flip count. When attribution
    data is unavailable, falls back to a flip-rate-led headline so the
    card still renders cleanly on a degenerate / empty audit.

    2026-05-18 fix: a top non-terminal agent at ``attribution_pct == 0%``
    must NOT lead the headline as "agent X contributed most" — that
    sentence is a buyer-visible contradiction with the 0% stat. The
    first-branch gate now requires ``attribution_pct > 0`` to fire;
    the all-zero subcase falls through to the inconclusive branch with
    a lede that surfaces the inconclusive-rejudge rate when present
    (the most common root cause: every Phase-2 leave-one-out re-judge
    returned an unparseable verdict, e.g. ``Score: NONE`` on a HOLD
    response, so attribution math has no usable evidence).

    Caveat (non-negotiable): no branch says "your model is biased".
    Copy describes signal magnitudes worth a follow-up audit.
    """
    # The first non-terminal row is the top attributable agent.
    top_agent: dict[str, Any] | None = None
    for r in by_node_rows:
        if not r.get("is_terminal"):
            top_agent = r
            break

    # The first swap-pair row is the most-flipping substitution.
    top_swap = swap_pair_rows[0] if swap_pair_rows else None

    # 2026-05-18 fix: did any non-terminal agent close any flips at
    # all? If every non-terminal row reports 0 attribution_pct, the
    # leave-one-out pass produced NO usable evidence for which upstream
    # agent drove the flip — usually because every Phase-2 re-judge
    # returned an unparseable verdict (e.g. all 3 jury vendors emitted
    # ``Score: NONE`` on a HOLD response). The card MUST NOT name a
    # specific agent in that state.
    any_nonterminal_attribution = any(
        (not r.get("is_terminal"))
        and isinstance(r.get("attribution_pct"), (int, float))
        and not isinstance(r.get("attribution_pct"), bool)
        and float(r["attribution_pct"]) > 0.0
        for r in by_node_rows
    )

    # Pass-1 MAJOR M3 fix: n_canonical_pairs == 0 means no demographic
    # markers were found in the eval set, so the generators returned
    # zero swaps. The prior "held its verdict under demographic
    # counterfactual substitutions" headline was a stability claim
    # against a measurement that NEVER RAN — actively misleading. Lead
    # with the no-op explanation so the operator's eye lands on
    # "nothing measured" instead of "nothing flipped".
    if n_canonical_pairs == 0:
        title = (
            "No counterfactual measurements were possible on this run."
        )
        lede = (
            f"None of the canonical eval rows in {topology_name} "
            "contained demographic markers our counterfactual "
            "generators recognise (Bertrand-Mullainathan names, "
            "metro-matched ZIPs, age phrases, or gendered pronouns). "
            "This card cannot certify a stability or fairness claim. "
            "Add demographic-marker rows to the eval set or run with "
            "`--counterfactual <dim>` filtered to dims whose markers "
            "actually appear in your input. The headline number "
            "below is therefore zero by construction, not a measured "
            "stability finding — see the caveat box."
        )
        stat_headline = "0"
        stat_label = (
            "ablation-eligible verdict flips under demographic "
            "substitution (no markers found in eval)"
        )
        sub_parts = ["0 pairs audited (no markers in eval)"]
        if partial:
            sub_parts.append("audit run partial")
        sub_line = " · ".join(sub_parts)
        return title, lede, stat_headline, stat_label, sub_line

    # 2026-05-18 fix: REQUIRE top_agent attribution_pct > 0 before
    # naming a specific agent in the headline. Without this gate, an
    # audit where every non-terminal node closed 0 flips still selects
    # ``by_node_rows[0]`` (sort is stable on attribution_pct desc,
    # ties on name asc — so the alphabetically-first node wins on a
    # full-zero tie) and renders "Agent X contributed most …" over a
    # 0.0% stat panel. That contradiction was the buyer-PDF failure
    # surfaced on the 2026-05-17 partial-Phase-2 lending bias run.
    # Pass-1 MINOR Mi1 fix (folded from 2026-05-18 dual-audit):
    # ``_extract_by_node_rows`` guarantees ``attribution_pct`` is a
    # non-NaN float in [0, 100] on every row it returns (see lines
    # 1141-1144 + 1158 in this file). The previous draft fold used
    # ``... or 0.0`` to defensively coerce a hypothetical ``None``,
    # but that short-circuit is dead code today AND silently coerces
    # ``None`` → 0.0 if the extractor's contract ever weakens — quiet
    # in exactly the way the 2026-05-17 buyer-PDF bug was quiet. Drop
    # the ``or 0.0`` so a contract violation surfaces as TypeError
    # rather than as a silent zero.
    if (
        top_agent
        and top_swap
        and n_flipping_pairs > 0
        and any_nonterminal_attribution
        and float(top_agent.get("attribution_pct", 0.0)) > 0.0
    ):
        top_name = top_agent["name"]
        # Pass-1 MINOR Mi3 fix: rename ``top_pct`` → ``top_pct_str``.
        # The field carries the pre-formatted "%.1f%%" string (e.g.
        # ``"64.0%"``); the bare name suggested a float and invited
        # arithmetic comparison. Use ``top_agent["attribution_pct"]``
        # if the float form is needed.
        top_pct_str = top_agent["attribution_pct_str"]
        flipping_pairs = top_swap["flipping_pairs"]
        total_pairs = top_swap["total_pairs"]
        from_token = top_swap["from_token"]
        to_token = top_swap["to_token"]
        title = (
            f"The {top_name} agent contributed most to "
            "demographic-counterfactual verdict flips on "
            f"{topology_name}."
        )
        # Pass-1 BLOCKER B1 fix: the denominator on the swap-pair row
        # (``total_pairs``) is now the count of Phase-1 ATTEMPTS for
        # that substitution, not a structurally-100% denominator.
        # Reword the lede so it reads as a measurement against
        # canonical attempts, not as "N/M times" with a hidden
        # denominator.
        lede = (
            f"Swapping {from_token} → {to_token} flipped the verdict "
            f"{flipping_pairs} of {total_pairs} canonical attempts in "
            "this audit; leave-one-out ablation traced "
            f"{top_pct_str} of audit-wide flip-closure back to the "
            f"{top_name} role. The by-agent table below shows the "
            "full attribution distribution — scope a follow-up "
            "audit against the top contributors before treating the "
            "headline as actionable."
        )
        stat_headline = top_pct_str
        stat_label = (
            f"of audit-wide ablation-eligible flips closed by removing "
            f"the {top_name} agent"
        )
        sub_parts = [f"{n_flipping_pairs} flipping pairs measured"]
        if partial:
            sub_parts.append("audit run partial")
        sub_line = " · ".join(sub_parts)
        return title, lede, stat_headline, stat_label, sub_line

    if top_swap and n_flipping_pairs > 0:
        # No usable by_node attribution. Two subcases:
        #   (a) every non-terminal row had attribution_pct == 0%
        #       (the post-2026-05-18 all-zero subcase — usually
        #       every Phase-2 re-judge collapsed to unparseable);
        #   (b) all meaningful results landed on the terminal node
        #       (the original fall-through case before the all-zero
        #       gate was added).
        # Either way: surface the flip-pair as the headline metric
        # and keep the by-agent framing in the lede.
        flipping_pairs = top_swap["flipping_pairs"]
        total_pairs = top_swap["total_pairs"]
        from_token = top_swap["from_token"]
        to_token = top_swap["to_token"]
        title = (
            f"Demographic counterfactuals flipped {topology_name}'s "
            "verdict, but the leave-one-out pass did not isolate a "
            "single upstream agent."
        )
        # 2026-05-18 fix (dual-audit pass-1 BLOCKER + MEDIUM M3 folded):
        #
        # The all-zero attribution branch must distinguish THREE
        # subcases to be honest in the lede a buyer / opposing-counsel
        # could cite:
        #
        # (1) ``nap > napm`` (some ablation re-judges did not produce a
        #     usable verdict — n_ablation_points - n_ablation_points_meaningful):
        #     attribution math has fewer signals than the per-swap-pair
        #     flip count suggests; rerun would recover the missing
        #     signal.
        # (2) ``nap == napm > 0`` (every ablation re-judge WAS parseable
        #     AND every non-terminal attribution_pct == 0): the math
        #     ran cleanly and concluded no single non-terminal role's
        #     removal closes any flip — responsibility likely sits on
        #     the terminal node, is distributed below the per-role
        #     detection floor, or genuinely cannot be attributed by
        #     leave-one-out. NOT a tooling failure — a legitimate
        #     finding. Buyer should NOT rerun chasing a wrong root
        #     cause.
        # (3) ``nap == 0`` (no ablation re-judges recorded at all,
        #     e.g. Phase 2 cap-hit before any triple completed): the
        #     attribution machinery never ran — different from (1) and
        #     (2).
        #
        # The previous draft fold (the BLOCKER pass-1 found) overloaded
        # all three into a single "unparseable verdict (most often a
        # 'HOLD'-style response)" clause. That is a deposition-citation
        # risk: ``audit.json.results[*].kind`` carries FOUR distinct
        # non-meaningful kinds (``failed``, ``skip_no_canonical_output``,
        # ``skip_no_counterfactual_output``, ``skip_downstream`` — see
        # counterfactual_audit._KIND_*); only ``failed`` is an
        # "unparseable verdict". The other three are skip-with-no-LLM-
        # call states (canonical / CF / downstream output absent). A
        # buyer-PDF that asserts "unparseable verdict" when the truth
        # is "skipped because canonical output was missing" can be
        # picked apart by a lawyer pulling the sidecar JSON.
        inconclusive_clause = ""
        n_inconclusive = 0
        nap_napm_well_formed = (
            isinstance(n_ablation_points, int)
            and isinstance(n_ablation_points_meaningful, int)
            and n_ablation_points >= 0
            and n_ablation_points_meaningful >= 0
            and n_ablation_points >= n_ablation_points_meaningful
        )
        if nap_napm_well_formed and n_ablation_points > 0:
            n_inconclusive = n_ablation_points - n_ablation_points_meaningful
            if n_inconclusive > 0:
                inconclusive_clause = (
                    f" {n_inconclusive} of {n_ablation_points} "
                    "ablation re-judges did not produce a usable "
                    "verdict (some skipped because canonical / "
                    "counterfactual / downstream outputs never "
                    "reached the verdict node; some returned text "
                    "the jury could not score as approve/deny); the "
                    "attribution math therefore has fewer signals "
                    "than the per-swap-pair flip count suggests."
                )
        if not any_nonterminal_attribution and n_flipping_pairs > 0:
            if n_inconclusive > 0:
                # Subcase (1): math failed on some cells.
                attribution_clause = (
                    "By-agent attribution math could not isolate a "
                    "single non-terminal role: every Phase-2 cell that "
                    "DID produce a usable verdict left the flip open."
                )
            elif (
                nap_napm_well_formed
                and n_ablation_points > 0
                and n_ablation_points_meaningful > 0
            ):
                # Subcase (2): math ran cleanly on every cell and
                # produced a zero answer. This is a legitimate finding,
                # NOT a tooling failure — the buyer should not rerun.
                #
                # Pass-2 RT MAJOR-2 fold (folded from 2026-05-18
                # dual-audit pass 2): the previous draft asserted
                # "responsibility likely sits on the terminal node"
                # without inspecting whether the terminal row has
                # positive attribution. When the terminal control row
                # ALSO reports 0% (entirely non-attributable: every
                # leave-one-out pivot left the flip open), the
                # "terminal node" claim is unverifiable from
                # ``audit.json.by_node`` and is a deposition-citation
                # risk. Only surface the terminal-node hypothesis when
                # a terminal row actually exists in by_node_rows with
                # a positive attribution_pct; otherwise hedge cleanly
                # on the remaining hypotheses ("distributed below the
                # detection floor" / "non-attributable by leave-one-out
                # on this fixture").
                terminal_has_positive = any(
                    r.get("is_terminal") is True
                    and isinstance(r.get("attribution_pct"), (int, float))
                    and not isinstance(r.get("attribution_pct"), bool)
                    and float(r["attribution_pct"]) > 0.0
                    for r in by_node_rows
                )
                if terminal_has_positive:
                    attribution_clause = (
                        "By-agent attribution ran on every ablation "
                        "re-judge in this audit and no non-terminal "
                        "role's removal closed any measured flip — "
                        "responsibility likely sits on the terminal "
                        "node, is distributed across roles below the "
                        "per-role detection floor, or cannot be "
                        "cleanly attributed by leave-one-out on this "
                        "fixture."
                    )
                else:
                    attribution_clause = (
                        "By-agent attribution ran on every ablation "
                        "re-judge in this audit and no non-terminal "
                        "role's removal closed any measured flip — "
                        "responsibility is likely distributed across "
                        "roles below the per-role detection floor, or "
                        "cannot be cleanly attributed by leave-one-out "
                        "on this fixture."
                    )
            else:
                # Subcase (3): nap == 0 — the attribution machinery
                # never ran (most often Phase 2 cap-hit before any
                # triple completed).
                attribution_clause = (
                    "By-agent attribution did not run on this audit "
                    "(no ablation re-judges recorded — most often "
                    "because Phase 2 hit `--max-cost-usd` before any "
                    "triple completed)."
                )
        else:
            attribution_clause = (
                "By-agent attribution was inconclusive (no "
                "non-terminal role closed enough flips to lead the "
                "ranking)."
            )
        # Pass-1 BLOCKER B1 fix: denominator wording per above.
        lede = (
            f"Swapping {from_token} → {to_token} flipped the verdict "
            f"{flipping_pairs} of {total_pairs} canonical attempts "
            "across the audit. "
            f"{attribution_clause}"
            f"{inconclusive_clause} The drilldown table below scopes "
            "the substitutions that did the most damage."
        )
        stat_headline = f"{flipping_pairs}/{total_pairs}"
        stat_label = (
            f"verdict flips on the {from_token} → {to_token} "
            "substitution"
        )
        sub_parts = [f"{n_flipping_pairs} flipping pairs measured"]
        if partial:
            sub_parts.append("audit run partial")
        sub_line = " · ".join(sub_parts)
        return title, lede, stat_headline, stat_label, sub_line

    # n_canonical_pairs > 0 but zero flips — the audit ran AND the
    # pipeline held its verdict under demographic substitution. This
    # is the legitimate "stable under counterfactual" branch.
    title = (
        f"This {topology_name} pipeline held its verdict under "
        "demographic counterfactual substitutions."
    )
    lede = (
        f"{n_canonical_pairs} canonical / counterfactual pairs were "
        "audited and none produced an ablation-eligible verdict flip. "
        "Treat this as a preliminary cross-demographic-stability "
        "signal, not a disparate-impact certification — see the "
        "caveat box below."
    )
    stat_headline = "0"
    stat_label = (
        "ablation-eligible verdict flips under demographic "
        "substitution"
    )
    sub_parts = [f"{n_canonical_pairs} pairs audited"]
    if partial:
        sub_parts.append("audit run partial")
    sub_line = " · ".join(sub_parts)
    return title, lede, stat_headline, stat_label, sub_line


def _build_counterfactual_card_context(
    *,
    metadata: dict[str, str],
    manifest: dict[str, Any],
    audit: dict[str, Any],
    target: str | None,
) -> dict[str, Any]:
    """Assemble the Jinja context for counterfactual_card.html.j2.

    Byte-deterministic given (metadata, manifest, audit, target). Tests
    pin the rendered context without spinning up Chromium.
    """
    topology_name = _truncate_topology_name(
        str(manifest.get("topology_name") or "(unknown)")
    )

    n_canonical_pairs = audit.get("n_canonical_pairs")
    n_flipping_pairs = audit.get("n_flipping_pairs")
    if (
        isinstance(n_canonical_pairs, bool)
        or not isinstance(n_canonical_pairs, int)
    ):
        n_canonical_pairs = 0
    if (
        isinstance(n_flipping_pairs, bool)
        or not isinstance(n_flipping_pairs, int)
    ):
        n_flipping_pairs = 0

    # F7 BLOCKER guard (belt-and-suspenders): refuse to render the
    # buyer-facing PDF when the audit's top-level flip counter says
    # "flips happened" but every per-swap-pair entry reports zero.
    # Pre-fix audits cap-hit during Phase 2 produced exactly this
    # state — the audit-side fix prevents new sidecars from carrying
    # it, but stale sidecars on disk would still render a
    # contradictory PDF without this guard. Narrow on purpose:
    # ``n != sum(...)`` would catch benign test fixtures with looser
    # fidelity; ``n > 0 AND sum == 0`` catches the exact misleading
    # state and nothing else.
    #
    # Pass-1 RT MAJOR fix: when the inconsistency is caused by
    # malformed by_swap_pair entries (string-typed / missing /
    # negative ``flipping_pairs``) the diagnostic must point at the
    # MALFORMED SIDECAR, not at the cap-hit story — otherwise the
    # buyer reruns a multi-dollar audit chasing the wrong root
    # cause. Pass-1 RT MINOR fix: negative ints don't silently drop;
    # they emit a WARNING (producers should never write negatives)
    # and count toward the malformed tally.
    if n_flipping_pairs > 0:
        by_swap_pair_raw = audit.get("by_swap_pair")
        sp_flip_sum = 0
        sp_entries_total = 0
        sp_entries_malformed = 0
        if isinstance(by_swap_pair_raw, dict):
            for payload in by_swap_pair_raw.values():
                # Pass-2 RT FINDING N1 fix: non-dict payloads count
                # toward the total + malformed tallies so a sidecar
                # whose entries are mostly garbage strings / nulls
                # routes to the malformed-sidecar diagnostic instead
                # of being skipped and dropping into the cap-hit
                # branch (which would waste the buyer's audit budget
                # on a wrong root cause).
                sp_entries_total += 1
                if not isinstance(payload, dict):
                    sp_entries_malformed += 1
                    continue
                fp = payload.get("flipping_pairs")
                if isinstance(fp, bool) or not isinstance(fp, int):
                    sp_entries_malformed += 1
                    continue
                if fp < 0:
                    _log.warning(
                        "publish: by_swap_pair entry has negative "
                        "flipping_pairs=%r; treating as malformed "
                        "(producers should never emit negatives).",
                        fp,
                    )
                    sp_entries_malformed += 1
                    continue
                sp_flip_sum += fp
        # Pass-2 RT FINDING N3 fix: a sidecar that mixes valid
        # positives with stray malformed entries (e.g. one negative
        # in a 10-entry table) raises a DEDICATED PublishError rather
        # than silently dropping the malformed entries from
        # ``sp_flip_sum``. A _log.warning to stderr is invisible to a
        # buyer reading the rendered PDF, so the malformed signal has
        # to surface as a hard render-stop with an actionable message
        # naming the count of dropped entries.
        if sp_entries_malformed > 0 and sp_flip_sum > 0:
            raise PublishError(
                f"audit.json by_swap_pair has "
                f"{sp_entries_malformed} of {sp_entries_total} "
                "entries with a malformed flipping_pairs field "
                "(missing, non-int, negative, or non-dict payload). "
                "Rendering would silently drop these from the "
                "drilldown total — refusing to publish. Rerun "
                "`topofuzz audit` against the source run to "
                "regenerate a well-formed sidecar."
            )
        if sp_flip_sum == 0:
            if (
                sp_entries_total > 0
                and sp_entries_malformed == sp_entries_total
            ):
                raise PublishError(
                    f"audit.json reports n_flipping_pairs={n_flipping_pairs} "
                    f"but every by_swap_pair entry "
                    f"({sp_entries_malformed} / {sp_entries_total}) has a "
                    "malformed flipping_pairs field (missing, non-int, "
                    "negative, or non-dict payload). The sidecar was "
                    "produced by an incompatible writer or hand-edited. "
                    "Rerun `topofuzz audit` against the source run to "
                    "regenerate a well-formed sidecar — DO NOT raise "
                    "the cost cap; the existing run is structurally "
                    "unreadable, not under-budgeted."
                )
            raise PublishError(
                f"audit.json reports n_flipping_pairs={n_flipping_pairs} "
                "but every by_swap_pair[*].flipping_pairs is zero — the "
                "card cannot honestly lead with a 'counterfactuals "
                "flipped' headline against an all-zero drilldown. This "
                "usually means the audit was hard-stopped mid-Phase-2 "
                "(see --max-cost-usd) under a pre-fix engine; rerun "
                "`topofuzz audit` with a higher cap to regenerate the "
                "sidecar."
            )

    total_cost_usd = audit.get("total_cost_usd")
    if isinstance(total_cost_usd, (int, float)) and not isinstance(
        total_cost_usd, bool
    ):
        cost_str = f"${float(total_cost_usd):.2f}"
    else:
        cost_str = "—"

    partial = audit.get("partial") is True

    # Pass-1 BLOCKER B4 fix: surface the unpriced-call footnote when
    # any re-execution calls hit a model missing from the cost rate
    # table. Without this the buyer reads "$0.42" as the audit's real
    # spend even though billing excludes unpriced calls.
    # Pass-2 RT MAJOR M3 fix: changed wording from "lower bound" →
    # "EXCLUDES UNPRICED CALLS — billing could be ~5x higher" because
    # Phase-2 verdict-node fallback ($0.05) sits in the middle of a
    # $0.03-$0.10 premium-model band; a compliance buyer reading the
    # PDF in deposition deserves the stronger framing.
    unpriced_count = audit.get("unpriced_call_count")
    unpriced_models_field = audit.get("unpriced_models")
    if (
        isinstance(unpriced_count, int)
        and not isinstance(unpriced_count, bool)
        and unpriced_count > 0
    ):
        models_repr: list[str] = []
        if isinstance(unpriced_models_field, list):
            for m in unpriced_models_field:
                if isinstance(m, str) and m:
                    models_repr.append(m)
        if models_repr:
            models_str = ", ".join(models_repr)
        else:
            models_str = "unknown vendor model(s)"
        unpriced_note = (
            f"total_cost_usd EXCLUDES UNPRICED CALLS — {unpriced_count} "
            f"re-execution call(s) used models missing from the cost "
            f"rate table ({models_str}); actual vendor billing could be "
            f"~5× higher on premium verdict-node models."
        )
    else:
        unpriced_note = ""

    by_node_rows, terminal_row = _extract_by_node_rows(audit)
    swap_pair_rows = _extract_swap_pair_rows(audit)
    by_dim_rows = _extract_by_dim_rows(audit)

    # 2026-05-18 fix: plumb the inconclusive-rejudge accounting into
    # the headline picker so the all-zero attribution branch can
    # surface WHY attribution is inconclusive. Both fields are
    # already in audit.json (counterfactual_audit._write_audit_sidecar
    # wrote ``n_ablation_points`` + ``n_ablation_points_meaningful``
    # before 2026-05-18); read them defensively.
    #
    # Pass-2 RT MEDIUM-2 fold (folded from 2026-05-18 dual-audit
    # pass 2): a missing field on an older sidecar degrades to 0
    # (legitimate back-compat — pre-2026-05-18 producers didn't write
    # them). But malformed types (string, float, dict, negative int)
    # are PRODUCER BUGS and must NOT be silently coerced to 0 — that
    # would let the subcase-3 lede ("attribution did not run; most
    # often Phase 2 hit max-cost-usd") lie about the cause when the
    # real cause is sidecar corruption. Match the existing
    # ``napm > nap`` posture: corruption → render-stop with an
    # actionable message.
    raw_nap = audit.get("n_ablation_points")
    raw_napm = audit.get("n_ablation_points_meaningful")

    def _read_ablation_count(raw: Any, field_name: str) -> int:
        if raw is None:
            # Legitimate back-compat path for sidecars produced by a
            # pre-2026-05-18 writer that never set the field.
            return 0
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise PublishError(
                f"audit.json[{field_name!r}] is {type(raw).__name__} "
                f"({raw!r}); expected a non-negative int (or missing for "
                "legacy sidecars). The sidecar was hand-edited or "
                "produced by an incompatible writer — refusing to "
                "publish. Rerun `topofuzz audit` against the source "
                "run to regenerate a well-formed sidecar."
            )
        if raw < 0:
            raise PublishError(
                f"audit.json[{field_name!r}]={raw} is negative; ablation "
                "counters are structurally non-negative. The sidecar "
                "was hand-edited or produced by an incompatible writer "
                "— refusing to publish. Rerun `topofuzz audit` against "
                "the source run to regenerate a well-formed sidecar."
            )
        return raw

    nap = _read_ablation_count(raw_nap, "n_ablation_points")
    napm = _read_ablation_count(
        raw_napm, "n_ablation_points_meaningful"
    )
    if napm > nap:
        # Pass-1 RT MAJOR fix (folded from 2026-05-18 dual-audit):
        # n_ablation_points_meaningful > n_ablation_points is
        # structurally impossible from a well-formed writer — the
        # producer at counterfactual_audit._aggregate computes
        # n_meaningful as a count over a subset of ``results`` so by
        # construction n_meaningful ≤ len(results) == n_ablation_points.
        # The only way to reach this state is a hand-edited sidecar or
        # a schema-incompatible writer. Match the F7 counter-mismatch
        # guard pattern (see lines 1696-1723 in this file): a producer
        # bug or sidecar corruption that mis-labels the buyer-PDF
        # numbers must surface as a render-stop, NOT a silent log +
        # clamp. _log.warning to stderr is invisible to a buyer reading
        # the rendered PDF, so silently rendering would hide the
        # corruption signal.
        raise PublishError(
            f"audit.json reports n_ablation_points_meaningful={napm} "
            f"> n_ablation_points={nap}, which is structurally "
            "impossible from a well-formed writer (meaningful is a "
            "strict subset of total). The sidecar was hand-edited or "
            "produced by an incompatible writer — refusing to "
            "publish. Rerun `topofuzz audit` against the source run "
            "to regenerate a well-formed sidecar."
        )
    headline_title, headline_lede, stat_headline, stat_label, headline_sub = (
        _format_counterfactual_headline(
            by_node_rows=by_node_rows,
            swap_pair_rows=swap_pair_rows,
            n_flipping_pairs=n_flipping_pairs,
            n_canonical_pairs=n_canonical_pairs,
            topology_name=topology_name,
            partial=partial,
            n_ablation_points=nap,
            n_ablation_points_meaningful=napm,
        )
    )

    terminal_control_note = (
        "The terminal node is listed as a control: its attribution is "
        "always 100% by construction (substitute_terminal ablation "
        "swaps the canonical answer directly into the counterfactual "
        "run), so a high value there reflects mechanics, not a "
        "finding."
        if terminal_row is not None
        else ""
    )

    target_phrase = target.strip() if target else None

    return {
        "headline_title": headline_title,
        "headline_lede": headline_lede,
        "stat_headline": stat_headline,
        "stat_label": stat_label,
        "headline_sub_line": headline_sub,
        "by_node_rows": by_node_rows,
        "swap_pair_rows": swap_pair_rows,
        "by_dim_rows": by_dim_rows,
        "terminal_control_note": terminal_control_note,
        "topology_name": topology_name,
        "n_canonical_pairs": n_canonical_pairs,
        "n_flipping_pairs": n_flipping_pairs,
        "cost_str": cost_str,
        # Pass-1 BLOCKER B4 fix: pass the unpriced-call footnote
        # through to the template; rendered next to the cost when
        # nonzero, empty string otherwise.
        "unpriced_note": unpriced_note,
        "partial": partial,
        "target_phrase": target_phrase,
        "run_id": metadata.get("run_id") or "(unknown)",
        "tds_version": metadata.get("tds_version") or "(unknown)",
        "signed_at": metadata.get("signed_at") or "(unsigned)",
    }


def _render_counterfactual_card_html(ctx: dict[str, Any]) -> str:
    """Render counterfactual_card.html.j2 with the assembled context."""
    try:
        import jinja2
    except ImportError as e:
        raise PublishError(
            "topofuzz publish --counterfactual-card requires the "
            "'full' install extra — jinja2 not found. Install with "
            "`pip install topofuzz[full]`."
        ) from e

    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template("counterfactual_card.html.j2")
    return template.render(**ctx)


_DEGRADED_THRESHOLD = 0.0995
"""Failure-rate threshold for the degraded branch (F8 RT pass-1 M3).

Set just below 0.10 so a rate of 0.0999 — which displays as ``10.0%``
under our ``.1f`` formatter — does NOT slip through into the stable
branch. Prevents a card that says ``"stable"`` while displaying
``"10.0%"`` in the same panel."""

_VALID_SEVERITIES = frozenset(
    {"concentrated", "degraded", "stable", "unknown", "attribution"}
)
"""Severity strings that map to a defined CSS class in the template
(F8 RT pass-1 M10). Anything outside this set falls back to ``unknown``
so a future naming drift cannot produce an unstyled white card.

``"attribution"`` is the F8 Stage A class: F4 per-agent ablation
headline wins when ``ablation.json`` is present in the run_dir, and
the vendor-concentration narrative demotes to a secondary notes
block. Stylesheet maps it to a neutral blue chip (not concentrated
red / degraded amber / stable green) because high attribution to one
role is a finding-to-investigate, not a failure verdict."""

_ATTRIBUTION_TABLE_LIMIT = 3
"""How many top non-terminal nodes to render in the attribution table
(F8 Stage A). Pragmatic upper bound: a 4-role pipeline like the
lending-4role gallery has 3 non-terminal roles (intake/risk/policy);
a 6-role pipeline still fits the 1-page card if we cap at 3.
Truncation tail is shown as a note when more roles exist."""

_VENDOR_OVERLAP_DISPLAY_LIMIT = 3
"""How many vendor names to show literally in the lede before
collapsing to ``... (+N more)`` (F8 RT pass-1 M6). Bounds the joined
string so a cosmetically poisoned manifest cannot push the card over
its 1-page spec."""

_TOPOLOGY_NAME_DISPLAY_LIMIT = 60
"""Hard cap on the topology name shown in the headline strip and
``<title>`` (F8 CR pass-1 N3). A 5KB hostile topology_name would not
render on-page but would inflate the ``<title>`` tag."""


def _build_consistency_card_context(
    *,
    metadata: dict[str, str],
    manifest: dict[str, Any],
    independence: dict[str, Any],
    target_company: str | None,
    ablation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the Jinja context for consistency_card.html.j2.

    Byte-deterministic given (metadata, manifest, independence,
    target_company, ablation) — tests can pin headline / stats without
    spinning up Chromium. Defensive against malformed independence
    blocks: any missing, wrong-typed, or contradictory field collapses
    to the conservative ``unknown`` framing rather than fabricating an
    audit-grade headline from absent data (F8 BLOCKERs B1/B2).

    F8 Stage A: when ``ablation`` is non-None and yields at least one
    non-terminal node with attribution, the F4 per-agent attribution
    headline takes precedence over the vendor-concentration headline.
    Rationale: a CRO reading "risk-agent caused 64% of flips on this
    pipeline" gets the actionable finding directly, whereas
    "2-of-3 jurors are Google" requires a second hop to "what do I
    fix?". Vendor-concentration narrative demotes to a secondary
    ``vendor_independence_note`` rendered as a notes block.
    """
    topology_name = _truncate_topology_name(
        str(manifest.get("topology_name") or "(unknown)")
    )
    n_variants = manifest.get("n_variants")
    # F8 RT pass-1 M5: real production manifests write ``n_eval_rows``,
    # not ``n_evals``. Fall back so the card doesn't render ``Evals: —``
    # on every real gallery run.
    n_evals = manifest.get("n_evals")
    if not isinstance(n_evals, int) or isinstance(n_evals, bool):
        n_evals = manifest.get("n_eval_rows")
    cost_actual = manifest.get("cost_actual_usd")

    overlap, failure_rate, disjoint, juror_rows, distinct_vendor_count, total_jurors = (
        _extract_vendor_independence(independence, manifest)
    )

    # F8 RT pass-1 B4: surface self_preference_warning when present.
    # The real lending-tricky-jury gallery run has it ("Agents and
    # judges share vendor(s): google. ..."). Silently dropping it
    # lets an analyst diff the PDF against report.json and catch us.
    self_preference = independence.get("self_preference_warning")
    self_preference_str = (
        str(self_preference).strip()
        if isinstance(self_preference, str) and self_preference.strip()
        else ""
    )

    # F8 RT pass-1 M9: format the failure rate ONCE so headline and
    # body don't diverge by 0.4pp. Done up here so all branches can
    # interpolate the same string.
    if isinstance(failure_rate, float) and not isinstance(failure_rate, bool):
        failure_rate_str = f"{failure_rate * 100:.1f}%"
    else:
        failure_rate_str = "—"

    target_phrase = target_company.strip() if target_company else None
    target = target_phrase or "your stack"

    # F8 Stage A: pick the attribution headline when ablation has at
    # least one non-terminal node with parseable attribution; else fall
    # back to the existing vendor-concentration headline. Both branches
    # compute the vendor-concentration narrative; only one publishes it
    # as the headline. The other surfaces it as a secondary notes block.
    attribution_rows = _extract_attribution_rows(ablation) if ablation else []
    attribution_n_truncated = 0
    attribution_n_flipping_pairs: int | None = None
    attribution_partial = False
    if ablation and isinstance(ablation, dict):
        nfp_raw = ablation.get("n_flipping_pairs")
        if isinstance(nfp_raw, int) and not isinstance(nfp_raw, bool):
            attribution_n_flipping_pairs = nfp_raw
        # F8 Stage A pass-1 RT-B2: surface ablation.partial so a CRO who
        # opens the source can't catch us silently dropping the flag.
        attribution_partial = ablation.get("partial") is True
        # Track whether more non-terminal nodes existed than we render.
        by_node = ablation.get("by_node")
        if isinstance(by_node, dict):
            non_terminal_total = sum(
                1
                for v in by_node.values()
                if isinstance(v, dict) and not v.get("is_terminal", False)
            )
            if non_terminal_total > len(attribution_rows):
                attribution_n_truncated = non_terminal_total - len(attribution_rows)

    if attribution_rows:
        # F8 Stage A: F4 attribution headline wins. Attribution mode
        # also publishes its own sub-line (the chip's lower text) —
        # the vendor-concentration "Jury-vote failure rate: X%" is
        # off-topic under an attribution headline (RT pass-1 M3).
        (
            headline_title,
            headline_lede,
            stat_headline,
            stat_label,
            headline_sub_line,
        ) = _format_attribution_headline(
            attribution_rows=attribution_rows,
            topology_name=topology_name,
            n_flipping_pairs=attribution_n_flipping_pairs,
            n_truncated=attribution_n_truncated,
            partial=attribution_partial,
        )
        severity = "attribution"
        # Vendor-concentration narrative demotes to a notes block.
        vc_title, vc_lede, _vc_stat, _vc_label, vc_severity = (
            _format_consistency_headline(
                overlap=overlap,
                failure_rate=failure_rate,
                failure_rate_str=failure_rate_str,
                disjoint=disjoint,
                total_jurors=total_jurors,
                distinct_vendor_count=distinct_vendor_count,
            )
        )
        # Only surface as a note when the vendor-concentration finding
        # is itself meaningful (concentrated or degraded). A "stable"
        # or "unknown" finding doesn't earn a notes block in
        # attribution mode — keep the card uncluttered.
        if vc_severity in ("concentrated", "degraded"):
            vendor_independence_note = f"{vc_title} {vc_lede}"
        else:
            vendor_independence_note = ""
    else:
        # Pre-Stage-A behaviour: vendor-concentration is the headline,
        # and the chip sub-line is the failure-rate line preserved
        # byte-for-byte from the original F8 ship for back-compat.
        headline_title, headline_lede, stat_headline, stat_label, severity = (
            _format_consistency_headline(
                overlap=overlap,
                failure_rate=failure_rate,
                failure_rate_str=failure_rate_str,
                disjoint=disjoint,
                total_jurors=total_jurors,
                distinct_vendor_count=distinct_vendor_count,
            )
        )
        # Pass-2 RT MAJOR M1 fold (folded from 2026-05-18 dual-audit
        # pass 2): the F7 pass-1 fold added a `pct <= 0.0` filter to
        # ``_extract_attribution_rows`` to suppress the all-zero F4
        # contradiction. Symmetric concern: when an ablation.json HAS
        # a populated ``by_node`` block (non-terminal nodes present)
        # but every parseable non-terminal attribution_pct is 0.0,
        # the extractor returns ``[]`` → this fallback branch fires →
        # the buyer sees a pure vendor-concentration narrative with
        # no breadcrumb that F4 attribution ran-and-zeroed. A CRO
        # later opens the source files, sees ``ablation.json`` with
        # attribution machinery output, and the PDF pretends it
        # doesn't exist — silent demotion of a signal an opposing
        # counsel can pull from the audit bundle. Surface a single
        # honest sentence so the buyer-PDF tracks the source state.
        f4_attempted_but_zero = False
        if (
            ablation
            and isinstance(ablation, dict)
            and not attribution_rows
        ):
            ab_by_node = ablation.get("by_node")
            if isinstance(ab_by_node, dict):
                # Did the ablation file carry non-terminal data we
                # could have read? Distinguishes "F4 ran and zeroed"
                # from "F4 didn't run / by_node entirely absent".
                f4_attempted_but_zero = any(
                    isinstance(v, dict)
                    and v.get("is_terminal") is not True
                    and isinstance(v.get("attribution_pct"), (int, float))
                    and not isinstance(v.get("attribution_pct"), bool)
                    for v in ab_by_node.values()
                )
        if f4_attempted_but_zero:
            vendor_independence_note = (
                "F4 per-role attribution ran on this audit and "
                "produced no non-terminal role with positive "
                "signal — see ablation.json. The headline below is "
                "the vendor-concentration finding."
            )
        else:
            vendor_independence_note = ""
        headline_sub_line = f"Jury-vote failure rate: {failure_rate_str}"

    # F8 RT pass-1 M10: whitelist the CSS class to a known severity
    # so a future naming drift falls back to ``unknown`` (styled)
    # instead of an unstyled white card.
    if severity not in _VALID_SEVERITIES:
        severity = "unknown"

    what_para = _format_consistency_what_paragraph(target, severity)
    cta_line = (
        f"Want this independence check rerun against {target_phrase}'s "
        "own jury composition? Reply to this email — repro is free."
        if target_phrase
        else (
            "Want this independence check rerun against your own jury "
            "composition? Reply to this email — repro is free."
        )
    )

    cost_str = (
        f"${float(cost_actual):.2f}"
        if isinstance(cost_actual, (int, float)) and not isinstance(cost_actual, bool)
        else "—"
    )

    def _safe_int(v: Any) -> int | str:
        if isinstance(v, bool) or not isinstance(v, int):
            return "—"
        return v

    return {
        "headline_title": headline_title,
        "headline_lede": headline_lede,
        "stat_headline": stat_headline,
        "stat_label": stat_label,
        "severity": severity,
        "failure_rate_str": failure_rate_str,
        # F8 Stage A pass-1 RT-M3: chip sub-line varies by mode —
        # attribution shows flip count + role count (+ "F4 run
        # partial" when ablation.partial), vendor-concentration shows
        # the original "Jury-vote failure rate: X%".
        "headline_sub_line": headline_sub_line,
        "self_preference_warning": self_preference_str,
        "what_paragraph": what_para,
        "cta_line": cta_line,
        "topology_name": topology_name,
        "n_variants": _safe_int(n_variants),
        "n_evals": _safe_int(n_evals),
        "cost_str": cost_str,
        "jurors": juror_rows,
        # F8 Stage A: empty list when ablation absent, populated list
        # (top-N non-terminal rows) when present. The template loops
        # over this with ``{% if attribution_rows %}`` so both modes
        # render correctly. ``vendor_independence_note`` is the
        # demoted vendor-concentration text shown as a notes block in
        # attribution mode; empty string suppresses the block in
        # vendor-concentration mode.
        "attribution_rows": attribution_rows,
        "attribution_n_truncated": attribution_n_truncated,
        "vendor_independence_note": vendor_independence_note,
        "run_id": metadata.get("run_id") or "(unknown)",
        "tds_version": metadata.get("tds_version") or "(unknown)",
        "signed_at": metadata.get("signed_at") or "(unsigned)",
    }


def _truncate_topology_name(name: str) -> str:
    """Bound the rendered topology_name (F8 CR pass-1 N3).

    Operator-supplied; an oversize value would inflate the HTML
    ``<title>`` and the .meta-row line without tripping the page-count
    guard (the page-count check only catches body overflows).
    """
    if len(name) > _TOPOLOGY_NAME_DISPLAY_LIMIT:
        return name[: _TOPOLOGY_NAME_DISPLAY_LIMIT - 1] + "…"
    return name


_ATTRIBUTION_ROLE_NAME_DISPLAY_LIMIT = 30
"""Hard cap on the role name shown in the attribution title and table
(F8 Stage A pass-1 code-review M-CR2 / RT-M1 — drops the
`-agent` suffix grafting AND bounds operator-supplied role IDs so
a 5000-char hostile node id can't inflate ``headline_title`` or the
HTML ``<title>`` tag past the page budget. Autoescape catches XSS;
this cap catches overflow."""


def _safe_role_name(name: str) -> str:
    """Length-bound a role name + visible-truncation marker. F8 Stage A.

    Reused by ``_extract_attribution_rows`` (table cells) and
    ``_format_attribution_headline`` (title + lede). Visible ``…``
    suffix so an auditor can tell the field was clipped rather than
    silently misreading a truncated identifier as the full one.
    """
    if len(name) > _ATTRIBUTION_ROLE_NAME_DISPLAY_LIMIT:
        return name[: _ATTRIBUTION_ROLE_NAME_DISPLAY_LIMIT - 1] + "…"
    return name


def _extract_attribution_rows(
    ablation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Pull the top-N non-terminal node attribution rows from ablation.json.

    F8 Stage A. The ablation.json shape (schema_version 1) has::

        {
          "n_flipping_pairs": int,
          "by_node": {
            "<node_id>": {
              "flips_closed": int,
              "total_meaningful": int,
              "attribution_pct": float,  # 0–100
              "is_terminal": bool
            },
            ...
          },
          ...
        }

    Returns up to :data:`_ATTRIBUTION_TABLE_LIMIT` rows sorted by
    ``attribution_pct`` descending, filtered to non-terminal nodes
    only (terminal attribution is always 100% by construction via the
    ``substitute_terminal`` ablation kind — mechanical, not signal).
    Each row has ``{name, attribution_pct, flips_closed,
    total_meaningful, attribution_pct_str}`` for the template.

    Defensive against malformed input: missing / wrong-typed /
    out-of-range entries are silently skipped. An empty list signals
    "no F4 headline available, fall back to vendor-concentration."
    Empty list is returned when ``ablation`` is None or when no
    non-terminal node has parseable attribution.
    """
    if not isinstance(ablation, dict):
        return []
    by_node = ablation.get("by_node")
    if not isinstance(by_node, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, payload in by_node.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        if payload.get("is_terminal") is True:
            continue
        pct = payload.get("attribution_pct")
        if isinstance(pct, bool) or not isinstance(pct, (int, float)):
            continue
        # NaN / out-of-range filter — protects the headline against a
        # mis-emitted attribution value that would render as "nan%".
        if pct != pct:  # NaN
            continue
        if pct > 100:
            continue
        # Pass-2 RT MAJOR-3 fold (folded from 2026-05-18 dual-audit
        # pass 2): mirror the swap-pair extractor's warn-on-negative
        # posture so a producer regression that mis-signs F4
        # attribution surfaces in stderr / CI rather than being
        # silently dropped into the vendor-concentration fallback.
        if pct < 0:
            _log.warning(
                "publish: ablation.by_node[%s] has negative "
                "attribution_pct=%r; dropping from F8 attribution "
                "table (producers should never emit negatives).",
                name, pct,
            )
            continue
        flips_closed = payload.get("flips_closed")
        total_meaningful = payload.get("total_meaningful")
        # F8 Stage A pass-2 code-review MINOR: drop degenerate rows
        # where total_meaningful == 0 (no eligible ablation cells).
        # These produce "0 of 0" copy in the lede and a zero-or-NaN
        # attribution pct that would be misleading in the table.
        if (
            isinstance(total_meaningful, int)
            and not isinstance(total_meaningful, bool)
            and total_meaningful == 0
        ):
            continue
        # Pass-1 RT MAJOR M1 fix (folded from 2026-05-18 dual-audit
        # silent-failure-hunter): the F7 counterfactual card just
        # gained an all-zero attribution gate (line ~1442) because a
        # parseable-but-zero attribution_pct rendering as "the X agent
        # contributed most …" over a "0.0%" stat is a buyer-visible
        # contradiction. The F8 Stage A consistency card has the
        # IDENTICAL silent-failure shape via this extractor: if every
        # ablation.by_node row has parseable-but-zero attribution_pct
        # (the post-Phase-2-cap-hit / all-rejudges-unparseable state),
        # the row list stays non-empty, the F4 attribution headline
        # wins in _build_consistency_card_context, and the consistency
        # card PDF renders e.g. "The intake role closed 0 of 1
        # ablation-eligible verdict flips (0.0%)" with the actually-
        # informative vendor-concentration narrative demoted to a
        # notes block. Drop 0%-attribution rows here so the consumer
        # falls back to the vendor-concentration headline whenever no
        # non-terminal role has a positive signal — same posture as
        # the F7 fix.
        if float(pct) <= 0.0:
            continue
        rows.append({
            "name": _safe_role_name(name),
            "attribution_pct": float(pct),
            "attribution_pct_str": f"{float(pct):.1f}%",
            "flips_closed": flips_closed if isinstance(flips_closed, int) and not isinstance(flips_closed, bool) else None,
            "total_meaningful": total_meaningful if isinstance(total_meaningful, int) and not isinstance(total_meaningful, bool) else None,
        })
    # Stable secondary sort on name so ties render deterministically
    # (important for byte-exact PDF reproducibility under repro).
    rows.sort(key=lambda r: (-r["attribution_pct"], r["name"]))
    return rows[:_ATTRIBUTION_TABLE_LIMIT]


def _format_attribution_headline(
    *,
    attribution_rows: list[dict[str, Any]],
    topology_name: str,
    n_flipping_pairs: int | None,
    n_truncated: int,
    partial: bool,
) -> tuple[str, str, str, str, str]:
    """Render the F4 per-agent attribution headline + stat copy.

    Caller guarantees ``attribution_rows`` is non-empty (the empty
    case falls back to the vendor-concentration headline upstream).
    Returns ``(title, lede, stat_headline, stat_label, sub_line)``.

    Honesty bar (F8 Stage A pass-1 RT-B1 / B2):

    * Per-role denominator: each ``attribution_pct`` is
      ``flips_closed / total_meaningful`` for THAT role, where
      ``total_meaningful`` is the number of flipping pairs where the
      role was on the ablation-eligible path — NOT ``n_flipping_pairs``.
      The lede must publish the per-role numerator/denominator so a
      reader who divides reaches the same ratio. Pre-fix, the lede
      read "Across {n_flipping_pairs} measured verdict flips ...
      attributes the highest share to the {top} role" and a CRO who
      did 7/12 = 58.3% (not 63.6%) caught us on a 30-second diff.
    * ``partial`` flag: when ``ablation.partial is True`` (F4 run was
      interrupted), the lede MUST disclose it. Pre-fix, the field was
      silently dropped — a CRO who opened ablation.json saw the flag
      and dismissed the report.

    Copy guardrails:

    * No "-agent" suffix grafting (operator-supplied role IDs may
      already include role-type words; suffix produces redundancy
      like "risk-agent-agent").
    * No ``.capitalize()`` — sentence-leads with "The {role}" so the
      role name renders as-typed (autoescape + ``_safe_role_name``
      bound length).
    * "Follow-up audit" instead of "re-engineering pass" — the engine
      measures attribution, not causation; "re-engineering" overclaims
      what F4 can substantiate.

    Sub-line (``headline_stat .ci`` slot in the template): in
    attribution mode this should NOT say "Jury-vote failure rate" —
    that's a vendor-concentration sub-line. Replace with F4-relevant
    context: flip count + non-terminal role count.
    """
    top = attribution_rows[0]
    top_name = str(top["name"])  # already bounded by _safe_role_name
    top_pct_str = top["attribution_pct_str"]
    top_closed = top.get("flips_closed")
    top_total = top.get("total_meaningful")

    # F8 Stage A pass-2 RT-MAJOR: surface the per-role denominator in
    # the title itself so a CRO scanning the headline can reconcile
    # the % against the ratio without scrolling to the lede. Pre-fix,
    # title "63.6% of attributable verdict flips" + chip sub-line "12
    # measured flips" invited the false-positive diff 7/12 = 58.3% ≠
    # 63.6%. Post-fix title leads with the ratio + pct in parens, so
    # the denominator is part of the headline (not buried).
    if isinstance(top_closed, int) and isinstance(top_total, int) and top_total > 0:
        title = (
            f"The {top_name} role closed {top_closed} of {top_total} "
            f"ablation-eligible verdict flips on {topology_name} "
            f"({top_pct_str})."
        )
    else:
        title = (
            f"The {top_name} role closed {top_pct_str} of "
            f"ablation-eligible verdict flips on {topology_name}."
        )

    others = attribution_rows[1:]
    if others:
        other_phrases = [
            (
                f"{r['name']} {r['flips_closed']}/{r['total_meaningful']} "
                f"({r['attribution_pct_str']})"
                if (
                    isinstance(r.get("flips_closed"), int)
                    and isinstance(r.get("total_meaningful"), int)
                )
                else f"{r['name']} ({r['attribution_pct_str']})"
            )
            for r in others
        ]
        if len(other_phrases) == 1:
            others_clause = f"; the next contributor was {other_phrases[0]}"
        else:
            others_clause = (
                "; the next contributors were " +
                ", ".join(other_phrases[:-1]) +
                f", and {other_phrases[-1]}"
            )
    else:
        others_clause = ""

    if isinstance(top_closed, int) and isinstance(top_total, int):
        top_ratio = f"{top_closed} of {top_total}"
    else:
        top_ratio = "the majority of"

    lede = (
        f"Leave-one-out ablation closed {top_ratio} ablation-"
        f"eligible flip pairs when the {top_name} role was removed "
        f"({top_pct_str}){others_clause}. The breakdown below shows "
        "the contribution magnitude a reviewer would use to scope a "
        "follow-up audit."
    )
    if partial:
        lede += (
            " The source ablation run was flagged ``partial`` — some "
            "ablation cells did not complete; treat this as a "
            "preliminary signal pending a full F4 rerun."
        )

    stat_headline = top_pct_str
    if n_truncated > 0:
        stat_label = (
            f"of attributable verdict flips closed by removing the "
            f"{top_name} role (top of {len(attribution_rows) + n_truncated} "
            "non-terminal roles ablated)"
        )
    else:
        stat_label = (
            f"of attributable verdict flips closed by removing the "
            f"{top_name} role"
        )

    # Attribution-mode sub-line: replace the vendor-concentration
    # "Jury-vote failure rate: X%" with F4 telemetry. n_flipping_pairs
    # is the count of distinct flipping pairs the run measured.
    sub_parts: list[str] = []
    if isinstance(n_flipping_pairs, int) and n_flipping_pairs > 0:
        sub_parts.append(f"{n_flipping_pairs} measured verdict flips")
    sub_parts.append(
        f"{len(attribution_rows) + n_truncated} non-terminal roles ablated"
        if n_truncated > 0
        else f"{len(attribution_rows)} non-terminal roles ablated"
    )
    if partial:
        sub_parts.append("F4 run partial")
    sub_line = " · ".join(sub_parts)

    return title, lede, stat_headline, stat_label, sub_line


def _format_consistency_headline(
    *,
    overlap: list[str],
    failure_rate: float | None,
    failure_rate_str: str,
    disjoint: bool,
    total_jurors: int | None,
    distinct_vendor_count: int | None,
) -> tuple[str, str, str, str, str]:
    """Pick the severity branch + render headline copy.

    Returns ``(title, lede, stat_headline, stat_label, severity)``.

    Branch precedence (highest-severity wins, so a run that triggers
    both concentrated AND degraded ships under concentrated — the
    overlap is the more actionable procurement finding because the
    jury's independence claim itself is conditional on disjointness).
    The degradation signal is preserved on the headline-stat
    sub-line ("Jury-vote failure rate: X%" in the template) AND, when
    non-zero, in a tail sentence appended to the concentrated lede
    (F8 RT pass-2 R1 — earlier docstring overclaimed surfacing in the
    ``what`` paragraph, which is not the case; the lede tail is the
    correct surface):

    1. ``concentrated`` — ``overlap`` non-empty AND we have enough
       juror telemetry (``total_jurors >= 2``, ``distinct_vendor_count``
       resolved) to make a defensible "N of K share a vendor" claim.
       Lede branches on ``distinct_vendor_count == 1`` (F8 RT pass-2
       R3: "Two or more jurors share a vendor" is tonally wrong when
       ALL jurors share a single vendor — use "All jurors share a
       vendor" in that case).
    2. ``degraded`` — failure rate at-or-above the displayed-10%
       boundary. (F8 RT pass-1 M3 donut-hole fix.)
    3. ``stable`` — disjoint, low failure rate, AND we can prove from
       juror telemetry that the jury has at least 2 distinct vendors
       (``total_jurors >= 2`` AND ``distinct_vendor_count ==
       total_jurors``). Without this guard the branch fabricated
       "3 / 3" headlines from hardcoded ``or 3`` literals (F8 B1/B2).
    4. ``unknown`` — anything else. Defaults the stat panel to ``?``
       so a procurement reader sees the absence of evidence, not a
       confected number.
    """
    have_juror_evidence = (
        total_jurors is not None
        and total_jurors >= 2
        and distinct_vendor_count is not None
    )

    if overlap and have_juror_evidence:
        # Bound the joined overlap list (F8 RT pass-1 M6).
        display = [_html_safe_vendor(v) for v in overlap[:_VENDOR_OVERLAP_DISPLAY_LIMIT]]
        if len(overlap) > _VENDOR_OVERLAP_DISPLAY_LIMIT:
            display.append(f"+{len(overlap) - _VENDOR_OVERLAP_DISPLAY_LIMIT} more")
        pretty_overlap = ", ".join(display)
        # F8 CR pass-1 M2: pluralise to avoid "1 distinct vendors".
        vendor_noun = "vendor" if distinct_vendor_count == 1 else "vendors"
        title = (
            f"This jury-of-{total_jurors} has only "
            f"{distinct_vendor_count} distinct {vendor_noun}."
        )
        # F8 RT pass-2 R3: branch lede tone on the all-same-vendor case.
        # "Two or more jurors share a vendor" reads wrong when every
        # juror shares the same vendor; "All jurors share a vendor"
        # is the correct copy.
        if distinct_vendor_count == 1:
            lede_head = f"All jurors share a vendor ({pretty_overlap})."
        else:
            lede_head = f"Two or more jurors share a vendor ({pretty_overlap})."
        # F8 RT pass-2 R1: surface non-zero degradation in the lede tail
        # so a procurement reader scanning bold copy sees BOTH signals
        # when concentrated AND degraded fire together. The headline-stat
        # sub-line carried the number before this fix but procurement
        # readers don't scan 9pt CSS sublines.
        lede = (
            f"{lede_head} A jury whose members can fail correlated is "
            "not the independence claim the audit assumed."
        )
        if failure_rate is not None and failure_rate >= _DEGRADED_THRESHOLD:
            lede += (
                f" Jury votes also failed on {failure_rate_str} of "
                "cells, compounding the concentration finding."
            )
        stat_headline = f"{distinct_vendor_count} / {total_jurors}"
        stat_label = f"distinct {vendor_noun} backing the jury verdict"
        return title, lede, stat_headline, stat_label, "concentrated"

    if failure_rate is not None and failure_rate >= _DEGRADED_THRESHOLD:
        # F8 RT pass-1 M9: same precision as the body line, computed once.
        if have_juror_evidence:
            title = (
                f"This jury-of-{total_jurors} degraded on "
                f"{failure_rate_str} of cells."
            )
        else:
            title = f"This jury degraded on {failure_rate_str} of cells."
        lede = (
            "Each degraded cell fell back to a smaller jury (or to a "
            "single juror) because a vendor call failed. A degraded-"
            "jury claim is not the audit-grade claim the report header "
            "advertises."
        )
        return (
            title,
            lede,
            failure_rate_str,
            "of cells where a jury vote failed",
            "degraded",
        )

    if (
        disjoint
        and (failure_rate is None or failure_rate < _DEGRADED_THRESHOLD)
        and have_juror_evidence
        and distinct_vendor_count == total_jurors
    ):
        title = (
            f"This jury-of-{total_jurors} held vendor disjointness."
        )
        lede = (
            "Each juror was backed by a distinct vendor and the jury "
            "voted without high-volume degraded fallbacks. Treat this "
            "as a preliminary cross-vendor-disjointness signal — "
            "confirm with a self-preference review and a rerun-"
            "stability pass before quoting it in a regulated workflow."
        )
        stat_headline = f"{distinct_vendor_count} / {total_jurors}"
        return (
            title,
            lede,
            stat_headline,
            "distinct vendors backing the jury verdict",
            "stable",
        )

    # Conservative default — partial telemetry, refuse to overstate
    # (F8 BLOCKERs B1/B2: no more hardcoded ``or 3``).
    title = "Jury vendor independence not verifiable."
    lede = (
        "This run's report.json does not contain the cross-vendor "
        "telemetry needed for an independence claim. Rerun with the "
        "latest topofuzz to populate the judge_independence block, or "
        "confirm the source run was a jury-of-N (not a single judge)."
    )
    return (
        title,
        lede,
        "?",
        "distinct vendors backing the jury verdict",
        "unknown",
    )


_VENDOR_DISAMBIGUATOR_RE = re.compile(r"_\d+$")


def _normalize_vendor_name(name: str) -> str:
    """Strip a trailing ``_N`` disambiguator suffix from a vendor name.

    The IR uses ``google``, ``google_2``, ``google_3`` to keep juror
    keys unique when a jury-of-N pulls multiple models from the same
    vendor (F8 RT pass-1 B1). The PDF cares about the human-readable
    vendor only, so we strip the suffix before deduping or rendering.
    Without this, the lending-tricky-jury gallery run prints
    ``distinct_vendor_count=3`` while ``vendor_overlap=['google']``
    says 2 — a contradiction a hostile analyst spots in 60 seconds by
    diffing the PDF against ``report.json``.
    """
    return _VENDOR_DISAMBIGUATOR_RE.sub("", name)


def _extract_vendor_independence(
    independence: dict[str, Any], manifest: dict[str, Any]
) -> tuple[
    list[str],          # vendor_overlap (normalized)
    float | None,       # vendor_call_failure_rate
    bool,               # vendor_disjoint
    list[dict[str, str]],  # juror_rows for the table (position re-numbered)
    int | None,         # distinct vendor count (post-normalisation)
    int | None,         # total juror count
]:
    """Pull the headline.judge_independence + manifest.jury_models fields.

    Defensive: each field falls back to a conservative shape on type
    mismatch so the consistency card always has something to render.
    """
    overlap_raw = independence.get("vendor_overlap")
    # F8 RT pass-1 B1: normalise ``google_2`` -> ``google`` so the
    # overlap list and the distinct-count derived from juror_rows agree.
    # De-dup post-normalisation so a future producer that returns
    # ["google", "google_2"] doesn't get counted twice.
    overlap_normalised: list[str] = []
    if isinstance(overlap_raw, list):
        for v in overlap_raw:
            if not isinstance(v, str):
                continue
            n = _normalize_vendor_name(v)
            if n not in overlap_normalised:
                overlap_normalised.append(n)
    overlap = overlap_normalised

    failure_rate_raw = independence.get("vendor_call_failure_rate")
    failure_rate: float | None
    if isinstance(failure_rate_raw, bool) or not isinstance(failure_rate_raw, (int, float)):
        failure_rate = None
    elif failure_rate_raw != failure_rate_raw:  # NaN
        failure_rate = None
    else:
        failure_rate = float(failure_rate_raw)

    disjoint_raw = independence.get("vendor_disjoint")
    disjoint = bool(disjoint_raw) if isinstance(disjoint_raw, bool) else False

    # Prefer the resolved jury list from manifest.cli_args.jury_models_resolved
    # (dict of jury_position -> "vendor/model"). Fall back to
    # headline.judge_independence.judges (list of {vendor, model}).
    # Vendor names are normalised through ``_normalize_vendor_name`` so the
    # ``google``/``google_2`` IR-disambiguator doesn't inflate the distinct
    # vendor count (F8 RT pass-1 B1). Position is re-numbered 1..N AFTER
    # the malformed-row filter so the rendered table never has a gap
    # (F8 CR pass-1 N1).
    juror_rows: list[dict[str, str]] = []
    cli_args = manifest.get("cli_args") if isinstance(manifest, dict) else None
    resolved = cli_args.get("jury_models_resolved") if isinstance(cli_args, dict) else None
    if isinstance(resolved, dict):
        for _key, model_id in sorted(resolved.items()):
            if not isinstance(model_id, str):
                continue
            if "/" in model_id:
                vendor, model = model_id.split("/", 1)
            else:
                vendor, model = model_id, ""
            juror_rows.append(
                {"vendor": _normalize_vendor_name(vendor), "model": model}
            )
    if not juror_rows:
        judges_raw = independence.get("judges")
        if isinstance(judges_raw, list):
            for j in judges_raw:
                if not isinstance(j, dict):
                    continue
                vendor = j.get("vendor")
                model_field = j.get("model") or j.get("model_id")
                if not isinstance(vendor, str):
                    continue
                # Strip the "vendor/" prefix from model_id when present so
                # the table column doesn't duplicate the Vendor cell.
                model_str = ""
                if isinstance(model_field, str):
                    if "/" in model_field:
                        _, model_str = model_field.split("/", 1)
                    else:
                        model_str = model_field
                juror_rows.append({
                    "vendor": _normalize_vendor_name(vendor),
                    "model": model_str,
                })
    # Re-number positions sequentially after the malformed-row filter.
    for idx, row in enumerate(juror_rows, start=1):
        row["position"] = str(idx)

    total_jurors = len(juror_rows) if juror_rows else None
    distinct_vendor_count = (
        len({r["vendor"] for r in juror_rows}) if juror_rows else None
    )

    return overlap, failure_rate, disjoint, juror_rows, distinct_vendor_count, total_jurors


def _format_consistency_what_paragraph(target: str, severity: str) -> str:
    """The "what this means for $TARGET" body paragraph.

    Four severity shapes (F8 Stage A added ``attribution``).
    Pitch-copy guardrail: no shape says "your model is unsafe" — they
    describe procurement-grade signals worth a follow-up audit.
    """
    if severity == "attribution":
        return (
            f"If {target} runs multi-agent LLM audits, the actionable "
            "finding is which role in the chain drives the verdict "
            "instability — not just that it exists. This run isolated "
            "the contribution of each non-terminal role via leave-one-"
            "out ablation. The top-contributing role is the candidate "
            "for a follow-up audit before the SOW signs; the second- "
            "and third-place rows say how much of the remaining "
            "instability is distributed across the rest of the "
            "pipeline. Per-role attribution shares can overlap and "
            "need not sum to 100% — leave-one-out semantics credit "
            "every role that, removed individually, would have "
            "prevented the flip. Read the vendor-independence note "
            "below for a secondary caveat orthogonal to attribution."
        )
    if severity == "concentrated":
        return (
            f"If {target} sources jury-of-3 model audits, this is the "
            "kind of finding a procurement reviewer flags before the "
            "SOW signs: the audit's independence claim only holds if "
            "the three jurors actually fail uncorrelated. Two jurors "
            "from the same vendor share infrastructure, share weights "
            "(when shared model family), and fail together on the same "
            "prompt class. The headline confidence interval the audit "
            "report quotes is conditional on vendor-disjointness it "
            "may not have."
        )
    if severity == "degraded":
        return (
            f"If {target} commissions jury-of-3 audits, a degraded-"
            "jury rate above ~10% means the headline confidence is "
            "computed against an audit that silently lost jurors on "
            "every tenth call. The auditable claim is no longer a "
            "jury-of-3 audit — it is a jury-of-the-jurors-that-"
            "answered audit. Worth a clarifying question to the "
            "vendor before the report ships to the model-risk "
            "committee."
        )
    if severity == "stable":
        return (
            f"If {target} commissions jury-of-N audits, this run is a "
            "preliminary cross-vendor-disjointness signal: distinct "
            "vendors voted and no high-volume degradation events were "
            "observed. The signal is necessary but not sufficient for "
            "a model-risk-management quote — confirm with a self-"
            "preference review (do agents and judges share vendors?) "
            "and a rerun-stability check before relying on it in a "
            "regulated workflow. The attestation footer below lets "
            "any reviewer re-run the test bit-for-bit."
        )
    return (
        f"This run does not surface enough cross-vendor telemetry to "
        f"give {target} an independence claim either way. Rerun on "
        "topofuzz v0.6.0+ with `--jury-of-3` and a vendor-disjoint "
        "juror list to populate the audit-grade headline."
    )


def _html_safe_vendor(name: str) -> str:
    """Length-bound a vendor name before it lands in the comma list.

    Defends the headline's one-liner length against a hostile or sloppy
    vendor name (>40 chars). Real vendor names are short (``google``,
    ``openai``, ``anthropic``, ``deepseek``) — anything past 40 chars
    is either a misconfig or a payload attempt; truncate visibly so an
    auditor can tell the field was clipped.
    """
    if len(name) > 40:
        return name[:37] + "…"
    return name


def _render_consistency_card_html(ctx: dict[str, Any]) -> str:
    """Render consistency_card.html.j2 with the assembled context."""
    try:
        import jinja2
    except ImportError as e:
        raise PublishError(
            "topofuzz publish --consistency-card requires the 'full' "
            "install extra — jinja2 not found. Install with "
            "`pip install topofuzz[full]`."
        ) from e

    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template("consistency_card.html.j2")
    return template.render(**ctx)


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
    # The orchestrator writes ``n_eval_rows``; fall back so the card does
    # not render ``Evals: —`` on a real run.
    n_evals = manifest.get("n_evals")
    if not isinstance(n_evals, int) or isinstance(n_evals, bool):
        n_evals = manifest.get("n_eval_rows")
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

    cli_args = manifest.get("cli_args")
    jury_of_3 = isinstance(cli_args, dict) and cli_args.get("jury_of_3") is True
    headline_title, headline_lede = _format_headline(
        topology_name, flip_rate, n_flips, jury_of_3=jury_of_3
    )

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
    # Schema note: the 3-state verdict work renamed the count keys from
    # ``n_pairs``/``n_flips`` to ``denominator``/``numerator`` (and added
    # ``clear_pairs``/``conditional_pairs``). Read the current keys first,
    # fall back to the legacy names so cards render on both old and new runs.
    # Without this, a current-schema run has neither legacy key -> both come
    # back None -> the card wrongly falls into the "stable run" branch while
    # _build_flip_card still shows a real flip (self-contradiction).
    n_pairs_raw = verdict_flip_rate.get("denominator", verdict_flip_rate.get("n_pairs"))
    n_flips_raw = verdict_flip_rate.get("numerator", verdict_flip_rate.get("n_flips"))

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
    topology_name: str,
    flip_rate: float | None,
    n_flips: int | None,
    *,
    jury_of_3: bool = True,
) -> tuple[str, str]:
    """Headline + lede tuned to the run shape.

    ``jury_of_3`` must reflect how the source run was actually judged:
    the card is a buyer-facing artifact, and a single-judge run must not
    claim it was "measured under a jury-of-3".
    """
    pretty_name = topology_name.replace("_", " ").replace("-", " ")
    if flip_rate is not None and n_flips and n_flips > 0:
        title = f"This {pretty_name} pipeline flips its verdict under re-ordering."
        if jury_of_3:
            lede = (
                "Same model, same prompts, same eval. Reorder the agents and "
                "the answer changes — measured under a jury-of-3, not a "
                "single judge."
            )
        else:
            lede = (
                "Same model, same prompts, same eval. Reorder the agents and "
                "the answer changes — measured with a single AI judge; "
                "re-run with a jury-of-3 to rule out judge noise."
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
        if not eval_prompt:
            # run.sqlite has no ``evals`` table in current schemas — the
            # case text lives in the eval file the run was launched with.
            eval_prompt = _eval_input_from_eval_file(manifest, run_dir, eid)
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
    # Cross-consumer NFKC + Cc/Cf/M normalize via shared helper
    # (BACKLOG 2026-05-16 MAJOR). Without this step a hand-edited
    # manifest with ZWSP / combining marks / fullwidth Latin / NBSP
    # padding survives the consumer normalize even though the producer
    # IR validator would have rejected it at construction time. The
    # helper is byte-faithful to `ir._normalize_verdict_label` so a
    # well-formed manifest round-trips identically here and a tampered
    # one collapses to the same bytes the IR would have produced —
    # closing the gap that lets visually-as-default content ship with
    # bytewise-tampered labels.
    #
    # Normalize FIRST so the pair-level XOR check below operates on the
    # POST-normalize results. Pass-1 dual-audit (CR M-1 + RT M1,
    # 2026-05-16): a label like `"​"` (ZWSP only) is truthy under
    # `.strip()` (which does NOT strip Cf characters), passes the XOR
    # as "set", then collapses to empty in the helper and falls back to
    # the default partner — yielding a half-custom pair (custom-true +
    # default-false) that defeats the pair-level fallback this consumer
    # claims to enforce. Evaluating XOR on the post-normalize result
    # catches all such asymmetric tampers.
    norm_true = (
        _normalize_verdict_label_for_consumer(label_true_raw)
        if isinstance(label_true_raw, str) else ""
    )
    norm_false = (
        _normalize_verdict_label_for_consumer(label_false_raw)
        if isinstance(label_false_raw, str) else ""
    )
    # Step 20a/F6 follow-up dual-audit pass-1 RT M2: pair-level XOR
    # check on the manifest. The orchestrator persists labels only as
    # a pair (both-or-neither, gated on differs-from-default), so a
    # manifest with exactly one key resolving to non-empty after
    # normalize is an obvious tamper or hand-edit. Log and fall back to
    # BOTH defaults rather than letting publish silently default one
    # half against an explicit other half.
    if bool(norm_true) != bool(norm_false):
        _log.warning(
            "publish: manifest has exactly one of verdict_label_true/false "
            "non-empty after NFKC + Cc/Cf/M strip (true=%r, false=%r) — "
            "both must resolve to a non-empty pair or neither (orchestrator "
            "persists as a pair). Falling back to APPROVE/DENY defaults.",
            label_true_raw, label_false_raw,
        )
        norm_true = ""
        norm_false = ""
    # Step 20a/F6 follow-up dual-audit pass-1 RT M1 + CR MAJOR-2: a
    # manifest with a label that normalizes to >12 chars used to
    # silently slice [:12], which (a) renders truncated nonsense in
    # the CEO PDF and (b) can collide with the partner label post-slice
    # in ways the producer IR validator would reject. Hard-reject and
    # fall back to defaults so the publish path matches the producer's
    # policy. Cap measured POST-normalize (not POST-strip) to mirror
    # producer: a ZWSP-padded 15-char raw collapsing to 7 chars passes,
    # a 14-char fullwidth raw collapsing to 14 ASCII chars does not.
    for raw_label, norm, key in (
        (label_true_raw, norm_true, "verdict_label_true"),
        (label_false_raw, norm_false, "verdict_label_false"),
    ):
        if len(norm) > 12:
            _log.warning(
                "publish: manifest %s=%r normalizes to >12 chars after "
                "NFKC + Cc/Cf/M strip — exceeds the side-by-side flip-"
                "card cap. Falling back to APPROVE/DENY for BOTH labels "
                "(pair-level fallback; the partner label is discarded "
                "too so the buyer sees the same default pair rather "
                "than a half-custom pair).",
                key, raw_label,
            )
            norm_true = ""
            norm_false = ""
            break
    label_true = norm_true if norm_true else "APPROVE"
    label_false = norm_false if norm_false else "DENY"
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
        "variant_badge": _variant_badge(
            canonical_row[0] if canonical_row else None,
            variant_row[0] if variant_row else None,
        ),
        "canonical_excerpt": _truncate(
            canonical_excerpt, _CEO_CARD_VERDICT_EXCERPT_LIMIT
        ),
        "variant_excerpt": _truncate(
            variant_excerpt, _CEO_CARD_VERDICT_EXCERPT_LIMIT
        ),
    }


def _eval_input_from_eval_file(
    manifest: dict[str, Any], run_dir: Path, eval_id: str
) -> str:
    """Best-effort: the ``input`` text of ``eval_id`` from the run's eval file.

    Returns ``""`` when the file can't be located or parsed — the caller
    then renders its "(eval prompt unavailable)" placeholder. Never
    raises: the excerpt is decoration, not evidence.
    """
    cli_args = manifest.get("cli_args")
    raw = cli_args.get("eval") if isinstance(cli_args, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return ""
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [
        run_dir / path, run_dir.parent / path, Path.cwd() / path
    ]
    try:
        from topofuzz.loaders import load_eval_rows

        for cand in candidates:
            if cand.is_file():
                for row in load_eval_rows(cand):
                    if str(row.get("id")) == str(eval_id):
                        text = row.get("input")
                        return text if isinstance(text, str) else ""
                return ""
    except Exception:  # noqa: BLE001 — decoration only; never fail the card
        _log.debug("publish: could not read eval input for %s", eval_id)
    return ""


def _variant_badge(canonical_json: str | None, variant_json: str | None) -> str:
    """Name what the flipping variant changed, for the flip-card badge.

    Mirrors :func:`topofuzz.inspect._describe_variant_change`: only an
    order_swap is a re-ordering; a role_swap keeps the slots and trades
    two agents' roles; a decoding_param_swap changes neither.
    """
    try:
        c = json.loads(canonical_json) if canonical_json else []
        v = json.loads(variant_json) if variant_json else []
        c_ids = [o.get("node_id") for o in c]
        v_ids = [o.get("node_id") for o in v]
        if c_ids != v_ids:
            return "Re-ordered"
        if [o.get("role") for o in c] != [o.get("role") for o in v]:
            return "Agents swapped"
        return "Same order, model setting changed"
    except (ValueError, TypeError, AttributeError):
        return "Re-ordered"


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
