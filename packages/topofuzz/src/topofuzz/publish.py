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
