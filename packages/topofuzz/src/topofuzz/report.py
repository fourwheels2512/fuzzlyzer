# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 11 (Weekend 3): render the run-summary HTML report.

Reads ``<run_dir>/manifest.json`` (required), ``<run_dir>/run.sqlite``
(required), and ``<run_dir>/attestation.json`` (optional, presence
implies the ``[DEV-KEY]`` banner). Produces ``<run_dir>/report.html``
with two inline matplotlib SVGs — per-eval voo bar + score-matrix
heatmap — plus two hand-rolled SVGs from step 18.5b (flip-composition
donut + per-eval variance sparkline) and banner conditionals per
BLUEPRINT_v5plus.md §11.

Verify-path safety: this module is **never** imported at module scope
by :mod:`topofuzz.cli`, :mod:`topofuzz.attest`,
:mod:`fuzzeval.attestation`, or :mod:`fuzzeval.keys` (the four guarded
by ``tests/test_import_guard.py``). The CLI's ``report`` subcommand
imports :func:`render_report` lazily inside its function body so a
``topofuzz[verify]`` install (no matplotlib / jinja2 / numpy) can still
run ``verify`` / ``keygen`` without pulling ~50 MB of plotting deps.

The two SVGs are produced by matplotlib's Agg backend (set unconditionally
inside :func:`render_report` BEFORE :mod:`matplotlib.pyplot` is imported —
the only safe time to switch backends). Each SVG has its XML preamble
stripped so it inlines cleanly inside HTML5 without a stray
``<?xml ... ?>`` PI breaking the document tree.

Banner conditionals (§11 line 304):

* ``[DEV-KEY]`` — emitted iff ``attestation.json`` exists in ``run_dir``.
  This is a presence-only check; cryptographic verification is the job
  of ``topofuzz verify``. The v0.5 attestation pipeline only knows dev
  keys (BLUEPRINT_v5plus.md §17a), so file presence == dev-key.
* ``[JURY]`` — emitted iff ``manifest['cli_args']['jury_of_3']`` is
  truthy. Step 12b (Weekend 4) wired ``--jury-of-3`` past the prior
  bad-usage gate; the banner now lights up for live jury runs, not
  just forward-compat manifest fixtures.
* ``[DEGRADED]`` — emitted iff the ``runs.status`` row reads exactly
  ``"degraded"``. The other §A non-success states (``failed``,
  ``cost_cap_hit``, ``interrupted``) are surfaced through the status
  line; the blueprint specifically names ``degraded`` as the banner
  trigger.
"""

from __future__ import annotations

import datetime as _datetime
import json
import logging
import math
import re
import sqlite3
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from topofuzz import __version__ as _topofuzz_version

if TYPE_CHECKING:
    import numpy as np

_log = logging.getLogger(__name__)

# Strip everything up to (but not including) the first '<svg' tag so the
# emitted SVG inlines cleanly into an HTML document. matplotlib's
# default SVG output starts with `<?xml version="1.0" ...?>` + a DOCTYPE;
# both are valid XML but are not allowed inside an HTML5 <body>.
_SVG_PRE_TAG_RE = re.compile(rb"^.*?(?=<svg)", re.DOTALL)

# Hard cap on cell labels in the heatmap so a 50-eval × 50-variant run
# doesn't produce a wall of unreadable tick text. Above the cap, axis
# tick labels are dropped and the user is expected to cross-reference
# the eval-ids list at the bottom of the report. 32 keeps the axis
# legible at default figure size.
_HEATMAP_TICK_LABEL_CAP = 32


class ReportInputError(FileNotFoundError):
    """``manifest.json`` or ``run.sqlite`` is missing from the run dir.

    Subclass of :class:`FileNotFoundError` so the CLI's existing
    bad-usage branch (``except FileNotFoundError`` patterns) catches
    it without a dedicated handler. The custom type lets tests assert
    the specific failure mode.
    """


class ReportDependencyError(RuntimeError):
    """A required heavy dependency (numpy / matplotlib / jinja2) is not
    importable in the current environment.

    Raised by :func:`render_report` when invoked from a
    ``topofuzz[verify]``-only install (no ``[full]`` extras) so the CLI
    can surface a single friendly install hint rather than a raw
    ``ImportError`` traceback from deep inside the plotting path
    (RT pass-1 MAJOR-2).
    """


# Hard cap on rendered eval-id list in the HTML footer to bound output
# size on pathologically large manifests (RT pass-1 MINOR-3). At 500 the
# HTML stays well under 1 MB even for 100k-eval runs; the manifest still
# carries the full list for downstream tooling.
_EVAL_ID_RENDER_CAP = 500


def render_report(run_dir: Path) -> Path:
    """Render ``<run_dir>/report.html`` and return the written path.

    Raises:
        ReportInputError: ``manifest.json`` or ``run.sqlite`` is missing.
        ValueError: ``run.sqlite`` has zero or multiple ``runs`` rows
            (the v0.5 contract is one run per file).
        ReportDependencyError: numpy / matplotlib / jinja2 unavailable.
            Most likely the user installed ``topofuzz`` without the
            ``[full]`` extras.

    Heavy deps (matplotlib, numpy, jinja2) are imported here, not at
    module scope. The CLI's ``report`` subcommand imports this module
    lazily so a verify-only install (``pip install topofuzz[verify]``)
    that never invokes ``topofuzz report`` stays lean.

    Existing ``report.html`` files in ``run_dir`` are silently
    overwritten — every invocation regenerates from scratch.
    """
    # Probe heavy deps up front so the failure mode is one friendly
    # exception, not a traceback from deep inside a matplotlib call.
    # The actual modules are imported again at use sites (idempotent;
    # Python caches the loader result) — this is a cheap probe.
    _check_heavy_deps()

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    sqlite_path = run_dir / "run.sqlite"
    attestation_path = run_dir / "attestation.json"

    if not manifest_path.exists():
        raise ReportInputError(
            f"manifest.json not found in {run_dir} "
            f"(expected at {manifest_path})"
        )
    if not sqlite_path.exists():
        raise ReportInputError(
            f"run.sqlite not found in {run_dir} "
            f"(expected at {sqlite_path})"
        )

    manifest = _read_json_capped(manifest_path)
    status, run_id = _read_run_row(sqlite_path)

    # Score matrix (variant × eval). Empty matrix is a legitimate outcome
    # for an aborted run with no continuous-judge cells; the plotting
    # path falls back to a placeholder SVG rather than crashing. Any
    # read failure is bubbled up to the user as a banner+log warning
    # (RT pass-1 MAJOR-5 — don't silently swallow at INFO).
    (
        matrix,
        variant_ids,
        eval_ids_from_db,
        matrix_warning,
        per_vendor_matrix,
    ) = _read_matrix_or_empty(run_dir, run_id)

    voo_svg = _render_voo_bar_svg(matrix, eval_ids_from_db, per_vendor_matrix)
    heatmap_svg = _render_heatmap_svg(matrix, variant_ids, eval_ids_from_db)
    influence_svg, influence_rows = _render_influence_panel(
        matrix, variant_ids, manifest
    )
    worstcase_ctx = _render_worstcase_panel(
        matrix, variant_ids, eval_ids_from_db, run_dir, run_id, manifest
    )
    # Render the §14 flip-composition donut BEFORE the §17 summary so
    # the summary panel can consume the donut's signed-vs-live
    # ``drift_flag`` and emit a complementary stale-prose warning.
    # Without this ordering, a reader who only glances at the NLS card
    # silently consumes stale prose when the donut detects drift.
    # (closes BACKLOG: drift propagation into §17 NLS)
    flip_composition_ctx = _render_flip_composition_panel(
        matrix, variant_ids, run_dir, manifest
    )
    summary_ctx = _render_summary_panel(
        run_dir,
        drift_flag=bool(flip_composition_ctx.get("drift_flag", False)),
    )
    variance_distribution_ctx = _render_variance_distribution_panel(
        matrix, eval_ids_from_db
    )

    # Banner gates. attestation.json presence is the structural check,
    # but we additionally require the file to parse as JSON shaped like
    # a DSSE envelope (``signatures`` array). A zero-byte or arbitrary-
    # garbage file would otherwise light up the watermark and mislead
    # downstream consumers (RT pass-1 MAJOR-4). The v0.5 attestation
    # pipeline signs with dev keys (§17a), so a real envelope here
    # implies dev-key by construction.
    show_dev_key = _attestation_looks_signed(attestation_path)
    show_jury = bool(
        manifest.get("cli_args", {}).get("jury_of_3", False)
    )
    show_degraded = status == "degraded"
    # Step 18 (§11 line 306): [REKOR-DEFERRED] gate. Lights iff a
    # ``.rekor-deferred`` marker is on disk AND there is no successful
    # LogEntry alongside it — a re-attest that succeeded should clear
    # the marker, but we AND with looks_like_rekor_logged so a race
    # between marker write and unlink can't strand a stale deferred
    # banner on a successful run.
    from topofuzz.rekor import looks_like_rekor_deferred, looks_like_rekor_logged

    show_rekor_deferred = (
        looks_like_rekor_deferred(run_dir)
        and not looks_like_rekor_logged(run_dir)
    )

    # Prefer manifest run_id (it's the kickoff-time truth) but fall back
    # to the DB row if the manifest is somehow missing the field. A
    # mismatch between the two is a real corruption signal — log
    # WARNING so the operator notices.
    manifest_run_id = manifest.get("run_id")
    if manifest_run_id and manifest_run_id != run_id:
        _log.warning(
            "report: manifest run_id %r differs from runs.sqlite run_id %r — "
            "preferring manifest; this usually means the run dir was edited",
            manifest_run_id, run_id,
        )
    display_run_id = manifest_run_id or run_id

    kickoff_ts = manifest.get("kickoff_ts")
    if isinstance(kickoff_ts, (int, float)):
        kickoff_iso = (
            _datetime.datetime.fromtimestamp(
                float(kickoff_ts), tz=_datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    else:
        kickoff_iso = "(unknown)"

    generated_iso = _datetime.datetime.now(
        tz=_datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Cap the rendered eval-id list to bound HTML size on huge manifests.
    raw_eval_ids = list(manifest.get("eval_ids", []) or [])
    eval_ids_truncated = max(0, len(raw_eval_ids) - _EVAL_ID_RENDER_CAP)
    eval_ids_shown = raw_eval_ids[:_EVAL_ID_RENDER_CAP]
    if eval_ids_truncated:
        # Surface the truncation in the operator's log too (RT pass-2
        # MINOR-3) — the HTML footer is the load-bearing channel but a
        # log line catches scripted invocations that never open the file.
        _log.info(
            "report: capped eval_ids list at %d (%d truncated)",
            _EVAL_ID_RENDER_CAP, eval_ids_truncated,
        )

    wilson_addenda = _build_wilson_addenda(flip_composition_ctx)

    html = _render_template(
        run_id=display_run_id,
        topology_name=manifest.get("topology_name") or "(unknown)",
        status=status,
        judge_model=manifest.get("judge_model") or "(unknown)",
        rubric_hash=manifest.get("rubric_hash") or "",
        n_variants=manifest.get("n_variants", 0),
        n_eval_rows=manifest.get("n_eval_rows", 0),
        kickoff_iso=kickoff_iso,
        generated_iso=generated_iso,
        eval_ids=eval_ids_shown,
        eval_ids_truncated=eval_ids_truncated,
        voo_svg=voo_svg,
        heatmap_svg=heatmap_svg,
        influence_svg=influence_svg,
        influence_rows=influence_rows,
        show_dev_key=show_dev_key,
        show_jury=show_jury,
        show_degraded=show_degraded,
        show_rekor_deferred=show_rekor_deferred,
        matrix_warning=matrix_warning,
        worstcase=worstcase_ctx,
        summary=summary_ctx,
        flip_composition=flip_composition_ctx,
        variance_distribution=variance_distribution_ctx,
        wilson_addenda=wilson_addenda,
    )

    out_path = run_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


# 64 MiB ceiling on the JSON files topofuzz report parses. A
# pathological multi-GB manifest or report.json crafted by a hostile
# run_dir would otherwise pin RAM during ``topofuzz verify`` / ``report``
# (audit step 18.5a v0.5.1 POLISH — closes BACKLOG: 64 MiB cap on
# json.loads). Real artefacts are sub-megabyte; lift if a legitimate use
# case demands more.
_MAX_JSON_BYTES: int = 64 * 1024 * 1024


def _read_json_capped(path: "Path", *, max_bytes: int = _MAX_JSON_BYTES) -> Any:
    """Read+parse a JSON file, refusing files larger than ``max_bytes``.

    Raises ``ValueError`` (the canonical caller-catchable type at every
    existing call site that already handles ``json.JSONDecodeError``)
    when the file is over the cap, so existing try/except blocks pick
    up the new error without surface changes.
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"{path.name} is {size} bytes (> {max_bytes} cap); refusing "
            "to parse to bound RAM. Edit _MAX_JSON_BYTES if intentional."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _check_heavy_deps() -> None:
    """Probe numpy / matplotlib / jinja2 importability; raise on miss.

    The CLI catches :class:`ReportDependencyError` and surfaces a one-
    line install hint. Without this gate, ``topofuzz report`` on a
    verify-only install produces a raw ``ImportError`` from inside
    :func:`_read_matrix_or_empty` (numpy) or :func:`_setup_matplotlib`
    (matplotlib) — much harder for the user to diagnose.

    Uses :func:`importlib.util.find_spec` rather than ``__import__`` so
    the probe has zero side-effects (RT pass-2 NIT-1). ``__import__``
    would run each module's top-level code, which for matplotlib
    initialises rcparams and lazy-binds a backend; ``find_spec`` is a
    pure metadata lookup.
    """
    import importlib.util

    missing = [
        mod for mod in ("numpy", "matplotlib", "jinja2")
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        raise ReportDependencyError(
            f"topofuzz report needs {', '.join(missing)} (not installed). "
            "Install the [full] extras: `pip install 'topofuzz[full]'`."
        )


def _attestation_looks_signed(path: Path) -> bool:
    """Return True iff ``path`` parses as a DSSE envelope.

    We require the file to (a) exist, (b) parse as JSON, and (c) carry a
    non-empty ``signatures`` array — the DSSE v1.0 invariant the topofuzz
    signer always emits. Anything weaker (file-exists-only) lets a
    zero-byte ``attestation.json`` light up the ``[DEV-KEY]`` watermark
    on an unsigned run (RT pass-1 MAJOR-4). Full cryptographic verify is
    still ``topofuzz verify``'s job; this is the watermark gate only.

    On read failure (permission denied, broken symlink) we log WARNING
    and return False so the operator sees a reason for the missing
    watermark rather than puzzling over an unsigned-looking run (RT
    pass-2 NIT-2).
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        # Common case: unsigned run never wrote attestation.json. The
        # call site already expected this — return quietly. Closes
        # BACKLOG: misleading "present but unreadable" log.
        return False
    except OSError as e:
        _log.warning(
            "report: attestation.json present but unreadable (%s) — "
            "[DEV-KEY] banner suppressed", e,
        )
        return False
    if not raw:
        return False
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(envelope, dict):
        return False
    sigs = envelope.get("signatures")
    return isinstance(sigs, list) and len(sigs) > 0


# ---------------------------------------------------------------------------
# DB reads
# ---------------------------------------------------------------------------


def _read_run_row(sqlite_path: Path) -> tuple[str, str]:
    """Return ``(status, run_id)`` from the single ``runs`` row.

    The v0.5 contract is one run per ``run.sqlite``. A zero-row or
    multi-row DB is a hard error here — the report needs a definitive
    status to drive the ``[DEGRADED]`` banner gate.
    """
    db = sqlite3.connect(str(sqlite_path))
    try:
        # SELECT order is (status, run_id) to match the return tuple
        # order — keeps the cognitive load on the caller side at zero.
        rows = db.execute("SELECT status, run_id FROM runs").fetchall()
    except sqlite3.DatabaseError as e:
        raise ReportInputError(
            f"run.sqlite at {sqlite_path} could not be read: {e}"
        ) from e
    finally:
        db.close()
    if not rows:
        raise ValueError(
            f"run.sqlite at {sqlite_path} has no rows in the runs table"
        )
    if len(rows) != 1:
        raise ValueError(
            f"run.sqlite at {sqlite_path} has {len(rows)} runs rows; "
            "the v0.5 report contract is one run per file"
        )
    status, run_id = rows[0]
    return status, run_id


def _build_wilson_addenda(
    flip_composition_ctx: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Wilson 95% score interval for the verdict-flip rate.

    "wilsons at the bottom for anyone who cares" — Wilson is a tighter,
    boundary-safe CI for a binomial proportion. The bootstrap CI
    already lives on ``headline.verdict_flip_rate`` (the "if I
    resampled evals" question); Wilson answers the orthogonal "is the
    observed rate distinguishable from zero?" question. Same closure as
    :func:`topofuzz.attest.compute_verdict_flip_for_attestation`.

    Returns ``None`` when the flip-composition context is missing or
    carries no live flip counts — the addenda block is silently
    omitted from the report, not rendered with placeholders.
    """
    if not isinstance(flip_composition_ctx, dict):
        return None
    n_flips = flip_composition_ctx.get("n_flips")
    n_pairs = flip_composition_ctx.get("n_pairs")
    # Pass-2 M-RT-P2-1: ``bool`` is a subclass of ``int`` in Python;
    # without the explicit reject a True/False slipping through would
    # crash :func:`wilson_score_interval`. Mirrors the score-validation
    # discipline in :func:`_try_build_per_vendor_matrix`.
    if (
        isinstance(n_flips, bool)
        or isinstance(n_pairs, bool)
        or not isinstance(n_flips, int)
        or not isinstance(n_pairs, int)
    ):
        return None
    if n_pairs <= 0 or n_flips < 0 or n_flips > n_pairs:
        return None
    from topofuzz.metrics import wilson_score_interval

    lo, hi = wilson_score_interval(n_flips, n_pairs, conf=0.95)
    if lo is None or hi is None:
        return None
    return {
        "n_flips": n_flips,
        "n_pairs": n_pairs,
        "point": float(n_flips) / float(n_pairs),
        "ci_lower": lo,
        "ci_upper": hi,
        "conf_pct": 95,
    }


def _read_matrix_or_empty(
    run_dir: Path, run_id: str
) -> tuple[
    "np.ndarray",
    tuple[str, ...],
    tuple[str, ...],
    str | None,
    "np.ndarray | None",
]:
    """Read the continuous-score matrix.

    Returns ``(matrix, variant_ids, eval_ids, warning, per_vendor_matrix)``.
    ``warning`` is ``None`` on a clean read and a short human-readable
    string when the matrix had to be degraded to empty (RT pass-1
    MAJOR-5 — empty matrix is silent today; we now surface the reason
    so the user sees it both in the log AND in the rendered HTML).
    ``per_vendor_matrix`` is the F3 jury-aware companion slab (``None``
    for single-judge / legacy-v5 runs); the report's bar SVG dispatches
    on it to escape the Bernoulli-bounded majority for jury runs.

    An empty matrix is a legitimate outcome for an aborted run with no
    continuous-judge cells; the plotting helpers handle that case with a
    placeholder SVG so the report still renders. Any read error
    (corrupted DB, schema mismatch) also degrades to empty + warning so
    the user gets a report instead of a crash, but they're told why the
    figures are blank.
    """
    from topofuzz.metrics import read_score_matrix

    try:
        result = read_score_matrix(
            run_dir,
            run_id=run_id,
            allow_empty_rubric_hash=True,
        )
    except (FileNotFoundError, ValueError, sqlite3.DatabaseError) as e:
        msg = f"score matrix unavailable: {e}"
        _log.warning("report: %s — rendering placeholders", msg)
        return _empty_matrix(), (), (), msg, None
    return (
        result.matrix,
        result.variant_ids,
        result.eval_ids,
        None,
        result.per_vendor_matrix,
    )


def _empty_matrix() -> "np.ndarray":
    import numpy as np

    return np.empty((0, 0), dtype=np.float64)


# ---------------------------------------------------------------------------
# SVG renderers
# ---------------------------------------------------------------------------


def _setup_matplotlib() -> Any:
    """Force Agg backend, then return :mod:`matplotlib.pyplot`.

    Backend selection MUST happen before pyplot is first imported in
    the process — once matplotlib has bound a backend, ``matplotlib.use``
    only warns. Headless servers (CI, Docker, no DISPLAY) require Agg
    or the import crashes.

    ``force=False`` (RT pass-1 NIT-1): if a *prior* import in the same
    process already bound a non-Agg backend (e.g. a notebook running
    ``%matplotlib inline``), this call is a silent no-op and figures
    render on the previously-bound backend. For the CLI process (which
    only enters here) the first call is the Agg-bound one and we're
    safe. Library callers embedding :func:`render_report` in a notebook
    should pre-bind Agg themselves.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _strip_svg_preamble(svg_bytes: bytes) -> str:
    """Drop the ``<?xml ...?>`` + DOCTYPE so the SVG embeds inside HTML.

    Defensive: if the input is missing a ``<svg`` tag (would mean
    matplotlib's SVG backend produced output we don't recognise — a
    real regression, not a recoverable case), raise loudly rather than
    let an empty body silently propagate to the rendered HTML (RT pass-1
    NIT-4).
    """
    if b"<svg" not in svg_bytes:
        # "internal:" prefix flags this as a matplotlib regression /
        # backend bug, NOT bad user input (RT pass-2 MINOR-4). The CLI's
        # generic ValueError-→-exit-2 path would otherwise conflate this
        # with a bad-usage error.
        raise ValueError(
            "internal: SVG bytes from matplotlib backend contain no "
            "'<svg' tag — likely a matplotlib regression or wrong "
            "output format. Please file an issue."
        )
    return _SVG_PRE_TAG_RE.sub(b"", svg_bytes).decode("utf-8")


def _placeholder_svg(message: str, *, subtitle: str | None = None) -> str:
    """Tiny static SVG used when a chart can't be rendered.

    ``message`` is the headline — keep under ~55 chars to fit.
    ``subtitle`` is an optional second line below it, e.g. plain-English
    explanation of why the chart is empty.
    """
    def _safe(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    height = 120 if subtitle is None else 150
    y_head = 65 if subtitle is None else 58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="{height}" '
        f'viewBox="0 0 480 {height}">',
        f'<rect width="480" height="{height}" fill="#f3f3f3" stroke="#bbb"/>',
        f'<text x="240" y="{y_head}" text-anchor="middle" '
        'font-family="ui-monospace, Menlo, Consolas, monospace" '
        f'font-size="14" fill="#444">{_safe(message)}</text>',
    ]
    if subtitle:
        parts.append(
            '<text x="240" y="92" text-anchor="middle" '
            'font-family="-apple-system, Segoe UI, sans-serif" '
            f'font-size="12" fill="#777">{_safe(subtitle)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_voo_bar_svg(
    matrix: "np.ndarray",
    eval_ids: tuple[str, ...],
    per_vendor_matrix: "np.ndarray | None" = None,
) -> str:
    """Per-eval variance across orderings as a bar chart.

    Dispatches via :func:`topofuzz.metrics.per_eval_variance`:
    single-judge runs keep the canonical
    ``np.var(matrix[:, j], ddof=0)`` closure; F3 jury runs use the
    per-vendor-averaged per-eval variance so the bar isn't
    Bernoulli-bounded at 0.25.
    """
    import numpy as np

    from topofuzz.metrics import per_eval_variance

    if matrix.size == 0 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        return _placeholder_svg("no continuous-judge cells to score")

    plt = _setup_matplotlib()
    per_eval = per_eval_variance(matrix, per_vendor_matrix)
    # NaN-column evals (one or more failed cells in that column) get a
    # zero-height bar — the alternative (skip them) would hide the fact
    # that the cell exists at all. NaN propagation in np.var is the
    # contract from `read_score_matrix` (failed cells are NaN, not 0).
    plot_values = np.where(np.isnan(per_eval), 0.0, per_eval)
    n = len(plot_values)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    try:
        x = np.arange(n)
        bars = ax.bar(x, plot_values, color="#3b6fb0")
        # Mark NaN-column bars distinctly so the reader knows a 0 here
        # means "data missing" rather than "zero variance".
        for i, v in enumerate(per_eval):
            if np.isnan(v):
                bars[i].set_color("#cccccc")
                bars[i].set_hatch("//")
        ax.set_xlabel("eval (column)")
        ax.set_ylabel("variance across variants")
        ax.set_title("Per-eval variance across orderings")
        if n <= _HEATMAP_TICK_LABEL_CAP:
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_short(e) for e in eval_ids],
                rotation=45,
                ha="right",
                fontsize=8,
            )
        else:
            ax.set_xticks([])
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(
            buf,
            format="svg",
            # metadata={'Date': None} suppresses matplotlib's embedded
            # generation timestamp. SVG output is best-effort byte-stable
            # across renders on the same matrix at the same matplotlib
            # version — `fig.tight_layout` is allowed to drift sub-pixel
            # across matplotlib upgrades, so a golden-file test would need
            # to pin the matplotlib version too (CR pass-1 MINOR-2).
            metadata={"Date": None},
        )
    finally:
        plt.close(fig)
    return _strip_svg_preamble(buf.getvalue())


def _render_heatmap_svg(
    matrix: "np.ndarray",
    variant_ids: tuple[str, ...],
    eval_ids: tuple[str, ...],
) -> str:
    """Continuous-judge score heatmap (variants × evals)."""
    import numpy as np

    if matrix.size == 0 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        return _placeholder_svg("no continuous-judge cells to render")

    plt = _setup_matplotlib()
    # Mask NaN entries so they render as the colormap's `bad` color
    # (we set this to a flat gray below — visually unambiguous against
    # the viridis-cool palette).
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("viridis").with_extremes(bad="#cccccc")

    n_variants, n_evals = matrix.shape
    # Heatmaps with too many rows turn into a thin sliver; clamp the
    # aspect ratio so a 1-variant matrix still has a visible row.
    height = max(2.0, min(6.0, 1.0 + 0.4 * n_variants))
    width = max(4.0, min(9.0, 2.0 + 0.35 * n_evals))
    fig, ax = plt.subplots(figsize=(width, height))
    try:
        im = ax.imshow(
            masked,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel("eval")
        ax.set_ylabel("variant")
        ax.set_title("Score matrix (continuous judge)")
        if n_evals <= _HEATMAP_TICK_LABEL_CAP:
            ax.set_xticks(np.arange(n_evals))
            ax.set_xticklabels(
                [_short(e) for e in eval_ids],
                rotation=45,
                ha="right",
                fontsize=7,
            )
        else:
            ax.set_xticks([])
        if n_variants <= _HEATMAP_TICK_LABEL_CAP:
            ax.set_yticks(np.arange(n_variants))
            ax.set_yticklabels(
                [_short(v) for v in variant_ids], fontsize=7
            )
        else:
            ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
    finally:
        plt.close(fig)
    return _strip_svg_preamble(buf.getvalue())


def _short(label: str, *, limit: int = 12) -> str:
    """Truncate canonical_hash / eval-id labels for axis legibility."""
    s = str(label)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


# Hard cap on rows rendered in the influence-panel table so a topology
# with ~50 agents (unlikely in v0.5 but not impossible) doesn't blow up
# the HTML footprint. 32 is the same heuristic as _HEATMAP_TICK_LABEL_CAP.
_INFLUENCE_ROW_RENDER_CAP = 32


def _render_influence_panel(
    matrix: "np.ndarray",
    variant_ids: tuple[str, ...],
    manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Render the §6a per-agent influence panel.

    Returns ``(svg, rows)``. ``svg`` is a horizontal bar chart (one bar
    per agent, error bars = 95% bootstrap CI) or the placeholder SVG
    when influence is not computable (old run, too few variants, no
    score cells). ``rows`` is the list of dicts the template iterates
    over for the textual table fallback — exactly what the attestation
    ``findings.per_agent_influence`` field carries.

    The manifest path is the source of truth for ``variant_orderings``;
    we align them against the score matrix's canonical-hash axis the
    same way :func:`topofuzz.attest.compute_influence_for_attestation`
    does. The two callers compute identical influence rows from the
    same run dir — verified in tests so the signed and rendered views
    cannot drift.
    """
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return _placeholder_svg("influence unavailable: score matrix empty"), []

    try:
        from topofuzz.metrics import (
            OrderingAlignmentError,
            align_orderings_to_matrix,
            compute_influence,
        )
    except ImportError:
        return _placeholder_svg("influence unavailable: metrics not installed"), []

    # Strict alignment (audit RT-B-1 / RT-B-2): duplicates / mismatches
    # / unknown hashes degrade to a placeholder + logged WARNING. The
    # placeholder message uses the structured reason slug from the
    # exception, not the raw ValueError text, so we never accidentally
    # leak diagnostic detail into the rendered HTML.
    try:
        aligned = align_orderings_to_matrix(manifest, variant_ids)
    except OrderingAlignmentError as e:
        if e.reason == "missing":
            # Pre-step-15 run dir; the operator already knows the field
            # is new, no reason to spam the log on every report render.
            # Mirrors the attest path's policy (RT pass-2 M-1).
            return (
                _placeholder_svg("influence unavailable: no orderings in manifest"),
                [],
            )
        _log.warning("report: influence skipped (%s): %s", e.reason, e)
        return (
            _placeholder_svg(f"influence unavailable: manifest {e.reason}"),
            [],
        )

    try:
        comp = compute_influence(matrix, aligned, seed=0)
    except (ValueError, ImportError) as e:
        _log.warning("report: influence computation failed: %s", e)
        return _placeholder_svg("influence unavailable: computation error"), []

    rows: list[dict[str, Any]] = []
    for r in comp.results:
        rows.append(
            {
                "node_id": r.node_id,
                "influence": r.influence,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "degenerate": r.degenerate,
                "degenerate_reason": r.degenerate_reason,
            }
        )
    if not rows:
        return _placeholder_svg("influence unavailable: no agents"), []
    # The SVG bar chart caps to keep the figure legible at default size,
    # but the rendered rows table keeps every row so it matches the
    # attestation `findings.per_agent_influence` list verbatim
    # (audit CR-M2 — signed and rendered views must not silently diverge).
    bar_rows = rows[:_INFLUENCE_ROW_RENDER_CAP]
    svg = _render_influence_bar_svg(bar_rows)
    return svg, rows


def _render_influence_bar_svg(rows: list[dict[str, Any]]) -> str:
    """Horizontal bar chart, one bar per agent, with 95% CI error bars."""
    import numpy as np

    plot_rows = [r for r in rows if r["influence"] is not None]
    if not plot_rows:
        # Every agent's per-agent attribution collapsed — no row has a
        # usable influence number. Most common cause: zero verdict flips
        # in the run (nothing to split between agents). Less common
        # causes: each agent moved <2 times, or no scored cells covered
        # the contrast. Surface this in plain English instead of the old
        # cryptic "every agent is degenerate" line.
        return _placeholder_svg(
            "No agent stood out as the cause.",
            subtitle=(
                "Most likely your AI gave the same answer no matter how the "
                "agents were ordered."
            ),
        )

    plt = _setup_matplotlib()
    labels = [_short(r["node_id"], limit=20) for r in plot_rows]
    values = np.array([float(r["influence"]) for r in plot_rows], dtype=float)
    # CI bounds and the point estimate are emitted together by
    # compute_influence — degenerate rows carry None for all three and
    # are already filtered out above. n_bootstrap=0 leaves the point
    # populated and both bounds None; in that path the error bars
    # collapse to zero width.
    lo = np.array(
        [
            float(r["ci_lower"]) if r["ci_lower"] is not None else float(r["influence"])
            for r in plot_rows
        ],
        dtype=float,
    )
    hi = np.array(
        [
            float(r["ci_upper"]) if r["ci_upper"] is not None else float(r["influence"])
            for r in plot_rows
        ],
        dtype=float,
    )
    # matplotlib errorbar expects (lower_offset, upper_offset), both >= 0.
    err_lo = np.maximum(values - lo, 0.0)
    err_hi = np.maximum(hi - values, 0.0)

    n = len(plot_rows)
    height = max(2.0, min(6.0, 0.4 * n + 1.0))
    fig, ax = plt.subplots(figsize=(7.5, height))
    try:
        y_pos = np.arange(n)
        # Stable two-tone colouring: positive = blue (driver), negative
        # = green (stabilising). No clamping per the step-15 decision.
        colors = ["#3b6fb0" if v >= 0 else "#3da06e" for v in values]
        ax.barh(y_pos, values, color=colors, alpha=0.85)
        ax.errorbar(
            values,
            y_pos,
            xerr=[err_lo, err_hi],
            fmt="none",
            ecolor="#1a1a1a",
            elinewidth=1.0,
            capsize=3,
        )
        ax.axvline(0.0, color="#888", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()  # largest influence at top, matches ranking
        ax.set_xlabel("influence  (P(flip | A moved) − P(flip | A same slot))")
        ax.set_title("Per-agent influence with 95% bootstrap CI")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
    finally:
        plt.close(fig)
    return _strip_svg_preamble(buf.getvalue())


# ---------------------------------------------------------------------------
# Worst-case viewer (§10a / step 16) — Jinja-side adapter
# ---------------------------------------------------------------------------


def _render_worstcase_panel(
    matrix: "np.ndarray",
    variant_ids: tuple[str, ...],
    eval_ids: tuple[str, ...],
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the Jinja context dict for the §10a worst-case section.

    The data layer lives in :mod:`topofuzz.metrics.worstcase` (step 16).
    Here we degrade gracefully on any failure — a crash inside the
    panel must not take down the rest of the report (banners, voo,
    heatmap, influence). On ANY exception we surface a one-line
    warning and return a disabled-state context with a diagnostic
    reason.
    """
    try:
        from topofuzz.metrics import compute_worstcase_panel
    except ImportError as e:
        _log.warning(
            "report: worstcase metrics unavailable (%s) — section disabled",
            e,
        )
        return {
            "disabled": True,
            "disabled_reason": "metrics module unavailable",
            "cases": [],
        }

    try:
        panel = compute_worstcase_panel(
            matrix,
            variant_ids,
            eval_ids,
            run_dir=run_dir,
            run_id=run_id,
            manifest=manifest,
        )
    except (
        ValueError, ImportError, TypeError, OSError,
        sqlite3.DatabaseError, json.JSONDecodeError,
    ) as e:
        # Audit pass-2 RT-MAJOR-4: narrowed from ``Exception`` so
        # ``MemoryError``/``RecursionError``/``KeyboardInterrupt`` /
        # ``SystemExit`` propagate. The exception text is NOT
        # interpolated into the log because the underlying message can
        # carry chunks of user prompt data (e.g. a malformed
        # ``node_outputs_json`` row whose ``e.args[0]`` is the
        # offending JSON); centralised log shippers would then quietly
        # exfiltrate. Type + the file/run pair is enough for an
        # operator to find the row themselves.
        _log.warning(
            "report: worstcase panel raised %s — section disabled "
            "(run_dir=%s, run_id=%r)",
            type(e).__name__, run_dir, run_id,
        )
        return {
            "disabled": True,
            "disabled_reason": "computation error",
            "cases": [],
        }

    if panel.disabled:
        return {
            "disabled": True,
            "disabled_reason": panel.disabled_reason,
            "cases": [],
        }

    # Convert the frozen dataclasses to plain dicts. Jinja can iterate
    # dataclass attrs directly but the template stays readable when it
    # walks dicts, and the test suite asserts panel-side dataclass
    # semantics separately.
    cases_out = []
    for case in panel.cases:
        cases_out.append(
            {
                "eval_id": case.eval_id,
                "flip_count": case.flip_count,
                "n_pairs": case.n_pairs,
                "variant_a": _side_to_dict(case.variant_a),
                "variant_b": _side_to_dict(case.variant_b),
            }
        )
    return {
        "disabled": False,
        "disabled_reason": None,
        "cases": cases_out,
    }


def _side_to_dict(side: Any) -> dict[str, Any]:
    """Convert :class:`VariantSideView` to a Jinja-friendly dict.

    Both the truncated display label and the full canonical_hash are
    surfaced so the rendered card can show a clipped hash inline AND
    a ``title=`` tooltip for the operator who needs the full value
    (audit RT-MINOR-6).
    """
    return {
        "variant_id": side.variant_id,
        "variant_id_short": _short(side.variant_id, limit=16),
        "ordering": list(side.ordering),
        "score": side.score,
        "verdict": bool(side.verdict),
        "rationale": side.rationale,
        "node_outputs": [
            {
                "node_id": n.node_id,
                "role": n.role,
                "response_text": n.response_text,
            }
            for n in side.node_outputs
        ],
    }


# ---------------------------------------------------------------------------
# Natural-language summary (§17)
# ---------------------------------------------------------------------------


def _render_summary_panel(
    run_dir: Path, *, drift_flag: bool = False,
) -> dict[str, Any]:
    """Build the §17 natural-language summary Jinja context.

    Sources both ``headline.verdict_flip_rate`` and
    ``findings.per_agent_influence`` from ``<run_dir>/report.json`` —
    the plain-JSON mirror persisted by step 15a, which is in turn the
    plain mirror of the signed attestation. Reading the persisted
    artifact (rather than recomputing) means the rendered summary
    matches what was signed and what downstream consumers reading
    ``report.json`` will see. A pre-step-15a run (no ``report.json``)
    degrades cleanly to the ``unavailable`` placeholder.

    On any exception we degrade to a placeholder context with
    ``verdict_word="unavailable"`` so the rest of the report renders.
    The summary section is best-effort, not load-bearing.

    ``drift_flag`` is the §14 donut's signed-vs-live drift signal
    (``flip_composition.drift_flag`` — see
    :func:`_render_flip_composition_panel`). When True, the donut
    detected that ``report.json``'s signed ``verdict_flip_rate.point``
    disagrees with the live ``n_flips / n_pairs`` recount by more
    than :data:`_FLIP_DRIFT_TOLERANCE`. The §17 prose is derived from
    the same signed value, so the prose is silently stale on drift —
    we surface a complementary warning so a reader who only glances
    at the NLS card doesn't consume stale data. The warning is added
    ONLY on the success path; the ``unavailable`` placeholder paths
    have no prose to be stale, so they suppress the flag.
    """
    try:
        from topofuzz.summary import render_summary
    except ImportError as e:
        _log.warning(
            "report: summary module unavailable (%s) — section disabled",
            e,
        )
        return {
            "verdict_word": "unavailable",
            "summary_text": (
                "Topology stability summary unavailable "
                "(topofuzz.summary module not importable)."
            ),
            "summary_html": (
                "Topology stability summary unavailable "
                "(topofuzz.summary module not importable)."
            ),
            # Shape stability for Python consumers using ``ctx["drift_flag"]``
            # rather than ``.get(...)``; the placeholder has no prose to be
            # stale, so the flag is explicitly suppressed (audit pass-1 MINOR).
            "drift_flag": False,
        }

    report_json_path = run_dir / "report.json"
    verdict_flip_rate: dict[str, Any] | None = None
    per_agent_influence: list[dict[str, Any]] | None = None
    if report_json_path.is_file():
        try:
            doc = _read_json_capped(report_json_path)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            _log.warning(
                "report: report.json unreadable (%s) — summary degrades to "
                "unavailable (run_dir=%s)",
                type(e).__name__, run_dir,
            )
            doc = None
        if isinstance(doc, dict):
            headline = doc.get("headline")
            if isinstance(headline, dict):
                vf = headline.get("verdict_flip_rate")
                if isinstance(vf, dict):
                    verdict_flip_rate = vf
            findings = doc.get("findings")
            if isinstance(findings, dict):
                rows = findings.get("per_agent_influence")
                if isinstance(rows, list):
                    per_agent_influence = rows

    try:
        ctx = render_summary(verdict_flip_rate, per_agent_influence)
    except (
        ValueError, TypeError, ArithmeticError, AttributeError,
    ) as e:
        # Narrowed catch-list mirroring step 16's _render_worstcase_panel
        # audit pass-2 RT-MAJOR-4: ``MemoryError`` / ``RecursionError`` /
        # ``KeyboardInterrupt`` / ``SystemExit`` must propagate. The
        # realistic failure modes of pure-stdlib text formatting are
        # arithmetic (overflow / NaN format), type confusion, attribute
        # errors on a malformed input row, and value errors from format
        # string mismatches. Exception type only (no repr) keeps PII
        # out of operator logs (audit pass-2 RT log-leak discipline).
        _log.warning(
            "report: summary render raised %s — section disabled "
            "(run_dir=%s)",
            type(e).__name__, run_dir,
        )
        return {
            "verdict_word": "unavailable",
            "summary_text": (
                "Topology stability summary unavailable "
                "(internal render error)."
            ),
            "summary_html": (
                "Topology stability summary unavailable "
                "(internal render error)."
            ),
            # Shape stability — see import-failure placeholder above.
            "drift_flag": False,
        }

    out: dict[str, Any] = {
        "verdict_word": ctx.verdict_word,
        "verdict_flip_rate": ctx.verdict_flip_rate,
        "primary_driver": ctx.primary_driver,
        "primary_influence": ctx.primary_influence,
        "summary_text": ctx.summary_text,
        "summary_html": ctx.summary_html,
        "drift_flag": bool(drift_flag),
    }
    if drift_flag:
        # Static HTML — no untrusted interpolation. The donut's drift
        # warning already cites the numeric drift; this companion warning
        # speaks to prose staleness specifically and points the reader
        # at the §14 visual for the numeric details. Kept short on
        # purpose so it doesn't visually compete with the verdict prose.
        out["drift_warning_html"] = (
            'Note: the verdict above was derived from the attested '
            '<code>report.json</code>, but the live results database '
            'disagrees with that attestation '
            '(see the §14 donut below for the numeric drift). The '
            'prose may be stale — re-run <code>topofuzz attest</code> '
            'to refresh.'
        )
    return out


# ---------------------------------------------------------------------------
# Step 18.5b — flip-composition donut + per-eval variance sparkline
#
# Both helpers hand-roll their SVG (no matplotlib) per the Variant D sketch
# contract — these are the at-a-glance "headline" visuals that complement
# the existing matplotlib voo bar + heatmap. Hand-rolling keeps the SVG
# byte-stable across matplotlib upgrades and avoids spinning a figure
# canvas just to draw 50 rectangles.
# ---------------------------------------------------------------------------


# Verdict-flip threshold sourced from :mod:`topofuzz.metrics.verdict_flip`.
# Closes BACKLOG: lift _FLIP_THRESHOLD. The metrics submodule is stdlib-
# only at import time (numpy stays behind TYPE_CHECKING + lazy func-level
# imports), so this stays cheap for verify-only callers.
from topofuzz.metrics.verdict_flip import DEFAULT_FLIP_THRESHOLD as _FLIP_THRESHOLD

# §10a worst-case gate: per-eval variance ≥ this value promotes the
# eval into the worst-case viewer's candidate set. Sourced from
# :mod:`topofuzz.metrics.worstcase` lazily (inside
# :func:`_render_variance_distribution_panel`) so the sparkline's gate
# matches the §10a promoter's gate exactly. Uses ``>=`` (audit RT-3:
# the worstcase promoter at ``worstcase.py:356`` is inclusive; the
# sparkline must agree on boundary values).
_WORSTCASE_GATE_FALLBACK: float = 0.1

# Hard cap on bars rendered in the sparkline. Above ~200 the bar widths
# collapse below 1 SVG-unit and bars stop being visually distinguishable;
# the headline counts (n_evals, n_above_gate) still cover the full set.
_SPARKLINE_BAR_RENDER_CAP: int = 200

# Tolerance for the donut's signed-vs-live drift check. A clean run
# satisfies ``abs(live_rate - signed_rate) ≤ 0.005`` by construction
# (both numerators come from the same masking, threshold, and orig_idx);
# the only way they diverge is when ``run.sqlite`` was edited between
# attestation and ``topofuzz report``. We surface the drift with a log
# WARNING + a Jinja-side annotation; we do NOT disable the donut, since
# the drift case still has informational content (the user just needs
# to know which number is the signed truth).
_FLIP_DRIFT_TOLERANCE: float = 0.005


def _render_flip_composition_panel(
    matrix: "np.ndarray",
    variant_ids: tuple[str, ...],
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the Jinja context for the §14 flip-composition donut.

    Sources the headline number from
    ``<run_dir>/report.json``'s ``headline.verdict_flip_rate.point`` (the
    step-15a plain mirror of the signed attestation) so the rendered
    percentage matches what was signed. Computes the approve→deny vs
    deny→approve directional split from the in-memory score matrix
    using the SAME threshold (``_FLIP_THRESHOLD``) and the SAME §6
    "original variant = manifest insertion order" convention as
    :func:`compute_verdict_flip_rate` — that's why we delegate the
    original-index resolution to :func:`_resolve_original_variant_index`
    in :mod:`topofuzz.attest`. Sharing a single resolver means the
    donut and the signed payload cannot disagree on which variant was
    the "original".

    Returns one of:

    * ``{"disabled": True, "disabled_reason": str}`` — donut hidden.
      Reasons: ``"verdict-flip rate unavailable"`` (no report.json
      or malformed), ``"score matrix empty"``, ``"single-variant
      run (no permuted variants to compare)"``, ``"score matrix
      has zero eval columns"``, ``"attest helper unavailable"``,
      ``"original variant unresolvable"``, ``"direction split
      computation error"``.
    * ``{"disabled": False, "flip_rate": float,
      "flip_rate_pct_int": int, "live_pct_int": int, "n_flips":
      int, "n_pairs": int, "approve_to_deny": int,
      "deny_to_approve": int, "drift_flag": bool, "donut_svg":
      str}`` — donut rendered. ``flip_rate_pct_int`` is the signed
      percent (donut center); ``live_pct_int`` is the live recount
      percent (legend). They agree in clean runs; ``drift_flag``
      is set when they diverge by more than
      :data:`_FLIP_DRIFT_TOLERANCE`.

    Degrades on any exception (same pattern as
    :func:`_render_worstcase_panel`) — a crash here must not take
    down the rest of the report.
    """
    flip_rate = _read_flip_rate_from_report_json(run_dir)
    if flip_rate is None:
        return {
            "disabled": True,
            "disabled_reason": "verdict-flip rate unavailable",
        }

    # Audit RT pass-1 MINOR-2: branch the disabled reason so the user
    # sees a precise diagnostic. The three matrix-shape failure modes
    # are distinct and the operator's next step differs for each.
    if matrix.size == 0:
        return {
            "disabled": True,
            "disabled_reason": "score matrix empty",
        }
    if matrix.shape[0] < 2:
        return {
            "disabled": True,
            "disabled_reason": "single-variant run (no permuted variants to compare)",
        }
    if matrix.shape[1] < 1:
        return {
            "disabled": True,
            "disabled_reason": "score matrix has zero eval columns",
        }

    # Lazy import — :mod:`topofuzz.attest` is heavy enough that we don't
    # want it on the verify-only fast path. The lazy import mirrors
    # :func:`_render_summary_panel`'s pattern.
    try:
        from topofuzz.attest import _resolve_original_variant_index
    except ImportError as e:
        # Audit pass-2 Cross-step MAJOR-2: log-discipline — exception
        # type only, never the bare exception object. Locked from
        # step 18a commit ac28ac7.
        _log.warning(
            "report: flip-composition disabled (attest helper "
            "unavailable: %s)", type(e).__name__,
        )
        return {
            "disabled": True,
            "disabled_reason": "attest helper unavailable",
        }

    try:
        orig_idx = _resolve_original_variant_index(manifest, variant_ids)
    except (TypeError, ValueError, KeyError) as e:
        _log.warning(
            "report: flip-composition raised %s during original-variant "
            "resolution — section disabled (run_dir=%s)",
            type(e).__name__, run_dir,
        )
        return {
            "disabled": True,
            "disabled_reason": "original variant unresolvable",
        }
    if orig_idx is None:
        return {
            "disabled": True,
            "disabled_reason": "original variant unresolvable",
        }

    try:
        a2d, d2a, n_pairs = _count_flip_directions(matrix, orig_idx)
    except (ValueError, ImportError, TypeError, IndexError) as e:
        # Audit RT pass-1 MINOR-1: ``IndexError`` added for
        # defense-in-depth against a future resolver refactor that
        # could return an out-of-range int (today's
        # ``_resolve_original_variant_index`` cannot, but the donut
        # must not crash and take down the rest of the report).
        _log.warning(
            "report: flip-composition direction count raised %s — "
            "section disabled (run_dir=%s)",
            type(e).__name__, run_dir,
        )
        return {
            "disabled": True,
            "disabled_reason": "direction split computation error",
        }

    n_flips = a2d + d2a

    # Audit consensus (Code-reviewer MAJOR + RT MAJOR + Blueprint
    # MAJOR): detect signed-vs-live drift. The signed ``flip_rate``
    # from ``report.json`` and the live ``n_flips/n_pairs`` produce
    # an identical numerator and denominator on cells that
    # ``compute_verdict_flip_rate`` includes in ``finite_cols``: on
    # zero-finite-pair columns both sums add zero, so the rate
    # matches. They diverge ONLY when ``run.sqlite`` was edited
    # between attestation and ``topofuzz report``. Surface a small
    # annotation so the legend doesn't quietly contradict the donut
    # center; do NOT disable the section, since both numbers are
    # still informational and the user needs to see them to diagnose.
    #
    # Audit pass-2 Code-reviewer MAJOR-2: the ``n_pairs > 0`` guard
    # also covers the silent-contradiction edge case. When the live
    # matrix has zero finite cell-pairs (every column has either a
    # NaN original or a NaN permuted) AND the signed flip_rate is
    # non-zero, the donut still has to surface drift — otherwise the
    # legend reads "0 of 0 pairs flipped (44%)" which is the exact
    # contradiction the detector exists for.
    drift_flag = False
    if n_pairs > 0:
        live_rate = float(n_flips) / float(n_pairs)
        if abs(live_rate - flip_rate) > _FLIP_DRIFT_TOLERANCE:
            drift_flag = True
            _log.warning(
                "report: signed verdict_flip_rate.point=%.4f differs "
                "from live n_flips/n_pairs=%.4f (run_dir=%s) — "
                "report.json may be stale relative to run.sqlite",
                flip_rate, live_rate, run_dir,
            )
    elif flip_rate > _FLIP_DRIFT_TOLERANCE:
        # Zero pairs but a signed non-zero rate — definitionally a
        # stale-mirror case. Set the flag so the template renders
        # the warning instead of "0 of 0 pairs flipped (X%)".
        drift_flag = True
        _log.warning(
            "report: signed verdict_flip_rate.point=%.4f but live "
            "matrix has zero finite cell-pairs (run_dir=%s) — "
            "report.json may be stale relative to run.sqlite",
            flip_rate, run_dir,
        )

    return {
        "disabled": False,
        "flip_rate": flip_rate,
        # Pre-rounded for the donut center text — Jinja-side
        # ``'%d' % (flip_rate * 100)`` would be a banker's-rounding
        # surprise for the 0.005 boundary. Pre-rounding once means the
        # donut percentage and the legend percentage are guaranteed
        # identical without the template having to re-derive.
        "flip_rate_pct_int": int(round(flip_rate * 100)),
        # Live percentage for the legend on drift — keeps the legend
        # numerically self-consistent (audit pass-2 Code-reviewer
        # MAJOR-3: previously the legend rendered "X of Y pairs
        # flipped (Z%)" where Z was the SIGNED %, contradicting X/Y).
        "live_pct_int": (
            int(round(100.0 * n_flips / n_pairs)) if n_pairs > 0 else 0
        ),
        "n_flips": n_flips,
        "n_pairs": n_pairs,
        "approve_to_deny": a2d,
        "deny_to_approve": d2a,
        "drift_flag": drift_flag,
        "donut_svg": _render_donut_svg(flip_rate),
    }


def _read_flip_rate_from_report_json(run_dir: Path) -> float | None:
    """Read ``headline.verdict_flip_rate.point`` from ``report.json``.

    Returns ``None`` if the file is missing, unreadable, malformed, or
    the field is absent / out of ``[0, 1]``. Logs WARNING on read
    failure — silent on missing-file (a pre-step-15a run dir is a
    legitimate state and the report still renders for those).
    """
    report_json_path = run_dir / "report.json"
    if not report_json_path.is_file():
        return None
    try:
        doc = _read_json_capped(report_json_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _log.warning(
            "report: report.json unreadable for flip-composition (%s) — "
            "donut disabled (run_dir=%s)",
            type(e).__name__, run_dir,
        )
        return None
    if not isinstance(doc, dict):
        return None
    headline = doc.get("headline")
    if not isinstance(headline, dict):
        return None
    vf = headline.get("verdict_flip_rate")
    if not isinstance(vf, dict):
        return None
    point = vf.get("point")
    # ``isinstance(True, int)`` is True in Python — exclude bools so a
    # serialiser that wrote ``true`` doesn't sneak in as ``1.0``.
    if not isinstance(point, (int, float)) or isinstance(point, bool):
        return None
    point_f = float(point)
    if not math.isfinite(point_f):
        return None
    if not (0.0 <= point_f <= 1.0):
        return None
    return point_f


def _count_flip_directions(
    matrix: "np.ndarray", orig_idx: int,
) -> tuple[int, int, int]:
    """Count approve→deny / deny→approve transitions vs the original.

    Returns ``(a2d, d2a, n_pairs)``:

    * ``a2d`` — count of (permuted_variant, eval) cells where the
      original variant's binarised verdict was PASS but the permuted
      variant's verdict was FAIL (and both cells are finite).
    * ``d2a`` — symmetric count for (FAIL → PASS).
    * ``n_pairs`` — total denominator (both cells finite).

    ``a2d + d2a`` is the same numerator that
    :func:`compute_verdict_flip_rate` uses for the headline rate, by
    construction: the masking, the threshold, and the original-index
    resolver are all shared. The donut's split therefore decomposes
    the signed headline exactly in clean runs.
    """
    import numpy as np

    valid = np.isfinite(matrix)
    preds = matrix >= _FLIP_THRESHOLD

    orig_preds = preds[orig_idx, :]
    orig_valid = valid[orig_idx, :]

    perm_mask = np.ones(matrix.shape[0], dtype=bool)
    perm_mask[orig_idx] = False
    perm_preds = preds[perm_mask, :]
    perm_valid = valid[perm_mask, :]

    denom_mask = perm_valid & orig_valid[None, :]
    # ``np.where`` mirrors compute_verdict_flip_rate's audit-RT-M1
    # pattern: NaN cells propagate as False through ``>= threshold``,
    # so without the explicit denom gate, "False == False == agreement"
    # would silently count failed cells as non-flips and skew the split.
    a2d_mask = denom_mask & orig_preds[None, :] & ~perm_preds
    d2a_mask = denom_mask & ~orig_preds[None, :] & perm_preds

    return int(a2d_mask.sum()), int(d2a_mask.sum()), int(denom_mask.sum())


def _render_donut_svg(flip_rate: float) -> str:
    """Hand-rolled donut SVG matching the Variant D sketch (130×130).

    Outer track is the Variant D border-soft cream; the orange arc is
    the Variant D ``--accent`` (#c25024). Center text is the integer-
    rounded percentage in serif (matches the report's editorial voice);
    the caption "flipped" sits below in italic muted serif.

    The arc is drawn as a stroked circle with ``stroke-dasharray``
    = ``"<filled> <rest>"`` and a ``rotate(-90)`` so the arc starts at
    12 o'clock. ``stroke-linecap`` is omitted (defaults to ``butt``) so
    the arc edges line up with the track exactly.
    """
    # Geometry: r=46 + stroke-width=18 → outer radius 55, leaves 5px
    # padding inside the 130×130 viewBox for the bevel.
    radius = 46
    circumference = 2.0 * math.pi * radius
    # Clamp defensively: report.json carries the signed point in [0,1],
    # but a future serialiser that emits, say, 1.0 + 1e-15 shouldn't
    # produce an SVG with a negative ``rest`` (browsers tolerate it but
    # the rendered arc gets a hairline overshoot).
    fr = max(0.0, min(1.0, float(flip_rate)))
    filled = fr * circumference
    rest = circumference - filled
    pct = int(round(fr * 100))
    # Audit UX pass-2 MAJOR-1+2: a darkened track (#e7d8c2 ≈ 1.40:1
    # vs #ffffff card surface) was strictly an improvement over the
    # invisible #f0eae0 (~1.20:1) but ALSO weakened the orange arc's
    # contrast vs the track on the common non-zero case (3.93:1 →
    # 3.36:1). The right fix is to KEEP the lighter track (best
    # arc-vs-track contrast) and add a 1px outer ring in
    # ``--muted`` (#897764, ~4.0:1 vs surface) that defines the
    # donut shape regardless of arc length — so a 0%-flip donut
    # still reads as a deliberate "no flips" donut, not a rendering
    # bug. The outer-ring radius is r=55 (track radius 46 + half
    # stroke-width 9) so it sits exactly at the track's outer edge.
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 130 130" '
        'width="130" height="130" role="img" '
        f'aria-label="Verdict flip rate: {pct}%. {pct}% of '
        'permuted-variant answers disagreed with the original variant '
        'on the same eval.">'
        '<circle cx="65" cy="65" r="55" fill="none" '
        'stroke="#897764" stroke-width="1"/>'
        '<circle cx="65" cy="65" r="46" fill="none" '
        'stroke="#f0eae0" stroke-width="18"/>'
        f'<circle cx="65" cy="65" r="46" fill="none" '
        f'stroke="#c25024" stroke-width="18" '
        f'stroke-dasharray="{filled:.3f} {rest:.3f}" '
        f'stroke-dashoffset="0" transform="rotate(-90 65 65)"/>'
        '<text x="65" y="68" text-anchor="middle" '
        'font-family="Source Serif Pro, Georgia, serif" '
        f'font-size="26" font-weight="500" fill="#1f1c19">{pct}%</text>'
        '<text x="65" y="86" text-anchor="middle" '
        'font-family="Source Serif Pro, Georgia, serif" '
        'font-size="11" font-style="italic" fill="#6b5f54">flipped</text>'
        '</svg>'
    )


def _render_variance_distribution_panel(
    matrix: "np.ndarray",
    eval_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Build the Jinja context for the §9 variance-distribution sparkline.

    Computes per-eval variance via ``np.var(matrix, axis=0, ddof=0)``
    (same closure as :func:`_render_voo_bar_svg` and
    :func:`compute_voo_for_attestation`). NaN columns (failed cells
    in that column propagate to NaN variance) are zero-filled — same
    policy as :func:`_render_voo_bar_svg`'s
    ``np.where(np.isnan(per_eval), 0.0, per_eval)`` so the sparkline
    and the §9 detail chart agree on ``n_evals``. The result is sorted
    descending; bars at-or-above the §10a gate are accent-colored,
    bars below are muted gray. A dashed threshold line at the gate
    marks which evals cleared.

    The §10a gate value is sourced from
    :data:`topofuzz.metrics.worstcase.DEFAULT_VARIANCE_GATE` (lazy
    import) rather than re-encoded locally, so a future blueprint
    revision that bumps the gate updates one site, not two.

    Returns a disabled context (mirroring :func:`_render_worstcase_panel`)
    when the matrix is empty.
    """
    if matrix.size == 0 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        return {
            "disabled": True,
            "disabled_reason": "no continuous-judge cells to score",
        }

    import numpy as np

    try:
        # Audit pass-2 Code-reviewer MAJOR-1: ``np.nanvar`` matches
        # ``worstcase.py:300`` exactly. The earlier ``np.var`` →
        # zero-fill landed the WRONG invariant (matched the voo bar,
        # which is a secondary chart) and silently desynced the
        # sparkline from the §10a candidate set on any run with a
        # partial-failure cell. ``np.nanvar`` ignores NaN cells per
        # column, returning NaN only for fully-NaN columns; we
        # zero-fill those (matching the voo bar's display policy)
        # so two charts on the same page still agree on ``n_evals``.
        per_eval = np.nanvar(matrix, axis=0, ddof=0)
    except (ValueError, TypeError) as e:
        _log.warning(
            "report: variance-distribution np.nanvar raised %s — "
            "section disabled", type(e).__name__,
        )
        return {
            "disabled": True,
            "disabled_reason": "variance computation error",
        }

    # Audit RT pass-1 MAJOR-3: source the gate from the §10a
    # promoter so the sparkline and the worstcase candidate set
    # agree on boundary values. The fallback is the v0.5 contract
    # value, used only if the import fails (verify-only install
    # without metrics — should never happen here since we already
    # need numpy to compute per_eval).
    try:
        from topofuzz.metrics.worstcase import DEFAULT_VARIANCE_GATE as _gate
        gate = float(_gate)
    except ImportError as e:
        # Audit pass-2 Cross-step MAJOR-2: log-discipline — exception
        # type only, never the bare exception object. Locked decision
        # from step 18a commit ac28ac7.
        _log.warning(
            "report: variance-distribution falling back to local gate "
            "constant (%s)", type(e).__name__,
        )
        gate = _WORSTCASE_GATE_FALLBACK

    # Audit pass-2 Code-reviewer MAJOR-1 (cont.): zero-fill columns
    # that ``np.nanvar`` left NaN (i.e., FULLY-NaN columns where
    # nothing finite remains to compute variance from). Per-column
    # partial-failure cells are already correctly handled by nanvar.
    plot_values = np.where(np.isnan(per_eval), 0.0, per_eval)

    # Sort descending; ties broken by original column index for
    # cross-platform determinism. ``np.argsort(-x, kind="stable")``
    # alone is unstable across BLAS/SIMD paths when many columns
    # have nominally identical variance (audit Code-reviewer
    # MINOR-1). ``lexsort`` with the primary key ``-plot_values``
    # and secondary key ``arange`` enforces a single canonical
    # ordering everywhere numpy runs.
    primary = -plot_values.astype(np.float64)
    secondary = np.arange(plot_values.size, dtype=np.int64)
    sort_idx = np.lexsort((secondary, primary))
    sorted_vals = plot_values[sort_idx].astype(np.float64)

    n = int(sorted_vals.size)
    # Audit RT pass-1 MAJOR-3 (cont.): use ``>=`` to match
    # ``worstcase.py:356`` exactly. With strict ``>`` the sparkline
    # would say "this eval did NOT clear the gate" while §10a
    # surfaces it as a worst-case card — boundary-case desync.
    n_above_gate = int((sorted_vals >= gate).sum())
    max_var = float(sorted_vals.max())

    # Cap the rendered bar count so a 500-eval run doesn't render
    # invisible sub-pixel bars. Headline numbers (n, n_above_gate) still
    # cover the FULL set — only the SVG bar list is capped.
    rendered_vals = sorted_vals[:_SPARKLINE_BAR_RENDER_CAP]
    n_truncated = max(0, n - _SPARKLINE_BAR_RENDER_CAP)

    return {
        "disabled": False,
        "n_evals": n,
        "n_above_gate": n_above_gate,
        "max_variance": max_var,
        "threshold": gate,
        "n_truncated": n_truncated,
        "sparkline_svg": _render_sparkline_svg(
            rendered_vals,
            max_variance=max_var,
            threshold=gate,
            n_above_gate_total=n_above_gate,
            n_total=n,
        ),
    }


def _render_sparkline_svg(
    sorted_vals: "np.ndarray",
    *,
    max_variance: float,
    threshold: float,
    n_above_gate_total: int,
    n_total: int,
) -> str:
    """Render the per-eval variance sparkline as inline SVG.

    Layout: 500×100 viewBox, baseline at y=95 (5px bottom padding),
    bars grow upward to y=5 (5px top padding) → 90px draw area capped
    to 85px so the topmost bar stops just below the top edge. Bars
    at-or-above ``threshold`` are accent (#c25024); bars below are
    muted gray (#cccccc). A dashed gray line marks the threshold.

    ``n_above_gate_total`` and ``n_total`` are the FULL-set counts
    (before truncation), surfaced into the aria-label so screen-reader
    users hear the same numbers the visible caption shows. Audit
    Blueprint MAJOR-2: a previous in-function recount over the
    truncated ``sorted_vals`` could undercount on capped renders.
    """
    n = int(sorted_vals.size)
    if n == 0:
        # Defensive only — _render_variance_distribution_panel
        # short-circuits on empty matrix before reaching here.
        return _placeholder_svg("no evals to sparkline")

    # Y-axis: always show the threshold line inside the chart area.
    # If max_variance is below threshold, expand y_max to threshold*1.2
    # so the line sits at ~80% of the bar area instead of touching the
    # top edge (where it'd visually disappear).
    y_max = max(max_variance, threshold * 1.2, 1e-9)
    chart_top = 5.0
    chart_bottom = 95.0
    bar_area = chart_bottom - chart_top  # = 90

    # Bar width: divide 500 by n with a 1px gap on either side per bar
    # boundary. Floor to keep bars integer-pixel-aligned in browsers
    # that pixel-snap (Chromium does, Safari pixel-snaps differently;
    # the .3f below gives sub-pixel placement either way).
    canvas_w = 500.0
    bar_w = max(1.0, canvas_w / n - 1.0)
    # Stride = canvas_w / n centers the bar group across the full width
    # regardless of n (small-n looks balanced; large-n tiles densely).
    stride = canvas_w / n

    bars: list[str] = []
    for i, v in enumerate(sorted_vals):
        h = (float(v) / y_max) * bar_area
        # Defensive: a numerical ε can drive h slightly below 0 for a
        # zero-variance column; clamp so we never emit a negative-
        # height rect (browsers handle it but it lights up SVG linters).
        h = max(0.0, min(bar_area, h))
        x = i * stride
        y = chart_bottom - h
        # ``>=`` matches worstcase.py:356 exactly so boundary-case
        # bars don't desync from the §10a candidate set.
        color = "#c25024" if float(v) >= threshold else "#cccccc"
        bars.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" '
            f'width="{bar_w:.3f}" height="{h:.3f}" fill="{color}"/>'
        )
    bars_svg = "".join(bars)

    # Threshold line: dashed, gray, with a text label at the right
    # edge. Bumped from font-size 9 to 11 (audit UX MAJOR-4): 9pt at
    # 100px chart height is below WCAG's user-readable threshold
    # without zoom; 11pt matches the donut caption.
    threshold_y = chart_bottom - (threshold / y_max) * bar_area
    threshold_label = (
        f"variance = {threshold:g} "
        "(topology-fragility threshold, not a safety threshold)"
    )

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 500 100" width="100%" height="100" '
        'preserveAspectRatio="none" role="img" '
        f'aria-label="Per-eval variance sparkline, sorted descending. '
        f'{n_above_gate_total} of {n_total} evals cleared the '
        f'topology-fragility threshold (variance ≥ {threshold:g}); '
        f'remaining {n_total - n_above_gate_total} are stable.">'
        f'{bars_svg}'
        f'<line x1="0" y1="{threshold_y:.3f}" x2="500" y2="{threshold_y:.3f}" '
        'stroke="#897764" stroke-dasharray="3 3" stroke-width="1"/>'
        f'<text x="498" y="{threshold_y - 3:.3f}" text-anchor="end" '
        'font-family="ui-monospace, Menlo, Consolas, monospace" '
        f'font-size="11" fill="#6b5f54">{threshold_label}</text>'
        '</svg>'
    )


# ---------------------------------------------------------------------------
# Jinja template
# ---------------------------------------------------------------------------


def _render_template(**ctx: Any) -> str:
    """Render the packaged ``report.html.j2`` with the given context.

    Loads the template via :mod:`importlib.resources` so editable +
    wheel installs both work — the hatch wheel-build target packs every
    file under ``src/topofuzz/`` so ``templates/report.html.j2`` rides
    along (see ``packages/topofuzz/pyproject.toml`` ``[tool.hatch.build.targets.wheel]``).
    """
    import jinja2

    source = (
        resources.files("topofuzz")
        .joinpath("templates/report.html.j2")
        .read_text(encoding="utf-8")
    )
    env = jinja2.Environment(
        autoescape=True,
        # No filesystem loader — we read the template text once via
        # importlib.resources and feed it directly. Removes the
        # zip-import / read-only-fs edge cases that FileSystemLoader
        # mishandles.
        loader=jinja2.BaseLoader(),
        keep_trailing_newline=True,
    )
    template = env.from_string(source)
    return template.render(
        topofuzz_version=_topofuzz_version,
        **ctx,
    )
