"""Step 11 (Weekend 3): render the run-summary HTML report.

Reads ``<run_dir>/manifest.json`` (required), ``<run_dir>/run.sqlite``
(required), and ``<run_dir>/attestation.json`` (optional, presence
implies the ``[DEV-KEY]`` banner). Produces ``<run_dir>/report.html``
with two inline matplotlib SVGs — per-eval voo bar + score-matrix
heatmap — and banner conditionals per BLUEPRINT_v5plus.md §11.

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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status, run_id = _read_run_row(sqlite_path)

    # Score matrix (variant × eval). Empty matrix is a legitimate outcome
    # for an aborted run with no continuous-judge cells; the plotting
    # path falls back to a placeholder SVG rather than crashing. Any
    # read failure is bubbled up to the user as a banner+log warning
    # (RT pass-1 MAJOR-5 — don't silently swallow at INFO).
    matrix, variant_ids, eval_ids_from_db, matrix_warning = _read_matrix_or_empty(
        run_dir, run_id
    )

    voo_svg = _render_voo_bar_svg(matrix, eval_ids_from_db)
    heatmap_svg = _render_heatmap_svg(matrix, variant_ids, eval_ids_from_db)
    influence_svg, influence_rows = _render_influence_panel(
        matrix, variant_ids, manifest
    )
    worstcase_ctx = _render_worstcase_panel(
        matrix, variant_ids, eval_ids_from_db, run_dir, run_id, manifest
    )
    summary_ctx = _render_summary_panel(run_dir)

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
        matrix_warning=matrix_warning,
        worstcase=worstcase_ctx,
        summary=summary_ctx,
    )

    out_path = run_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


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


def _read_matrix_or_empty(
    run_dir: Path, run_id: str
) -> tuple["np.ndarray", tuple[str, ...], tuple[str, ...], str | None]:
    """Read the continuous-score matrix.

    Returns ``(matrix, variant_ids, eval_ids, warning)``. ``warning`` is
    ``None`` on a clean read and a short human-readable string when the
    matrix had to be degraded to empty (RT pass-1 MAJOR-5 — empty matrix
    is silent today; we now surface the reason so the user sees it both
    in the log AND in the rendered HTML).

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
        return _empty_matrix(), (), (), msg
    return result.matrix, result.variant_ids, result.eval_ids, None


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


def _placeholder_svg(message: str) -> str:
    """Tiny static SVG used when the score matrix is empty."""
    # Hand-written so it doesn't pull matplotlib when the matrix is
    # empty (common for aborted-very-early runs). Inline so it embeds
    # straight into the template without escaping.
    safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="120" '
        'viewBox="0 0 480 120">'
        '<rect width="480" height="120" fill="#f3f3f3" stroke="#bbb"/>'
        '<text x="240" y="65" text-anchor="middle" '
        'font-family="ui-monospace, Menlo, Consolas, monospace" '
        f'font-size="14" fill="#666">{safe}</text>'
        "</svg>"
    )


def _render_voo_bar_svg(
    matrix: "np.ndarray", eval_ids: tuple[str, ...]
) -> str:
    """Per-eval variance across orderings as a bar chart.

    Per-eval point estimate is ``np.var(matrix[:, j], ddof=0)`` — same
    closure as :func:`topofuzz.metrics.variance_across_orderings` and
    :func:`topofuzz.attest.compute_voo_for_attestation` (axis convention:
    rows=variants, cols=evals).
    """
    import numpy as np

    if matrix.size == 0 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        return _placeholder_svg("no continuous-judge cells to score")

    plt = _setup_matplotlib()
    per_eval = np.var(matrix, axis=0, ddof=0)
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
        return _placeholder_svg(
            "influence available but every agent is degenerate"
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


def _render_summary_panel(run_dir: Path) -> dict[str, Any]:
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
        }

    report_json_path = run_dir / "report.json"
    verdict_flip_rate: dict[str, Any] | None = None
    per_agent_influence: list[dict[str, Any]] | None = None
    if report_json_path.is_file():
        try:
            raw = report_json_path.read_text(encoding="utf-8")
            doc = json.loads(raw)
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
        }

    return {
        "verdict_word": ctx.verdict_word,
        "verdict_flip_rate": ctx.verdict_flip_rate,
        "primary_driver": ctx.primary_driver,
        "primary_influence": ctx.primary_influence,
        "summary_text": ctx.summary_text,
        "summary_html": ctx.summary_html,
    }


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
