"""Step 11: ``topofuzz report <run_dir>`` end-to-end coverage.

Covers:

* Happy path: report.html written, file:// URL printed to stdout.
* Banner conditionals: ``[DEV-KEY]`` iff ``attestation.json`` exists,
  ``[JURY]`` iff ``manifest.cli_args.jury_of_3`` truthy, ``[DEGRADED]``
  iff ``runs.status == "degraded"``.
* Missing inputs: no ``manifest.json`` / no ``run.sqlite`` / bad
  ``run_dir`` all exit ``_BAD_USAGE_EXIT`` (==2) with a friendly error.
* Empty score matrix (no continuous-judge cells) still renders, with
  placeholder SVGs.
* Verify-path safety: ``topofuzz.report`` is NOT importable from
  ``topofuzz.cli`` at module scope (defended further by the dedicated
  import-guard test; this asserts the inverse — that the lazy-import
  fires when ``report`` is actually invoked).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import textwrap
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topofuzz.cli import app


# ---------------------------------------------------------------------------
# Fixture helpers — build a minimal run dir without spinning the orchestrator
# ---------------------------------------------------------------------------


def _make_run_dir(
    tmp_path: Path,
    *,
    status: str = "succeeded",
    jury_of_3: bool = False,
    judge_kind: str = "continuous",
    rubric_hash: str = "rh-test",
    with_attestation: bool = False,
    n_variants: int = 2,
    n_evals: int = 3,
) -> Path:
    """Build a v0.5-shaped run_dir under ``tmp_path``.

    Schema mirrors :data:`topofuzz.runner._SCHEMA` so a future schema bump
    breaks this fixture loudly — desirable, since the report consumes
    that schema directly.
    """
    run_dir = tmp_path / "run-test"
    run_dir.mkdir()

    eval_ids = [f"eval-{i}" for i in range(n_evals)]
    variant_ids = [f"var{i:02d}" for i in range(n_variants)]

    manifest = {
        "run_id": "run-test",
        "topology_canonical_hash": "topo-hash",
        "topology_name": "lending_4role",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": n_variants,
        "n_eval_rows": n_evals,
        "variant_canonical_hashes": variant_ids,
        "eval_ids": eval_ids,
        "max_cost_usd": None,
        "judge_model": "google/gemini-2.5-pro",
        "rubric_hash": rubric_hash,
        "semaphore_size": 8,
        "cli_args": {"jury_of_3": jury_of_3},
        "kickoff_ts": 1700000000.0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    db_path = run_dir / "run.sqlite"
    db = sqlite3.connect(str(db_path))
    try:
        db.executescript("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                topology_canonical_hash TEXT NOT NULL,
                topology_name TEXT NOT NULL,
                tds_version TEXT NOT NULL,
                created_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                finished_at REAL,
                n_variants INTEGER,
                n_evals INTEGER,
                judge_protocol TEXT,
                cost_actual_usd REAL,
                tenant_id TEXT NOT NULL DEFAULT 'local',
                rubric_hash TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE cells (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                eval_id TEXT NOT NULL,
                ir_canonical_hash TEXT NOT NULL,
                node_outputs_json TEXT NOT NULL,
                error TEXT,
                cell_cost_usd REAL,
                PRIMARY KEY (run_id, variant_id, eval_id)
            );
            CREATE TABLE judge_results (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                eval_id TEXT NOT NULL,
                judge_kind TEXT NOT NULL,
                score_value REAL,
                passed INTEGER,
                rationale TEXT,
                failure_reason TEXT,
                PRIMARY KEY (run_id, variant_id, eval_id)
            );
            PRAGMA user_version = 4;
        """)
        db.execute(
            "INSERT INTO runs "
            "(run_id, topology_canonical_hash, topology_name, "
            " tds_version, created_at, status, rubric_hash) "
            "VALUES (?, 'h', 'name', '0.5', 0.0, ?, ?)",
            ("run-test", status, rubric_hash),
        )
        # Populate a small deterministic score grid so the SVGs render
        # with real data. Pattern: matrix[i, j] = (i + j) / (V+E) — gives
        # the heatmap a visible gradient and non-zero per-eval variance.
        denom = max(1, n_variants + n_evals)
        for i, vid in enumerate(variant_ids):
            for j, eid in enumerate(eval_ids):
                score = (i + j) / denom
                db.execute(
                    "INSERT INTO judge_results "
                    "(run_id, variant_id, eval_id, judge_kind, "
                    " score_value, passed, rationale, failure_reason) "
                    "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                    ("run-test", vid, eid, judge_kind, score),
                )
        db.commit()
    finally:
        db.close()

    if with_attestation:
        # DSSE-envelope-shaped placeholder. The banner gate
        # (`_attestation_looks_signed`) requires a parseable JSON dict
        # carrying a non-empty `signatures` array — a zero-byte or
        # garbage file would otherwise falsely light up [DEV-KEY] on an
        # unsigned run (RT pass-1 MAJOR-4). Full cryptographic verify is
        # still `topofuzz verify`'s job; this is the watermark gate only.
        (run_dir / "attestation.json").write_bytes(
            b'{"payloadType":"x","payload":"y",'
            b'"signatures":[{"keyid":"dev","sig":"placeholder"}]}'
        )

    return run_dir


# ---------------------------------------------------------------------------
# Happy-path + banner conditionals
# ---------------------------------------------------------------------------


def test_report_writes_html_and_prints_file_uri(tmp_path: Path) -> None:
    """The vanilla happy path: HTML file written, file:// URL on stdout."""
    run_dir = _make_run_dir(tmp_path)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    html_path = run_dir / "report.html"
    assert html_path.exists(), "report.html should be written"
    expected_uri = html_path.resolve().as_uri()
    assert expected_uri in result.output
    body = html_path.read_text(encoding="utf-8")
    assert "<title>topofuzz report" in body
    assert "lending_4role" in body
    # Both SVGs present and inlined (no XML preamble inside <body>).
    assert body.count("<svg") >= 2
    assert "<?xml" not in body


def test_report_dev_key_banner_when_attestation_present(tmp_path: Path) -> None:
    """[DEV-KEY] iff attestation.json exists in run_dir."""
    run_dir = _make_run_dir(tmp_path, with_attestation=True)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEV-KEY]" in body


def test_report_no_dev_key_banner_when_attestation_absent(tmp_path: Path) -> None:
    """No [DEV-KEY] banner when attestation.json is missing."""
    run_dir = _make_run_dir(tmp_path, with_attestation=False)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEV-KEY]" not in body


def test_report_jury_banner_when_manifest_flag_set(tmp_path: Path) -> None:
    """[JURY] iff manifest.cli_args.jury_of_3 is truthy."""
    run_dir = _make_run_dir(tmp_path, jury_of_3=True)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[JURY]" in body


def test_report_no_jury_banner_when_flag_false(tmp_path: Path) -> None:
    """[JURY] banner absent when jury_of_3 is false / missing."""
    run_dir = _make_run_dir(tmp_path, jury_of_3=False)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[JURY]" not in body


def test_report_degraded_banner_when_status_degraded(tmp_path: Path) -> None:
    """[DEGRADED] banner only when runs.status == 'degraded'."""
    run_dir = _make_run_dir(tmp_path, status="degraded")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEGRADED]" in body


@pytest.mark.parametrize("status", ["succeeded", "failed", "cost_cap_hit", "interrupted"])
def test_report_no_degraded_banner_for_other_statuses(
    tmp_path: Path, status: str
) -> None:
    """The blueprint names 'degraded' specifically; other non-success
    statuses surface through the status row, NOT the banner."""
    run_dir = _make_run_dir(tmp_path, status=status)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEGRADED]" not in body
    # Status itself should still appear in the meta table.
    assert status in body


def test_report_all_three_banners_together(tmp_path: Path) -> None:
    """Banners are independent — verify they coexist in one render."""
    run_dir = _make_run_dir(
        tmp_path,
        status="degraded",
        jury_of_3=True,
        with_attestation=True,
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEV-KEY]" in body
    assert "[JURY]" in body
    assert "[DEGRADED]" in body


# ---------------------------------------------------------------------------
# Missing-input failure modes
# ---------------------------------------------------------------------------


def test_report_missing_run_dir_exits_bad_usage(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does-not-exist"
    result = CliRunner().invoke(app, ["report", str(nonexistent)])
    assert result.exit_code == 2, result.output
    assert "not a directory" in result.output.lower()


def test_report_missing_manifest_exits_bad_usage(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "manifest.json").unlink()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 2, result.output
    assert "manifest.json" in result.output


def test_report_missing_sqlite_exits_bad_usage(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "run.sqlite").unlink()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 2, result.output
    assert "run.sqlite" in result.output


def test_report_corrupt_sqlite_exits_bad_usage(tmp_path: Path) -> None:
    """A non-sqlite file at run.sqlite surfaces as a friendly error,
    not a raw sqlite3.DatabaseError traceback."""
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "run.sqlite").write_bytes(b"not a database")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 2, result.output


# ---------------------------------------------------------------------------
# Empty-data degraded path: matrix is 0×0, SVGs fall back to placeholders
# ---------------------------------------------------------------------------


def test_report_empty_score_matrix_still_renders(tmp_path: Path) -> None:
    """A run with no continuous-judge cells (e.g. fully-aborted run) still
    produces a readable report rather than crashing on empty arrays."""
    run_dir = _make_run_dir(tmp_path)
    # Wipe judge_results so the score matrix is empty.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DELETE FROM judge_results")
        db.commit()
    finally:
        db.close()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    # Placeholder copy from `_placeholder_svg` should appear since the
    # matrix is empty.
    assert "no continuous-judge cells" in body


# ---------------------------------------------------------------------------
# RT pass-1 fix coverage: banner gate strictness, matrix_warning surface,
# subprocess cwd, eval_ids truncation cap.
# ---------------------------------------------------------------------------


def test_report_dev_key_banner_rejects_empty_attestation(tmp_path: Path) -> None:
    """RT pass-1 MAJOR-4: a zero-byte attestation.json must NOT trigger
    the [DEV-KEY] watermark — only a parseable DSSE-shaped envelope does."""
    run_dir = _make_run_dir(tmp_path, with_attestation=False)
    (run_dir / "attestation.json").write_bytes(b"")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEV-KEY]" not in body


def test_report_dev_key_banner_rejects_garbage_attestation(tmp_path: Path) -> None:
    """A non-JSON / non-DSSE-shaped attestation.json doesn't trigger
    the banner — defends against accidental file collisions."""
    run_dir = _make_run_dir(tmp_path, with_attestation=False)
    (run_dir / "attestation.json").write_bytes(b"not json at all")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEV-KEY]" not in body


def test_report_dev_key_banner_rejects_missing_signatures(tmp_path: Path) -> None:
    """JSON without a non-empty `signatures` array does not light up
    the watermark — DSSE invariant must hold."""
    run_dir = _make_run_dir(tmp_path, with_attestation=False)
    (run_dir / "attestation.json").write_bytes(b'{"signatures": []}')
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[DEV-KEY]" not in body


def test_report_corrupt_db_surfaces_matrix_warning(tmp_path: Path) -> None:
    """RT pass-1 MAJOR-5: when the score matrix can't be read (corrupt
    DB, schema mismatch), the user sees a visible warning in the HTML
    — not a silent 'no cells' placeholder."""
    run_dir = _make_run_dir(tmp_path)
    # Drop the judge_results table entirely — read_score_matrix will
    # raise sqlite3.OperationalError ("no such table"), caught and
    # surfaced through `matrix_warning`.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DROP TABLE judge_results")
        db.commit()
    finally:
        db.close()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "score matrix unavailable" in body


def test_report_eval_ids_capped_in_html(tmp_path: Path) -> None:
    """RT pass-1 MINOR-3: a 1000-eval manifest must not produce a
    1000-item HTML list — the cap is 500 with a 'and N more' footer."""
    run_dir = _make_run_dir(tmp_path, n_evals=1)
    # Rewrite manifest with 1000 fake eval_ids (don't touch the DB —
    # the cap is purely on the rendered HTML list).
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["eval_ids"] = [f"e{i:04d}" for i in range(1000)]
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    # 500 rendered + 500 truncated; the footer should announce 500 more.
    assert "and 500 more" in body
    # The 501st eval id must not have rendered.
    assert "e0500" not in body
    # The 1st eval id must still render.
    assert "e0000" in body


# ---------------------------------------------------------------------------
# Verify-path safety: topofuzz.report is NOT loaded by importing topofuzz.cli
# ---------------------------------------------------------------------------


def test_cli_import_does_not_load_report_module() -> None:
    """Importing ``topofuzz.cli`` in a fresh interpreter MUST NOT
    transitively load ``topofuzz.report``.

    Complements the broader ``test_import_guard.py`` (which forbids
    matplotlib/jinja2/numpy after cli import); this test pins the
    intermediate hop. If a future refactor moves
    ``from topofuzz.report import ...`` to module scope, the heavy
    deps come with it and the verify-only install grows by ~50 MB.
    """
    script = textwrap.dedent(
        """
        import sys, importlib
        importlib.import_module("topofuzz.cli")
        if "topofuzz.report" in sys.modules:
            print("LEAKED: topofuzz.report")
            sys.exit(1)
        sys.exit(0)
        """
    )
    # Pin cwd to the repo root (RT pass-1 MINOR-5) so the subprocess
    # picks up the same in-development topofuzz install rather than a
    # globally-installed one that might be on the user's PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        f"topofuzz.cli transitively loaded topofuzz.report.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
