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
    variant_orderings: list[list[str]] | None = None,
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

    if variant_orderings is None:
        # Default: every variant is the same single-agent ordering. With
        # 2 variants × 1 agent, influence is degenerate (no same-slot
        # AND moved pairs both exist) — the report renders a placeholder
        # SVG and an empty rows table. Step-15 influence-panel tests
        # override this to exercise the live-render path.
        variant_orderings = [["A"] for _ in range(n_variants)]
    manifest = {
        "run_id": "run-test",
        "topology_canonical_hash": "topo-hash",
        "topology_name": "lending_4role",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": n_variants,
        "n_eval_rows": n_evals,
        "variant_canonical_hashes": variant_ids,
        "variant_orderings": variant_orderings,
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
# Step 18: [REKOR-DEFERRED] banner gate
# ---------------------------------------------------------------------------


def test_report_rekor_deferred_banner_when_marker_present(
    tmp_path: Path,
) -> None:
    """[REKOR-DEFERRED] banner lights iff .rekor-deferred marker exists
    AND no successful rekor LogEntry sits beside it. BLUEPRINT line 306."""
    from topofuzz.rekor import REKOR_DEFERRED_MARKER

    run_dir = _make_run_dir(tmp_path, with_attestation=True)
    (run_dir / REKOR_DEFERRED_MARKER).write_text("", encoding="utf-8")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[REKOR-DEFERRED]" in body


def test_report_no_rekor_deferred_banner_when_marker_absent(
    tmp_path: Path,
) -> None:
    """No marker → no banner — the default state for a non-opt-in run."""
    run_dir = _make_run_dir(tmp_path, with_attestation=True)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[REKOR-DEFERRED]" not in body


def test_report_rekor_deferred_suppressed_when_log_entry_present(
    tmp_path: Path,
) -> None:
    """A stale marker beside a successful rekor.json should NOT light
    the banner — looks_like_rekor_logged AND-gate prevents the deferred
    banner from sticking on a now-logged run (race-window guard)."""
    import json as _json

    from topofuzz.rekor import REKOR_DEFERRED_MARKER, REKOR_OUTPUT_FILENAME

    run_dir = _make_run_dir(tmp_path, with_attestation=True)
    (run_dir / REKOR_DEFERRED_MARKER).write_text("", encoding="utf-8")
    (run_dir / REKOR_OUTPUT_FILENAME).write_text(
        _json.dumps({"abc-uuid": {"logIndex": 1}}), encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "[REKOR-DEFERRED]" not in body


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


# ---------------------------------------------------------------------------
# Step 15 — §6a per-agent influence panel
# ---------------------------------------------------------------------------


def test_influence_panel_rendered_when_orderings_supplied(tmp_path: Path) -> None:
    """A run with 4 variants × 3 evals and a dominant agent's orderings
    produces a non-placeholder influence panel with at least one bar."""
    orderings = [
        ["A", "B", "C"],
        ["A", "C", "B"],
        ["B", "A", "C"],
        ["B", "C", "A"],
    ]
    run_dir = _make_run_dir(
        tmp_path,
        n_variants=4,
        n_evals=3,
        variant_orderings=orderings,
    )
    # Overwrite the default per-cell scores with values that polarise on
    # agent A's slot so influence has signal. variants 0,1 have A in
    # slot 0 → high score; variants 2,3 have A elsewhere → low score.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DELETE FROM judge_results")
        cells = [
            (0, 0.9), (0, 0.85), (0, 0.92),
            (1, 0.88), (1, 0.91), (1, 0.86),
            (2, 0.10), (2, 0.05), (2, 0.12),
            (3, 0.08), (3, 0.11), (3, 0.09),
        ]
        eval_ids = [f"eval-{i}" for i in range(3)]
        for i, (v_idx, score) in enumerate(cells):
            db.execute(
                "INSERT INTO judge_results "
                "(run_id, variant_id, eval_id, judge_kind, score_value, "
                " passed, rationale, failure_reason) "
                "VALUES (?, ?, ?, 'continuous', ?, NULL, NULL, NULL)",
                ("run-test", f"var{v_idx:02d}", eval_ids[i % 3], score),
            )
        db.commit()
    finally:
        db.close()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Per-agent influence" in body
    # The placeholder copy "influence unavailable" should NOT appear.
    assert "influence unavailable" not in body
    # All three agents appear in the influence-rows table.
    assert ">A<" in body or ">A</" in body
    assert ">B<" in body or ">B</" in body
    assert ">C<" in body or ">C</" in body
    # Three SVGs: voo bar, heatmap, influence chart.
    assert body.count("<svg") >= 3


def test_influence_panel_placeholder_when_orderings_missing(tmp_path: Path) -> None:
    """A pre-step-15 run (no variant_orderings in manifest) renders a
    placeholder SVG, not a crash, and produces an empty rows table."""
    run_dir = _make_run_dir(tmp_path, n_variants=4, n_evals=3)
    # Strip variant_orderings from the manifest to simulate an old run.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("variant_orderings", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Per-agent influence" in body
    assert "influence unavailable" in body
    assert "no orderings in manifest" in body


def test_attestation_findings_per_agent_influence_populated(tmp_path: Path) -> None:
    """End-to-end: ``emit_run_attestation`` lands ``per_agent_influence``
    inside the signed payload's ``findings`` dict."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from fuzzeval.attestation import Ed25519Signer, dsse_verify

    from topofuzz.attest import emit_run_attestation

    orderings = [
        ["A", "B", "C"],
        ["A", "C", "B"],
        ["B", "A", "C"],
        ["B", "C", "A"],
    ]
    run_dir = _make_run_dir(
        tmp_path,
        n_variants=4,
        n_evals=3,
        variant_orderings=orderings,
    )
    # Same polarised cells as the panel test so influence is computable.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DELETE FROM judge_results")
        cells = [
            (0, 0.9), (0, 0.85), (0, 0.92),
            (1, 0.88), (1, 0.91), (1, 0.86),
            (2, 0.10), (2, 0.05), (2, 0.12),
            (3, 0.08), (3, 0.11), (3, 0.09),
        ]
        eval_ids = [f"eval-{i}" for i in range(3)]
        for i, (v_idx, score) in enumerate(cells):
            db.execute(
                "INSERT INTO judge_results "
                "(run_id, variant_id, eval_id, judge_kind, score_value, "
                " passed, rationale, failure_reason) "
                "VALUES (?, ?, ?, 'continuous', ?, NULL, NULL, NULL)",
                ("run-test", f"var{v_idx:02d}", eval_ids[i % 3], score),
            )
        db.commit()
    finally:
        db.close()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    public_key_pem = signer.public_key_pem
    out_path = emit_run_attestation(
        run_dir=run_dir,
        run_id="run-test",
        manifest=manifest,
        status="succeeded",
        successful_cells=12,
        total_cells=12,
        n_variants=4,
        n_evals=3,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        signer=signer,
    )
    envelope = out_path.read_bytes()
    verified = dsse_verify(envelope, public_key_pem)
    payload = json.loads(verified.payload_bytes)
    assert "findings" in payload
    findings = payload["findings"]
    assert "per_agent_influence" in findings
    rows = findings["per_agent_influence"]
    assert isinstance(rows, list)
    assert len(rows) == 3
    for r in rows:
        assert set(r.keys()) >= {
            "node_id", "influence", "ci_lower", "ci_upper", "degenerate"
        }
    # The dominant agent (A) carries a positive, non-degenerate influence.
    by_id = {r["node_id"]: r for r in rows}
    assert by_id["A"]["degenerate"] is False
    assert by_id["A"]["influence"] is not None
    assert by_id["A"]["influence"] > 0.0


def test_attestation_influence_alignment_survives_non_lex_hashes(
    tmp_path: Path,
) -> None:
    """Audit CR-M1 + RT-B-1 + RT-B-2 cover: a manifest whose
    variant_canonical_hashes do NOT come out lex-sorted matches them
    back to orderings via the hash table, not by row index. A is the
    dominant agent in variants whose hash sorts last, so a naive
    row-index alignment would attribute A's signal to the wrong agent.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from fuzzeval.attestation import Ed25519Signer, dsse_verify

    from topofuzz.attest import emit_run_attestation

    # Non-lex hashes: lex order would be c0, d1, e2, f3.
    # Insertion order (manifest order): f3, c0, e2, d1.
    hashes = ["f3", "c0", "e2", "d1"]
    # Per the dominant fixture: A is in slot 0 for the first two
    # manifest-order variants → polarised high. Place those at the
    # hashes that DON'T sort first lex, to prove alignment.
    orderings = [
        ["A", "B", "C"],   # f3 → manifest row 0, polarised high
        ["A", "C", "B"],   # c0 → manifest row 1, polarised high
        ["B", "A", "C"],   # e2 → manifest row 2, polarised low
        ["B", "C", "A"],   # d1 → manifest row 3, polarised low
    ]
    run_dir = tmp_path / "non-lex-run"
    run_dir.mkdir()

    # Build the manifest + run.sqlite by hand with non-lex hashes so
    # the read_score_matrix lex sort genuinely shuffles the axis.
    manifest = {
        "run_id": "non-lex-run",
        "topology_canonical_hash": "topo-hash",
        "topology_name": "lending_4role",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": 4,
        "n_eval_rows": 3,
        "variant_canonical_hashes": hashes,
        "variant_orderings": orderings,
        "eval_ids": ["eval-0", "eval-1", "eval-2"],
        "max_cost_usd": None,
        "judge_model": "google/gemini-2.5-pro",
        "rubric_hash": "rh",
        "semaphore_size": 8,
        "cli_args": {},
        "kickoff_ts": 1700000000.0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    db = sqlite3.connect(str(run_dir / "run.sqlite"))
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
        """)
        db.execute(
            "INSERT INTO runs (run_id, topology_canonical_hash, "
            "topology_name, tds_version, created_at, status, rubric_hash) "
            "VALUES ('non-lex-run', 'h', 'name', '0.5', 0.0, 'succeeded', 'rh')"
        )
        # Insert per-variant cells with the polarised pattern, keyed
        # by the actual hash (not by row index).
        cells_per_variant = {
            "f3": [0.90, 0.85, 0.92],  # A in slot 0 → high
            "c0": [0.88, 0.91, 0.86],  # A in slot 0 → high
            "e2": [0.10, 0.05, 0.12],  # A in slot 1 → low
            "d1": [0.08, 0.11, 0.09],  # A in slot 2 → low
        }
        eval_ids = ["eval-0", "eval-1", "eval-2"]
        for vid, scores in cells_per_variant.items():
            for eid, sc in zip(eval_ids, scores, strict=True):
                db.execute(
                    "INSERT INTO judge_results "
                    "(run_id, variant_id, eval_id, judge_kind, "
                    " score_value, passed, rationale, failure_reason) "
                    "VALUES ('non-lex-run', ?, ?, 'continuous', ?, "
                    " NULL, NULL, NULL)",
                    (vid, eid, sc),
                )
        db.commit()
    finally:
        db.close()

    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    public_key_pem = signer.public_key_pem
    out_path = emit_run_attestation(
        run_dir=run_dir,
        run_id="non-lex-run",
        manifest=manifest,
        status="succeeded",
        successful_cells=12,
        total_cells=12,
        n_variants=4,
        n_evals=3,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        signer=signer,
    )
    payload = json.loads(dsse_verify(out_path.read_bytes(), public_key_pem).payload_bytes)
    rows = payload["findings"]["per_agent_influence"]
    by_id = {r["node_id"]: r for r in rows}
    # A is the polarised agent; correctly aligned to slot 0 in f3/c0
    # variants → positive non-degenerate influence.
    assert by_id["A"]["degenerate"] is False
    assert by_id["A"]["influence"] is not None
    assert by_id["A"]["influence"] > 0.0


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


# ---------------------------------------------------------------------------
# Step 16: §10a worst-case viewer integration coverage
# ---------------------------------------------------------------------------


def _populate_cells_for_worstcase(
    run_dir: Path,
    *,
    run_id: str = "run-test",
    variant_ids: list[str],
    eval_ids: list[str],
    response_text: str = "trace excerpt",
) -> None:
    """Insert minimal cells rows for every (variant, eval) pair.

    Step 16 reads ``cells.node_outputs_json`` to build side-by-side
    trace excerpts. Existing :func:`_make_run_dir` populates
    ``judge_results`` but not ``cells`` — for the report-level
    worst-case tests we seed cells too.
    """
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        for vid in variant_ids:
            for eid in eval_ids:
                node_outs = [
                    {
                        "node_id": f"agent-1-{vid}",
                        "role": "decision",
                        "response_text": (
                            f"{response_text} for ({vid}, {eid})"
                        ),
                    }
                ]
                db.execute(
                    "INSERT INTO cells "
                    "(run_id, variant_id, eval_id, ir_canonical_hash, "
                    " node_outputs_json) VALUES (?, ?, ?, ?, ?)",
                    (run_id, vid, eid, vid, json.dumps(node_outs)),
                )
        db.commit()
    finally:
        db.close()


def _rewrite_scores_for_flip(
    run_dir: Path,
    *,
    flips: dict[str, dict[str, float]],
    run_id: str = "run-test",
) -> None:
    """Overwrite ``judge_results.score_value`` with the given pattern.

    ``flips`` is ``{variant_id: {eval_id: score}}``. Wipes existing
    judge_results rows for the run_id first so the deterministic
    pattern from :func:`_make_run_dir` doesn't bleed through.
    """
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute(
            "DELETE FROM judge_results WHERE run_id = ?", (run_id,)
        )
        for vid, by_eval in flips.items():
            for eid, score in by_eval.items():
                db.execute(
                    "INSERT INTO judge_results "
                    "(run_id, variant_id, eval_id, judge_kind, "
                    " score_value, passed, rationale, failure_reason) "
                    "VALUES (?, ?, ?, 'continuous', ?, NULL, ?, NULL)",
                    (
                        run_id, vid, eid, score,
                        f"rationale for ({vid}, {eid})",
                    ),
                )
        db.commit()
    finally:
        db.close()


_WC_SECTION_BEGIN = "<!-- WORSTCASE-SECTION-BEGIN -->"
_WC_SECTION_END = "<!-- WORSTCASE-SECTION-END -->"


def _extract_worstcase_section(body: str) -> str:
    """Return the §10a section between sentinel HTML comments.

    Audit pass-2 RT-MAJOR-1 / RT-MAJOR-2: assertions on the full body
    can match SVG-placeholder copy or future-section content elsewhere.
    Scoping to the worst-case section makes the substring/grep tests
    a real localised contract.
    """
    start = body.index(_WC_SECTION_BEGIN)
    end = body.index(_WC_SECTION_END, start + len(_WC_SECTION_BEGIN))
    return body[start : end + len(_WC_SECTION_END)]


def test_report_worstcase_section_renders_when_verdicts_flip(
    tmp_path: Path,
) -> None:
    """When the score matrix has a flipping pair AND per-column
    variance clears the DoD #13 gate, the §10a section renders with at
    least one WORST card and the eval_id appears in that section."""
    run_dir = _make_run_dir(tmp_path, n_variants=2, n_evals=2)
    _populate_cells_for_worstcase(
        run_dir,
        variant_ids=["var00", "var01"],
        eval_ids=["eval-0", "eval-1"],
    )
    # eval-0: var=0.16 above gate; eval-1: var=0.16 above gate.
    _rewrite_scores_for_flip(
        run_dir,
        flips={
            "var00": {"eval-0": 0.9, "eval-1": 0.1},
            "var01": {"eval-0": 0.1, "eval-1": 0.9},
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_worstcase_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Worst-case viewer" in section
    assert "WORST #1" in section
    assert "eval-0" in section
    # Trace excerpt + rationale propagation, scoped to the section.
    assert "trace excerpt for (var00, eval-0)" in section
    assert "rationale for (var01, eval-0)" in section
    assert "No instability detected" not in section


def test_report_worstcase_banner_when_all_agree(tmp_path: Path) -> None:
    """When every variant produces the same verdict on every eval, the
    §10a section renders the 'no instability detected' banner copy."""
    run_dir = _make_run_dir(tmp_path, n_variants=2, n_evals=3)
    _populate_cells_for_worstcase(
        run_dir,
        variant_ids=["var00", "var01"],
        eval_ids=["eval-0", "eval-1", "eval-2"],
    )
    _rewrite_scores_for_flip(
        run_dir,
        flips={
            v: {e: 0.85 for e in ("eval-0", "eval-1", "eval-2")}
            for v in ("var00", "var01")
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_worstcase_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Worst-case viewer" in section
    assert "No instability detected" in section
    assert "WORST #" not in section


def test_report_worstcase_disabled_when_score_matrix_empty(
    tmp_path: Path,
) -> None:
    """An empty score matrix routes through a distinct 'no continuous-
    judge cells to rank' banner — NOT the 'no instability detected'
    copy reserved for the BLUEPRINT line 300 stable case (audit pass-1
    RT MAJOR-3 + pass-2 MAJOR-1 scoping)."""
    run_dir = _make_run_dir(tmp_path)
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DELETE FROM judge_results")
        db.commit()
    finally:
        db.close()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_worstcase_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Worst-case viewer" in section
    assert "no continuous-judge cells to rank" in section
    # MUST NOT masquerade as the BLUEPRINT line 300 stable banner.
    assert "No instability detected" not in section
    assert "WORST #" not in section


def test_report_worstcase_banner_copy_is_blueprint_verbatim(
    tmp_path: Path,
) -> None:
    """The stable-case banner reproduces BLUEPRINT line 300 verbatim
    (audit pass-1 RT-MAJOR-2 / pass-2 RT-MAJOR-1 scoping). Step 22's
    CEO outreach PDF will grep-verify this copy against the spec; any
    drift breaks the screenshot."""
    run_dir = _make_run_dir(tmp_path, n_variants=2, n_evals=3)
    _populate_cells_for_worstcase(
        run_dir,
        variant_ids=["var00", "var01"],
        eval_ids=["eval-0", "eval-1", "eval-2"],
    )
    _rewrite_scores_for_flip(
        run_dir,
        flips={
            v: {e: 0.85 for e in ("eval-0", "eval-1", "eval-2")}
            for v in ("var00", "var01")
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_worstcase_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    # The BLUEPRINT_v5plus.md §10a line 300 copy, verbatim, scoped to
    # the §10a section so a stray placeholder string elsewhere can't
    # satisfy the assertion.
    spec_copy = (
        "No instability detected — your topology produced consistent "
        "answers across all orderings."
    )
    assert spec_copy in section, (
        "Banner copy must reproduce BLUEPRINT §10a line 300 verbatim "
        "for step 22 grep verification."
    )


def test_report_worstcase_visual_regression_snapshot(tmp_path: Path) -> None:
    """BLUEPRINT DoD #13 (line 533): visual regression on a fixture run.

    Builds a deterministic fixture run dir, renders the report twice,
    and asserts (a) the §10a section is byte-identical across renders
    (determinism) and (b) the rendered fragment contains the expected
    structural anchors. Extraction uses HTML sentinel comments (audit
    pass-2 RT-MAJOR-2) so a future template re-flow (e.g. step 17
    moving sections) doesn't silently mask the snapshot's coverage.
    """
    run_dir = _make_run_dir(tmp_path, n_variants=2, n_evals=3)
    _populate_cells_for_worstcase(
        run_dir,
        variant_ids=["var00", "var01"],
        eval_ids=["eval-0", "eval-1", "eval-2"],
    )
    # Two evals flip with high per-column variance; one is stable.
    # Expected: cards for eval-0 and eval-1 in lex order, eval-2 omitted.
    _rewrite_scores_for_flip(
        run_dir,
        flips={
            "var00": {"eval-0": 0.95, "eval-1": 0.90, "eval-2": 0.85},
            "var01": {"eval-0": 0.05, "eval-1": 0.10, "eval-2": 0.80},
        },
    )

    def _render_section() -> str:
        result = CliRunner().invoke(app, ["report", str(run_dir)])
        assert result.exit_code == 0, result.output
        body = (run_dir / "report.html").read_text(encoding="utf-8")
        return _extract_worstcase_section(body)

    first = _render_section()
    second = _render_section()
    # Determinism: same fixture → byte-identical rendered section.
    assert first == second, (
        "Worst-case section is not deterministic across renders"
    )

    # Structural anchors, scoped to the section.
    assert "WORST #1" in first
    assert "WORST #2" in first
    assert "WORST #3" not in first  # only two flipping evals
    # Top-K selection is ordered by flip count (both 1 here) then by
    # eval-id lex — so eval-0 precedes eval-1 in the rendered HTML.
    assert first.index("eval-0") < first.index("eval-1")
    # Both variants appear in side labels.
    assert "var00" in first
    assert "var01" in first
    # Stable eval (eval-2) is NOT promoted to a card.
    assert "eval-2" not in first


def test_report_worstcase_section_sentinels_present(tmp_path: Path) -> None:
    """The HTML sentinel comments wrap the §10a section in every
    rendered report — the snapshot extractor depends on them, so we
    pin the contract here so a template refactor that drops the
    sentinels surfaces immediately (audit pass-2 RT-MAJOR-2)."""
    run_dir = _make_run_dir(tmp_path)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert _WC_SECTION_BEGIN in body
    assert _WC_SECTION_END in body
    assert body.index(_WC_SECTION_BEGIN) < body.index(_WC_SECTION_END)


# ---------------------------------------------------------------------------
# §17 natural-language summary (step 17)
# ---------------------------------------------------------------------------


_SUMMARY_SECTION_BEGIN = "<!-- SUMMARY-SECTION-BEGIN -->"
_SUMMARY_SECTION_END = "<!-- SUMMARY-SECTION-END -->"


def _extract_summary_section(body: str) -> str:
    """Return the §17 summary section between sentinel HTML comments.

    Same pattern as :func:`_extract_worstcase_section` — scoped assertions
    keep tests honest even if a future section introduces overlapping
    copy.
    """
    start = body.index(_SUMMARY_SECTION_BEGIN)
    end = body.index(_SUMMARY_SECTION_END, start + len(_SUMMARY_SECTION_BEGIN))
    return body[start : end + len(_SUMMARY_SECTION_END)]


def _write_report_json(
    run_dir: Path,
    *,
    verdict_flip_rate: dict | None,
    per_agent_influence: list[dict] | None = None,
) -> None:
    """Write a step-15a-shaped ``report.json`` mirror under ``run_dir``.

    Only the fields :func:`_render_summary_panel` reads are populated;
    other fields (manifest, schema_version, signed_at) are omitted on
    purpose so the test fails loudly if the reader ever grows a new
    dependency we forgot to wire.
    """
    doc = {
        "headline": {
            "verdict_flip_rate": verdict_flip_rate,
        },
        "findings": {
            "per_agent_influence": per_agent_influence or [],
        },
    }
    (run_dir / "report.json").write_text(json.dumps(doc), encoding="utf-8")


def test_report_summary_section_sentinels_present(tmp_path: Path) -> None:
    """The §17 sentinel HTML comments wrap the summary div in every
    rendered report so a future template refactor that drops them
    surfaces immediately."""
    run_dir = _make_run_dir(tmp_path)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert _SUMMARY_SECTION_BEGIN in body
    assert _SUMMARY_SECTION_END in body
    # And the §17 sentinels precede the §10a sentinels (summary is at top).
    assert body.index(_SUMMARY_SECTION_BEGIN) < body.index(_WC_SECTION_BEGIN)


def test_report_summary_unstable_with_driver(tmp_path: Path) -> None:
    """A real report.json with high flip rate + a positive-influence row
    renders the unstable-tier paragraph naming the driver."""
    run_dir = _make_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.31, "ci_lower": 0.24, "ci_upper": 0.38,
            "degenerate_flag": False,
        },
        per_agent_influence=[
            {"node_id": "risk", "influence": 0.67,
             "ci_lower": 0.55, "ci_upper": 0.78,
             "degenerate": False, "degenerate_reason": None},
            {"node_id": "intake", "influence": 0.10,
             "ci_lower": 0.05, "ci_upper": 0.15,
             "degenerate": False, "degenerate_reason": None},
        ],
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_summary_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "<strong>unstable</strong>" in section
    assert "nls-unstable" in section
    assert "<code>risk</code>" in section
    assert "31% of answers flipped" in section
    assert "pin its position or add a verification step downstream" in section


def test_report_summary_stable_no_action(tmp_path: Path) -> None:
    """Low flip rate renders the stable tier and no recommendation."""
    run_dir = _make_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.02, "ci_lower": 0.00, "ci_upper": 0.04,
            "degenerate_flag": False,
        },
        per_agent_influence=[
            {"node_id": "risk", "influence": 0.08,
             "ci_lower": 0.05, "ci_upper": 0.12,
             "degenerate": False, "degenerate_reason": None},
        ],
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_summary_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "<strong>stable</strong>" in section
    assert "nls-stable" in section
    assert "No action recommended" in section
    assert "pin its position or add a verification step downstream" not in section


def test_report_summary_unavailable_without_report_json(tmp_path: Path) -> None:
    """When ``report.json`` is missing, the section degrades to the
    'unavailable' placeholder and the rest of the report renders fine."""
    run_dir = _make_run_dir(tmp_path)
    assert not (run_dir / "report.json").exists()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    section = _extract_summary_section(body)
    assert "nls-unavailable" in section
    # Sanity: rest of the report still rendered.
    assert "<title>topofuzz report" in body
    assert "Worst-case viewer" in body


def test_report_summary_unavailable_with_malformed_report_json(
    tmp_path: Path,
) -> None:
    """Garbage in ``report.json`` degrades to 'unavailable' without crash."""
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "report.json").write_text("{ not valid json", encoding="utf-8")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_summary_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "nls-unavailable" in section


def test_report_summary_escapes_malicious_node_id(tmp_path: Path) -> None:
    """A node_id containing HTML metacharacters in report.json is
    HTML-escaped, never injected. Defence in depth — the IR restricts
    node_id syntax, but this boundary is load-bearing for the report."""
    run_dir = _make_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.31, "ci_lower": 0.24, "ci_upper": 0.38,
            "degenerate_flag": False,
        },
        per_agent_influence=[
            {"node_id": "</div><script>alert(1)</script>",
             "influence": 0.67,
             "ci_lower": 0.55, "ci_upper": 0.78,
             "degenerate": False, "degenerate_reason": None},
        ],
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_summary_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "<script>alert(1)</script>" not in section
    assert "&lt;script&gt;" in section


def test_report_summary_unstable_no_driver(tmp_path: Path) -> None:
    """Unstable verdict but every influence row is degenerate → fallback
    wording, no recommendation."""
    run_dir = _make_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.31, "ci_lower": 0.24, "ci_upper": 0.38,
            "degenerate_flag": False,
        },
        per_agent_influence=[
            {"node_id": "a", "influence": None,
             "ci_lower": None, "ci_upper": None,
             "degenerate": True, "degenerate_reason": "no moved pairs"},
        ],
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_summary_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "<strong>unstable</strong>" in section
    assert "No single agent" in section
    assert "pin its position or add a verification step downstream" not in section


def test_report_summary_matches_real_attestation_chain(
    tmp_path: Path,
) -> None:
    """Audit pass-1 RT-MINOR-1: cover the full attest → report.json →
    summary chain on a real multi-agent fixture.

    Without this, the §17 integration tests all hand-craft ``report.json``
    via ``_write_report_json`` and the contract that "summary's
    primary_driver matches the highest-influence row in the rendered
    panel" is enforced only by construction, not by test. A future
    contributor who switches the summary back to in-process recompute
    (or who changes the field name on either side) would slip past CI.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from fuzzeval.attestation import Ed25519Signer

    from topofuzz.attest import emit_run_attestation

    # Same polarised cell layout as
    # ``test_attestation_findings_per_agent_influence_populated`` — A
    # dominates outcomes, so the real influence computation produces a
    # clean ranking with A at top.
    orderings = [
        ["A", "B", "C"],
        ["A", "C", "B"],
        ["B", "A", "C"],
        ["B", "C", "A"],
    ]
    run_dir = _make_run_dir(
        tmp_path, n_variants=4, n_evals=3, variant_orderings=orderings,
    )
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DELETE FROM judge_results")
        cells = [
            (0, 0.9), (0, 0.85), (0, 0.92),
            (1, 0.88), (1, 0.91), (1, 0.86),
            (2, 0.10), (2, 0.05), (2, 0.12),
            (3, 0.08), (3, 0.11), (3, 0.09),
        ]
        eval_ids = [f"eval-{i}" for i in range(3)]
        for i, (v_idx, score) in enumerate(cells):
            db.execute(
                "INSERT INTO judge_results "
                "(run_id, variant_id, eval_id, judge_kind, score_value, "
                " passed, rationale, failure_reason) "
                "VALUES (?, ?, ?, 'continuous', ?, NULL, NULL, NULL)",
                ("run-test", f"var{v_idx:02d}", eval_ids[i % 3], score),
            )
        db.commit()
    finally:
        db.close()

    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    emit_run_attestation(
        run_dir=run_dir,
        run_id="run-test",
        manifest=manifest,
        status="succeeded",
        successful_cells=12,
        total_cells=12,
        n_variants=4,
        n_evals=3,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        signer=signer,
    )

    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output

    body = (run_dir / "report.html").read_text(encoding="utf-8")

    # Pull the primary driver out of the persisted report.json — this is
    # what the summary reads — and assert the rendered HTML names it.
    report_doc = json.loads(
        (run_dir / "report.json").read_text(encoding="utf-8")
    )
    rows = report_doc["findings"]["per_agent_influence"]
    usable = [
        r for r in rows
        if not r.get("degenerate") and r.get("influence") is not None
        and r.get("influence") > 0
    ]
    assert usable, "fixture is supposed to produce >=1 usable driver row"
    top = max(usable, key=lambda r: r["influence"])

    section = _extract_summary_section(body)
    assert f"<code>{top['node_id']}</code>" in section, (
        f"summary should name the highest-influence agent "
        f"({top['node_id']!r}); got section:\n{section!r}"
    )
    # And the same agent appears in the §6a influence table further down.
    # The table renders rows with the raw node_id; this is the "panel and
    # summary do not drift" contract.
    assert top["node_id"] in body


def test_report_summary_idempotent_render(tmp_path: Path) -> None:
    """Rendering the same run_dir twice produces a byte-identical summary
    section — pins determinism so a future contributor doesn't introduce
    a timestamp or hash into the rendered text."""
    run_dir = _make_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.31, "ci_lower": 0.24, "ci_upper": 0.38,
            "degenerate_flag": False,
        },
        per_agent_influence=[
            {"node_id": "risk", "influence": 0.67,
             "ci_lower": 0.55, "ci_upper": 0.78,
             "degenerate": False, "degenerate_reason": None},
        ],
    )
    runner = CliRunner()
    runner.invoke(app, ["report", str(run_dir)])
    first = _extract_summary_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    runner.invoke(app, ["report", str(run_dir)])
    second = _extract_summary_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert first == second


# ---------------------------------------------------------------------------
# Step 18.5b — flip-composition donut + per-eval variance sparkline
# ---------------------------------------------------------------------------


_FLIP_SECTION_BEGIN = "<!-- FLIP-COMPOSITION-SECTION-BEGIN -->"
_FLIP_SECTION_END = "<!-- FLIP-COMPOSITION-SECTION-END -->"
_VARIANCE_SECTION_BEGIN = "<!-- VARIANCE-DISTRIBUTION-SECTION-BEGIN -->"
_VARIANCE_SECTION_END = "<!-- VARIANCE-DISTRIBUTION-SECTION-END -->"


def _extract_flip_composition_section(body: str) -> str:
    """Slice the §14 flip-composition section between sentinel comments.

    Same scoping pattern as :func:`_extract_summary_section` /
    :func:`_extract_worstcase_section` — assertions on the whole body
    risk matching disclaimer copy or future-section content elsewhere.
    """
    start = body.index(_FLIP_SECTION_BEGIN)
    end = body.index(_FLIP_SECTION_END, start + len(_FLIP_SECTION_BEGIN))
    return body[start : end + len(_FLIP_SECTION_END)]


def _extract_variance_distribution_section(body: str) -> str:
    """Slice the §9 variance-distribution sparkline section."""
    start = body.index(_VARIANCE_SECTION_BEGIN)
    end = body.index(_VARIANCE_SECTION_END, start + len(_VARIANCE_SECTION_BEGIN))
    return body[start : end + len(_VARIANCE_SECTION_END)]


def _make_flipping_run_dir(tmp_path: Path) -> Path:
    """Build a 4-variant × 3-eval fixture with a known flip pattern.

    var00 is the §6 original (insertion-order-first); the three permuted
    variants flip on a controlled subset of cells so the directional
    split is mechanically computable and easy to grep in tests.

    Score layout (threshold = 0.5):

    *   var00 (original): all PASS (0.9, 0.9, 0.9)
    *   var01: all FAIL (0.1, 0.1, 0.1) → 3 approve→deny vs original
    *   var02: PASS, FAIL, PASS (0.9, 0.1, 0.9) → 1 approve→deny
    *   var03: all PASS (0.9, 0.9, 0.9) → 0 flips

    Total flips = 4 (all approve→deny). n_pairs = 9 (3 perm variants
    × 3 evals). Verdict-flip rate = 4/9 ≈ 0.444.
    """
    orderings = [["A", "B"]] * 4
    run_dir = _make_run_dir(
        tmp_path, n_variants=4, n_evals=3, variant_orderings=orderings,
    )
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DELETE FROM judge_results")
        cells = {
            "var00": [0.9, 0.9, 0.9],
            "var01": [0.1, 0.1, 0.1],
            "var02": [0.9, 0.1, 0.9],
            "var03": [0.9, 0.9, 0.9],
        }
        eval_ids = [f"eval-{i}" for i in range(3)]
        for vid, scores in cells.items():
            for eid, sc in zip(eval_ids, scores, strict=True):
                db.execute(
                    "INSERT INTO judge_results "
                    "(run_id, variant_id, eval_id, judge_kind, "
                    " score_value, passed, rationale, failure_reason) "
                    "VALUES (?, ?, ?, 'continuous', ?, NULL, NULL, NULL)",
                    ("run-test", vid, eid, sc),
                )
        db.commit()
    finally:
        db.close()
    return run_dir


# --- flip-composition section -----------------------------------------------


def test_report_flip_composition_section_sentinels_present(tmp_path: Path) -> None:
    """Sentinel HTML comments wrap the §14 section in every render — the
    test extractor depends on them, and any future template refactor
    that drops them must surface immediately."""
    run_dir = _make_run_dir(tmp_path)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert _FLIP_SECTION_BEGIN in body
    assert _FLIP_SECTION_END in body
    assert body.index(_FLIP_SECTION_BEGIN) < body.index(_FLIP_SECTION_END)
    # And the donut sentinels precede the worst-case sentinels (donut
    # sits above the meta-grid, well above §10a worst-case).
    assert body.index(_FLIP_SECTION_BEGIN) < body.index(_WC_SECTION_BEGIN)


def test_report_flip_composition_disabled_without_report_json(
    tmp_path: Path,
) -> None:
    """A pre-step-15a run dir (no report.json) suppresses the §14
    section entirely (audit UX pass-2 MAJOR-3: a question-shaped
    void above the meta-grid is worse UX than no section). Sentinels
    still wrap the (empty) section so test extractors keep working.
    """
    run_dir = _make_run_dir(tmp_path)
    assert not (run_dir / "report.json").exists()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    section = _extract_flip_composition_section(body)
    # H2 + explainer + body MUST all be suppressed when disabled.
    assert "Where did the flips happen?" not in section
    assert "Flip composition unavailable" not in section
    assert "verdict-flip rate" not in section
    # Sentinels still present + bracket only whitespace.
    assert _FLIP_SECTION_BEGIN in body
    assert _FLIP_SECTION_END in body


def test_report_flip_composition_disabled_with_malformed_report_json(
    tmp_path: Path,
) -> None:
    """Garbage JSON in report.json suppresses the section, not crash."""
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "report.json").write_text("{not valid", encoding="utf-8")
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Where did the flips happen?" not in section


def test_report_flip_composition_disabled_with_missing_headline_field(
    tmp_path: Path,
) -> None:
    """report.json present but ``headline.verdict_flip_rate.point`` is
    missing → section suppressed."""
    run_dir = _make_run_dir(tmp_path)
    _write_report_json(run_dir, verdict_flip_rate=None)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Where did the flips happen?" not in section


def test_report_flip_composition_disabled_with_out_of_range_point(
    tmp_path: Path,
) -> None:
    """A nonsense point value (> 1 or < 0) is rejected — defence in
    depth against a serialiser bug that wrote a percentage as 31.0
    instead of 0.31. Section suppressed."""
    run_dir = _make_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 1.5, "ci_lower": 0.0, "ci_upper": 1.0,
            "degenerate_flag": False,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Where did the flips happen?" not in section


def test_report_flip_composition_disabled_for_single_variant(
    tmp_path: Path,
) -> None:
    """A 1-variant fixture has no permuted variants → score matrix
    has 1 row → directional split is structurally undefined → section
    is suppressed (audit UX pass-2 MAJOR-3). The internal disabled
    reason "single-variant run (no permuted variants to compare)"
    is logged but not user-facing."""
    run_dir = _make_run_dir(tmp_path, n_variants=1, n_evals=3)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.0, "ci_lower": None, "ci_upper": None,
            "degenerate_flag": True,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Where did the flips happen?" not in section


def test_report_flip_composition_renders_with_real_data(tmp_path: Path) -> None:
    """Happy-path render: donut SVG present, integer percent in centre,
    legend names the flip count, directional split decomposes the total."""
    run_dir = _make_flipping_run_dir(tmp_path)
    # 4/9 ≈ 0.444 — matches what the fixture's matrix produces. Pre-write
    # the headline so the section renders even without an attestation pass.
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 4.0 / 9.0, "ci_lower": 0.20, "ci_upper": 0.70,
            "degenerate_flag": False,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    # Donut SVG present + center text rounds to 44%.
    assert "<svg" in section
    assert ">44%<" in section
    # Legend mentions 4 of 9 (variant, eval) pairs (audit Legal MINOR:
    # "pair" was undefined; legend now spells the unit out).
    assert "4 of 9 (variant, eval) pairs flipped" in section
    assert "5 of 9 pairs stable" in section
    # Directional split: all 4 flips are approve→deny by construction.
    # The non-color cue (↓ glyph + descriptive "(more denials)"
    # annotation; audit Legal pass-2 MINOR-1 reworded "got stricter"
    # → "(more denials)" to stay descriptive instead of evaluative).
    assert "↓ 4 went APPROVE → DENY" in section
    assert "more denials" in section
    assert "↑ 0 went DENY → APPROVE" in section
    assert "more approvals" in section


def test_report_flip_composition_split_decomposes_count(tmp_path: Path) -> None:
    """For any fixture, approve→deny + deny→approve == n_flips —
    pinning the donut's invariant so a future refactor of either
    counter can't silently desynchronise the legend total."""
    run_dir = _make_flipping_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 4.0 / 9.0, "ci_lower": 0.0, "ci_upper": 1.0,
            "degenerate_flag": False,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    # Pull the three numbers out of the section via simple substring
    # parsing. Legend after pass-1 + pass-2 fixes:
    # "X of Y (variant, eval) pairs flipped" / "↓ X went APPROVE → DENY (more denials)"
    # / "↑ X went DENY → APPROVE (more approvals)". A regex would buy
    # nothing — the fixture is deterministic.
    assert "4 of 9 (variant, eval) pairs flipped" in section
    assert "↓ 4 went APPROVE → DENY" in section
    assert "↑ 0 went DENY → APPROVE" in section
    # 4 = 4 + 0 by inspection — the invariant.


def test_report_flip_composition_idempotent_render(tmp_path: Path) -> None:
    """Two renders of the same run dir produce a byte-identical §14
    section — pinning determinism so a future contributor doesn't
    sneak a wall-clock or hash into the donut."""
    run_dir = _make_flipping_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 4.0 / 9.0, "ci_lower": 0.0, "ci_upper": 1.0,
            "degenerate_flag": False,
        },
    )
    runner = CliRunner()
    runner.invoke(app, ["report", str(run_dir)])
    first = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    runner.invoke(app, ["report", str(run_dir)])
    second = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert first == second


# --- variance-distribution sparkline ----------------------------------------


def test_report_variance_distribution_section_sentinels_present(
    tmp_path: Path,
) -> None:
    """Sentinel HTML comments wrap the §9 sparkline section in every
    render."""
    run_dir = _make_run_dir(tmp_path)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")
    assert _VARIANCE_SECTION_BEGIN in body
    assert _VARIANCE_SECTION_END in body
    assert body.index(_VARIANCE_SECTION_BEGIN) < body.index(_VARIANCE_SECTION_END)
    # The sparkline sits ABOVE the existing matplotlib voo-bar §9 detail
    # chart per the Variant D sketch (sparkline = at-a-glance teaser,
    # matplotlib = engineer detail).
    assert "§9 detail" in body
    assert body.index(_VARIANCE_SECTION_END) < body.index("§9 detail")


def test_report_variance_distribution_renders_with_real_data(
    tmp_path: Path,
) -> None:
    """Happy-path sparkline render: SVG present, threshold line drawn,
    caption announces n_above_gate / n_evals."""
    run_dir = _make_flipping_run_dir(tmp_path)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_variance_distribution_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "<svg" in section
    # Threshold line label sits in the SVG as an inline text element.
    assert "variance = 0.1" in section
    assert "topology-fragility threshold" in section
    # 3 evals in the flipping fixture; per-eval variance computed by
    # np.nanvar(matrix, axis=0, ddof=0) across var00=[0.9,0.9,0.9],
    # var01=[0.1,0.1,0.1], var02=[0.9,0.1,0.9], var03=[0.9,0.9,0.9]
    # is [0.12, 0.16, 0.12] — all three clear the 0.1 gate. Audit
    # pass-2 Code-reviewer MINOR-5: pin "3 of 3" exactly so a future
    # regression that flipped the count to "0 of 3" or "1 of 3"
    # wouldn't pass the looser "of 3 evals cleared" substring.
    assert "3 of 3 evals cleared the threshold" in section


def test_report_variance_distribution_disabled_when_matrix_empty(
    tmp_path: Path,
) -> None:
    """An empty score matrix → sparkline disabled with placeholder copy.
    Same fallback pattern as the existing voo-bar matplotlib SVG."""
    run_dir = _make_run_dir(tmp_path)
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute("DELETE FROM judge_results")
        db.commit()
    finally:
        db.close()
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_variance_distribution_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "Variance distribution unavailable" in section
    assert "no continuous-judge cells to score" in section


def test_report_variance_distribution_truncation_label_present(
    tmp_path: Path,
) -> None:
    """When the matrix has more evals than the sparkline render cap, the
    caption announces the truncation count so the user knows the
    headline numbers cover more than the chart shows."""
    from topofuzz.report import _SPARKLINE_BAR_RENDER_CAP

    n_evals = _SPARKLINE_BAR_RENDER_CAP + 5
    run_dir = _make_run_dir(tmp_path, n_variants=2, n_evals=n_evals)
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_variance_distribution_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    # Caption shows "showing top 200 of 205 evals in the chart" when
    # the bar render cap kicks in (audit UX MINOR: cap exposed as a
    # clean truncation message instead of "rank N omitted").
    assert (
        f"showing top {_SPARKLINE_BAR_RENDER_CAP} of {n_evals} evals in the chart"
        in section
    )


def test_report_variance_distribution_idempotent_render(tmp_path: Path) -> None:
    """Two renders → byte-identical sparkline section, pinning
    determinism on the hand-rolled SVG."""
    run_dir = _make_flipping_run_dir(tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["report", str(run_dir)])
    first = _extract_variance_distribution_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    runner.invoke(app, ["report", str(run_dir)])
    second = _extract_variance_distribution_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert first == second


# --- audit pass-1 fix coverage ----------------------------------------------


def test_report_flip_composition_directional_split_survives_non_lex_hashes(
    tmp_path: Path,
) -> None:
    """Audit RT pass-1 MAJOR-4: the donut's directional split MUST
    pick the §6 original variant via the manifest hash list, NOT by
    score-matrix row index. Mirrors the existing influence-panel
    non-lex test (`test_attestation_influence_alignment_survives_non_lex_hashes`).
    The score-matrix axis is canonical_hash lex-sorted; the §6
    original is `manifest.variant_canonical_hashes[0]`. If the donut
    silently drifted to "row 0 of the matrix" the lex-coincident
    happy-path tests would still pass, but a non-lex hash list would
    attribute the wrong variant's verdict to the original.
    """
    # Hash list in NON-lex order so var_a (the polarised "all PASS"
    # row that should be the original by manifest-insertion order)
    # does NOT happen to sit at matrix row 0.
    hashes = ["zvar_a", "avar_b", "mvar_c", "qvar_d"]
    # Lex-sorted: avar_b, mvar_c, qvar_d, zvar_a → matrix axis order
    # places zvar_a at index 3, NOT 0.
    orderings = [["X"]] * 4
    run_dir = tmp_path / "non-lex-flip"
    run_dir.mkdir()

    manifest = {
        "run_id": "non-lex-flip",
        "topology_canonical_hash": "topo-hash",
        "topology_name": "lending_4role",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": 4,
        "n_eval_rows": 3,
        "variant_canonical_hashes": hashes,
        "variant_orderings": orderings,
        "eval_ids": ["eval-0", "eval-1", "eval-2"],
        "max_cost_usd": None,
        "judge_model": "google/gemini-2.5-pro",
        "rubric_hash": "rh",
        "semaphore_size": 8,
        "cli_args": {},
        "kickoff_ts": 1700000000.0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    db = sqlite3.connect(str(run_dir / "run.sqlite"))
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
        """)
        db.execute(
            "INSERT INTO runs (run_id, topology_canonical_hash, "
            "topology_name, tds_version, created_at, status, rubric_hash) "
            "VALUES ('non-lex-flip', 'h', 'name', '0.5', 0.0, 'succeeded', 'rh')"
        )
        # Score layout — keyed by HASH not by row index. zvar_a is
        # the §6 original (manifest insertion-order index 0); only
        # avar_b flips against it. Expected counts under correct
        # alignment: 3 a2d (avar_b's three FAIL cells vs zvar_a's
        # three PASS), 0 d2a, 9 n_pairs.
        cells_per_variant = {
            "zvar_a": [0.9, 0.9, 0.9],   # original (§6 insertion-order index 0)
            "avar_b": [0.1, 0.1, 0.1],   # flips on every eval
            "mvar_c": [0.9, 0.9, 0.9],   # no flips
            "qvar_d": [0.9, 0.9, 0.9],   # no flips
        }
        eval_ids = ["eval-0", "eval-1", "eval-2"]
        for vid, scores in cells_per_variant.items():
            for eid, sc in zip(eval_ids, scores, strict=True):
                db.execute(
                    "INSERT INTO judge_results "
                    "(run_id, variant_id, eval_id, judge_kind, "
                    " score_value, passed, rationale, failure_reason) "
                    "VALUES ('non-lex-flip', ?, ?, 'continuous', ?, "
                    " NULL, NULL, NULL)",
                    (vid, eid, sc),
                )
        db.commit()
    finally:
        db.close()
    # 3/9 = 0.333... — what compute_verdict_flip_rate would give too.
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 3.0 / 9.0, "ci_lower": 0.0, "ci_upper": 1.0,
            "degenerate_flag": False,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    # If the donut had silently picked matrix-row-0 (= avar_b) as
    # "original", we'd see 3 d2a (FAIL→PASS, since zvar_a/mvar_c/
    # qvar_d all PASS against avar_b's FAIL) and 0 a2d. The correct
    # alignment yields 3 a2d / 0 d2a.
    assert "↓ 3 went APPROVE → DENY" in section
    assert "↑ 0 went DENY → APPROVE" in section


def test_report_flip_composition_drift_warning_renders(tmp_path: Path) -> None:
    """Audit consensus MAJOR (Code-reviewer + RT + Blueprint pass-1):
    when the signed flip_rate from report.json disagrees with the live
    n_flips/n_pairs from run.sqlite by more than the drift tolerance,
    the legend renders a stale-mirror warning. Audit pass-2
    Code-reviewer MAJOR-3: the legend's `(X%)` parenthetical is dropped
    on drift to avoid the contradiction "4 of 9 pairs flipped (10%)";
    the drift band names BOTH the signed and live percentages explicitly."""
    run_dir = _make_flipping_run_dir(tmp_path)
    # Live matrix produces 4 of 9 flips (≈ 44%); write a stale
    # report.json claiming 10% to force the drift detector.
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.10, "ci_lower": 0.0, "ci_upper": 0.5,
            "degenerate_flag": False,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    # Drift annotation is visible + spells out both percentages.
    assert "donut percentage (10%)" in section
    assert "legend counts (44%)" in section
    assert "topofuzz attest" in section
    # Donut center renders signed 10%; legend renders the live count.
    assert ">10%<" in section
    assert "4 of 9 (variant, eval) pairs flipped" in section
    # Critical: the legend MUST NOT render "(10%)" or "(90%)" after
    # the count when drift is active — that would be the
    # internally-inconsistent "4 of 9 (variant, eval) pairs flipped (10%)"
    # that audit pass-2 MAJOR-3 was about.
    assert "pairs flipped (10%)" not in section
    assert "pairs stable (90%)" not in section


def test_report_flip_composition_no_drift_warning_on_clean_run(
    tmp_path: Path,
) -> None:
    """Inverse of the drift test: when signed and live agree within
    tolerance, the warning does NOT appear (no false positives)."""
    run_dir = _make_flipping_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 4.0 / 9.0, "ci_lower": 0.0, "ci_upper": 1.0,
            "degenerate_flag": False,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    section = _extract_flip_composition_section(
        (run_dir / "report.html").read_text(encoding="utf-8")
    )
    assert "report.json is stale" not in section


def test_report_variance_sparkline_gate_uses_inclusive_comparator(
    tmp_path: Path,
) -> None:
    """Audit RT pass-1 MAJOR-3 + pass-2 MAJOR-1: the sparkline gate
    is `>= DEFAULT_VARIANCE_GATE` (matches `worstcase.py:356`), not
    `>`. Test directly at the function level with FP-free synthetic
    bars: the previous fixture-based approach used `sqrt(0.1)` whose
    `np.var` returns `0.10000000000000002` — passes both `>` AND
    `>=`, so a regression to strict `>` would slip through.

    Direct call to `_render_sparkline_svg` with three explicit bars:
    one at exactly the gate (must be accent-coloured under `>=`,
    muted under `>`), one above, one below. The bar-fill color is
    the comparator-discriminating signal.
    """
    from topofuzz.report import _render_sparkline_svg
    import numpy as np

    # Exactly at the gate, exactly above, exactly below — all
    # bit-identical 0.1-multiples so the comparison is FP-exact.
    sorted_vals = np.array([0.2, 0.1, 0.05], dtype=np.float64)
    svg = _render_sparkline_svg(
        sorted_vals,
        max_variance=0.2,
        threshold=0.1,
        n_above_gate_total=2,
        n_total=3,
    )
    # First bar (0.2) is above the gate → accent.
    # Second bar (0.1) is AT the gate → accent under `>=`, muted under `>`.
    # Third bar (0.05) is below the gate → muted.
    accent_rects = svg.count('fill="#c25024"')
    muted_rects = svg.count('fill="#cccccc"')
    assert accent_rects == 2, (
        f"Expected 2 accent bars (0.2 + 0.1 boundary) under `>=` "
        f"semantics; got {accent_rects}. A regression to strict `>` "
        f"would yield 1 accent bar."
    )
    assert muted_rects == 1, (
        f"Expected 1 muted bar (0.05 below gate); got {muted_rects}."
    )


def test_report_full_section_chain_ordering(tmp_path: Path) -> None:
    """Audit pass-2 Cross-step MAJOR-3: pin the full top-down section
    order so a future template refactor cannot legally rearrange any
    pair of sections without a test failing. The previously-pinned
    pairwise checks (summary < worstcase, flip < worstcase, variance <
    §9 detail) left several legal-but-undesired reorderings unguarded.
    """
    run_dir = _make_flipping_run_dir(tmp_path)
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 4.0 / 9.0, "ci_lower": 0.0, "ci_upper": 1.0,
            "degenerate_flag": False,
        },
    )
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    body = (run_dir / "report.html").read_text(encoding="utf-8")

    # The end-to-end chain: SUMMARY (§17 prose) → FLIP-COMPOSITION
    # (§14 donut) → "What we tested" (manifest H2) → "Per-agent
    # influence" (§6a H2) → WORSTCASE (§10a) → VARIANCE-DISTRIBUTION
    # (§9 sparkline) → "§9 detail" (matplotlib voo bar) → score
    # matrix → eval IDs → footer disclaimer.
    #
    # Each marker is anchored to the HTML body (H2 tag prefix or
    # sentinel comment) so a CSS-block substring match can't satisfy
    # the assertion accidentally.
    chain = [
        _SUMMARY_SECTION_BEGIN,
        _FLIP_SECTION_BEGIN,
        "<h2>What we tested",
        "<h2>Per-agent influence",
        _WC_SECTION_BEGIN,
        _VARIANCE_SECTION_BEGIN,
        ">§9 detail</span>",
        "<h2>Score matrix",
        "<h2>Eval IDs",
        "Informational only",  # footer disclaimer fragment
    ]
    positions = [body.index(marker) for marker in chain]
    assert positions == sorted(positions), (
        "Section chain is out of order. Got positions:\n"
        + "\n".join(f"  {pos:>6} — {marker!r}" for pos, marker in zip(positions, chain))
    )


def test_report_flip_composition_drift_warning_on_zero_pairs(tmp_path: Path) -> None:
    """Audit pass-2 Code-reviewer MAJOR-2: when the live matrix has
    zero finite cell-pairs (every column has either a NaN original
    or a NaN permuted) AND the signed flip_rate is non-zero, the
    drift detector MUST fire. Otherwise the legend reads
    "0 of 0 pairs flipped (44%)" — the exact silent contradiction
    the detector exists for.

    Tested at the function level (not via the CLI) because the
    all-NaN-cells fixture path would route through
    ``_read_matrix_or_empty``'s degraded-empty branch (which sets
    the matrix shape to (0, 0)) before reaching the drift logic.
    The drift-on-zero-pairs invariant lives inside
    ``_render_flip_composition_panel`` itself.
    """
    import numpy as np

    from topofuzz.report import _render_flip_composition_panel

    # Build a 2×2 NaN-everywhere matrix paired with a manifest that
    # carries var_a as the §6 original. ``denom_mask = perm_valid &
    # orig_valid`` will be all-False → n_pairs = 0.
    matrix = np.full((2, 2), np.nan, dtype=np.float64)
    variant_ids = ("var_a", "var_b")
    # Write a real report.json so _read_flip_rate_from_report_json
    # picks up a non-zero signed point.
    rd = tmp_path / "zero-pairs-run"
    rd.mkdir()
    (rd / "report.json").write_text(
        json.dumps({"headline": {"verdict_flip_rate": {"point": 4.0 / 9.0}}}),
        encoding="utf-8",
    )
    manifest = {
        "variant_canonical_hashes": ["var_a", "var_b"],
        "variant_orderings": [["A"], ["A"]],
    }

    ctx = _render_flip_composition_panel(matrix, variant_ids, rd, manifest)
    assert ctx["disabled"] is False
    assert ctx["n_pairs"] == 0
    assert ctx["n_flips"] == 0
    # The drift flag MUST be set when n_pairs == 0 and signed rate
    # is non-zero, otherwise the rendered legend silently reads
    # "0 of 0 pairs flipped (44%)" — the bug audit pass-2
    # Code-reviewer MAJOR-2 caught.
    assert ctx["drift_flag"] is True, (
        "drift_flag must be True when n_pairs == 0 and signed "
        "flip_rate exceeds the drift tolerance — otherwise the "
        "legend silently contradicts the donut center"
    )


def test_render_summary_panel_drift_flag_adds_warning(tmp_path: Path) -> None:
    """When ``drift_flag=True`` is passed, the §17 summary context
    carries ``drift_flag=True`` plus a ``drift_warning_html`` snippet
    that points the reader at ``topofuzz attest`` for refresh. The
    snippet is omitted on clean (drift_flag=False) runs so the
    happy-path template renders unchanged.

    Unit-level (not CLI) because the propagation contract is between
    ``_render_flip_composition_panel`` and ``_render_summary_panel``;
    rendering the full CLI just to assert on a returned-dict field
    would be heavyweight. The end-to-end propagation through the
    Jinja template is covered separately by
    :func:`test_report_summary_section_renders_drift_warning_on_flip_drift`.
    """
    from topofuzz.report import _render_summary_panel

    run_dir = tmp_path / "drift-summary-run"
    run_dir.mkdir()
    # Real report.json so the summary doesn't degrade to "unavailable" —
    # drift warning only attaches on the success path, never on the
    # placeholder path (the placeholder has no prose to be stale).
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.31, "ci_lower": 0.24, "ci_upper": 0.38,
            "degenerate_flag": False,
        },
        per_agent_influence=[
            {"node_id": "risk", "influence": 0.67,
             "ci_lower": 0.55, "ci_upper": 0.78,
             "degenerate": False, "degenerate_reason": None},
        ],
    )

    ctx_clean = _render_summary_panel(run_dir)
    assert ctx_clean["drift_flag"] is False
    assert "drift_warning_html" not in ctx_clean, (
        "drift_warning_html must not be present on clean runs — "
        "rendering an empty drift-warning would visually clutter the "
        "happy-path NLS card"
    )

    ctx_drift = _render_summary_panel(run_dir, drift_flag=True)
    assert ctx_drift["drift_flag"] is True
    warning_html = ctx_drift["drift_warning_html"]
    # The warning must name the operator action (attest refresh) and
    # the artefact that became stale (report.json). Without both, the
    # reader can't act.
    assert "topofuzz attest" in warning_html
    assert "report.json" in warning_html



def test_report_summary_section_renders_drift_warning_on_flip_drift(
    tmp_path: Path,
) -> None:
    """End-to-end: when the §14 donut detects signed-vs-live drift,
    the §17 NLS panel renders a complementary stale-prose warning
    inside the same ``.natural-language-summary`` div. Failure mode
    this guards against: a reader who glances only at the NLS card
    silently consumes stale prose (BACKLOG: drift propagation into
    §17 NLS).

    Drift is provoked by writing a ``report.json`` whose signed
    ``verdict_flip_rate.point`` does NOT match the live recount from
    ``run.sqlite``. ``_make_run_dir`` populates run.sqlite with a
    clean score matrix; overwriting only report.json keeps the
    contradiction localised, which is the same shape a stale-mirror
    edit would produce in the field.
    """
    run_dir = _make_run_dir(tmp_path)
    # A signed rate of 0.99 with a live recount near 0 (the
    # _make_run_dir matrix is well-separated, no flips) is far above
    # the 0.005 drift tolerance, so the donut fires the flag and the
    # NLS panel inherits it.
    _write_report_json(
        run_dir,
        verdict_flip_rate={
            "point": 0.99, "ci_lower": 0.95, "ci_upper": 1.00,
            "degenerate_flag": False,
        },
        per_agent_influence=[
            {"node_id": "risk", "influence": 0.67,
             "ci_lower": 0.55, "ci_upper": 0.78,
             "degenerate": False, "degenerate_reason": None},
        ],
    )

    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output

    body = (run_dir / "report.html").read_text(encoding="utf-8")
    section = _extract_summary_section(body)
    # The warning must live INSIDE the §17 sentinels, not just
    # somewhere in the body (the §14 donut also emits a drift warning
    # under a different class).
    assert "nls-drift-warning" in section, (
        "§17 panel must render the .nls-drift-warning element when "
        "the §14 donut's drift_flag is True — otherwise the prose "
        "above silently reflects stale data"
    )
    assert "topofuzz attest" in section
