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
