"""Step 9: read_score_matrix — DB round-trip + ordering + refusals."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from topofuzz.metrics import read_score_matrix


# ---------------------------------------------------------------------------
# Helpers: hand-craft a minimal v4 run.sqlite without going through the full
# orchestrator. Step 9's loader doesn't care HOW the DB was populated, only
# that the schema matches; an inline fixture is much faster than spinning
# up the orchestrator for boundary tests.
# ---------------------------------------------------------------------------


def _make_v4_db(
    db_path,
    *,
    runs: list[tuple[str, str]],  # (run_id, rubric_hash)
    judge_rows: list[tuple[str, str, str, str, float | None, int | None, str | None]],
    # judge_rows: (run_id, variant_id, eval_id, judge_kind, score_value, passed, failure_reason)
) -> None:
    """Create a v4-schema run.sqlite with the supplied rows.

    The columns mirror :data:`topofuzz.runner._SCHEMA` byte-for-byte so a
    schema drift breaks this fixture loudly (good — drifted fixtures
    miss real regressions).
    """
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
        for rid, rhash in runs:
            db.execute(
                "INSERT INTO runs "
                "(run_id, topology_canonical_hash, topology_name, "
                " tds_version, created_at, rubric_hash) "
                "VALUES (?, 'h', 'name', '0.5', 0.0, ?)",
                (rid, rhash),
            )
        for rid, vid, eid, kind, score, passed, fail in judge_rows:
            db.execute(
                "INSERT INTO judge_results "
                "(run_id, variant_id, eval_id, judge_kind, score_value, "
                " passed, rationale, failure_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                (rid, vid, eid, kind, score, passed, fail),
            )
        db.commit()
    finally:
        db.close()


def test_read_score_matrix_happy_path(tmp_path):
    """Two variants × three evals → 2×3 matrix; variants sorted by
    canonical_hash lex."""
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "rh-abc")],
        judge_rows=[
            ("r1", "vB", "e1", "continuous", 0.7, None, None),
            ("r1", "vB", "e2", "continuous", 0.3, None, None),
            ("r1", "vB", "e3", "continuous", 0.5, None, None),
            ("r1", "vA", "e1", "continuous", 0.9, None, None),
            ("r1", "vA", "e2", "continuous", 0.1, None, None),
            ("r1", "vA", "e3", "continuous", 0.8, None, None),
        ],
    )
    out = read_score_matrix(tmp_path)
    assert out.rubric_hash == "rh-abc"
    # variant_ids sorted lex: vA before vB.
    assert out.variant_ids == ("vA", "vB")
    assert out.matrix.shape == (2, 3)
    # Row order matches sorted variant_ids.
    np.testing.assert_allclose(out.matrix[0], [0.9, 0.1, 0.8])
    np.testing.assert_allclose(out.matrix[1], [0.7, 0.3, 0.5])


def test_read_score_matrix_filters_failure_and_binary(tmp_path):
    """judge_kind != 'continuous' and failure rows are excluded;
    missing cells appear as NaN."""
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 0.7, None, None),
            ("r1", "vA", "e2", "failure", None, None, "bad parse"),
            ("r1", "vB", "e1", "binary", None, 1, None),
            ("r1", "vB", "e2", "continuous", 0.4, None, None),
        ],
    )
    out = read_score_matrix(tmp_path)
    # vA, vB discovered via the kept continuous rows only.
    assert out.variant_ids == ("vA", "vB")
    assert out.eval_ids == ("e1", "e2")
    # (vA, e2) and (vB, e1) excluded → NaN.
    assert np.isnan(out.matrix[0, 1])
    assert np.isnan(out.matrix[1, 0])
    assert out.matrix[0, 0] == 0.7
    assert out.matrix[1, 1] == 0.4


def test_read_score_matrix_refuses_empty_rubric_hash(tmp_path):
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "")],  # empty → v3-migrated / no-judge run
        judge_rows=[("r1", "vA", "e1", "continuous", 0.5, None, None)],
    )
    with pytest.raises(ValueError, match="empty rubric_hash"):
        read_score_matrix(tmp_path)


def test_read_score_matrix_allow_empty_rubric_hash(tmp_path):
    """Forensic path: opt-in to load a rubric-unknown run."""
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "")],
        judge_rows=[("r1", "vA", "e1", "continuous", 0.5, None, None)],
    )
    out = read_score_matrix(tmp_path, allow_empty_rubric_hash=True)
    assert out.rubric_hash == ""
    assert out.matrix.shape == (1, 1)


def test_read_score_matrix_requires_run_id_for_multi_run_db(tmp_path):
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "rh"), ("r2", "rh")],
        judge_rows=[("r1", "vA", "e1", "continuous", 0.5, None, None)],
    )
    with pytest.raises(ValueError, match="pass run_id="):
        read_score_matrix(tmp_path)


def test_read_score_matrix_run_id_disambiguation(tmp_path):
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "rh-1"), ("r2", "rh-2")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 0.5, None, None),
            ("r2", "vA", "e1", "continuous", 0.9, None, None),
        ],
    )
    out_r1 = read_score_matrix(tmp_path, run_id="r1")
    out_r2 = read_score_matrix(tmp_path, run_id="r2")
    assert out_r1.rubric_hash == "rh-1"
    assert out_r2.rubric_hash == "rh-2"
    assert out_r1.matrix[0, 0] == 0.5
    assert out_r2.matrix[0, 0] == 0.9


def test_read_score_matrix_unknown_run_id(tmp_path):
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[("r1", "vA", "e1", "continuous", 0.5, None, None)],
    )
    with pytest.raises(ValueError, match="not found"):
        read_score_matrix(tmp_path, run_id="r-nope")


def test_read_score_matrix_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.sqlite not found"):
        read_score_matrix(tmp_path)


def test_read_score_matrix_empty_run(tmp_path):
    """A run with no continuous rows → empty matrix, no crash."""
    db_path = tmp_path / "run.sqlite"
    _make_v4_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            # Only failure / binary rows → no continuous data.
            ("r1", "vA", "e1", "failure", None, None, "x"),
        ],
    )
    out = read_score_matrix(tmp_path)
    assert out.matrix.shape == (0, 0)
    assert out.variant_ids == ()
    assert out.eval_ids == ()
    assert out.rubric_hash == "rh"
