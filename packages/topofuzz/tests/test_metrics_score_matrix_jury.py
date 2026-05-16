# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""F3 MAJOR (2026-05-16): per-vendor jury matrix plumbing in
:func:`topofuzz.metrics.read_score_matrix`.

Covers:
* Legacy v5 DB (no ``votes_json`` column) → ``per_vendor_matrix=None``.
* v6 single-judge run (column exists, every row NULL) →
  ``per_vendor_matrix=None``.
* v6 jury run (every row populated) → 3D matrix with deterministic
  sorted vendor axis; per-cell values agree with the parsed JSON.
* Mixed-protocol run (some rows populated, some NULL) → ``None``
  (refused; a half-populated matrix would silently lie).
* Malformed payloads (bad JSON, missing keys, duplicate vendor) →
  ``None`` (refused, WARNING logged).
* Vendor-set drift across cells → ``None`` (refused).
* NaN propagation: cells filtered by failure_reason / non-continuous
  judge_kind are NaN in the 2D matrix and NaN in the per-vendor slab.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from topofuzz.metrics import read_score_matrix


# ---------------------------------------------------------------------------
# v6 fixture helper — mirrors the runner's CREATE TABLE byte-for-byte
# including the ``votes_json`` column.
# ---------------------------------------------------------------------------


def _make_v6_db(
    db_path,
    *,
    runs: list[tuple[str, str]],
    judge_rows: list,
    # rows: (run_id, variant_id, eval_id, judge_kind, score_value,
    #        passed, failure_reason, votes_json)
) -> None:
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
                votes_json TEXT,
                PRIMARY KEY (run_id, variant_id, eval_id)
            );
            PRAGMA user_version = 6;
        """)
        for rid, rhash in runs:
            db.execute(
                "INSERT INTO runs "
                "(run_id, topology_canonical_hash, topology_name, "
                " tds_version, created_at, rubric_hash) "
                "VALUES (?, 'h', 'name', '0.5', 0.0, ?)",
                (rid, rhash),
            )
        for rid, vid, eid, kind, score, passed, fail, vj in judge_rows:
            db.execute(
                "INSERT INTO judge_results "
                "(run_id, variant_id, eval_id, judge_kind, score_value, "
                " passed, rationale, failure_reason, votes_json) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (rid, vid, eid, kind, score, passed, fail, vj),
            )
        db.commit()
    finally:
        db.close()


def _votes(*pairs: tuple[str, float]) -> str:
    """JSON-encode ``[{"vendor": v, "score": s, "rationale": ""}, ...]``."""
    return json.dumps(
        [{"vendor": v, "score": s, "rationale": ""} for v, s in pairs]
    )


# ---------------------------------------------------------------------------
# Happy path: jury run populates per_vendor_matrix.
# ---------------------------------------------------------------------------


def test_jury_run_populates_per_vendor_matrix(tmp_path):
    """A 2-variant × 2-eval × 3-vendor jury run builds a (3, 2, 2) slab."""
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             _votes(("google", 0.9), ("deepseek", 1.0), ("openai", 0.7))),
            ("r1", "vA", "e2", "continuous", 0.0, None, None,
             _votes(("google", 0.1), ("deepseek", 0.0), ("openai", 0.3))),
            ("r1", "vB", "e1", "continuous", 1.0, None, None,
             _votes(("google", 0.8), ("deepseek", 0.9), ("openai", 1.0))),
            ("r1", "vB", "e2", "continuous", 1.0, None, None,
             _votes(("google", 0.6), ("deepseek", 0.7), ("openai", 0.8))),
        ],
    )
    out = read_score_matrix(tmp_path)
    # 2D matrix preserved (majority scores).
    assert out.matrix.shape == (2, 2)
    # Per-vendor matrix populated; vendor axis sorted lex.
    assert out.per_vendor_matrix is not None
    assert out.per_vendor_matrix.shape == (3, 2, 2)
    assert out.vendor_ids == ("deepseek", "google", "openai")
    # Spot-check: deepseek's vote for (vA, e1) is 1.0.
    deepseek_idx = out.vendor_ids.index("deepseek")
    vA_idx = out.variant_ids.index("vA")
    e1_idx = out.eval_ids.index("e1")
    assert out.per_vendor_matrix[deepseek_idx, vA_idx, e1_idx] == 1.0
    # Spot-check: google's vote for (vB, e2) is 0.6.
    google_idx = out.vendor_ids.index("google")
    vB_idx = out.variant_ids.index("vB")
    e2_idx = out.eval_ids.index("e2")
    assert out.per_vendor_matrix[google_idx, vB_idx, e2_idx] == 0.6


# ---------------------------------------------------------------------------
# Legacy v5 DB: no votes_json column → per_vendor_matrix is None.
# ---------------------------------------------------------------------------


def _make_v5_db(db_path, *, runs, judge_rows):
    """v5 schema: judge_results WITHOUT votes_json column."""
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
            PRAGMA user_version = 5;
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


def test_legacy_v5_db_returns_none_per_vendor(tmp_path):
    db_path = tmp_path / "run.sqlite"
    _make_v5_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 0.7, None, None),
            ("r1", "vB", "e1", "continuous", 0.3, None, None),
        ],
    )
    out = read_score_matrix(tmp_path)
    assert out.matrix.shape == (2, 1)
    assert out.per_vendor_matrix is None
    assert out.vendor_ids == ()


# ---------------------------------------------------------------------------
# v6 single-judge run: votes_json column exists but every row is NULL.
# ---------------------------------------------------------------------------


def test_v6_single_judge_run_returns_none_per_vendor(tmp_path):
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 0.7, None, None, None),
            ("r1", "vB", "e1", "continuous", 0.3, None, None, None),
        ],
    )
    out = read_score_matrix(tmp_path)
    assert out.matrix.shape == (2, 1)
    assert out.per_vendor_matrix is None
    assert out.vendor_ids == ()


# ---------------------------------------------------------------------------
# Refusal paths — silent degrade with logged WARNING.
# ---------------------------------------------------------------------------


def test_mixed_protocol_run_returns_none_with_warning(tmp_path, caplog):
    """Some cells have votes_json, others don't → refuse the per-vendor
    matrix (the 2D ``matrix`` still populates)."""
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             _votes(("google", 0.9), ("deepseek", 1.0))),
            ("r1", "vB", "e1", "continuous", 0.5, None, None, None),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.matrix.shape == (2, 1)
    assert out.per_vendor_matrix is None
    assert any("mixed-protocol" in r.message for r in caplog.records)


def test_malformed_votes_json_returns_none(tmp_path, caplog):
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None, "{not json"),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("parse failed" in r.message for r in caplog.records)


def test_partial_vendor_roster_is_tolerated_with_nan(tmp_path, caplog):
    """A jury cell with a partial vendor roster (per-vendor judge
    failure inside the jury) populates with NaN for the missing entry
    rather than refusing the whole per-vendor matrix.

    The 2D ``matrix``'s NaN-on-missing-cell semantics already give
    downstream metrics the right "no data" sentinel; mirroring that
    behavior on the per-vendor slab lets jury VOO use nanvar/nanmean
    on the slab without losing the rest of the per-vendor data.

    Physically validated against the lending tricky run on 2026-05-16:
    one cell of 38 had a single-vendor roster while the other 37 had
    all three; refusing would have thrown away 37 real measurements.
    """
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             _votes(("deepseek", 0.9), ("google", 1.0), ("openai", 0.7))),
            ("r1", "vA", "e2", "continuous", 1.0, None, None,
             _votes(("deepseek", 0.9), ("google", 1.0), ("openai", 0.7))),
            # Partial roster: only one vendor returned a score for vB×e1.
            ("r1", "vB", "e1", "continuous", 1.0, None, None,
             _votes(("google", 0.5))),
            ("r1", "vB", "e2", "continuous", 0.0, None, None,
             _votes(("deepseek", 0.0), ("google", 0.1), ("openai", 0.2))),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    # Per-vendor matrix populated — not refused.
    assert out.per_vendor_matrix is not None
    assert out.vendor_ids == ("deepseek", "google", "openai")
    # Roster union across the run (all three vendors observed somewhere).
    assert out.per_vendor_matrix.shape == (3, 2, 2)
    # Missing entries are NaN.
    vB_idx = out.variant_ids.index("vB")
    e1_idx = out.eval_ids.index("e1")
    deepseek_idx = out.vendor_ids.index("deepseek")
    openai_idx = out.vendor_ids.index("openai")
    google_idx = out.vendor_ids.index("google")
    assert np.isnan(out.per_vendor_matrix[deepseek_idx, vB_idx, e1_idx])
    assert np.isnan(out.per_vendor_matrix[openai_idx, vB_idx, e1_idx])
    # The one vendor that voted persists exactly.
    assert out.per_vendor_matrix[google_idx, vB_idx, e1_idx] == 0.5
    # One-line summary WARNING logged — operator sees the signal.
    assert any(
        "partial vendor roster" in r.message for r in caplog.records
    )


def test_duplicate_vendor_in_cell_returns_none(tmp_path, caplog):
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             _votes(("google", 0.9), ("google", 0.5))),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("duplicate vendor" in r.message for r in caplog.records)


def test_votes_json_non_dict_entry_returns_none(tmp_path, caplog):
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             json.dumps([["google", 0.9]])),  # list-of-lists, not dicts
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("not a dict" in r.message for r in caplog.records)


def test_votes_json_missing_score_returns_none(tmp_path, caplog):
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             json.dumps([{"vendor": "google"}])),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("missing/invalid" in r.message for r in caplog.records)


@pytest.mark.parametrize("bad_score", [
    float("inf"), float("-inf"), float("nan"), -0.5, 1.5, 999.0
])
def test_votes_json_score_out_of_range_returns_none(tmp_path, caplog, bad_score):
    """Audit M3: ``inf`` / ``NaN`` / out-of-[0,1] scores would poison the
    jury VOO (entire vendor's slab inflates to inf, NaN propagates). The
    JuryJudge contract says score ∈ [0, 1]; refuse violations with a
    WARNING."""
    db_path = tmp_path / "run.sqlite"
    # ``json.dumps`` rejects inf/nan; use a hand-built JSON string.
    if bad_score != bad_score:  # NaN
        score_json = "NaN"
    elif bad_score == float("inf"):
        score_json = "Infinity"
    elif bad_score == float("-inf"):
        score_json = "-Infinity"
    else:
        score_json = repr(bad_score)
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             '[{"vendor": "google", "score": ' + score_json + ', "rationale": ""}]'),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any(
        "out of [0,1]" in r.message or "missing/invalid" in r.message
        for r in caplog.records
    )


def test_votes_json_score_bool_refused(tmp_path, caplog):
    """``bool`` is a subclass of ``int`` in Python; ``True``/``False``
    would silently coerce to 1.0/0.0 and skew the headline. Refuse."""
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             '[{"vendor": "google", "score": true, "rationale": ""}]'),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("missing/invalid" in r.message for r in caplog.records)


def test_votes_json_empty_vendor_name_refused(tmp_path, caplog):
    """An empty vendor string would collide silently with another empty
    vendor in the lex-sort; refuse upfront."""
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             '[{"vendor": "", "score": 0.5, "rationale": ""}]'),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("missing/invalid" in r.message for r in caplog.records)


def test_votes_json_huge_vendor_name_refused(tmp_path, caplog):
    """Defense-in-depth: a hand-edited DB carrying a multi-MB vendor
    string is bounded at 256 chars."""
    db_path = tmp_path / "run.sqlite"
    huge_v = "x" * 1000
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             json.dumps([{"vendor": huge_v, "score": 0.5}])),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("missing/invalid" in r.message for r in caplog.records)


def test_empty_votes_list_returns_none(tmp_path, caplog):
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None, "[]"),
        ],
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="topofuzz.metrics.score_matrix"):
        out = read_score_matrix(tmp_path)
    assert out.per_vendor_matrix is None
    assert any("shape invalid" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Failure / non-continuous cells filter as before.
# ---------------------------------------------------------------------------


def test_failure_cells_filtered_from_per_vendor_matrix(tmp_path):
    """A non-continuous / failure cell is filtered from both ``matrix``
    and ``per_vendor_matrix`` — NaN in the slab where the cell would
    have been."""
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             _votes(("google", 0.9), ("deepseek", 1.0))),
            # Failure cell for vA × e2: filtered from continuous query
            # entirely; e2 only appears via vB below.
            ("r1", "vA", "e2", "failure", None, None, "bad", None),
            ("r1", "vB", "e1", "continuous", 1.0, None, None,
             _votes(("google", 0.8), ("deepseek", 0.7))),
            ("r1", "vB", "e2", "continuous", 0.0, None, None,
             _votes(("google", 0.1), ("deepseek", 0.0))),
        ],
    )
    out = read_score_matrix(tmp_path)
    # vA × e2 is filtered: NaN in the 2D matrix.
    e2_idx = out.eval_ids.index("e2")
    vA_idx = out.variant_ids.index("vA")
    assert np.isnan(out.matrix[vA_idx, e2_idx])
    # Per-vendor matrix has matching NaN for the filtered cell.
    assert out.per_vendor_matrix is not None
    assert np.isnan(out.per_vendor_matrix[:, vA_idx, e2_idx]).all()
    # The kept cells populate as expected.
    google_idx = out.vendor_ids.index("google")
    e1_idx = out.eval_ids.index("e1")
    assert out.per_vendor_matrix[google_idx, vA_idx, e1_idx] == 0.9


def test_per_vendor_matrix_axes_align_with_2d_matrix(tmp_path):
    """``per_vendor_matrix.mean(axis=0)`` is the continuous per-vendor
    AVERAGE — emphatically NOT the persisted majority. Tests that
    assumption holds so callers don't accidentally double-count."""
    db_path = tmp_path / "run.sqlite"
    _make_v6_db(
        db_path,
        runs=[("r1", "rh")],
        judge_rows=[
            ("r1", "vA", "e1", "continuous", 1.0, None, None,
             _votes(("a", 0.6), ("b", 0.8), ("c", 1.0))),  # avg 0.8, majority 1.0
        ],
    )
    out = read_score_matrix(tmp_path)
    # Persisted majority in the 2D matrix.
    assert out.matrix[0, 0] == 1.0
    # Per-vendor average (continuous) — distinct from the majority.
    assert out.per_vendor_matrix is not None
    avg = float(out.per_vendor_matrix[:, 0, 0].mean())
    assert abs(avg - 0.8) < 1e-9
    assert avg != out.matrix[0, 0]
