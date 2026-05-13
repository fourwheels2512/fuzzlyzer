"""Step 16: §10a worst-case viewer data layer coverage.

Covers :mod:`topofuzz.metrics.worstcase`:

* Structural disable paths (empty / too-small matrix).
* Semantic disable path ("no instability detected" — all verdicts
  agree across every eval column).
* Top-K eval selection by binary-flip count with deterministic
  eval_id-lex tiebreak.
* Best-pair selection per eval by ``|score_a - score_b|`` with
  deterministic variant_id-lex tiebreak; pair must have differing
  verdicts.
* NaN-cell handling (excluded from both flip counts and pair search).
* DB read of ``cells.node_outputs_json`` + ``judge_results.rationale``;
  graceful degradation when the cell row is missing or the JSON is
  malformed.
* Truncation of long ``response_text`` and ``rationale``.
* Manifest ordering lookup wiring (pre-step-15 manifests render an
  empty ordering tuple).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from topofuzz.metrics.worstcase import (
    DEFAULT_K,
    NodeOutputView,
    VariantSideView,
    WorstCaseEntry,
    WorstCasePanel,
    _truncate,
    compute_worstcase_panel,
)


# ---------------------------------------------------------------------------
# Run-dir fixture builders (no orchestrator dependency)
# ---------------------------------------------------------------------------


_CELLS_DDL = """
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
"""


def _empty_run_dir(tmp_path: Path, *, run_id: str = "run-test") -> Path:
    """Create a run dir with the cells + judge_results schema only."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.executescript(_CELLS_DDL)
        db.commit()
    finally:
        db.close()
    return run_dir


def _populate_cells(
    run_dir: Path,
    run_id: str,
    cells: dict[tuple[str, str], list[dict]],
    rationales: dict[tuple[str, str], str | None] | None = None,
) -> None:
    """Bulk-insert ``cells`` and ``judge_results`` rows."""
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        for (vid, eid), node_outs in cells.items():
            db.execute(
                "INSERT INTO cells "
                "(run_id, variant_id, eval_id, ir_canonical_hash, "
                " node_outputs_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, vid, eid, vid, json.dumps(node_outs)),
            )
        if rationales:
            for (vid, eid), rat in rationales.items():
                db.execute(
                    "INSERT OR REPLACE INTO judge_results "
                    "(run_id, variant_id, eval_id, judge_kind, "
                    " score_value, passed, rationale, failure_reason) "
                    "VALUES (?, ?, ?, 'continuous', NULL, NULL, ?, NULL)",
                    (run_id, vid, eid, rat),
                )
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Structural disables
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_empty_matrix(tmp_path: Path) -> None:
    """Empty matrix → disabled with 'matrix empty' reason."""
    run_dir = _empty_run_dir(tmp_path)
    panel = compute_worstcase_panel(
        np.empty((0, 0), dtype=np.float64),
        (),
        (),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is True
    assert panel.disabled_reason == "matrix empty"
    assert panel.cases == ()
    assert panel.k == DEFAULT_K


def test_compute_worstcase_panel_single_variant_is_disabled(
    tmp_path: Path,
) -> None:
    """n_variants < 2 → disabled (single agent has no permuted partner)."""
    run_dir = _empty_run_dir(tmp_path)
    matrix = np.array([[0.1, 0.9, 0.5]], dtype=np.float64)
    panel = compute_worstcase_panel(
        matrix,
        ("vA",),
        ("e0", "e1", "e2"),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is True
    assert panel.disabled_reason == "matrix empty"


def test_compute_worstcase_panel_zero_evals_is_disabled(
    tmp_path: Path,
) -> None:
    """n_evals == 0 → disabled."""
    run_dir = _empty_run_dir(tmp_path)
    matrix = np.empty((3, 0), dtype=np.float64)
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC"),
        (),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is True


# ---------------------------------------------------------------------------
# Semantic disable — all verdicts agree
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_all_agree_is_disabled(
    tmp_path: Path,
) -> None:
    """All cells > threshold → no flips → disabled 'no instability'."""
    run_dir = _empty_run_dir(tmp_path)
    # 3 variants × 4 evals, every cell 0.8 (PASS). Zero flip pairs.
    matrix = np.full((3, 4), 0.8, dtype=np.float64)
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC"),
        ("e0", "e1", "e2", "e3"),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is True
    assert panel.disabled_reason == "no instability detected"
    assert panel.cases == ()


def test_compute_worstcase_panel_all_fail_is_also_disabled(
    tmp_path: Path,
) -> None:
    """All cells < threshold → no flips → disabled."""
    run_dir = _empty_run_dir(tmp_path)
    matrix = np.full((3, 4), 0.1, dtype=np.float64)
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC"),
        ("e0", "e1", "e2", "e3"),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is True
    assert panel.disabled_reason == "no instability detected"


def test_compute_worstcase_panel_variance_gate_disables_below_threshold(
    tmp_path: Path,
) -> None:
    """DoD #13 gate (BLUEPRINT line 533): max per-eval variance ≤ 0.1
    routes to the disabled banner even when binary verdicts flip.

    Scores tightly clustered around the 0.5 cutoff produce many flip
    pairs (every (0.51, 0.49) pair flips) but variance ≈ 0.0001 — the
    spec explicitly excludes that case from rendering the panel.
    """
    run_dir = _empty_run_dir(tmp_path)
    matrix = np.array(
        [
            [0.51, 0.49],
            [0.49, 0.51],
            [0.51, 0.49],
        ],
        dtype=np.float64,
    )
    # Verify the design before the assert: lots of flips, no variance.
    assert float(np.var(matrix, axis=0).max()) < 0.1
    preds = matrix >= 0.5
    assert (preds[1:, :] != preds[:1, :]).any()
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC"),
        ("e0", "e1"),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is True
    assert panel.disabled_reason == "no instability detected"


def test_compute_worstcase_panel_variance_gate_is_per_column(
    tmp_path: Path,
) -> None:
    """Audit pass-2 RT-BLOCKER-1: the variance gate must apply
    **per-column** at the candidate stage, not just as a whole-matrix
    max. A high-variance column can't lift a low-variance noise-flip
    column into the rendered cards.

    Design:
      e0: var=0.16 (above 0.1 gate), 1 flip pair  ← expected card
      e1: var≈0.0001 (well below gate), 1 flip pair ← MUST be excluded
    Without the per-column filter, e1 would be ranked alongside e0
    purely on flip count.
    """
    run_dir = _empty_run_dir(tmp_path)
    matrix = np.array(
        [
            [0.9, 0.51],
            [0.1, 0.49],
        ],
        dtype=np.float64,
    )
    var_per_col = np.var(matrix, axis=0)
    assert float(var_per_col[0]) > 0.1
    assert float(var_per_col[1]) < 0.1
    _populate_cells(
        run_dir, "run-test",
        {(v, e): [] for v in ("vA", "vB") for e in ("e0", "e1")},
    )
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB"),
        ("e0", "e1"),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is False
    assert len(panel.cases) == 1
    assert panel.cases[0].eval_id == "e0"


def test_compute_worstcase_panel_variance_gate_param_override(
    tmp_path: Path,
) -> None:
    """``variance_gate=0.0`` disables the gate so the flip-count
    ranker runs regardless of variance."""
    run_dir = _empty_run_dir(tmp_path)
    matrix = np.array(
        [
            [0.51, 0.49],
            [0.49, 0.51],
        ],
        dtype=np.float64,
    )
    _populate_cells(
        run_dir, "run-test",
        {(v, e): [] for v in ("vA", "vB") for e in ("e0", "e1")},
    )
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB"),
        ("e0", "e1"),
        run_dir=run_dir,
        run_id="run-test",
        variance_gate=0.0,
    )
    assert panel.disabled is False
    assert len(panel.cases) == 2


# ---------------------------------------------------------------------------
# Top-K eval selection
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_picks_top_k_by_flip_count(
    tmp_path: Path,
) -> None:
    """Top-K selection ranks evals by binary-flip pair count.

    Uses extreme 0.9/0.1 scores so the per-column variance (≈ 0.16)
    sits above the DoD #13 variance gate of 0.1 and we exercise the
    flip-count ranker rather than the gate-disabled path.
    """
    run_dir = _empty_run_dir(tmp_path)
    # 4 variants × 5 evals. Design flip counts per eval (n_true, n_false):
    #   e0: (2, 2) -> 4 flip pairs    ← top
    #   e1: (1, 3) -> 3
    #   e2: (3, 1) -> 3               (tiebreak: eval_id lex puts e1 first)
    #   e3: (4, 0) -> 0               (no flips, excluded)
    #   e4: (1, 3) -> 3
    # Expected top-3: e0, e1, e2 (eval-id lex tiebreaks 3-flip ties).
    matrix = np.array(
        [
            # vA   vB   vC   vD on axis-0; columns are evals.
            [0.9, 0.9, 0.1, 0.1, 0.1],  # vA
            [0.9, 0.1, 0.9, 0.1, 0.9],  # vB
            [0.1, 0.1, 0.9, 0.1, 0.1],  # vC
            [0.1, 0.1, 0.9, 0.1, 0.1],  # vD
        ],
        dtype=np.float64,
    )
    # Sanity check the design:
    preds = matrix >= 0.5
    n_true = preds.sum(axis=0)
    expected_pair_counts = n_true * (4 - n_true)
    assert list(expected_pair_counts) == [4, 3, 3, 0, 3]
    # And that per-column variance clears the DoD #13 gate.
    assert float(np.var(matrix, axis=0).max()) > 0.1

    # cells: empty trace per cell is fine; we only test selection here.
    cells = {
        (v, e): [] for v in ("vA", "vB", "vC", "vD") for e in
        ("e0", "e1", "e2", "e3", "e4")
    }
    _populate_cells(run_dir, "run-test", cells)

    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC", "vD"),
        ("e0", "e1", "e2", "e3", "e4"),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert panel.disabled is False
    assert len(panel.cases) == 3
    # Top eval by flip count is e0.
    assert panel.cases[0].eval_id == "e0"
    assert panel.cases[0].flip_count == 4
    # Ties (e1, e2, e4 all at 3 flips) break by eval_id lex.
    assert [c.eval_id for c in panel.cases[1:]] == ["e1", "e2"]
    assert all(c.flip_count == 3 for c in panel.cases[1:])


def test_compute_worstcase_panel_k_override(tmp_path: Path) -> None:
    """`k=1` returns at most one case; `k > n_candidates` is clamped."""
    run_dir = _empty_run_dir(tmp_path)
    matrix = np.array(
        [
            [0.9, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    _populate_cells(
        run_dir, "run-test",
        {(v, e): [] for v in ("vA", "vB") for e in ("e0", "e1")},
    )
    panel_one = compute_worstcase_panel(
        matrix, ("vA", "vB"), ("e0", "e1"),
        run_dir=run_dir, run_id="run-test", k=1,
    )
    assert len(panel_one.cases) == 1
    assert panel_one.k == 1

    panel_many = compute_worstcase_panel(
        matrix, ("vA", "vB"), ("e0", "e1"),
        run_dir=run_dir, run_id="run-test", k=10,
    )
    # Only 2 evals have flips; k=10 clamps to 2.
    assert len(panel_many.cases) == 2


# ---------------------------------------------------------------------------
# Best-pair selection per eval
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_pair_picks_max_score_gap(
    tmp_path: Path,
) -> None:
    """Among differing-verdict pairs, the largest |score_a - score_b|
    pair is selected."""
    run_dir = _empty_run_dir(tmp_path)
    # 4 variants × 1 eval. Verdicts: vA=PASS (0.95), vB=FAIL (0.05),
    # vC=PASS (0.60), vD=FAIL (0.40).
    # Differing-verdict pairs and gaps:
    #   (vA, vB): 0.90  ← max
    #   (vA, vD): 0.55
    #   (vB, vC): 0.55
    #   (vC, vD): 0.20
    matrix = np.array(
        [[0.95], [0.05], [0.60], [0.40]], dtype=np.float64
    )
    _populate_cells(
        run_dir, "run-test",
        {(v, "e0"): [] for v in ("vA", "vB", "vC", "vD")},
    )
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC", "vD"),
        ("e0",),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert len(panel.cases) == 1
    case = panel.cases[0]
    assert case.variant_a.variant_id == "vA"
    assert case.variant_b.variant_id == "vB"
    assert case.variant_a.verdict is True
    assert case.variant_b.verdict is False


def test_compute_worstcase_panel_pair_tiebreak_by_variant_id(
    tmp_path: Path,
) -> None:
    """Equal max gaps tiebreak by ascending (variant_id_a, variant_id_b)."""
    run_dir = _empty_run_dir(tmp_path)
    # 4 variants × 1 eval. Build two differing-verdict pairs with the
    # SAME gap, so the tiebreak rule fires.
    # vA=0.9 (PASS), vB=0.9 (PASS), vC=0.1 (FAIL), vD=0.1 (FAIL).
    # Pairs (vA, vC) gap 0.8; (vA, vD) 0.8; (vB, vC) 0.8; (vB, vD) 0.8.
    # Tiebreak ascending → (vA, vC).
    matrix = np.array(
        [[0.9], [0.9], [0.1], [0.1]], dtype=np.float64
    )
    _populate_cells(
        run_dir, "run-test",
        {(v, "e0"): [] for v in ("vA", "vB", "vC", "vD")},
    )
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC", "vD"),
        ("e0",),
        run_dir=run_dir,
        run_id="run-test",
    )
    assert len(panel.cases) == 1
    case = panel.cases[0]
    assert case.variant_a.variant_id == "vA"
    assert case.variant_b.variant_id == "vC"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_nan_cells_excluded(
    tmp_path: Path,
) -> None:
    """NaN cells drop out of valid pairs; flip counts reflect valid only."""
    run_dir = _empty_run_dir(tmp_path)
    # vA has all NaN; vB and vC flip on e0.
    matrix = np.array(
        [
            [np.nan, np.nan],  # vA — excluded
            [0.9, 0.9],         # vB
            [0.1, 0.9],         # vC
        ],
        dtype=np.float64,
    )
    _populate_cells(
        run_dir, "run-test",
        {(v, e): [] for v in ("vB", "vC") for e in ("e0", "e1")},
    )
    panel = compute_worstcase_panel(
        matrix,
        ("vA", "vB", "vC"),
        ("e0", "e1"),
        run_dir=run_dir,
        run_id="run-test",
    )
    # Only one flip pair survives, on e0; e1 has all-PASS among valid.
    assert len(panel.cases) == 1
    case = panel.cases[0]
    assert case.eval_id == "e0"
    assert case.flip_count == 1  # 1 PASS × 1 FAIL among valid
    assert case.n_pairs == 1     # C(2, 2) = 1


# ---------------------------------------------------------------------------
# DB read — node outputs + rationale
# ---------------------------------------------------------------------------


def _flip_matrix_2x1() -> np.ndarray:
    """A 2-variant single-eval flipping matrix."""
    return np.array([[0.9], [0.1]], dtype=np.float64)


def test_compute_worstcase_panel_loads_node_outputs_from_db(
    tmp_path: Path,
) -> None:
    """The viewer side carries node_outputs + rationale from sqlite."""
    run_dir = _empty_run_dir(tmp_path)
    cells = {
        ("vA", "e0"): [
            {"node_id": "intake", "role": "risk", "response_text": "ok"},
            {"node_id": "decide", "role": "decision",
             "response_text": "approve"},
        ],
        ("vB", "e0"): [
            {"node_id": "decide", "role": "decision",
             "response_text": "deny"},
            {"node_id": "intake", "role": "risk",
             "response_text": "fail"},
        ],
    }
    rationales = {
        ("vA", "e0"): "Reasonable approve given policy exception.",
        ("vB", "e0"): "Reasonable deny under threshold 4.2.",
    }
    _populate_cells(run_dir, "run-test", cells, rationales)
    panel = compute_worstcase_panel(
        _flip_matrix_2x1(),
        ("vA", "vB"),
        ("e0",),
        run_dir=run_dir,
        run_id="run-test",
    )
    case = panel.cases[0]
    a = case.variant_a
    b = case.variant_b
    assert [n.node_id for n in a.node_outputs] == ["intake", "decide"]
    assert a.node_outputs[1].response_text == "approve"
    assert a.rationale == "Reasonable approve given policy exception."
    assert [n.node_id for n in b.node_outputs] == ["decide", "intake"]
    assert b.rationale == "Reasonable deny under threshold 4.2."


def test_compute_worstcase_panel_missing_cell_row_renders_empty_trace(
    tmp_path: Path,
) -> None:
    """Cell row absent → empty node_outputs, but case still surfaces."""
    run_dir = _empty_run_dir(tmp_path)
    # Insert only one of the two cells.
    _populate_cells(
        run_dir, "run-test",
        {("vA", "e0"): [{"node_id": "n", "role": "r",
                          "response_text": "t"}]},
    )
    panel = compute_worstcase_panel(
        _flip_matrix_2x1(),
        ("vA", "vB"),
        ("e0",),
        run_dir=run_dir,
        run_id="run-test",
    )
    case = panel.cases[0]
    assert case.variant_a.node_outputs[0].node_id == "n"
    assert case.variant_b.node_outputs == ()
    assert case.variant_b.rationale is None


def test_compute_worstcase_panel_malformed_node_outputs_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed JSON in cells.node_outputs_json → WARNING + empty trace."""
    run_dir = _empty_run_dir(tmp_path)
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        db.execute(
            "INSERT INTO cells "
            "(run_id, variant_id, eval_id, ir_canonical_hash, "
            " node_outputs_json) VALUES "
            "('run-test', 'vA', 'e0', 'vA', '{not json'),"
            " ('run-test', 'vB', 'e0', 'vB', '[]')"
        )
        db.commit()
    finally:
        db.close()

    with caplog.at_level("WARNING"):
        panel = compute_worstcase_panel(
            _flip_matrix_2x1(),
            ("vA", "vB"),
            ("e0",),
            run_dir=run_dir,
            run_id="run-test",
        )
    case = panel.cases[0]
    assert case.variant_a.node_outputs == ()
    # The warning surfaces the malformed payload.
    assert any(
        "node_outputs_json invalid" in r.getMessage() for r in caplog.records
    )


def test_compute_worstcase_panel_truncates_long_text(
    tmp_path: Path,
) -> None:
    """`node_text_cap` and `rationale_cap` truncate with an ellipsis."""
    run_dir = _empty_run_dir(tmp_path)
    long_text = "x" * 1000
    _populate_cells(
        run_dir, "run-test",
        {
            ("vA", "e0"): [
                {"node_id": "n", "role": "r",
                 "response_text": long_text}
            ],
            ("vB", "e0"): [],
        },
        {("vA", "e0"): long_text},
    )
    panel = compute_worstcase_panel(
        _flip_matrix_2x1(),
        ("vA", "vB"),
        ("e0",),
        run_dir=run_dir,
        run_id="run-test",
        node_text_cap=50,
        rationale_cap=30,
    )
    a = panel.cases[0].variant_a
    assert len(a.node_outputs[0].response_text) == 51  # 50 chars + …
    assert a.node_outputs[0].response_text.endswith("…")
    assert a.rationale is not None
    assert len(a.rationale) == 31  # 30 + …
    assert a.rationale.endswith("…")


# ---------------------------------------------------------------------------
# Manifest ordering wiring
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_uses_manifest_ordering(
    tmp_path: Path,
) -> None:
    """`manifest.variant_orderings` propagates into VariantSideView."""
    run_dir = _empty_run_dir(tmp_path)
    _populate_cells(
        run_dir, "run-test",
        {(v, "e0"): [] for v in ("vA", "vB")},
    )
    manifest = {
        "variant_canonical_hashes": ["vA", "vB"],
        "variant_orderings": [
            ["intake", "risk", "policy", "decide"],
            ["policy", "intake", "risk", "decide"],
        ],
    }
    panel = compute_worstcase_panel(
        _flip_matrix_2x1(),
        ("vA", "vB"),
        ("e0",),
        run_dir=run_dir,
        run_id="run-test",
        manifest=manifest,
    )
    case = panel.cases[0]
    assert case.variant_a.ordering == (
        "intake", "risk", "policy", "decide",
    )
    assert case.variant_b.ordering == (
        "policy", "intake", "risk", "decide",
    )


def test_compute_worstcase_panel_missing_manifest_ordering_is_empty_tuple(
    tmp_path: Path,
) -> None:
    """Pre-step-15 manifest (no variant_orderings) → empty tuple, no crash."""
    run_dir = _empty_run_dir(tmp_path)
    _populate_cells(
        run_dir, "run-test",
        {(v, "e0"): [] for v in ("vA", "vB")},
    )
    panel = compute_worstcase_panel(
        _flip_matrix_2x1(),
        ("vA", "vB"),
        ("e0",),
        run_dir=run_dir,
        run_id="run-test",
        manifest={"variant_canonical_hashes": ["vA", "vB"]},
    )
    case = panel.cases[0]
    assert case.variant_a.ordering == ()
    assert case.variant_b.ordering == ()


def test_compute_worstcase_panel_mismatched_manifest_lengths_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Length mismatch in manifest → WARNING + empty ordering tuples."""
    run_dir = _empty_run_dir(tmp_path)
    _populate_cells(
        run_dir, "run-test",
        {(v, "e0"): [] for v in ("vA", "vB")},
    )
    manifest = {
        "variant_canonical_hashes": ["vA", "vB"],
        "variant_orderings": [["only_one_entry"]],
    }
    with caplog.at_level("WARNING"):
        panel = compute_worstcase_panel(
            _flip_matrix_2x1(),
            ("vA", "vB"),
            ("e0",),
            run_dir=run_dir,
            run_id="run-test",
            manifest=manifest,
        )
    assert panel.cases[0].variant_a.ordering == ()
    assert any("length mismatch" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_validates_threshold(tmp_path: Path) -> None:
    run_dir = _empty_run_dir(tmp_path)
    with pytest.raises(ValueError, match="threshold"):
        compute_worstcase_panel(
            _flip_matrix_2x1(),
            ("vA", "vB"),
            ("e0",),
            run_dir=run_dir,
            run_id="run-test",
            threshold=1.5,
        )


def test_compute_worstcase_panel_validates_k(tmp_path: Path) -> None:
    run_dir = _empty_run_dir(tmp_path)
    with pytest.raises(ValueError, match="k must"):
        compute_worstcase_panel(
            _flip_matrix_2x1(),
            ("vA", "vB"),
            ("e0",),
            run_dir=run_dir,
            run_id="run-test",
            k=0,
        )


def test_compute_worstcase_panel_validates_axis_label_length(
    tmp_path: Path,
) -> None:
    run_dir = _empty_run_dir(tmp_path)
    with pytest.raises(ValueError, match="variant_ids length"):
        compute_worstcase_panel(
            _flip_matrix_2x1(),
            ("vA",),  # wrong length
            ("e0",),
            run_dir=run_dir,
            run_id="run-test",
        )
    with pytest.raises(ValueError, match="eval_ids length"):
        compute_worstcase_panel(
            _flip_matrix_2x1(),
            ("vA", "vB"),
            (),  # wrong length
            run_dir=run_dir,
            run_id="run-test",
        )


def test_compute_worstcase_panel_validates_ndim(tmp_path: Path) -> None:
    run_dir = _empty_run_dir(tmp_path)
    with pytest.raises(ValueError, match="2D"):
        compute_worstcase_panel(
            np.array([1.0, 2.0]),
            ("vA", "vB"),
            ("e0",),
            run_dir=run_dir,
            run_id="run-test",
        )


# ---------------------------------------------------------------------------
# Missing-DB graceful degradation
# ---------------------------------------------------------------------------


def test_compute_worstcase_panel_missing_run_sqlite_is_empty_traces(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """`run.sqlite` absent under `run_dir` → WARNING + empty traces, not crash."""
    run_dir = tmp_path / "run-no-db"
    run_dir.mkdir()
    with caplog.at_level("WARNING"):
        panel = compute_worstcase_panel(
            _flip_matrix_2x1(),
            ("vA", "vB"),
            ("e0",),
            run_dir=run_dir,
            run_id="run-test",
        )
    assert panel.disabled is False  # selection layer succeeded
    case = panel.cases[0]
    assert case.variant_a.node_outputs == ()
    assert case.variant_b.node_outputs == ()
    assert any("run.sqlite missing" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Truncate helper
# ---------------------------------------------------------------------------


def test_truncate_below_cap_unchanged() -> None:
    assert _truncate("hello", 10) == "hello"


def test_truncate_above_cap_adds_ellipsis() -> None:
    assert _truncate("hello world", 5) == "hello…"


def test_truncate_strips_trailing_whitespace() -> None:
    assert _truncate("hello   world", 8) == "hello…"


def test_truncate_zero_cap_is_empty() -> None:
    assert _truncate("hello", 0) == ""
    assert _truncate("hello", -3) == ""


# ---------------------------------------------------------------------------
# Dataclass smoke (the public surface)
# ---------------------------------------------------------------------------


def test_worstcase_dataclasses_are_frozen() -> None:
    """The result types are frozen — defensive against in-place mutation
    by template/test code (matches the §9/§14 metric convention)."""
    node = NodeOutputView(node_id="n", role="r", response_text="t")
    with pytest.raises(Exception):
        node.node_id = "x"  # type: ignore[misc]

    side = VariantSideView(
        variant_id="v", ordering=(), score=0.5, verdict=True,
        rationale=None, node_outputs=(),
    )
    with pytest.raises(Exception):
        side.score = 0.0  # type: ignore[misc]

    entry = WorstCaseEntry(
        eval_id="e", flip_count=1, n_pairs=1,
        variant_a=side, variant_b=side,
    )
    with pytest.raises(Exception):
        entry.flip_count = 0  # type: ignore[misc]

    panel = WorstCasePanel(
        disabled=False, disabled_reason=None, cases=(entry,), k=3,
    )
    with pytest.raises(Exception):
        panel.disabled = True  # type: ignore[misc]
