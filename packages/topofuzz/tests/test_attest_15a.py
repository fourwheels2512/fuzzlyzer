"""Step 15a attestation tests: headline-metric persistence + report.json mirror.

Covers BLUEPRINT §14 sub-deliverables (b) through (h):

* (b) voo_per_eval entries gain {ci_lower, ci_upper, degenerate_flag}
* (c) headline.verdict_flip_rate populated from score matrix
* (d) headline.kappa_original_vs_permuted scalar
* (e) headline.balanced_accuracy_drop dict (when manifest has ground_truth)
* (f) headline.judge_failure_rate plumbed from kwarg
* (g) headline.judge_independence full §8 dict plumbed from kwarg
* (h) report.json mirror is byte-identical to the signed payload
"""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path

import pytest
import rfc8785
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fuzzeval.attestation import Ed25519Signer

from topofuzz.attest import (
    ATTESTATION_FILENAME,
    REPORT_FILENAME,
    compute_ba_drop_for_attestation,
    compute_headline,
    compute_kappa_orig_vs_perm_for_attestation,
    compute_verdict_flip_for_attestation,
    compute_voo_for_attestation,
    emit_run_attestation,
    load_and_verify_attestation,
)


# ---------------------------------------------------------------------------
# Helpers: minimal sqlite + manifest for the attestation path
# ---------------------------------------------------------------------------


def _signer_pair() -> tuple[Ed25519Signer, bytes]:
    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    return signer, signer.public_key_pem


def _seed_db(db_path: Path, run_id: str, rubric_hash: str, rows: list[tuple]) -> None:
    """Hand-craft a minimal v4 run.sqlite. Row shape:
    (variant_id, eval_id, score_value) — all rows are continuous, no failures."""
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
            "INSERT INTO runs (run_id, topology_canonical_hash, topology_name, "
            "tds_version, created_at, rubric_hash) "
            "VALUES (?, 'h', 'name', '0.5', 0.0, ?)",
            (run_id, rubric_hash),
        )
        for vid, eid, score in rows:
            db.execute(
                "INSERT INTO judge_results (run_id, variant_id, eval_id, "
                "judge_kind, score_value, passed, rationale, failure_reason) "
                "VALUES (?, ?, ?, 'continuous', ?, NULL, NULL, NULL)",
                (run_id, vid, eid, score),
            )
        db.commit()
    finally:
        db.close()


def _build_manifest(
    run_id: str,
    variant_canonical_hashes: list[str],
    eval_ids: list[str],
    *,
    eval_ground_truth: list[int | None] | None = None,
    agent_vendors: list[str] | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "topology_canonical_hash": "topo-hash",
        "topology_name": "fixture",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": len(variant_canonical_hashes),
        "n_eval_rows": len(eval_ids),
        "variant_canonical_hashes": variant_canonical_hashes,
        "variant_orderings": [["x"] for _ in variant_canonical_hashes],
        "eval_ids": eval_ids,
        "eval_ground_truth": (
            eval_ground_truth if eval_ground_truth is not None else [None] * len(eval_ids)
        ),
        "agent_vendors": agent_vendors if agent_vendors is not None else [],
        "max_cost_usd": None,
        "judge_model": "single",
        "rubric_hash": "rh",
        "semaphore_size": 8,
        "cli_args": {},
        "kickoff_ts": 0.0,
    }


# ---------------------------------------------------------------------------
# (b) voo per-eval CI persistence
# ---------------------------------------------------------------------------


def test_voo_per_eval_includes_ci_fields(tmp_path: Path):
    """compute_voo_for_attestation emits CI fields per eval, not just point."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.10), ("vA", "e2", 0.20), ("vA", "e3", 0.30),
            ("vB", "e1", 0.90), ("vB", "e2", 0.80), ("vB", "e3", 0.70),
        ],
    )
    out = compute_voo_for_attestation(tmp_path, run_id="r1", seed=0)
    assert out is not None
    assert len(out) == 3
    for entry in out:
        # Step 15a contract: every per-eval entry must carry these keys.
        assert {"eval_id", "point", "ci_lower", "ci_upper", "degenerate_flag"} <= set(
            entry.keys()
        )
        if entry["degenerate_flag"] is False:
            assert entry["ci_lower"] <= entry["point"] <= entry["ci_upper"]


def test_voo_constant_column_marked_degenerate(tmp_path: Path):
    """Two variants with identical scores on every eval → variance is 0;
    bootstrap is structurally a point; mark degenerate."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.5), ("vA", "e2", 0.5),
            ("vB", "e1", 0.5), ("vB", "e2", 0.5),
        ],
    )
    out = compute_voo_for_attestation(tmp_path, run_id="r1", seed=0)
    assert out is not None
    for entry in out:
        assert entry["point"] == 0.0
        assert entry["degenerate_flag"] is True
        assert entry["ci_lower"] is None
        assert entry["ci_upper"] is None


# ---------------------------------------------------------------------------
# (c)(d)(e) per-metric attest wire-up
# ---------------------------------------------------------------------------


def test_verdict_flip_for_attestation_happy_path(tmp_path: Path):
    """Score matrix where the original (lex-first hash "vA") and the
    permuted ("vB") disagree on every eval → flip rate = 1.0."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.9),
            ("vB", "e1", 0.1), ("vB", "e2", 0.1),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1", "e2"],
    )
    out = compute_verdict_flip_for_attestation(tmp_path, "r1", manifest, seed=0)
    assert out is not None
    assert out["point"] == 1.0
    assert out["degenerate_flag"] is False


def test_verdict_flip_returns_none_when_original_hash_missing(tmp_path: Path):
    """Manifest's variant_canonical_hashes[0] not in score matrix axis →
    metric returns None rather than guessing or crashing."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vB", "e1", 0.1),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["ghost-hash", "vA"],  # ghost-hash not in DB
        eval_ids=["e1"],
    )
    out = compute_verdict_flip_for_attestation(tmp_path, "r1", manifest)
    assert out is None


def test_attest_refuses_duplicate_canonical_hashes(tmp_path: Path):
    """Audit RT-B1: duplicate entries in manifest.variant_canonical_hashes
    make the [0]-resolved original ambiguous against a lex-sorted score
    matrix axis. Metric must refuse rather than silently pick a row."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.8),
            ("vB", "e1", 0.1), ("vB", "e2", 0.2),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vA", "vB"],  # duplicate vA
        eval_ids=["e1", "e2"],
    )
    assert compute_verdict_flip_for_attestation(tmp_path, "r1", manifest) is None
    assert compute_kappa_orig_vs_perm_for_attestation(tmp_path, "r1", manifest) is None
    assert compute_ba_drop_for_attestation(tmp_path, "r1", manifest) is None


def test_attest_handles_partial_success_manifest_superset(tmp_path: Path):
    """Audit Pass-2 M1: when score_matrix is missing a variant the
    manifest declared (e.g. that variant's cells all failed), the
    attest path must STILL compute metrics for the surviving
    variants — the manifest is legitimately a superset of the matrix
    axis. The original pass-1 fix's `issubset` check broke this
    real-world partial-success case.

    Fixture: manifest declares 3 variants (vA + vB + vC), but the
    score DB only contains rows for vA + vB (vC's cells all failed).
    Original is vA. Metric must compute, not return None."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.8),
            ("vB", "e1", 0.1), ("vB", "e2", 0.2),
            # vC has no rows — every cell failed; loader drops it.
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB", "vC"],
        eval_ids=["e1", "e2"],
    )
    # verdict_flip must compute on vA-vs-vB and not silently null out.
    out = compute_verdict_flip_for_attestation(tmp_path, "r1", manifest)
    assert out is not None, "partial-success run lost verdict_flip"
    assert out["point"] == 1.0  # vA all-pass, vB all-fail


def test_attest_refuses_when_original_hash_not_in_matrix(tmp_path: Path):
    """Audit RT-B2 (post Pass-2 narrowing): the original hash MUST be
    in the score-matrix axis. If kickoff wrote a canonical_hash for
    the original that the loader doesn't see (drift OR the original's
    cells all failed), no headline metric makes sense — return None."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vB", "e1", 0.1),
        ],
    )
    manifest = _build_manifest(
        "r1",
        # Original ghost-hash NOT in matrix axis → refuse.
        variant_canonical_hashes=["ghost-hash", "vA"],
        eval_ids=["e1"],
    )
    assert compute_verdict_flip_for_attestation(tmp_path, "r1", manifest) is None


def test_emit_attestation_envelope_first_then_mirror(tmp_path: Path):
    """Audit CR-M3 / RT-M2: when dsse_sign or envelope write fails,
    the report.json mirror must NOT be left orphaned on disk. We
    simulate the failure mode by passing a path that can't be written
    (parent already exists as a file)."""
    # Create a "directory" that's actually a file so out_path.write_bytes
    # raises OSError, blocking step 2. Step 3 (mirror write) must not
    # have run yet.
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"")
    # out_path inside the blocker — write_bytes will fail because
    # parent is a file. parent.mkdir(exist_ok=True) would raise too.
    bad_out = blocker / "attestation.json"
    report_path = tmp_path / "report.json"
    payload = {"schema_version": 1, "headline": {}, "manifest": {}}
    signer, _ = _signer_pair()
    with pytest.raises((OSError, FileExistsError, NotADirectoryError)):
        from topofuzz.attest import emit_attestation
        emit_attestation(payload, signer, bad_out, report_path=report_path)
    # Critical: report.json must NOT exist.
    assert not report_path.exists(), (
        "envelope failure left an orphan report.json — auditor would "
        "trust the orphan as authoritative"
    )


def test_kappa_orig_vs_perm_for_attestation(tmp_path: Path):
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.9), ("vA", "e3", 0.9),
            ("vB", "e1", 0.9), ("vB", "e2", 0.9), ("vB", "e3", 0.9),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1", "e2", "e3"],
    )
    out = compute_kappa_orig_vs_perm_for_attestation(tmp_path, "r1", manifest)
    # Identical preds → κ = 1.0.
    assert out == 1.0


def test_ba_drop_returns_none_without_ground_truth(tmp_path: Path):
    """No ground_truth in manifest → BA-drop is degenerate."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.9),
            ("vB", "e1", 0.1), ("vB", "e2", 0.1),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1", "e2"],
        eval_ground_truth=[None, None],
    )
    assert compute_ba_drop_for_attestation(tmp_path, "r1", manifest) is None


def test_ba_drop_with_ground_truth(tmp_path: Path):
    """Original perfect against GT, permuted inverse → drop = 1.0."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.1), ("vA", "e3", 0.9), ("vA", "e4", 0.1),
            ("vB", "e1", 0.1), ("vB", "e2", 0.9), ("vB", "e3", 0.1), ("vB", "e4", 0.9),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1", "e2", "e3", "e4"],
        eval_ground_truth=[1, 0, 1, 0],
    )
    out = compute_ba_drop_for_attestation(tmp_path, "r1", manifest)
    assert out is not None
    assert out["original"] == 1.0
    assert out["mean_permuted"] == 0.0
    assert out["drop"] == 1.0


def test_ba_drop_aligns_ground_truth_by_eval_id_not_position(tmp_path: Path):
    """Regression: the score-matrix eval axis is in judge-row insertion
    order, NOT manifest plan order. Ground truth must be re-indexed by
    eval_id. Here the db rows are seeded in a shuffled eval order (e3, e1,
    e4, e2) so read_score_matrix yields a non-manifest eval axis; BA-drop
    must still compute correctly. The previous positional-alignment code
    returned None on every such run."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            # vA (original): perfect vs GT. Inserted shuffled: e3,e1,e4,e2.
            ("vA", "e3", 0.9), ("vA", "e1", 0.9), ("vA", "e4", 0.1), ("vA", "e2", 0.1),
            # vB (permuted): inverse of GT.
            ("vB", "e3", 0.1), ("vB", "e1", 0.1), ("vB", "e4", 0.9), ("vB", "e2", 0.9),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        # Manifest is in plan order — deliberately different from the
        # e3,e1,e4,e2 axis read_score_matrix will discover.
        eval_ids=["e1", "e2", "e3", "e4"],
        eval_ground_truth=[1, 0, 1, 0],
    )
    out = compute_ba_drop_for_attestation(tmp_path, "r1", manifest)
    assert out is not None
    # If ground truth were aligned positionally against the shuffled axis,
    # original would NOT be 1.0. By-eval-id alignment recovers the truth.
    assert out["original"] == 1.0
    assert out["mean_permuted"] == 0.0
    assert out["drop"] == 1.0


def test_ba_drop_none_when_matrix_eval_absent_from_manifest(tmp_path: Path):
    """A score-matrix eval_id with no manifest ground-truth entry is a
    structural bug signal — BA-drop skips rather than guessing."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.1), ("vA", "e5", 0.9),
            ("vB", "e1", 0.1), ("vB", "e2", 0.9), ("vB", "e5", 0.1),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1", "e2"],  # no e5
        eval_ground_truth=[1, 0],
    )
    assert compute_ba_drop_for_attestation(tmp_path, "r1", manifest) is None


def test_ba_drop_none_on_manifest_array_length_mismatch(tmp_path: Path):
    """manifest.eval_ids and manifest.eval_ground_truth must agree in
    length — a producer bug if they don't; BA-drop skips."""
    _seed_db(
        tmp_path / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.1),
            ("vB", "e1", 0.1), ("vB", "e2", 0.9),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1", "e2"],
        eval_ground_truth=[1],  # length 1 vs eval_ids length 2
    )
    assert compute_ba_drop_for_attestation(tmp_path, "r1", manifest) is None


# ---------------------------------------------------------------------------
# (f) + (g) compute_headline plumbing
# ---------------------------------------------------------------------------


def test_compute_headline_carries_step15a_fields():
    """All step-15a kwargs land in the assembled headline dict."""
    h = compute_headline(
        status="succeeded",
        n_successful_cells=10,
        n_total_cells=10,
        n_variants=2,
        n_evals=5,
        total_cost_usd=0.1,
        elapsed_s=1.0,
        verdict_flip_rate={"point": 0.4, "ci_lower": 0.2, "ci_upper": 0.6,
                           "degenerate_flag": False},
        kappa_original_vs_permuted=0.35,
        balanced_accuracy_drop={"original": 0.9, "mean_permuted": 0.6, "drop": 0.3},
        judge_failure_rate=0.02,
        judge_independence={
            "judge_protocol": "jury_of_3",
            "judges": [],
            "vendor_overlap": ["google"],
            "vendor_disjoint": False,
            "self_preference_warning": "shared vendor: google",
            "vendor_call_failure_rate": 0.01,
        },
    )
    assert h["verdict_flip_rate"]["point"] == 0.4
    assert h["kappa_original_vs_permuted"] == 0.35
    assert h["balanced_accuracy_drop"]["drop"] == 0.3
    assert h["judge_failure_rate"] == 0.02
    assert h["judge_independence"]["judge_protocol"] == "jury_of_3"
    assert h["judge_independence"]["vendor_overlap"] == ["google"]


def test_compute_headline_step15a_fields_default_to_none():
    """Pre-15a callers (no new kwargs) still get a well-shaped headline."""
    h = compute_headline(
        status="succeeded",
        n_successful_cells=1,
        n_total_cells=1,
        n_variants=1,
        n_evals=1,
        total_cost_usd=0.0,
        elapsed_s=0.0,
    )
    assert h["verdict_flip_rate"] is None
    assert h["kappa_original_vs_permuted"] is None
    assert h["balanced_accuracy_drop"] is None
    assert h["judge_failure_rate"] is None
    assert h["judge_independence"] is None


# ---------------------------------------------------------------------------
# (h) report.json mirror byte-equals signed payload
# ---------------------------------------------------------------------------


def test_emit_run_attestation_writes_report_json_mirror(tmp_path: Path):
    """report.json bytes must equal base64-decoded attestation.payload."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1"],
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    signer, _ = _signer_pair()
    emit_run_attestation(
        run_dir=run_dir,
        run_id="r1",
        manifest=manifest,
        status="succeeded",
        successful_cells=0,
        total_cells=0,
        n_variants=2,
        n_evals=1,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        signer=signer,
    )
    # Both files exist.
    report_path = run_dir / REPORT_FILENAME
    att_path = run_dir / ATTESTATION_FILENAME
    assert report_path.exists()
    assert att_path.exists()
    # Decode the DSSE envelope's base64 payload and compare bytes.
    envelope = json.loads(att_path.read_bytes())
    decoded = base64.b64decode(envelope["payload"])
    assert decoded == report_path.read_bytes()


def test_report_json_is_valid_jcs(tmp_path: Path):
    """The mirror is JCS-canonical bytes (so re-canonicalizing the parsed
    dict yields the same bytes)."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA"],
        eval_ids=["e1"],
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    signer, _ = _signer_pair()
    emit_run_attestation(
        run_dir=run_dir,
        run_id="r1",
        manifest=manifest,
        status="succeeded",
        successful_cells=0,
        total_cells=0,
        n_variants=1,
        n_evals=1,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        signer=signer,
    )
    report_bytes = (run_dir / REPORT_FILENAME).read_bytes()
    parsed = json.loads(report_bytes.decode("utf-8"))
    assert rfc8785.dumps(parsed) == report_bytes


# ---------------------------------------------------------------------------
# Schema-shape e2e: headline carries every new field after emit_run_attestation
# ---------------------------------------------------------------------------


def test_emit_run_attestation_headline_carries_15a_fields(tmp_path: Path):
    """End-to-end: emit attestation with a real DB; verify the signed
    headline carries every step-15a field (or null where degenerate)."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    _seed_db(
        run_dir / "run.sqlite",
        "r1",
        "rh",
        rows=[
            ("vA", "e1", 0.9), ("vA", "e2", 0.9), ("vA", "e3", 0.9), ("vA", "e4", 0.9),
            ("vB", "e1", 0.1), ("vB", "e2", 0.1), ("vB", "e3", 0.1), ("vB", "e4", 0.1),
        ],
    )
    manifest = _build_manifest(
        "r1",
        variant_canonical_hashes=["vA", "vB"],
        eval_ids=["e1", "e2", "e3", "e4"],
        eval_ground_truth=[1, 1, 1, 1],  # all same → degenerate for BA-drop
        agent_vendors=["google"],
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    signer, pub_pem = _signer_pair()
    emit_run_attestation(
        run_dir=run_dir,
        run_id="r1",
        manifest=manifest,
        status="succeeded",
        successful_cells=8,
        total_cells=8,
        n_variants=2,
        n_evals=4,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        signer=signer,
        judge_failure_rate=0.0,
        judge_independence={
            "judge_protocol": "single",
            "judges": [{"vendor": "google", "model_id": "google/gemini-2.5-pro"}],
            "vendor_overlap": ["google"],
            "vendor_disjoint": False,
            "self_preference_warning": "Agents and judges share vendor(s): google.",
            "vendor_call_failure_rate": 0.0,
        },
    )
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem
    )
    assert outcome.ok, outcome.checks
    headline = outcome.payload["headline"]
    # Every new field is present (value may be null where degenerate).
    for k in (
        "voo_per_eval",
        "verdict_flip_rate",
        "kappa_original_vs_permuted",
        "balanced_accuracy_drop",
        "judge_failure_rate",
        "judge_independence",
    ):
        assert k in headline, f"missing headline.{k}"
    # voo_per_eval was computed (constant column → degenerate but populated)
    assert headline["voo_per_eval"] is not None
    # verdict_flip_rate was computed (vA & vB disagree on every eval → 1.0)
    assert headline["verdict_flip_rate"]["point"] == 1.0
    # F1: numerator/denominator + Wilson CI keys are populated end-to-end.
    vfr = headline["verdict_flip_rate"]
    assert {
        "point", "ci_lower", "ci_upper", "degenerate_flag",
        "numerator", "denominator",
        "wilson_ci_lower", "wilson_ci_upper", "wilson_conf_level",
    } <= set(vfr)
    assert isinstance(vfr["numerator"], int)
    assert isinstance(vfr["denominator"], int)
    assert vfr["denominator"] > 0
    # Wilson CI must bracket the point (defining property of any valid CI).
    assert vfr["wilson_ci_lower"] is not None
    assert vfr["wilson_ci_upper"] is not None
    assert vfr["wilson_ci_lower"] <= vfr["point"] <= vfr["wilson_ci_upper"]
    # Wilson conf level is persisted alongside the bounds; default 0.95.
    assert vfr["wilson_conf_level"] == 0.95
    # kappa_original_vs_permuted is computable (κ=-1 for opposite constants
    # would be 0 by Cohen's formula, but with mixed predictions, this
    # specific fixture yields 0.0 because both raters are constant).
    assert headline["kappa_original_vs_permuted"] == 0.0
    # BA-drop is None because GT is constant (no signal).
    assert headline["balanced_accuracy_drop"] is None
    assert headline["judge_failure_rate"] == 0.0
    assert headline["judge_independence"]["judge_protocol"] == "single"
