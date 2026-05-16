# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""F3 MAJOR (2026-05-16): jury-aware VOO over per-vendor continuous scores.

Covers the new public surface in ``topofuzz.metrics.voo``:

* :func:`variance_across_orderings_jury` — averages standard VOO across
  vendors; preserves §9 axis semantics (variance is across the variant
  axis, never across the judge axis).
* :func:`variance_across_orderings_jury_ci` — cluster-bootstrap CI over
  the eval axis.

The plumbing into :class:`topofuzz.metrics.ScoreMatrixResult` is covered
in ``test_metrics_score_matrix_jury.py``; this module exercises the
metric closure on raw 3D arrays so the two layers can be debugged
independently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzeval.stats import BootstrapResult
from topofuzz.metrics import (
    variance_across_orderings,
    variance_across_orderings_jury,
    variance_across_orderings_jury_ci,
)


def test_jury_voo_matches_per_vendor_average_of_canonical_formula():
    """The headline closure is ``mean_vendors(VOO(vendor_slab))``."""
    rng = np.random.default_rng(7)
    n_vendors, n_variants, n_evals = 3, 4, 6
    pv = rng.random((n_vendors, n_variants, n_evals))
    expected = float(
        np.mean(
            [variance_across_orderings(pv[j]) for j in range(n_vendors)]
        )
    )
    got = variance_across_orderings_jury(pv)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_jury_voo_zero_on_constant_slab():
    pv = np.full((3, 4, 5), 0.5)
    assert variance_across_orderings_jury(pv) == 0.0


def test_jury_voo_zero_on_singleton_variant_axis():
    pv = np.random.default_rng(0).random((3, 1, 5))
    # Single variant → variance across the variant axis is 0 for every
    # vendor; mean across vendors is 0.
    assert variance_across_orderings_jury(pv) == 0.0


def test_jury_voo_zero_on_empty_axes():
    """Empty vendor / variant / eval axes all return 0.0 (not NaN)."""
    for shape in [(0, 4, 5), (3, 0, 5), (3, 4, 0)]:
        pv = np.empty(shape)
        assert variance_across_orderings_jury(pv) == 0.0


def test_jury_voo_rejects_non_3d():
    with pytest.raises(ValueError, match="3D"):
        variance_across_orderings_jury(np.array([[1.0, 2.0]]))


def test_jury_voo_handles_nan_cells_via_nanvar():
    """A missing vote (NaN) in one (vendor, variant, eval) cell should
    drop that cell from the variance, not propagate to the headline.

    The contract: NaN is the loader's "no data" sentinel, not a numeric
    value to average over.
    """
    pv = np.full((2, 3, 4), 0.5)
    # Inject a single NaN — the per-eval variance for that vendor/eval
    # should still be 0 (the other two variants are constant), and the
    # rest of the slab is constant — overall headline is 0.
    pv[0, 1, 2] = np.nan
    got = variance_across_orderings_jury(pv)
    assert got == 0.0


def test_jury_voo_partial_roster_does_not_collapse_to_zero(recwarn):
    """Audit M1 fix: a vendor with only 1 non-NaN variant for an eval
    is structurally degenerate (variance estimator needs ≥2 points).
    The cell should be NaN-masked, not silently collapsed to 0.

    Without the count≥2 guard, ``np.nanvar`` returns 0 for the
    singleton-vote cell, which then averages in as a real "zero
    variance" measurement and deflates the headline.
    """
    # Two vendors, two variants, two evals. For vendor 0 / eval 0,
    # only one variant has a vote — singleton in the variance estimator.
    pv = np.full((2, 2, 2), np.nan)
    pv[0, 0, 0] = 0.5  # only one vendor-0 vote for eval 0
    pv[0, 0, 1] = 0.3
    pv[0, 1, 1] = 0.7  # vendor-0 eval 1 has both variants
    pv[1, 0, 0] = 0.6
    pv[1, 1, 0] = 0.4  # vendor-1 eval 0 has both variants
    pv[1, 0, 1] = 0.1
    pv[1, 1, 1] = 0.9
    got = variance_across_orderings_jury(pv)
    # vendor 0: eval 0 NaN-masked (singleton), eval 1 var = var([0.3, 0.7])
    #           = 0.04. Mean across non-NaN evals = 0.04.
    # vendor 1: eval 0 var = var([0.6, 0.4]) = 0.01, eval 1 var =
    #           var([0.1, 0.9]) = 0.16. Mean = 0.085.
    # Headline mean(vendor 0, vendor 1) = (0.04 + 0.085) / 2 = 0.0625.
    expected = (0.04 + (0.01 + 0.16) / 2) / 2
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_jury_voo_continuous_recovers_signal_that_bernoulli_majority_loses():
    """Demonstration of the §9 thresholds Bernoulli-boundedness fix.

    Build a scenario where every cell's majority is the SAME value
    (binary 1.0) yet the per-vendor continuous scores vary across
    variants. The majority-VOO closure (``np.var(majority, axis=0)``)
    is exactly 0 — the issue the F3 MAJOR was filed to close. The
    jury-aware closure recovers the continuous variance.
    """
    # Three vendors all say "passed" (score>0.5) for every cell, but
    # the continuous score wobbles across the 4 variants. The majority
    # column is all 1.0; the continuous per-vendor matrix has real
    # variance.
    n_vendors, n_variants, n_evals = 3, 4, 5
    rng = np.random.default_rng(0)
    pv = 0.6 + 0.3 * rng.random((n_vendors, n_variants, n_evals))
    # Majority over a per-cell vote of all ">0.5" is always 1.0 → VOO
    # on majority is exactly 0 (the Bernoulli-bound failure mode).
    majority = (pv > 0.5).astype(float).mean(axis=0)
    majority_binarised = (majority > 0.5).astype(float)
    assert variance_across_orderings(majority_binarised) == 0.0
    # Jury-aware VOO sees the continuous variance — positive.
    jury_voo = variance_across_orderings_jury(pv)
    assert jury_voo > 0.0
    # Sanity: anywhere strictly below the 0.25 Bernoulli ceiling
    # demonstrates the continuous-bounded fix works. The exact value
    # depends on numpy's nanvar precision; an upper bound at the
    # Bernoulli ceiling is robust across numpy versions.
    assert jury_voo < 0.25


def test_jury_voo_ci_round_trips():
    rng = np.random.default_rng(0)
    pv = rng.random((3, 5, 8))
    r = variance_across_orderings_jury_ci(pv, n_resamples=400, seed=42)
    assert isinstance(r, BootstrapResult)
    assert r.method == "percentile"
    assert r.n_resamples == 400
    assert not r.degenerate_flag
    assert r.ci_lower is not None and r.ci_upper is not None
    assert math.isclose(
        r.point_estimate, variance_across_orderings_jury(pv), rel_tol=1e-12
    )
    assert r.ci_lower <= r.point_estimate <= r.ci_upper


def test_jury_voo_ci_degenerate_on_constant_slab():
    pv = np.full((3, 4, 6), 0.5)
    r = variance_across_orderings_jury_ci(pv, n_resamples=200, seed=0)
    assert r.degenerate_flag
    assert r.ci_lower is None and r.ci_upper is None


def test_jury_voo_ci_degenerate_carries_n_resamples_zero_sentinel():
    """Audit M4: degenerate ``BootstrapResult`` must carry
    ``n_resamples=0`` to match the project-wide convention used by
    ``percentile_cluster_bootstrap`` and the attestation rendering path
    (which uses ``r.n_resamples == 0`` as a degenerate sentinel)."""
    pv = np.full((3, 4, 6), 0.5)
    r = variance_across_orderings_jury_ci(pv, n_resamples=300, seed=0)
    assert r.degenerate_flag
    assert r.n_resamples == 0  # sentinel, NOT the user-passed 300


def test_jury_voo_ci_determinism_with_seed():
    rng = np.random.default_rng(0)
    pv = rng.random((3, 4, 6))
    r1 = variance_across_orderings_jury_ci(pv, n_resamples=300, seed=11)
    r2 = variance_across_orderings_jury_ci(pv, n_resamples=300, seed=11)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper
    assert r1.point_estimate == r2.point_estimate


def test_jury_voo_ci_rejects_non_3d():
    with pytest.raises(ValueError, match="3D"):
        variance_across_orderings_jury_ci(
            np.array([[1.0, 2.0]]), n_resamples=10
        )


def test_attest_compute_voo_dispatches_to_jury_on_v6_run(tmp_path):
    """Pass-2 M-RT-P2-3: the wire-in in ``compute_voo_for_attestation``
    must actually call the jury-aware path when ``per_vendor_matrix``
    is present.

    Constructs a v6 DB where the binarised majority is constantly 1.0
    (so the canonical 2D path returns 0 per eval — exactly the
    Bernoulli-bound failure mode F3 was filed to close) but the
    per-vendor continuous scores wobble across variants. The attest
    helper should emit a NON-zero ``point`` for every eval, proving
    the dispatch flows through to the signed JSON.
    """
    import json
    import sqlite3

    from topofuzz.attest import compute_voo_for_attestation

    db_path = tmp_path / "run.sqlite"
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
        db.execute(
            "INSERT INTO runs (run_id, topology_canonical_hash, "
            "topology_name, tds_version, created_at, rubric_hash) "
            "VALUES ('r1', 'h', 'name', '0.5', 0.0, 'rh')"
        )
        # 2 variants × 2 evals × 3 vendors. Each vendor's continuous
        # score is >0.5 (so majority is always 1.0) but they wobble
        # across the variant axis — the F3 fix recovers this signal.
        cells = [
            ("vA", "e1", [("a", 0.6), ("b", 0.7), ("c", 0.8)]),
            ("vA", "e2", [("a", 0.9), ("b", 0.6), ("c", 0.7)]),
            ("vB", "e1", [("a", 0.95), ("b", 0.85), ("c", 0.75)]),
            ("vB", "e2", [("a", 0.55), ("b", 0.95), ("c", 0.6)]),
        ]
        for vid, eid, votes in cells:
            votes_json = json.dumps(
                [{"vendor": v, "score": s, "rationale": ""} for v, s in votes]
            )
            db.execute(
                "INSERT INTO judge_results "
                "(run_id, variant_id, eval_id, judge_kind, score_value, "
                " passed, rationale, failure_reason, votes_json) "
                "VALUES ('r1', ?, ?, 'continuous', 1.0, NULL, NULL, NULL, ?)",
                (vid, eid, votes_json),
            )
        db.commit()
    finally:
        db.close()

    out = compute_voo_for_attestation(tmp_path, "r1", n_resamples=200, seed=0)
    assert out is not None
    # Every eval was constant-majority (all 1.0). The OLD path would
    # have returned point=0.0 for every eval (Bernoulli-bounded). The
    # jury dispatch surfaces the per-vendor continuous variance > 0.
    for entry in out:
        assert entry["point"] is not None
        assert entry["point"] > 0.0, (
            f"jury dispatch did not surface continuous signal for "
            f"{entry['eval_id']!r}: point={entry['point']}"
        )
