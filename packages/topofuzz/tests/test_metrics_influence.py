# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 15: §6a per-agent influence — point + bootstrap CI.

Covers the contract in :mod:`topofuzz.metrics.influence`:

* point estimate matches the pairwise P(flip | moved) − P(flip | same) formula
* sign is preserved (negative influence surfaces, no clamping)
* dominant-agent fixture ranks the obvious driver first
* degenerate paths (single variant, agent never moves, agent never stays
  put, missing scores) flag without crashing
* bootstrap CI brackets the point estimate and shrinks with more resamples
* ranking is stable across reseeds when the dominant agent's signal is
  well-separated (the §6a "statistically clean vs. tied" claim)
* NaN cells are excluded, not silently coerced
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topofuzz.metrics import (
    AgentInfluence,
    InfluenceComputation,
    OrderingAlignmentError,
    align_orderings_to_matrix,
    compute_influence,
)


def _dominant_fixture():
    """3-agent, 4-variant fixture where slot-of-A perfectly predicts the answer.

    Variants 0,1 have A in slot 0 → high score (binarised 1).
    Variants 2,3 have A in slots 1, 2 → low score (binarised 0).
    B and C move around but their slot doesn't determine the answer.
    """
    score = np.array(
        [
            [0.90, 0.85, 0.92, 0.88, 0.91],
            [0.88, 0.91, 0.86, 0.89, 0.93],
            [0.10, 0.05, 0.12, 0.08, 0.11],
            [0.08, 0.11, 0.09, 0.07, 0.10],
        ]
    )
    orderings = [
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "A", "C"),
        ("B", "C", "A"),
    ]
    return score, orderings


def test_returns_influence_computation_with_signed_values():
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, seed=0, n_bootstrap=200)
    assert isinstance(out, InfluenceComputation)
    assert out.n_bootstrap == 200
    assert math.isclose(out.conf, 0.95, rel_tol=1e-12)
    assert math.isclose(out.flip_threshold, 0.5, rel_tol=1e-12)
    assert len(out.results) == 3
    for row in out.results:
        assert isinstance(row, AgentInfluence)


def test_dominant_agent_ranks_first():
    """A's slot fully explains the binarised answer → A has the largest influence."""
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, seed=0, n_bootstrap=100)
    top = out.results[0]
    # A or B may tie at the top under perfectly-correlated synthetic
    # signal; A's row MUST be in the top tier with positive influence.
    a_row = next(r for r in out.results if r.node_id == "A")
    assert a_row.influence is not None
    assert a_row.influence > 0.5
    assert top.node_id in {"A", "B"}


def test_signed_influence_preserved_for_stabilising_agent():
    """An agent whose slot is anticorrelated with the answer surfaces signed-negative.

    Constructed so the only same-slot pairs of C are also same-answer
    pairs, and moved-C pairs straddle the answer boundary.
    """
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, seed=0, n_bootstrap=100)
    c_row = next(r for r in out.results if r.node_id == "C")
    assert c_row.influence is not None
    # C's same-slot pairs land on different-answer pairs in this
    # fixture; moved C pairs straddle answer boundary → negative.
    assert c_row.influence < 0


def test_point_matches_manual_formula():
    """Hand-computed influence(A) for the dominant fixture is 0.8."""
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, seed=0, n_bootstrap=0)
    a_row = next(r for r in out.results if r.node_id == "A")
    # Moved-A pairs: 5 (every cross-group pair + (2,3) where A is in
    # slots 1 vs 2). Same-A pairs: 1 ((0,1) — both slot 0). 5 evals
    # binarised → all moved pairs that span groups flip every eval,
    # the (2,3) pair never flips (both 0). 4 × 5 = 20 flips / 25
    # cells = 0.8; same-slot (0,1) never flips → 0/5 = 0.0.
    assert a_row.influence is not None
    assert math.isclose(a_row.influence, 0.8, abs_tol=1e-12)


def test_degenerate_when_agent_never_moves():
    score = np.array([[0.9, 0.1], [0.8, 0.2]])
    orderings = [("A", "B"), ("A", "B")]
    out = compute_influence(score, orderings, n_bootstrap=0)
    a = next(r for r in out.results if r.node_id == "A")
    assert a.degenerate is True
    assert a.influence is None
    assert a.ci_lower is None and a.ci_upper is None
    assert a.degenerate_reason is not None
    assert "same-slot" in a.degenerate_reason or "moved" in a.degenerate_reason


def test_degenerate_when_agent_never_stays_put():
    # Every variant has A in a distinct slot, and every variant has B
    # in a distinct slot → no same-slot pairs for either.
    score = np.array([[0.9, 0.85], [0.1, 0.05], [0.5, 0.55]])
    orderings = [("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B")]
    out = compute_influence(score, orderings, n_bootstrap=0)
    for r in out.results:
        if r.node_id in {"A", "B", "C"}:
            # All three rotate across all three slots → no same-slot pairs.
            assert r.degenerate is True


def test_degenerate_when_agent_in_too_few_variants():
    """An agent appearing in <2 variants cannot have any pair."""
    score = np.array([[0.9, 0.1], [0.1, 0.9]])
    orderings = [("A", "B"), ("C", "D")]
    out = compute_influence(score, orderings, n_bootstrap=0)
    # Every node only appears once → every row is degenerate.
    for r in out.results:
        assert r.degenerate is True


def test_nan_cells_excluded_from_counts():
    score = np.array(
        [
            [0.9, 0.85, np.nan],
            [0.1, 0.05, 0.12],
        ]
    )
    orderings = [("A", "B"), ("B", "A")]
    out = compute_influence(score, orderings, n_bootstrap=0)
    a = next(r for r in out.results if r.node_id == "A")
    # 2 evals contribute (the NaN eval is excluded). One pair (0,1)
    # is moved for A. Same-slot pairs: zero. → degenerate.
    assert a.degenerate is True


def test_bootstrap_ci_brackets_point():
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, seed=42, n_bootstrap=500)
    a = next(r for r in out.results if r.node_id == "A")
    assert a.influence is not None
    assert a.ci_lower is not None and a.ci_upper is not None
    assert a.ci_lower <= a.influence <= a.ci_upper


def test_ranking_stable_across_reseeds_for_dominant_signal():
    """§6a property: 'statistically clean vs. tied'. When the signal is
    well-separated the top-ranked agent should be the same across reseeds."""
    score, orderings = _dominant_fixture()
    rankings = []
    for seed in (0, 1, 2, 3, 4):
        out = compute_influence(score, orderings, seed=seed, n_bootstrap=300)
        rankings.append(out.results[0].node_id)
    # Permit ties between A and B in this synthetic but reject any other
    # agent winning.
    assert set(rankings) <= {"A", "B"}


def test_ranking_descending_by_signed_influence():
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, seed=0, n_bootstrap=100)
    non_deg = [r for r in out.results if not r.degenerate]
    deg = [r for r in out.results if r.degenerate]
    # Non-degenerate rows precede degenerate rows.
    if non_deg and deg:
        non_deg_ids = [r.node_id for r in out.results if not r.degenerate]
        first_deg = next(
            (i for i, r in enumerate(out.results) if r.degenerate), len(out.results)
        )
        assert all(
            not out.results[i].degenerate for i in range(first_deg)
        )
    # Among non-degenerate rows, influence is non-increasing.
    infls = [r.influence for r in non_deg]
    assert all(
        a >= b for a, b in zip(infls, infls[1:], strict=False)
    )


def test_rejects_mismatched_orderings_length():
    score = np.zeros((3, 4))
    orderings = [("A", "B")]
    with pytest.raises(ValueError, match="variant_orderings"):
        compute_influence(score, orderings)


def test_rejects_non_2d_matrix():
    with pytest.raises(ValueError, match="2D"):
        compute_influence(np.array([1.0, 2.0, 3.0]), [("A",)])


def test_rejects_bad_conf():
    score = np.zeros((2, 2))
    orderings = [("A",), ("A",)]
    with pytest.raises(ValueError, match="conf"):
        compute_influence(score, orderings, conf=1.5)


def test_rejects_negative_bootstrap():
    score = np.zeros((2, 2))
    orderings = [("A",), ("A",)]
    with pytest.raises(ValueError, match="n_bootstrap"):
        compute_influence(score, orderings, n_bootstrap=-1)


def test_rejects_non_finite_threshold():
    score = np.zeros((2, 2))
    orderings = [("A",), ("A",)]
    with pytest.raises(ValueError, match="flip_threshold"):
        compute_influence(score, orderings, flip_threshold=float("nan"))


def test_zero_bootstrap_returns_point_only():
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, n_bootstrap=0)
    a = next(r for r in out.results if r.node_id == "A")
    assert a.influence is not None
    assert a.ci_lower is None and a.ci_upper is None
    assert a.degenerate is False


def test_empty_matrix_degenerate_for_every_row():
    score = np.empty((0, 0))
    out = compute_influence(score, [], n_bootstrap=0)
    assert out.results == ()


def test_results_immutable():
    score, orderings = _dominant_fixture()
    out = compute_influence(score, orderings, seed=0, n_bootstrap=10)
    # Tuple + frozen dataclass — both immutable.
    assert isinstance(out.results, tuple)
    with pytest.raises((AttributeError, TypeError)):
        out.results[0].influence = 0.0  # type: ignore[misc]


def test_rejects_duplicate_node_id_within_one_variant_ordering():
    """Audit CR-m1: duplicate node_id in a single variant ordering
    would silently overwrite slot_map. Hard error at construction."""
    score = np.zeros((2, 2))
    orderings = [("A", "A", "B"), ("B", "A", "A")]
    with pytest.raises(ValueError, match="duplicate node_ids"):
        compute_influence(score, orderings, n_bootstrap=0)


# ---------------------------------------------------------------------------
# Step 15 audit fix: align_orderings_to_matrix strictness
# ---------------------------------------------------------------------------


def _manifest_with(hashes, orderings):
    return {
        "variant_canonical_hashes": list(hashes),
        "variant_orderings": [list(o) for o in orderings],
    }


def test_align_returns_in_axis_order_for_lex_sorted_hashes():
    manifest = _manifest_with(
        ["aaa", "bbb", "ccc"],
        [["A", "B"], ["B", "A"], ["A", "C"]],
    )
    aligned = align_orderings_to_matrix(manifest, ("aaa", "bbb", "ccc"))
    assert aligned == [["A", "B"], ["B", "A"], ["A", "C"]]


def test_align_realigns_when_axis_differs_from_insertion_order():
    """Audit CR-M1: cover the lex-reorder branch — score matrix axis
    is in canonical_hash lex order, manifest is in insertion order.
    When they disagree the helper MUST reattach each ordering to the
    correct row."""
    manifest = _manifest_with(
        # Insertion order: c, a, b. Lex order: a, b, c.
        ["ccc", "aaa", "bbb"],
        [["X", "Y"], ["A", "B"], ["B", "A"]],
    )
    aligned = align_orderings_to_matrix(manifest, ("aaa", "bbb", "ccc"))
    assert aligned == [["A", "B"], ["B", "A"], ["X", "Y"]]


def test_align_raises_on_missing_orderings():
    manifest = {"variant_canonical_hashes": ["a", "b"]}
    with pytest.raises(OrderingAlignmentError) as exc:
        align_orderings_to_matrix(manifest, ("a", "b"))
    assert exc.value.reason == "missing"


@pytest.mark.parametrize("bad_manifest", [None, [], "manifest", 42, object()])
def test_align_raises_on_non_dict_manifest(bad_manifest):
    """Post-step-15 cross-check: a non-dict manifest used to crash with
    AttributeError, which the silent-degrade callers (attest, report)
    don't catch. Must surface as OrderingAlignmentError."""
    with pytest.raises(OrderingAlignmentError) as exc:
        align_orderings_to_matrix(bad_manifest, ("a",))
    assert exc.value.reason == "malformed"


def test_align_raises_on_malformed_orderings():
    manifest = _manifest_with(["a", "b", "c"], [["A"], ["B"]])  # length mismatch
    with pytest.raises(OrderingAlignmentError) as exc:
        align_orderings_to_matrix(manifest, ("a", "b", "c"))
    assert exc.value.reason == "malformed"


def test_align_raises_on_duplicate_hashes():
    """Audit RT-B-1: dict overwrite on duplicate hashes would silently
    pick one ordering and sign the result. Must raise."""
    manifest = _manifest_with(
        ["aaa", "aaa", "bbb"],
        [["A", "B"], ["B", "A"], ["C", "D"]],
    )
    with pytest.raises(OrderingAlignmentError) as exc:
        align_orderings_to_matrix(manifest, ("aaa", "bbb"))
    assert exc.value.reason == "duplicate_hashes"


def test_align_raises_on_variant_count_mismatch():
    """Audit RT-B-2: matrix with fewer variants than manifest (partial
    resume / aborted mid-run) would silently drop agents."""
    manifest = _manifest_with(
        ["aaa", "bbb", "ccc"],
        [["A", "B"], ["B", "A"], ["A", "C"]],
    )
    with pytest.raises(OrderingAlignmentError) as exc:
        # Only 2 of 3 variants made it to the score matrix.
        align_orderings_to_matrix(manifest, ("aaa", "bbb"))
    assert exc.value.reason == "variant_count_mismatch"


def test_align_raises_on_unknown_hash_in_matrix_axis():
    """Audit RT-M-2: matrix references a variant the manifest never
    claimed. Corruption signal — refuse, don't silently degrade."""
    manifest = _manifest_with(["aaa", "bbb"], [["A"], ["B"]])
    with pytest.raises(OrderingAlignmentError) as exc:
        align_orderings_to_matrix(manifest, ("aaa", "ZZZ"))
    assert exc.value.reason == "unknown_hash"


def test_align_raises_on_non_string_ordering_entry():
    manifest = _manifest_with(["aaa", "bbb"], [["A", 7], ["B", "C"]])
    with pytest.raises(OrderingAlignmentError) as exc:
        align_orderings_to_matrix(manifest, ("aaa", "bbb"))
    assert exc.value.reason == "malformed"


def test_compute_influence_consumes_aligned_orderings_correctly():
    """Round-trip: align then compute — values match the same fixture
    fed to compute_influence directly. Cross-check on the helper."""
    score, orderings = _dominant_fixture()
    manifest = _manifest_with(
        ["zz", "yy", "xx", "ww"],
        orderings,
    )
    aligned = align_orderings_to_matrix(
        manifest, ("ww", "xx", "yy", "zz")
    )
    # axis order ww,xx,yy,zz maps back to insertion-order [3,2,1,0]
    # → expected orderings list, original orderings reversed.
    expected = [list(o) for o in reversed(orderings)]
    assert aligned == expected
    # The score matrix would also be in axis order, so reverse the rows
    # to match.
    reordered_score = score[::-1]
    out = compute_influence(reordered_score, aligned, seed=0, n_bootstrap=0)
    a = next(r for r in out.results if r.node_id == "A")
    assert a.influence is not None
    assert math.isclose(a.influence, 0.8, abs_tol=1e-12)
