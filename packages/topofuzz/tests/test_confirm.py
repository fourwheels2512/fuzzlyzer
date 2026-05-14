"""Tests for the noise-floor / verdict-confirmation feature.

Covers:

* ``flip_confirmations`` round-trip (RunWriter.write_flip_confirmation +
  read_flip_confirmations; the PK allows K repeats per cell).
* :func:`topofuzz.repro.find_all_flip_candidates` enumeration.
* :func:`topofuzz.confirm.run_confirm_pass` with a mocked LLM —
  candidates + reference re-run K times, rows land, cost-cap hard-stop.
* :func:`topofuzz.metrics.compute_verdict_stability`.
* :func:`topofuzz.metrics.compute_confirmed_verdict_flip_rate`.
* :func:`topofuzz.metrics.compute_voo_noise_floor` + ``voo_gate_passes``.

The LLM is mocked exactly the way ``test_runner.py`` mocks it — an async
callable returning a LiteLLM-shaped dict. No paid calls.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("diskcache")  # RunWriter path imports fuzzeval.cache
pytest.importorskip("numpy")

from fuzzeval.judge import ContinuousScore  # noqa: E402

from topofuzz.confirm import (  # noqa: E402
    ConfirmError,
    read_flip_confirmations,
    run_confirm_pass,
)
from topofuzz.ir import Edge, LLMConfig, Node, Topology  # noqa: E402
from topofuzz.metrics import (  # noqa: E402
    compute_confirmed_verdict_flip_rate,
    compute_verdict_stability,
    compute_voo_noise_floor,
    voo_gate_passes,
)
from topofuzz.repro import find_all_flip_candidates  # noqa: E402
from topofuzz.runner import Cell, NodeOutput, RunWriter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_topology(model: str = "google/gemini-2.5-flash") -> Topology:
    cfg = LLMConfig(model=model, temperature=0.0)
    nodes = (
        Node(id="n0", role="r0", llm_config=cfg),
        Node(id="n1", role="r1", llm_config=cfg),
    )
    edges = (Edge(from_id="n0", to_id="n1"),)
    return Topology(
        name="toy",
        nodes=nodes,
        edges=edges,
        entrypoint="n0",
        terminal=("n1",),
    )


def _make_cell(
    variant_id: str,
    eval_id: str,
    *,
    score: float | None = 0.5,
    judge_failure: bool = False,
    error: str | None = None,
) -> Cell:
    """A minimal Cell with one NodeOutput and a ContinuousScore judge result."""
    node = NodeOutput(
        node_id="n0",
        role="r0",
        response_text="x",
        model_served_version="google/gemini-2.5-flash",
        cache_hit=False,
        latency_ms=1,
    )
    if judge_failure:
        judge_result = None
    elif score is None:
        judge_result = None
    else:
        judge_result = ContinuousScore(value=score)
    return Cell(
        variant_id=variant_id,
        eval_id=eval_id,
        ir_canonical_hash=variant_id,
        node_outputs=(node,),
        judge_result=judge_result,
        judge_failure=judge_failure,
        judge_failure_reason="boom" if judge_failure else None,
        error=error,
    )


@dataclass
class FakeLLM:
    """Async LLM fake — same shape as test_runner.FakeLLM.

    ``scores_by_model`` is unused here; the fake just echoes a fixed
    response. Tests that need verdict control use synthetic
    flip_confirmations rows directly rather than driving the judge.
    """

    calls: list[dict] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "model": kwargs["model"],
        }


# ---------------------------------------------------------------------------
# flip_confirmations round-trip
# ---------------------------------------------------------------------------


def test_flip_confirmations_round_trip_pk_allows_repeats(tmp_path):
    """K repeats of one (variant, eval) cell must all persist — the PK
    includes repeat_idx, so they don't collide."""
    topo = _toy_topology()
    run_dir = tmp_path / "run-a"
    with RunWriter(run_dir, "run-a", topo) as w:
        for ridx in range(3):
            cell = _make_cell("v" * 64, "eval_001", score=0.1 + ridx * 0.1)
            w.write_flip_confirmation(cell, repeat_idx=ridx, role="reference")
        # A candidate repeat under a different variant_id, same eval.
        w.write_flip_confirmation(
            _make_cell("c" * 64, "eval_001", score=0.9),
            repeat_idx=0,
            role="candidate",
        )

    rows = read_flip_confirmations(run_dir, run_id="run-a")
    assert len(rows) == 4
    ref = [r for r in rows if r["role"] == "reference"]
    assert len(ref) == 3
    assert sorted(r["repeat_idx"] for r in ref) == [0, 1, 2]
    cand = [r for r in rows if r["role"] == "candidate"]
    assert len(cand) == 1 and cand[0]["variant_id"] == "c" * 64
    # score_value round-trips; judge_kind is 'continuous'.
    assert all(r["judge_kind"] == "continuous" for r in rows)
    assert ref[0]["score_value"] == pytest.approx(0.1)


def test_flip_confirmations_role_filter(tmp_path):
    topo = _toy_topology()
    run_dir = tmp_path / "run-b"
    with RunWriter(run_dir, "run-b", topo) as w:
        w.write_flip_confirmation(
            _make_cell("v" * 64, "e1", score=0.5), repeat_idx=0, role="reference"
        )
        w.write_flip_confirmation(
            _make_cell("c" * 64, "e1", score=0.5), repeat_idx=0, role="candidate"
        )
    assert len(read_flip_confirmations(run_dir, role="reference")) == 1
    assert len(read_flip_confirmations(run_dir, role="candidate")) == 1
    assert len(read_flip_confirmations(run_dir)) == 2


def test_write_flip_confirmation_rejects_bad_role(tmp_path):
    topo = _toy_topology()
    with RunWriter(tmp_path / "run-c", "run-c", topo) as w:
        with pytest.raises(ValueError, match="role must be"):
            w.write_flip_confirmation(
                _make_cell("v" * 64, "e1"), repeat_idx=0, role="bogus"
            )


def test_write_flip_confirmation_persists_judge_failure(tmp_path):
    """A judge-failure repeat persists with score_value NULL + a
    failure_reason — the metrics drop it as missing data."""
    topo = _toy_topology()
    run_dir = tmp_path / "run-d"
    with RunWriter(run_dir, "run-d", topo) as w:
        w.write_flip_confirmation(
            _make_cell("v" * 64, "e1", judge_failure=True),
            repeat_idx=0,
            role="reference",
        )
    rows = read_flip_confirmations(run_dir)
    assert len(rows) == 1
    assert rows[0]["score_value"] is None
    assert rows[0]["judge_kind"] == "failure"
    assert rows[0]["failure_reason"] == "boom"


def test_read_flip_confirmations_missing_db_raises(tmp_path):
    with pytest.raises(ConfirmError, match="run.sqlite not found"):
        read_flip_confirmations(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# find_all_flip_candidates
# ---------------------------------------------------------------------------


def _write_run_for_candidates(
    tmp_path: Path, judge_rows: list[tuple], *, hashes: list[str]
) -> Path:
    """Build a minimal run_dir: manifest with variant hashes + a
    judge_results table populated from ``judge_rows`` (run_id, variant_id,
    eval_id, judge_kind, score_value, passed, rationale, failure_reason)."""
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()
    manifest = {
        "run_id": "synth-0001",
        "variant_canonical_hashes": hashes,
        "variant_orderings": [["n0", "n1"]] * len(hashes),
        "eval_ids": ["eval_001", "eval_002"],
        "cli_args": {"fixture": "x.py:build_graph", "eval": "e.csv", "rng_seed": 0},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, failure_reason TEXT)"
    )
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", judge_rows
    )
    db.commit()
    db.close()
    return run_dir


def test_find_all_flip_candidates_enumerates_correctly(tmp_path):
    """Original (idx 0) APPROVE on eval_001; variant b DENY → candidate.
    Variant b APPROVE on eval_002 same as original → not a candidate."""
    A, B, C = "a" * 64, "b" * 64, "c" * 64
    rows = [
        # eval_001: original=0.9 (APPROVE), b=0.1 (DENY → flip), c=0.8 (no flip)
        ("synth-0001", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_001", "continuous", 0.1, None, None, None),
        ("synth-0001", C, "eval_001", "continuous", 0.8, None, None, None),
        # eval_002: original=0.2 (DENY), b=0.2 (DENY no flip), c=0.7 (APPROVE flip)
        ("synth-0001", A, "eval_002", "continuous", 0.2, None, None, None),
        ("synth-0001", B, "eval_002", "continuous", 0.2, None, None, None),
        ("synth-0001", C, "eval_002", "continuous", 0.7, None, None, None),
    ]
    run_dir = _write_run_for_candidates(tmp_path, rows, hashes=[A, B, C])
    candidates = find_all_flip_candidates(run_dir)
    # Sorted lexicographically by (eval_id, variant_id).
    assert candidates == [("eval_001", B), ("eval_002", C)]


def test_find_all_flip_candidates_skips_judge_failures(tmp_path):
    """A judge-failure cell (NULL score + NULL passed) contributes no
    verdict and is never a candidate; an eval whose ORIGINAL cell is a
    judge failure is skipped entirely (no reference frame)."""
    A, B = "a" * 64, "b" * 64
    rows = [
        # eval_001: original judge-failed → whole eval skipped
        ("synth-0001", A, "eval_001", "failure", None, None, None, "boom"),
        ("synth-0001", B, "eval_001", "continuous", 0.1, None, None, None),
        # eval_002: original APPROVE, b judge-failed → b not a candidate
        ("synth-0001", A, "eval_002", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_002", "failure", None, None, None, "boom"),
    ]
    run_dir = _write_run_for_candidates(tmp_path, rows, hashes=[A, B])
    assert find_all_flip_candidates(run_dir) == []


def test_find_all_flip_candidates_no_flip_returns_empty(tmp_path):
    """A non-fragile run legitimately has zero candidates — NOT an error."""
    A, B = "a" * 64, "b" * 64
    rows = [
        ("synth-0001", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_001", "continuous", 0.8, None, None, None),
    ]
    run_dir = _write_run_for_candidates(tmp_path, rows, hashes=[A, B])
    assert find_all_flip_candidates(run_dir) == []


# ---------------------------------------------------------------------------
# run_confirm_pass — full pass with a mocked LLM
# ---------------------------------------------------------------------------


_FIXTURE_SRC = """\
from langgraph.graph import END, START, StateGraph
from typing import TypedDict
from topofuzz.ir import LLMConfig


class _S(TypedDict, total=False):
    x: str


def _a(s):
    return s


def build_graph():
    g = StateGraph(_S)
    g.add_node('n0', _a)
    g.add_node('n1', _a)
    g.add_edge(START, 'n0')
    g.add_edge('n0', 'n1')
    g.add_edge('n1', END)
    return g.compile()


LLM_CONFIGS = {
    'n0': LLMConfig(model='google/gemini-2.5-flash', temperature=0.0),
    'n1': LLMConfig(model='google/gemini-2.5-flash', temperature=0.0),
}
"""


def _build_real_run_dir(tmp_path: Path) -> tuple[Path, list[str]]:
    """Write a langgraph fixture, regenerate the variant set the same way
    ``topofuzz run`` does, and lay down a manifest + run.sqlite whose
    judge_results make at least one variant a flip candidate.

    Returns ``(run_dir, variant_hashes)``.
    """
    pytest.importorskip("langgraph")
    from topofuzz.cli import _round_robin_select
    from topofuzz.loaders import load_topology
    from topofuzz.variants.generators import (
        decoding_param_swap,
        order_swap,
        role_swap,
    )

    fixture = tmp_path / "fx.py"
    fixture.write_text(_FIXTURE_SRC, encoding="utf-8")
    eval_file = tmp_path / "e.csv"
    eval_file.write_text(
        "id,input\neval_001,q1\neval_002,q2\n", encoding="utf-8"
    )

    topology = load_topology(f"{fixture}:build_graph")
    ord_list = list(order_swap(topology, rng_seed=0))
    role_list = list(role_swap(topology))
    dec_list = list(decoding_param_swap(topology))
    variant_list = _round_robin_select(
        original=ord_list[0],
        family_pools=(ord_list[1:], role_list, dec_list),
        total=4,
    )
    hashes = [v.canonical_hash() for v in variant_list]
    assert len(hashes) >= 2, "fixture must yield >= 2 variants for a flip"

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "real-run-0001",
        "topology_canonical_hash": topology.canonical_hash(),
        "topology_name": topology.name,
        "tds_version": topology.tds_version,
        "variant_canonical_hashes": hashes,
        "variant_orderings": [[n.id for n in v.nodes] for v in variant_list],
        "eval_ids": ["eval_001", "eval_002"],
        "cli_args": {
            "fixture": f"{fixture}:build_graph",
            "eval": str(eval_file),
            "rng_seed": 0,
            "variants": 4,
            "judge": "google/gemini-2.5-flash",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    # judge_results: original (hashes[0]) APPROVE both evals; variant
    # hashes[1] DENYs eval_001 → one flip candidate.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, failure_reason TEXT)"
    )
    jrows = []
    for eid in ("eval_001", "eval_002"):
        for i, h in enumerate(hashes):
            score = 0.9
            if h == hashes[1] and eid == "eval_001":
                score = 0.1  # the flip
            jrows.append(
                ("real-run-0001", h, eid, "continuous", score, None, None, None)
            )
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", jrows
    )
    db.commit()
    db.close()
    return run_dir, hashes


def test_run_confirm_pass_reruns_candidates_and_reference(tmp_path):
    """Confirm pass re-runs the original variant K× per candidate eval
    and each candidate (variant, eval) K×, persisting every repeat."""
    run_dir, hashes = _build_real_run_dir(tmp_path)
    fake = FakeLLM()
    summary = run_confirm_pass(
        run_dir, repeats=3, llm_callable=fake
    )
    # Exactly one flip candidate (variant hashes[1] on eval_001).
    assert summary.n_candidates == 1
    assert summary.n_candidate_variants == 1
    assert summary.repeats == 3
    # Reference: original × 3 repeats × 1 candidate eval = 3.
    assert summary.reference_repeats_run == 3
    # Candidate: 1 (variant, eval) × 3 repeats = 3.
    assert summary.candidate_repeats_run == 3
    assert not summary.partial

    rows = read_flip_confirmations(run_dir, run_id="real-run-0001")
    assert len(rows) == 6
    ref = [r for r in rows if r["role"] == "reference"]
    cand = [r for r in rows if r["role"] == "candidate"]
    assert len(ref) == 3 and len(cand) == 3
    # All reference rows are the original variant on eval_001.
    assert all(r["variant_id"] == hashes[0] for r in ref)
    assert all(r["eval_id"] == "eval_001" for r in ref)
    # All candidate rows are variant hashes[1] on eval_001.
    assert all(r["variant_id"] == hashes[1] for r in cand)
    # The mocked LLM was actually invoked (2 nodes × 6 cells = 12 calls).
    assert len(fake.calls) == 12


def test_run_confirm_pass_max_cost_hard_stop(tmp_path, monkeypatch):
    """--max-cost-usd hard-stop persists what completed and flags partial.

    The mocked LLM has no cost telemetry, so we monkeypatch
    ``_sum_cell_cost`` to bill a fixed amount per cell, then set a cap
    that trips after the first repeat."""
    run_dir, hashes = _build_real_run_dir(tmp_path)
    fake = FakeLLM()

    import topofuzz.confirm as confirm_mod

    monkeypatch.setattr(confirm_mod, "_sum_cell_cost", lambda cell: 1.0)
    # Cap = 1.5 → first repeat bills 1.0 (under), second would hit 2.0;
    # the cap check fires BEFORE the 2nd repoeat runs → 1 repeat total.
    summary = run_confirm_pass(
        run_dir, repeats=3, max_cost_usd=1.5, llm_callable=fake
    )
    assert summary.partial is True
    # Only reference repeats had a chance to run before the cap tripped.
    total_run = summary.reference_repeats_run + summary.candidate_repeats_run
    assert total_run < 6  # fewer than the full 3 ref + 3 cand
    assert total_run >= 1  # at least one completed and persisted
    rows = read_flip_confirmations(run_dir, run_id="real-run-0001")
    assert len(rows) == total_run


def test_run_confirm_pass_missing_manifest_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfirmError, match="manifest.json missing"):
        run_confirm_pass(empty)


def test_run_confirm_pass_rejects_bad_repeats(tmp_path):
    run_dir, _ = _build_real_run_dir(tmp_path)
    with pytest.raises(ValueError, match="repeats must be positive"):
        run_confirm_pass(run_dir, repeats=0)


# ---------------------------------------------------------------------------
# verdict_stability
# ---------------------------------------------------------------------------


def test_verdict_stability_unanimous_and_split():
    """eval_001: 3 unanimous APPROVE repeats. eval_002: 2 APPROVE + 1
    DENY → split. Stability = 1 unanimous / 2 evals = 0.5."""
    ref = [
        {"eval_id": "eval_001", "score_value": 0.9},
        {"eval_id": "eval_001", "score_value": 0.8},
        {"eval_id": "eval_001", "score_value": 0.95},
        {"eval_id": "eval_002", "score_value": 0.9},
        {"eval_id": "eval_002", "score_value": 0.7},
        {"eval_id": "eval_002", "score_value": 0.1},
    ]
    res = compute_verdict_stability(ref)
    assert res.n_evals == 2
    assert res.n_unanimous == 1
    assert res.point == pytest.approx(0.5)
    assert res.degenerate_flag is False


def test_verdict_stability_fully_stable():
    ref = [
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.1},
        {"eval_id": "e2", "score_value": 0.2},
    ]
    res = compute_verdict_stability(ref)
    assert res.point == pytest.approx(1.0)
    assert res.n_unanimous == 2


def test_verdict_stability_drops_judge_failures():
    """A NULL-score repeat is missing data — dropped from num + denom."""
    ref = [
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": None},  # judge failure
        {"eval_id": "e1", "score_value": 0.8},
    ]
    res = compute_verdict_stability(ref)
    # e1 still counts (2 scored repeats, both APPROVE → unanimous).
    assert res.n_evals == 1 and res.n_unanimous == 1
    assert res.point == pytest.approx(1.0)


def test_verdict_stability_degenerate_when_no_rows():
    res = compute_verdict_stability([])
    assert res.point is None
    assert res.degenerate_flag is True
    assert res.n_evals == 0


# ---------------------------------------------------------------------------
# confirmed_verdict_flip_rate
# ---------------------------------------------------------------------------


def test_confirmed_verdict_flip_confirmed_and_flaky():
    """Two candidates on two evals with stable baselines:
    - eval_001 / variant V: reference majority APPROVE, candidate
      majority DENY → CONFIRMED flip.
    - eval_002 / variant V: reference majority APPROVE, candidate
      majority APPROVE → FLAKY (screen flagged it, repeats don't back it).
    """
    V = "v" * 64
    ref = [
        # eval_001 reference: 3× APPROVE → stable APPROVE
        {"eval_id": "eval_001", "score_value": 0.9},
        {"eval_id": "eval_001", "score_value": 0.8},
        {"eval_id": "eval_001", "score_value": 0.85},
        # eval_002 reference: 3× APPROVE → stable APPROVE
        {"eval_id": "eval_002", "score_value": 0.9},
        {"eval_id": "eval_002", "score_value": 0.9},
        {"eval_id": "eval_002", "score_value": 0.7},
    ]
    cand = [
        # eval_001 / V: 3× DENY → majority DENY != APPROVE → CONFIRMED
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.2},
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.05},
        # eval_002 / V: 3× APPROVE → majority APPROVE == APPROVE → FLAKY
        {"eval_id": "eval_002", "variant_id": V, "score_value": 0.9},
        {"eval_id": "eval_002", "variant_id": V, "score_value": 0.8},
        {"eval_id": "eval_002", "variant_id": V, "score_value": 0.6},
    ]
    res = compute_confirmed_verdict_flip_rate(ref, cand, seed=0)
    assert res.confirmed == 1
    assert res.flaky == 1
    assert res.unstable_baseline == 0
    assert res.point == pytest.approx(0.5)  # 1 confirmed / 2 stable pairs
    # CI present (2 eval clusters).
    assert res.ci_lower is not None and res.ci_upper is not None
    assert 0.0 <= res.ci_lower <= res.point <= res.ci_upper <= 1.0
    assert res.degenerate_flag is False


def test_confirmed_verdict_flip_unstable_baseline_excluded():
    """A tied reference (1 APPROVE + 1 DENY) → unstable_baseline; its
    candidate is excluded from the numerator AND denominator."""
    V = "v" * 64
    ref = [
        # eval_001 reference: tie → unstable baseline
        {"eval_id": "eval_001", "score_value": 0.9},
        {"eval_id": "eval_001", "score_value": 0.1},
        # eval_002 reference: stable APPROVE
        {"eval_id": "eval_002", "score_value": 0.9},
        {"eval_id": "eval_002", "score_value": 0.8},
        {"eval_id": "eval_002", "score_value": 0.85},
    ]
    cand = [
        # eval_001 / V — baseline tied, excluded
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
        # eval_002 / V — stable baseline, candidate majority DENY → confirmed
        {"eval_id": "eval_002", "variant_id": V, "score_value": 0.1},
        {"eval_id": "eval_002", "variant_id": V, "score_value": 0.2},
        {"eval_id": "eval_002", "variant_id": V, "score_value": 0.05},
    ]
    res = compute_confirmed_verdict_flip_rate(ref, cand, seed=0)
    assert res.unstable_baseline == 1
    assert res.confirmed == 1
    assert res.flaky == 0
    # Denominator = confirmed + flaky = 1 (the tied-baseline pair excluded).
    assert res.point == pytest.approx(1.0)


def test_confirmed_verdict_flip_degenerate_when_no_stable_pairs():
    """Every candidate sits on a tied baseline → degenerate, point None."""
    V = "v" * 64
    ref = [
        {"eval_id": "eval_001", "score_value": 0.9},
        {"eval_id": "eval_001", "score_value": 0.1},
    ]
    cand = [
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(ref, cand, seed=0)
    assert res.point is None
    assert res.degenerate_flag is True
    assert res.unstable_baseline == 1
    assert res.confirmed == 0 and res.flaky == 0


def test_confirmed_verdict_flip_single_cluster_no_ci():
    """One stable-baseline eval cluster → point computable, CI is None."""
    V = "v" * 64
    ref = [
        {"eval_id": "eval_001", "score_value": 0.9},
        {"eval_id": "eval_001", "score_value": 0.9},
        {"eval_id": "eval_001", "score_value": 0.8},
    ]
    cand = [
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.2},
        {"eval_id": "eval_001", "variant_id": V, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(ref, cand, seed=0)
    assert res.point == pytest.approx(1.0)
    assert res.ci_lower is None and res.ci_upper is None
    assert res.degenerate_flag is True


def test_confirmed_verdict_flip_candidate_tie_is_flaky():
    """A candidate whose own repeats tie has no strict majority → it
    cannot confirm the screen flip → counted flaky, not confirmed."""
    V = "v" * 64
    ref = [
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.9},
    ]
    cand = [
        # e1 / V: 1 APPROVE + 1 DENY → tie → flaky
        {"eval_id": "e1", "variant_id": V, "score_value": 0.9},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        # e2 / V: 2 DENY → confirmed
        {"eval_id": "e2", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e2", "variant_id": V, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(ref, cand, seed=0)
    assert res.confirmed == 1
    assert res.flaky == 1
    assert res.point == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# voo_noise_floor + gate
# ---------------------------------------------------------------------------


def test_voo_noise_floor_zero_when_repeats_identical():
    """K identical repeats → zero variance → noise floor 0.0."""
    ref = [
        {"eval_id": "e1", "repeat_idx": 0, "score_value": 0.5},
        {"eval_id": "e1", "repeat_idx": 1, "score_value": 0.5},
        {"eval_id": "e2", "repeat_idx": 0, "score_value": 0.8},
        {"eval_id": "e2", "repeat_idx": 1, "score_value": 0.8},
    ]
    floor = compute_voo_noise_floor(ref)
    assert floor == pytest.approx(0.0)


def test_voo_noise_floor_positive_when_repeats_disagree():
    """Noisy repeats → positive noise floor (this is the whole point —
    the pipeline is non-deterministic on the SAME ordering)."""
    ref = [
        {"eval_id": "e1", "repeat_idx": 0, "score_value": 0.0},
        {"eval_id": "e1", "repeat_idx": 1, "score_value": 1.0},
        {"eval_id": "e2", "repeat_idx": 0, "score_value": 0.0},
        {"eval_id": "e2", "repeat_idx": 1, "score_value": 1.0},
    ]
    floor = compute_voo_noise_floor(ref)
    # var([0,1], ddof=0) = 0.25 for each eval; mean = 0.25.
    assert floor == pytest.approx(0.25)


def test_voo_noise_floor_none_on_ragged_block():
    """A judge-failure repeat makes the K×n_evals block incomplete →
    None (the floor is undefined over a ragged matrix)."""
    ref = [
        {"eval_id": "e1", "repeat_idx": 0, "score_value": 0.5},
        {"eval_id": "e1", "repeat_idx": 1, "score_value": None},  # failure
        {"eval_id": "e2", "repeat_idx": 0, "score_value": 0.5},
        {"eval_id": "e2", "repeat_idx": 1, "score_value": 0.5},
    ]
    assert compute_voo_noise_floor(ref) is None


def test_voo_noise_floor_none_when_under_two_repeats():
    ref = [{"eval_id": "e1", "repeat_idx": 0, "score_value": 0.5}]
    assert compute_voo_noise_floor(ref) is None
    assert compute_voo_noise_floor([]) is None


def test_voo_gate_passes_against_noise_floor():
    """The §22 gate: voo_ci_lower must clear the noise floor, not 0."""
    # ci_lower 0.10, floor 0.25 → does NOT clear the floor.
    assert voo_gate_passes(0.10, 0.25) is False
    # ci_lower 0.30, floor 0.25 → clears it.
    assert voo_gate_passes(0.30, 0.25) is True
    # No noise floor (no confirm pass) → falls back to > 0.
    assert voo_gate_passes(0.10, None) is True
    assert voo_gate_passes(0.0, None) is False
    # No voo CI at all → nothing to assert → False.
    assert voo_gate_passes(None, 0.25) is False
    assert voo_gate_passes(None, None) is False


# ---------------------------------------------------------------------------
# CLI: topofuzz confirm
# ---------------------------------------------------------------------------


def test_cli_confirm_bad_run_dir_exits_friendly():
    from typer.testing import CliRunner

    from topofuzz.cli import app

    result = CliRunner().invoke(app, ["confirm", "nonexistent-dir", "--yes"])
    assert result.exit_code == 2
    assert "not a directory" in result.output.lower()


def test_cli_confirm_invokes_run_confirm_pass(tmp_path, monkeypatch):
    """The CLI wires run_dir + --repeats into run_confirm_pass and prints
    the summary. run_confirm_pass itself is monkeypatched (it would
    otherwise bill real LLM calls — exercised directly elsewhere)."""
    from typer.testing import CliRunner

    import topofuzz.confirm as confirm_mod
    from topofuzz.cli import app

    run_dir, _ = _build_real_run_dir(tmp_path)

    captured = {}

    def fake_run_confirm_pass(rd, *, repeats, max_cost_usd, judge_model):
        captured["run_dir"] = Path(rd)
        captured["repeats"] = repeats
        captured["max_cost_usd"] = max_cost_usd
        return confirm_mod.ConfirmSummary(
            n_candidates=2,
            n_candidate_variants=1,
            repeats=repeats,
            reference_repeats_run=6,
            candidate_repeats_run=6,
            total_cost_usd=0.0,
            partial=False,
        )

    monkeypatch.setattr(confirm_mod, "run_confirm_pass", fake_run_confirm_pass)

    result = CliRunner().invoke(
        app, ["confirm", str(run_dir), "--repeats", "5", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert captured["run_dir"] == run_dir
    assert captured["repeats"] == 5
    assert "Candidates found: 2" in result.output
    assert "K=5" in result.output


def test_cli_confirm_partial_results_message(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import topofuzz.confirm as confirm_mod
    from topofuzz.cli import app

    run_dir, _ = _build_real_run_dir(tmp_path)

    def fake_run_confirm_pass(rd, *, repeats, max_cost_usd, judge_model):
        return confirm_mod.ConfirmSummary(
            n_candidates=1,
            n_candidate_variants=1,
            repeats=repeats,
            reference_repeats_run=1,
            candidate_repeats_run=0,
            total_cost_usd=1.5,
            partial=True,
        )

    monkeypatch.setattr(confirm_mod, "run_confirm_pass", fake_run_confirm_pass)
    result = CliRunner().invoke(
        app, ["confirm", str(run_dir), "--max-cost-usd", "1.5", "--yes"]
    )
    assert result.exit_code == 0
    assert "partial" in result.output.lower()


# ---------------------------------------------------------------------------
# Attestation wire-in
# ---------------------------------------------------------------------------


def _write_confirm_sidecar(
    run_dir: Path,
    *,
    repeats: int = 3,
    partial: bool = False,
    n_unmatched: int = 0,
    n_candidates: int = 1,
) -> None:
    """Lay down a minimal confirm.json sidecar for the attest-helper tests.

    The attest path reads confirm.json (B-RT-1 spec-owner decision) to
    decide whether the noise-floor headline is populatable.
    """
    payload = {
        "repeats": repeats,
        "partial": partial,
        "n_candidates": n_candidates,
        "n_candidate_variants": 1,
        "reference_repeats_run": repeats * n_candidates,
        "candidate_repeats_run": repeats * n_candidates,
        "n_unmatched_candidates": n_unmatched,
        "skipped_eval_ids": [],
        "total_cost_usd": 0.0,
    }
    (run_dir / "confirm.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_compute_noise_floor_for_attestation_none_without_confirm_pass(tmp_path):
    """No confirm.json sidecar → all three metric fields None + note None
    (B-RT-1: an absent sidecar means no confirm pass was run)."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-x"
    # A RunWriter creates the flip_confirmations table but writes no rows
    # and there is no confirm.json sidecar.
    with RunWriter(run_dir, "run-x", topo):
        pass
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-x"
    )
    assert stability is None
    assert confirmed is None
    assert floor is None
    assert note is None  # no confirm pass → no note, just stable nulls


def test_compute_noise_floor_for_attestation_populated(tmp_path):
    """With a complete confirm.json sidecar + reference + candidate rows,
    the attest helper returns the three populated metrics and note=None."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-y"
    V = "v" * 64
    with RunWriter(run_dir, "run-y", topo) as w:
        # Reference: original variant, 3 repeats on 2 evals, all stable.
        for eid in ("eval_001", "eval_002"):
            for ridx, score in enumerate((0.9, 0.85, 0.95)):
                w.write_flip_confirmation(
                    _make_cell(topo.canonical_hash(), eid, score=score),
                    repeat_idx=ridx,
                    role="reference",
                )
        # Candidate: variant V on eval_001, 3 repeats all DENY → confirmed.
        for ridx, score in enumerate((0.1, 0.2, 0.05)):
            w.write_flip_confirmation(
                _make_cell(V, "eval_001", score=score),
                repeat_idx=ridx,
                role="candidate",
            )
    _write_confirm_sidecar(run_dir, repeats=3)
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-y"
    )
    assert stability == pytest.approx(1.0)  # both reference evals unanimous
    assert confirmed is not None
    assert confirmed["confirmed"] == 1
    assert confirmed["flaky"] == 0
    assert confirmed["point"] == pytest.approx(1.0)
    assert floor is not None  # reference repeats vary slightly → small floor
    assert floor == pytest.approx(0.0, abs=0.01)
    assert note is None  # complete pass → no note


def test_compute_noise_floor_for_attestation_partial_refuses(tmp_path):
    """B-RT-1(c): a confirm.json with partial:true → metrics stay None
    and a recoverable-reason note is surfaced instead of a number."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-partial"
    with RunWriter(run_dir, "run-partial", topo) as w:
        # A ragged block — eval_001 got only 1 of 3 reference repeats.
        w.write_flip_confirmation(
            _make_cell(topo.canonical_hash(), "eval_001", score=0.9),
            repeat_idx=0,
            role="reference",
        )
    _write_confirm_sidecar(run_dir, repeats=3, partial=True)
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-partial"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "partial confirm pass" in note


def test_compute_noise_floor_for_attestation_unmatched_refuses(tmp_path):
    """M-RT-2: confirm.json with n_unmatched_candidates > 0 → metrics
    stay None + a drift note (reconstruction drift = incomplete data)."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-drift"
    with RunWriter(run_dir, "run-drift", topo) as w:
        for ridx, score in enumerate((0.9, 0.85, 0.95)):
            w.write_flip_confirmation(
                _make_cell(topo.canonical_hash(), "eval_001", score=score),
                repeat_idx=ridx,
                role="reference",
            )
    _write_confirm_sidecar(run_dir, repeats=3, n_unmatched=2)
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-drift"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "could not be reconstructed" in note


def test_headline_carries_noise_floor_fields():
    """compute_headline threads the three noise-floor fields through."""
    from topofuzz.attest import compute_headline

    h = compute_headline(
        status="succeeded",
        n_successful_cells=1,
        n_total_cells=1,
        n_variants=2,
        n_evals=1,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        verdict_stability=0.74,
        confirmed_verdict_flip_rate={
            "point": 0.1,
            "ci_lower": 0.0,
            "ci_upper": 0.2,
            "confirmed": 1,
            "flaky": 9,
            "unstable_baseline": 0,
            "degenerate_flag": False,
        },
        voo_noise_floor=0.03,
    )
    assert h["verdict_stability"] == 0.74
    assert h["confirmed_verdict_flip_rate"]["confirmed"] == 1
    assert h["voo_noise_floor"] == 0.03
    # confirm_note defaults to None when not supplied.
    assert h["confirm_note"] is None


def test_headline_carries_confirm_note():
    """compute_headline threads the recoverable-reason note through."""
    from topofuzz.attest import compute_headline

    h = compute_headline(
        status="succeeded",
        n_successful_cells=1,
        n_total_cells=1,
        n_variants=2,
        n_evals=1,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        confirm_note="partial confirm pass — re-run `topofuzz confirm`",
    )
    assert h["confirm_note"].startswith("partial confirm pass")
    # When the note is set, the metric fields stay null.
    assert h["verdict_stability"] is None
    assert h["confirmed_verdict_flip_rate"] is None
    assert h["voo_noise_floor"] is None


# ---------------------------------------------------------------------------
# B-RT-1 / M-RT-1 / M-RT-3: confirm-pass robustness regressions
# ---------------------------------------------------------------------------


def test_run_confirm_pass_writes_sidecar(tmp_path):
    """B-RT-1(a): a completed confirm pass writes confirm.json recording
    repeats / partial / candidate counts / unmatched / cost."""
    from topofuzz.confirm import read_confirm_sidecar

    run_dir, hashes = _build_real_run_dir(tmp_path)
    fake = FakeLLM()
    summary = run_confirm_pass(run_dir, repeats=3, llm_callable=fake)

    sidecar = read_confirm_sidecar(run_dir)
    assert sidecar is not None
    assert sidecar["repeats"] == 3
    assert sidecar["partial"] is False
    assert sidecar["n_candidates"] == summary.n_candidates
    assert sidecar["n_candidate_variants"] == summary.n_candidate_variants
    assert sidecar["n_unmatched_candidates"] == 0
    assert sidecar["reference_repeats_run"] == summary.reference_repeats_run
    assert sidecar["candidate_repeats_run"] == summary.candidate_repeats_run
    assert "total_cost_usd" in sidecar


def test_run_confirm_pass_partial_sidecar_flags_partial(tmp_path, monkeypatch):
    """B-RT-1(a)+(c): a cost-cap hard-stop writes confirm.json with
    partial:true so the attest path refuses to populate the headline."""
    import topofuzz.confirm as confirm_mod
    from topofuzz.confirm import read_confirm_sidecar

    run_dir, hashes = _build_real_run_dir(tmp_path)
    fake = FakeLLM()
    monkeypatch.setattr(confirm_mod, "_sum_cell_cost", lambda cell: 1.0)
    summary = run_confirm_pass(
        run_dir, repeats=3, max_cost_usd=1.5, llm_callable=fake
    )
    assert summary.partial is True
    sidecar = read_confirm_sidecar(run_dir)
    assert sidecar is not None and sidecar["partial"] is True


def test_run_confirm_pass_delete_then_write_idempotent_smaller_k(tmp_path):
    """M-RT-3: confirm --repeats 2 after --repeats 3 must NOT leave a
    stale repeat_idx 2 row behind — delete-then-write is idempotent."""
    run_dir, hashes = _build_real_run_dir(tmp_path)

    run_confirm_pass(run_dir, repeats=3, llm_callable=FakeLLM())
    rows_k3 = read_flip_confirmations(run_dir, run_id="real-run-0001")
    assert max(r["repeat_idx"] for r in rows_k3) == 2

    # Re-run with a SMALLER K. Without delete-then-write the idx-2 rows
    # from the K=3 pass would survive and contaminate the majority vote.
    run_confirm_pass(run_dir, repeats=2, llm_callable=FakeLLM())
    rows_k2 = read_flip_confirmations(run_dir, run_id="real-run-0001")
    assert max(r["repeat_idx"] for r in rows_k2) == 1, (
        "stale repeat_idx 2 row survived a smaller-K re-run"
    )
    # Every (variant, eval, role) group has exactly K=2 rows now.
    from collections import Counter

    counts = Counter(
        (r["variant_id"], r["eval_id"], r["role"]) for r in rows_k2
    )
    assert all(c == 2 for c in counts.values())


def test_run_confirm_pass_skipped_eval_ids_surfaced(tmp_path):
    """N5: a candidate eval_id absent from the run's eval file is recorded
    in ConfirmSummary.skipped_eval_ids + confirm.json, not just logged."""
    from topofuzz.confirm import read_confirm_sidecar

    run_dir, hashes = _build_real_run_dir(tmp_path)
    # The flip candidate is on eval_001. Drop eval_001 from the eval file
    # so its repeats cannot run.
    eval_file = run_dir.parent / "e.csv"
    eval_file.write_text("id,input\neval_002,q2\n", encoding="utf-8")

    summary = run_confirm_pass(run_dir, repeats=3, llm_callable=FakeLLM())
    assert "eval_001" in summary.skipped_eval_ids
    sidecar = read_confirm_sidecar(run_dir)
    assert sidecar is not None
    assert "eval_001" in sidecar["skipped_eval_ids"]


def test_run_confirm_pass_missing_rng_seed_raises(tmp_path):
    """N6: a manifest without cli_args.rng_seed must raise ConfirmError
    rather than silently guessing 0 (faithful regeneration required)."""
    run_dir, hashes = _build_real_run_dir(tmp_path)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    del manifest["cli_args"]["rng_seed"]
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ConfirmError, match="rng_seed missing"):
        run_confirm_pass(run_dir, repeats=3, llm_callable=FakeLLM())


def test_run_confirm_pass_missing_variants_raises(tmp_path):
    """N6: a manifest without cli_args.variants must raise ConfirmError."""
    run_dir, hashes = _build_real_run_dir(tmp_path)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    del manifest["cli_args"]["variants"]
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ConfirmError, match="variants missing"):
        run_confirm_pass(run_dir, repeats=3, llm_callable=FakeLLM())


def test_read_confirm_sidecar_absent_returns_none(tmp_path):
    """read_confirm_sidecar returns None (not an error) when no pass ran."""
    from topofuzz.confirm import read_confirm_sidecar

    d = tmp_path / "no-confirm"
    d.mkdir()
    assert read_confirm_sidecar(d) is None


def test_read_confirm_sidecar_corrupt_raises(tmp_path):
    """A present-but-corrupt confirm.json is a real signal → ConfirmError."""
    from topofuzz.confirm import read_confirm_sidecar

    d = tmp_path / "bad-confirm"
    d.mkdir()
    (d / "confirm.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfirmError, match="not valid JSON"):
        read_confirm_sidecar(d)


# ---------------------------------------------------------------------------
# M-RT-1 / m-RT-3: confirmed_verdict_flip sum invariant + bucket split
# ---------------------------------------------------------------------------


def test_confirmed_verdict_flip_sum_invariant_with_judge_failure_pair():
    """M-RT-1: a screened candidate pair whose every repeat is a judge
    failure lands in judge_failure_pairs so the sum invariant HOLDS:
    confirmed + flaky + unstable_baseline + missing_baseline +
    judge_failure_pairs == n_screened_pairs."""
    V, W = "v" * 64, "w" * 64
    ref = [
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
    ]
    cand = [
        # e1 / V: 3 DENY → confirmed.
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        # e1 / W: every repeat a judge failure → no candidate row at all.
        {"eval_id": "e1", "variant_id": W, "score_value": None},
        {"eval_id": "e1", "variant_id": W, "score_value": None},
        {"eval_id": "e1", "variant_id": W, "score_value": None},
    ]
    # The screened set has BOTH pairs.
    screened = [("e1", V), ("e1", W)]
    res = compute_confirmed_verdict_flip_rate(
        ref, cand, candidates=screened, repeats=3, seed=0
    )
    assert res.confirmed == 1
    assert res.judge_failure_pairs == 1
    assert res.n_screened_pairs == 2
    total = (
        res.confirmed
        + res.flaky
        + res.unstable_baseline
        + res.missing_baseline
        + res.judge_failure_pairs
    )
    assert total == res.n_screened_pairs


def test_confirmed_verdict_flip_missing_vs_tied_baseline_split():
    """m-RT-3: 'tied baseline' (unstable_baseline) and 'no reference
    rows at all' (missing_baseline) are separate buckets."""
    V, W = "v" * 64, "w" * 64
    ref = [
        # e1 reference: tied → unstable_baseline.
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.1},
        # e2 has NO reference rows at all → missing_baseline.
    ]
    cand = [
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e2", "variant_id": W, "score_value": 0.1},
        {"eval_id": "e2", "variant_id": W, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(
        ref, cand, candidates=[("e1", V), ("e2", W)], repeats=2, seed=0
    )
    assert res.unstable_baseline == 1  # e1: tied baseline
    assert res.missing_baseline == 1  # e2: no reference rows
    assert res.confirmed == 0 and res.flaky == 0
    total = (
        res.confirmed
        + res.flaky
        + res.unstable_baseline
        + res.missing_baseline
        + res.judge_failure_pairs
    )
    assert total == res.n_screened_pairs == 2


def test_confirmed_verdict_flip_drops_ragged_pair():
    """B-RT-1(b): a candidate pair with FEWER than K scored repeats is
    dropped (routed to judge_failure_pairs) — a ragged unit cannot be
    majority-voted honestly."""
    V = "v" * 64
    ref = [
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
    ]
    cand = [
        # Only 1 of K=3 repeats scored — ragged.
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(
        ref, cand, candidates=[("e1", V)], repeats=3, seed=0
    )
    # Without the ragged-drop, _majority([False]) would return False and
    # the pair would be scored "CONFIRMED" off a single draw.
    assert res.confirmed == 0
    assert res.judge_failure_pairs == 1
    assert res.degenerate_flag is True


def test_confirmed_verdict_flip_drops_ragged_reference():
    """B-RT-1(b): a reference eval with < K scored repeats is dropped;
    its candidate pair then has no callable baseline → missing_baseline."""
    V = "v" * 64
    ref = [
        # Only 1 of K=3 reference repeats — ragged baseline.
        {"eval_id": "e1", "score_value": 0.9},
    ]
    cand = [
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(
        ref, cand, candidates=[("e1", V)], repeats=3, seed=0
    )
    assert res.missing_baseline == 1
    assert res.confirmed == 0 and res.flaky == 0


# ---------------------------------------------------------------------------
# m-RT-1: verdict_stability — single-repeat evals are not "unanimous"
# ---------------------------------------------------------------------------


def test_verdict_stability_drops_single_repeat_eval():
    """m-RT-1: an eval with only ONE scored reference repeat is NOT
    counted 'unanimous' (len(set)==1 is trivially true for a singleton).
    It is dropped from BOTH numerator and denominator."""
    ref = [
        # e1: 2 scored repeats, both APPROVE → genuinely unanimous.
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.8},
        # e2: ONLY ONE scored repeat — must be dropped, not counted.
        {"eval_id": "e2", "score_value": 0.9},
    ]
    res = compute_verdict_stability(ref)
    assert res.n_evals == 1  # only e1 qualifies
    assert res.n_unanimous == 1
    assert res.point == pytest.approx(1.0)


def test_verdict_stability_repeats_drops_ragged_block():
    """B-RT-1(b): with repeats=K, an eval with < K scored reference
    repeats is dropped from the stability fraction."""
    ref = [
        # e1: full K=3.
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        # e2: ragged — only 2 of 3.
        {"eval_id": "e2", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.1},
    ]
    res = compute_verdict_stability(ref, repeats=3)
    assert res.n_evals == 1  # e2 dropped (ragged)
    assert res.point == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# B-RT-1(b) / m-RT-2: voo_noise_floor ragged drop + gate partial signal
# ---------------------------------------------------------------------------


def test_voo_noise_floor_repeats_drops_ragged_block():
    """B-RT-1(b): with repeats=K, an eval missing a repeat → None (the
    floor is undefined over a ragged K×n_evals block)."""
    ref = [
        {"eval_id": "e1", "repeat_idx": 0, "score_value": 0.5},
        {"eval_id": "e1", "repeat_idx": 1, "score_value": 0.5},
        {"eval_id": "e1", "repeat_idx": 2, "score_value": 0.5},
        # e2 only has 2 of 3.
        {"eval_id": "e2", "repeat_idx": 0, "score_value": 0.8},
        {"eval_id": "e2", "repeat_idx": 1, "score_value": 0.8},
    ]
    assert compute_voo_noise_floor(ref, repeats=3) is None
    # Without the K gate the looser check also catches this particular
    # case, but the K gate also catches "every eval has 2, K=3".
    ref_uniform_short = [
        {"eval_id": "e1", "repeat_idx": 0, "score_value": 0.5},
        {"eval_id": "e1", "repeat_idx": 1, "score_value": 0.5},
        {"eval_id": "e2", "repeat_idx": 0, "score_value": 0.8},
        {"eval_id": "e2", "repeat_idx": 1, "score_value": 0.8},
    ]
    assert compute_voo_noise_floor(ref_uniform_short, repeats=3) is None
    # repeats=2 → the uniform-short block is complete → computable.
    assert compute_voo_noise_floor(ref_uniform_short, repeats=2) == (
        pytest.approx(0.0)
    )


def test_voo_gate_passes_partial_does_not_fall_back():
    """m-RT-2: with confirm_partial=True a None floor must NOT silently
    fall back to the pre-correction `> 0` gate — it returns False."""
    # No confirm pass (confirm_partial=False) → historical > 0 fallback.
    assert voo_gate_passes(0.10, None) is True
    # Partial confirm pass → must NOT fall back; gate cannot pass.
    assert voo_gate_passes(0.10, None, confirm_partial=True) is False
    # A real floor is still honoured regardless of the partial flag.
    assert voo_gate_passes(0.30, 0.25, confirm_partial=True) is True


# ---------------------------------------------------------------------------
# m-RT-4 / N3: _resolve_eval_path resolution order
# ---------------------------------------------------------------------------


def test_resolve_eval_path_prefers_run_dir_over_cwd(tmp_path, monkeypatch):
    """m-RT-4: a same-named eval file in cwd must NOT shadow the one that
    belongs to the run directory."""
    from topofuzz.confirm import _resolve_eval_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "e.csv").write_text("id,input\nx,run-copy\n", encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "e.csv").write_text("id,input\nx,cwd-copy\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    resolved = _resolve_eval_path({"eval": "e.csv"}, run_dir)
    assert resolved == (run_dir / "e.csv").resolve()


def test_resolve_eval_path_absolute_no_duplicate_tries(tmp_path):
    """N3: an absolute eval path is tried once, not 4× — the not-found
    error message must not repeat the same path four times."""
    from topofuzz.confirm import _resolve_eval_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    missing_abs = (tmp_path / "nope.csv").resolve()
    with pytest.raises(ConfirmError) as ei:
        _resolve_eval_path({"eval": str(missing_abs)}, run_dir)
    # The absolute path appears exactly once in "tried ...".
    assert ei.value.args[0].count(str(missing_abs)) == 1


# ---------------------------------------------------------------------------
# Second-pass dual audit: RT-2 — exact-K (!= K) drop consistency
# ---------------------------------------------------------------------------


def test_confirmed_verdict_flip_drops_stale_over_k_pair():
    """RT-2: a candidate pair with MORE than K scored repeats is stale /
    contaminated and must be dropped (``!= K``, not ``< K``) — consistent
    with voo_noise_floor's exact-count gate."""
    V = "v" * 64
    ref = [
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
    ]
    cand = [
        # 4 scored repeats for K=3 — a stale row leaked in. Pre-fix the
        # ``< K`` predicate let this sail through and majority-vote over 4.
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(
        ref, cand, candidates=[("e1", V)], repeats=3, seed=0
    )
    assert res.confirmed == 0
    assert res.judge_failure_pairs == 1  # dropped → its own bucket
    assert res.degenerate_flag is True


def test_confirmed_verdict_flip_drops_stale_over_k_reference():
    """RT-2: a reference eval with MORE than K scored repeats is dropped;
    its candidate pair then has no callable baseline → missing_baseline."""
    V = "v" * 64
    ref = [
        # 4 scored reference repeats for K=3 — stale.
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
    ]
    cand = [
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
        {"eval_id": "e1", "variant_id": V, "score_value": 0.1},
    ]
    res = compute_confirmed_verdict_flip_rate(
        ref, cand, candidates=[("e1", V)], repeats=3, seed=0
    )
    assert res.missing_baseline == 1
    assert res.confirmed == 0 and res.flaky == 0


def test_verdict_stability_k1_is_degenerate():
    """RT-2 nuance: K=1 cannot measure agreement (a single repeat is
    trivially 'unanimous') — the whole metric is degenerate, not 1.0."""
    ref = [
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.1},
    ]
    res = compute_verdict_stability(ref, repeats=1)
    assert res.point is None
    assert res.degenerate_flag is True
    assert res.n_evals == 0


def test_verdict_stability_drops_stale_over_k_block():
    """RT-2: with repeats=K, an eval with MORE than K scored reference
    repeats (a stale block) is dropped — exact-K, not '>= K'."""
    ref = [
        # e1: exactly K=3.
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        {"eval_id": "e1", "score_value": 0.9},
        # e2: 4 of 3 — stale, must be dropped.
        {"eval_id": "e2", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.9},
        {"eval_id": "e2", "score_value": 0.1},
    ]
    res = compute_verdict_stability(ref, repeats=3)
    assert res.n_evals == 1  # only e1 (exactly K); e2 dropped as stale
    assert res.point == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Second-pass dual audit: RT-3 / CR-MINOR-2/3 — sidecar trust hardening
# ---------------------------------------------------------------------------


def test_noise_floor_refuses_sidecar_missing_repeats(tmp_path):
    """RT-3 / CR-MINOR-2: a sidecar missing the load-bearing ``repeats``
    key cannot be trusted — refuse + note, NOT silent ``repeats=None``."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-noreps"
    with RunWriter(run_dir, "run-noreps", topo) as w:
        for ridx, score in enumerate((0.9, 0.85, 0.95)):
            w.write_flip_confirmation(
                _make_cell(topo.canonical_hash(), "eval_001", score=score),
                repeat_idx=ridx,
                role="reference",
            )
    # A sidecar with no ``repeats`` key.
    payload = {
        "partial": False,
        "n_unmatched_candidates": 0,
        "n_candidates": 1,
    }
    (run_dir / "confirm.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-noreps"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "repeats" in note


def test_noise_floor_refuses_sidecar_missing_partial(tmp_path):
    """RT-3 / CR-MINOR-2: a sidecar missing the ``partial`` key cannot be
    trusted — refuse + note, not silent 'treat as complete'."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-nopart"
    with RunWriter(run_dir, "run-nopart", topo) as w:
        for ridx, score in enumerate((0.9, 0.85, 0.95)):
            w.write_flip_confirmation(
                _make_cell(topo.canonical_hash(), "eval_001", score=score),
                repeat_idx=ridx,
                role="reference",
            )
    payload = {"repeats": 3, "n_unmatched_candidates": 0}
    (run_dir / "confirm.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-nopart"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "partial" in note


def test_noise_floor_refuses_bool_n_unmatched(tmp_path):
    """CR-MINOR-3: JSON ``true`` is an int subclass — a bare isinstance
    check would accept ``n_unmatched_candidates: true`` and print a
    cosmetically wrong note. The bool guard refuses it as untrustworthy."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-boolunm"
    with RunWriter(run_dir, "run-boolunm", topo) as w:
        for ridx, score in enumerate((0.9, 0.85, 0.95)):
            w.write_flip_confirmation(
                _make_cell(topo.canonical_hash(), "eval_001", score=score),
                repeat_idx=ridx,
                role="reference",
            )
    payload = {
        "repeats": 3,
        "partial": False,
        "n_unmatched_candidates": True,  # bool, not a real count
    }
    (run_dir / "confirm.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-boolunm"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "n_unmatched_candidates" in note
    # The cosmetically-wrong "True screened candidate variant(s)" note
    # must NOT be emitted.
    assert "True screened candidate" not in note


def test_noise_floor_refuses_sidecar_repeats_disagrees_with_table(tmp_path):
    """RT-3: a hand-crafted sidecar whose ``repeats`` disagrees with the
    persisted ``repeat_idx`` range (here a repeat_idx >= repeats) is stale
    or tampered — refuse to populate the headline + emit a note."""
    from topofuzz.attest import compute_noise_floor_for_attestation

    topo = _toy_topology()
    run_dir = tmp_path / "run-disagree"
    V = "v" * 64
    with RunWriter(run_dir, "run-disagree", topo) as w:
        # The table actually holds 3 repeats (idx 0..2).
        for eid in ("eval_001",):
            for ridx, score in enumerate((0.9, 0.85, 0.95)):
                w.write_flip_confirmation(
                    _make_cell(topo.canonical_hash(), eid, score=score),
                    repeat_idx=ridx,
                    role="reference",
                )
        for ridx, score in enumerate((0.1, 0.2, 0.05)):
            w.write_flip_confirmation(
                _make_cell(V, "eval_001", score=score),
                repeat_idx=ridx,
                role="candidate",
            )
    # ...but the sidecar claims K=2 — repeat_idx 2 is out of range.
    _write_confirm_sidecar(run_dir, repeats=2)
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "run-disagree"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "disagrees" in note


# ---------------------------------------------------------------------------
# Second-pass dual audit: N-RT-1 — crash mid-pass marks the table partial
# ---------------------------------------------------------------------------


def test_run_confirm_pass_crash_writes_partial_sidecar(tmp_path, monkeypatch):
    """N-RT-1: ``delete_flip_confirmations`` runs before the re-run loop,
    so a crash mid-loop destroys the prior good rows. The pass must write
    a ``partial: true`` sidecar on the way out so the half-rewritten
    table is at least *marked* incomplete, then re-raise."""
    import topofuzz.confirm as confirm_mod
    from topofuzz.confirm import read_confirm_sidecar

    run_dir, _ = _build_real_run_dir(tmp_path)

    # Make the async re-run blow up after delete_flip_confirmations ran.
    async def boom(*_a, **_kw):
        raise RuntimeError("simulated mid-pass crash")

    monkeypatch.setattr(confirm_mod, "_confirm_async", boom)
    with pytest.raises(RuntimeError, match="simulated mid-pass crash"):
        run_confirm_pass(run_dir, repeats=3, llm_callable=FakeLLM())

    # The crash must have left a partial-marked sidecar behind.
    sidecar = read_confirm_sidecar(run_dir)
    assert sidecar is not None
    assert sidecar["partial"] is True


def test_noise_floor_refuses_after_crash_partial_sidecar(tmp_path, monkeypatch):
    """N-RT-1 end-to-end: after a crashed pass the attest path sees
    partial:true and refuses the headline with a partial-pass note."""
    import topofuzz.confirm as confirm_mod
    from topofuzz.attest import compute_noise_floor_for_attestation

    run_dir, _ = _build_real_run_dir(tmp_path)

    async def boom(*_a, **_kw):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(confirm_mod, "_confirm_async", boom)
    with pytest.raises(RuntimeError):
        run_confirm_pass(run_dir, repeats=3, llm_callable=FakeLLM())

    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "real-run-0001"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "partial confirm pass" in note


# ---------------------------------------------------------------------------
# Second-pass dual audit: N-RT-2 — --max-cost-usd 0 is a usage error
# ---------------------------------------------------------------------------


def test_cli_confirm_rejects_zero_max_cost(tmp_path):
    """N-RT-2: ``--max-cost-usd 0`` would hard-stop before the first
    repeat (``0 >= 0``) → an instant degenerate partial pass. The CLI
    rejects it as a usage error instead."""
    from typer.testing import CliRunner

    from topofuzz.cli import app

    run_dir, _ = _build_real_run_dir(tmp_path)
    result = CliRunner().invoke(
        app, ["confirm", str(run_dir), "--max-cost-usd", "0", "--yes"]
    )
    assert result.exit_code == 2
    assert "max-cost-usd must be a positive" in result.output


# ---------------------------------------------------------------------------
# Second-pass dual audit: N-RT-3 — honesty-invariant check at attest boundary
# ---------------------------------------------------------------------------


def test_noise_floor_refuses_when_buckets_dont_sum(tmp_path, monkeypatch):
    """N-RT-3: if the confirmed_verdict_flip disposition buckets do not
    sum to n_screened_pairs, the metric accounting is inconsistent — the
    attest path refuses the headline + emits a note (note, not assert).

    The honesty invariant is only guaranteed on the ``candidates=``
    path, so this uses a real run dir where ``find_all_flip_candidates``
    succeeds (``screened_candidates`` is non-None — the check arms)."""
    import topofuzz.metrics as metrics_mod
    from topofuzz.attest import compute_noise_floor_for_attestation
    from topofuzz.metrics.confirmed_verdict_flip import (
        ConfirmedVerdictFlipResult,
    )

    run_dir, hashes = _build_real_run_dir(tmp_path)
    topo = _toy_topology()
    # Lay down a complete reference + candidate block so the metric path
    # is reached (the table must be non-empty).
    with RunWriter(run_dir, "real-run-0001", topo) as w:
        for eid in ("eval_001",):
            for ridx, score in enumerate((0.9, 0.85, 0.95)):
                w.write_flip_confirmation(
                    _make_cell(hashes[0], eid, score=score),
                    repeat_idx=ridx,
                    role="reference",
                )
        for ridx, score in enumerate((0.1, 0.2, 0.05)):
            w.write_flip_confirmation(
                _make_cell(hashes[1], "eval_001", score=score),
                repeat_idx=ridx,
                role="candidate",
            )
    _write_confirm_sidecar(run_dir, repeats=3)

    # Force a result whose buckets do NOT sum to n_screened_pairs.
    def _bad_metric(*_a, **_kw):
        return ConfirmedVerdictFlipResult(
            point=1.0,
            ci_lower=None,
            ci_upper=None,
            confirmed=1,
            flaky=0,
            unstable_baseline=0,
            missing_baseline=0,
            judge_failure_pairs=0,
            n_screened_pairs=5,  # 1 + 0*4 != 5
            n_resamples=0,
            method="percentile",
            degenerate_flag=True,
        )

    # attest.py does a function-local ``from topofuzz.metrics import
    # compute_confirmed_verdict_flip_rate`` — patch it on the metrics
    # package so the local import picks up the fake.
    monkeypatch.setattr(
        metrics_mod, "compute_confirmed_verdict_flip_rate", _bad_metric
    )
    stability, confirmed, floor, note = compute_noise_floor_for_attestation(
        run_dir, "real-run-0001"
    )
    assert stability is None and confirmed is None and floor is None
    assert note is not None and "do not sum" in note
