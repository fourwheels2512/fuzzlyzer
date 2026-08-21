# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``topofuzz ablate`` — counterfactual leave-one-out attribution.

Covers:

* Sidecar JSON I/O round-trip + corrupt-file rejection.
* :func:`topofuzz.ablation.find_all_flipping_triples` — manifest canonical
  + flip-candidate pairing.
* :func:`topofuzz.ablation.run_ablation_pass` gates — rejects single-judge
  source runs, missing manifest / run.sqlite, empty flips writes empty
  sidecar.
* :func:`_ablate_async` with a hand-built mock JuryJudge and mock async
  LLM callable — exercises terminal re-execution + jury re-judgment +
  flip-closing math + by_node attribution aggregate.
* Skip kinds: skip_downstream (node ran after terminal in V's order),
  skip_no_canonical_output (canonical had no output for the node),
  failed (terminal re-exec raised).
* Cost-cap hard-stop persists what completed and writes partial sidecar.
* JuryJudge reconstruction path covers jury_models_resolved and the
  verbatim ``--jury-models`` fallback.
* Response-text truncation is codepoint-safe.

The async LLM mock has the same shape as ``test_confirm.FakeLLM`` — an
``async __call__`` that returns a LiteLLM-shaped dict. The jury mock is
a small object with ``.score(payload)`` + ``.vendors`` so the
ablation's reflection-based cost estimate doesn't need a real
JuryJudge.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("diskcache")

from fuzzeval.judge import ContinuousScore  # noqa: E402

from topofuzz.ablation import (  # noqa: E402
    ABLATION_SIDECAR_FILENAME,
    AblationError,
    AblationResult,
    AblationSummary,
    FlippingTriple,
    _ablate_async,
    _build_jury,
    _dicts_to_node_outputs,
    _truncate_response,
    find_all_flipping_triples,
    read_ablation_sidecar,
    run_ablation_pass,
    validate_ablation_source,
)
from topofuzz.ir import Edge, LLMConfig, Node, Topology  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _toy_topology(
    *, model: str = "google/gemini-2.5-flash", n_nodes: int = 4
) -> Topology:
    """Linear-chain Topology with N nodes. Terminal is the last node."""
    cfg = LLMConfig(model=model, temperature=0.0)
    nodes = tuple(
        Node(id=f"n{i}", role=f"r{i}", llm_config=cfg) for i in range(n_nodes)
    )
    edges = tuple(
        Edge(from_id=f"n{i}", to_id=f"n{i+1}") for i in range(n_nodes - 1)
    )
    return Topology(
        name="toy_linear",
        nodes=nodes,
        edges=edges,
        entrypoint="n0",
        terminal=(f"n{n_nodes - 1}",),
    )


def _make_node_output_dict(
    node_id: str,
    *,
    response_text: str = "ok",
    role: str | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "role": role or f"role-{node_id}",
        "response_text": response_text,
        "model_served_version": "google/gemini-2.5-flash",
        "cache_hit": False,
        "latency_ms": 1,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }


@dataclass
class MockJuryJudge:
    """Hand-built jury substitute for tests.

    Returns a fixed :class:`ContinuousScore` per invocation; tests
    sequence the per-call returns through ``scores_queue`` and the per-
    call jury vote breakdown through ``votes_queue``. ``vendors`` is the
    minimal surface ``_estimate_jury_cost`` reflects against.
    """

    scores_queue: list[float] = field(default_factory=list)
    votes_queue: list[tuple[tuple[str, float, str], ...]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    vendors: tuple[str, ...] = ("anthropic", "gemini", "openai")

    def score(self, payload: dict) -> ContinuousScore:
        self.calls.append(payload)
        score = self.scores_queue.pop(0) if self.scores_queue else 0.5
        votes = self.votes_queue.pop(0) if self.votes_queue else ()
        return ContinuousScore(value=score, rationale="mock", votes=votes)


@dataclass
class FakeLLM:
    """Async LLM mock — returns a fixed response_text per call."""

    response_text: str = "[MOCK terminal output]"
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": self.response_text}}],
            "model": kwargs.get("model", ""),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


# ---------------------------------------------------------------------------
# Sidecar I/O round-trip
# ---------------------------------------------------------------------------


def test_read_ablation_sidecar_returns_none_when_absent(tmp_path):
    """Missing sidecar reads as ``None`` (mirrors confirm.read_confirm_sidecar)."""
    assert read_ablation_sidecar(tmp_path) is None


def test_read_ablation_sidecar_raises_on_corrupt_json(tmp_path):
    (tmp_path / ABLATION_SIDECAR_FILENAME).write_text("{ malformed", encoding="utf-8")
    with pytest.raises(AblationError, match="unreadable / not valid JSON"):
        read_ablation_sidecar(tmp_path)


def test_read_ablation_sidecar_raises_on_non_object_root(tmp_path):
    (tmp_path / ABLATION_SIDECAR_FILENAME).write_text("[]", encoding="utf-8")
    with pytest.raises(AblationError, match="not a JSON object"):
        read_ablation_sidecar(tmp_path)


# ---------------------------------------------------------------------------
# find_all_flipping_triples
# ---------------------------------------------------------------------------


def _write_synthetic_run(
    tmp_path: Path,
    *,
    judge_rows: list[tuple],
    hashes: list[str],
    cli_args: dict[str, Any] | None = None,
) -> Path:
    """Lay down a manifest + run.sqlite (judge_results only) under ``run_dir``."""
    run_dir = tmp_path / "synth-run"
    run_dir.mkdir()
    manifest_cli_args = {
        "fixture": "x.py:build_graph",
        "eval": "e.csv",
        "rng_seed": 0,
        "variants": len(hashes),
        "jury_of_3": True,
        "rubric": "rubric.txt",
    }
    if cli_args:
        manifest_cli_args.update(cli_args)
    manifest = {
        "run_id": "synth-0001",
        "variant_canonical_hashes": hashes,
        "variant_orderings": [["n0", "n1", "n2", "n3"]] * len(hashes),
        "eval_ids": ["eval_001", "eval_002"],
        "cli_args": manifest_cli_args,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, "
        "failure_reason TEXT, votes_json TEXT)"
    )
    db.execute(
        "CREATE TABLE cells ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, "
        "ir_canonical_hash TEXT, node_outputs_json TEXT, error TEXT, "
        "cell_cost_usd REAL)"
    )
    for row in judge_rows:
        db.execute(
            "INSERT INTO judge_results "
            "(run_id, variant_id, eval_id, judge_kind, score_value, "
            "passed, rationale, failure_reason, votes_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row + (None,) if len(row) == 8 else row,
        )
    db.commit()
    db.close()
    return run_dir


def test_find_all_flipping_triples_pairs_canonical(tmp_path):
    """Canonical is hashes[0]; flipping variants are paired with it."""
    A, B, C = "a" * 64, "b" * 64, "c" * 64
    rows = [
        ("synth-0001", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_001", "continuous", 0.1, None, None, None),
        ("synth-0001", C, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", A, "eval_002", "continuous", 0.2, None, None, None),
        ("synth-0001", B, "eval_002", "continuous", 0.2, None, None, None),
        ("synth-0001", C, "eval_002", "continuous", 0.8, None, None, None),
    ]
    run_dir = _write_synthetic_run(tmp_path, judge_rows=rows, hashes=[A, B, C])
    triples = find_all_flipping_triples(run_dir)
    assert triples == [
        FlippingTriple(
            canonical_variant_id=A, flipping_variant_id=B, eval_id="eval_001"
        ),
        FlippingTriple(
            canonical_variant_id=A, flipping_variant_id=C, eval_id="eval_002"
        ),
    ]


def test_find_all_flipping_triples_empty_when_no_flips(tmp_path):
    A, B = "a" * 64, "b" * 64
    rows = [
        ("synth-0001", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_001", "continuous", 0.8, None, None, None),
    ]
    run_dir = _write_synthetic_run(tmp_path, judge_rows=rows, hashes=[A, B])
    assert find_all_flipping_triples(run_dir) == []


def test_find_all_flipping_triples_raises_on_empty_manifest_hashes(tmp_path):
    rows = [
        (
            "synth-0001", "a" * 64, "eval_001", "continuous",
            0.9, None, None, None,
        )
    ]
    run_dir = _write_synthetic_run(tmp_path, judge_rows=rows, hashes=[])
    # The manifest writer above used hashes=[], so the canonical
    # candidate set is empty — `find_all_flip_candidates` should raise
    # before we even reach the canonical-id lookup.
    with pytest.raises(AblationError):
        find_all_flipping_triples(run_dir)


# ---------------------------------------------------------------------------
# run_ablation_pass gates
# ---------------------------------------------------------------------------


def test_run_ablation_pass_rejects_single_judge_source(tmp_path):
    """The source run must have `jury_of_3=True`; single-judge runs are
    refused at the gate so attribution numbers don't inherit ~26% noise."""
    A, B = "a" * 64, "b" * 64
    rows = [
        ("synth-0001", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_001", "continuous", 0.1, None, None, None),
    ]
    run_dir = _write_synthetic_run(
        tmp_path, judge_rows=rows, hashes=[A, B],
        cli_args={"jury_of_3": False},
    )
    with pytest.raises(AblationError, match="jury-of-3"):
        run_ablation_pass(run_dir)


def test_run_ablation_pass_rejects_missing_manifest(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AblationError, match="manifest.json missing"):
        run_ablation_pass(empty)


def test_validate_ablation_source_rejects_missing_manifest(tmp_path):
    """ROB-5: :func:`validate_ablation_source` is the cheap, side-effect-
    free precondition check :func:`run_ablation_pass` runs first — it
    must raise the SAME ``AblationError`` on a missing manifest so the
    CLI can call it standalone, before the spend-confirmation prompt,
    without duplicating validation logic."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AblationError, match="manifest.json missing"):
        validate_ablation_source(empty)


def test_validate_ablation_source_accepts_valid_jury_manifest(tmp_path):
    """The happy path returns the parsed manifest dict (reused by
    run_ablation_pass so the source run isn't read twice)."""
    A, B = "a" * 64, "b" * 64
    rows = [
        ("synth-0001", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_001", "continuous", 0.1, None, None, None),
    ]
    run_dir = _write_synthetic_run(tmp_path, judge_rows=rows, hashes=[A, B])
    manifest = validate_ablation_source(run_dir)
    assert manifest["cli_args"]["jury_of_3"] is True


# ---------------------------------------------------------------------------
# ROB-5: `topofuzz ablate` CLI validates the source run BEFORE the
# interactive spend-confirmation prompt.
# ---------------------------------------------------------------------------


def test_cli_ablate_missing_manifest_errors_before_spend_prompt(tmp_path):
    """An existing-but-invalid run_dir (a real directory with no
    manifest.json) must error out immediately with no billing prompt.

    Verified against the pre-fix CLI ordering: the ``run_dir.is_dir()``
    check passed (the directory DOES exist), so the code proceeded to
    print "topofuzz: research tool... this bills real LLM calls" and
    the "Run counterfactual ablation ... ?" confirmation BEFORE
    ``run_ablation_pass`` ever got a chance to raise
    ``AblationError("manifest.json missing...")`` — exit code 0 with an
    "Aborted." line (CliRunner's stdin is empty). Post-fix, the
    validation runs first: exit 2, no prompt text anywhere in the
    output, no stdin needed.
    """
    from typer.testing import CliRunner

    from topofuzz.cli import app

    empty_dir = tmp_path / "empty-run"
    empty_dir.mkdir()

    result = CliRunner().invoke(app, ["ablate", str(empty_dir)])
    assert result.exit_code == 2, result.output
    assert "manifest.json missing" in result.output
    assert "bills real LLM calls" not in result.output
    assert "Aborted" not in result.output


def test_run_ablation_pass_no_flips_writes_empty_sidecar(tmp_path):
    """A non-fragile source run (no flips) writes an empty sidecar so the
    inspector can tell ablate-was-run from ablate-not-yet-run."""
    A, B = "a" * 64, "b" * 64
    rows = [
        ("synth-0001", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth-0001", B, "eval_001", "continuous", 0.8, None, None, None),
    ]
    run_dir = _write_synthetic_run(tmp_path, judge_rows=rows, hashes=[A, B])
    summary = run_ablation_pass(run_dir)
    assert summary.n_flipping_pairs == 0
    assert summary.n_ablation_points == 0
    assert summary.results == ()
    sidecar = read_ablation_sidecar(run_dir)
    assert sidecar is not None
    assert sidecar["n_flipping_pairs"] == 0
    assert sidecar["results"] == []


# ---------------------------------------------------------------------------
# _ablate_async with a mock judge + mock LLM — the math
# ---------------------------------------------------------------------------


def _build_run_dir_for_ablate(
    tmp_path: Path,
    *,
    canonical_outputs: list[dict],
    v_outputs: list[dict],
    canonical_score: float = 0.9,
    v_score: float = 0.1,
    eval_input: str = "applicant info",
) -> tuple[Path, str, str]:
    """Lay down a run_dir with one (canonical, V) pair on one eval.

    Returns ``(run_dir, canonical_id, v_id)``.
    """
    canonical_id = "c" * 64
    v_id = "v" * 64
    run_dir = tmp_path / "ablate-run"
    run_dir.mkdir()
    manifest = {
        "run_id": "ablate-001",
        "variant_canonical_hashes": [canonical_id, v_id],
        "variant_orderings": [["n0", "n1", "n2", "n3"], ["n3", "n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph",
            "eval": "e.csv",
            "rng_seed": 0,
            "variants": 2,
            "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, "
        "failure_reason TEXT, votes_json TEXT)"
    )
    db.execute(
        "CREATE TABLE cells ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, "
        "ir_canonical_hash TEXT, node_outputs_json TEXT, error TEXT, "
        "cell_cost_usd REAL)"
    )
    db.execute(
        "INSERT INTO judge_results VALUES "
        "('ablate-001', ?, 'eval_001', 'continuous', ?, NULL, NULL, NULL, NULL)",
        (canonical_id, canonical_score),
    )
    db.execute(
        "INSERT INTO judge_results VALUES "
        "('ablate-001', ?, 'eval_001', 'continuous', ?, NULL, NULL, NULL, NULL)",
        (v_id, v_score),
    )
    db.execute(
        "INSERT INTO cells VALUES ('ablate-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(canonical_outputs)),
    )
    db.execute(
        "INSERT INTO cells VALUES ('ablate-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (v_id, json.dumps(v_outputs)),
    )
    db.commit()
    db.close()
    return run_dir, canonical_id, v_id


def test_ablate_async_anchors_on_verdict_node_not_lex_first_terminal(tmp_path):
    """Step 20a BLOCKER B1 regression guard: when v_topology declares a
    v0.6 ``verdict_node``, ablation must anchor leave-one-out attribution
    to THAT node, not the lex-first terminal — otherwise F4 attribution
    silently scores fragility against the wrong upstream nodes.

    Build a topology with terminal=("a","z") (lex-first "a"), verdict_node
    "z", and assert the ``substitute_terminal`` ablation point names "z".
    """
    canonical_outputs = [
        _make_node_output_dict("a"),
        _make_node_output_dict("z", response_text="CANONICAL VERDICT: APPROVE"),
    ]
    v_outputs = [
        _make_node_output_dict("a"),
        _make_node_output_dict("z", response_text="V VERDICT: DENY"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
        canonical_score=0.9,
        v_score=0.1,
    )
    triples = find_all_flipping_triples(run_dir)
    assert len(triples) == 1

    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    nodes = (
        Node(id="a", role="r_a", llm_config=cfg),
        Node(id="z", role="r_z", llm_config=cfg),
    )
    edges = (Edge(from_id="a", to_id="z"),)
    # Both terminal — lex-first is "a", but verdict_node names "z".
    topo_canonical = Topology(
        name="vn", nodes=nodes, edges=edges,
        entrypoint="a", terminal=("a", "z"), verdict_node="z",
    )
    topo_v = Topology(
        name="vn_v", nodes=nodes, edges=edges,
        entrypoint="a", terminal=("a", "z"), verdict_node="z",
    )

    judge = MockJuryJudge(scores_queue=[1.0])
    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir, run_id="ablate-vn",
            triples=triples,
            by_hash={canonical_id: topo_canonical, v_id: topo_v},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge, max_cost_usd=None,
            llm_callable=FakeLLM(), dry_run=False,
        )
    )
    terminal_results = [r for r in summary.results if r.kind == "substitute_terminal"]
    assert len(terminal_results) == 1
    # Must anchor on verdict_node "z", NOT lex-first terminal "a".
    assert terminal_results[0].ablated_node_id == "z"


def test_ablate_async_terminal_substitution_always_closes_flip(tmp_path):
    """Terminal-node ablation substitutes canonical's terminal text and
    re-judges. With a mock jury returning canonical's score, the flip
    closes — this is the deliberate baseline column."""
    canonical_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2"),
        _make_node_output_dict("n3", response_text="CANONICAL VERDICT: APPROVE"),
    ]
    v_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2"),
        _make_node_output_dict("n3", response_text="V VERDICT: DENY"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
        canonical_score=0.9,  # APPROVE
        v_score=0.1,           # DENY
    )
    triples = find_all_flipping_triples(run_dir)
    assert len(triples) == 1

    # Build a topology that uses V's order (the runner judges declared
    # terminal `n3`, V's nodes-tuple order is the manifest's
    # variant_orderings[1] == ["n3","n0","n1","n2"]).
    # For the ablation pass we reconstruct by canonical_hash; the test
    # bypasses that by giving _ablate_async an explicit by_hash map.
    topo_canonical = _toy_topology(n_nodes=4)
    # V topology: same nodes but with n3 first to mirror manifest's
    # variant_orderings[1]. terminal is still ("n3",).
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    v_nodes = (
        Node(id="n3", role="r3", llm_config=cfg),
        Node(id="n0", role="r0", llm_config=cfg),
        Node(id="n1", role="r1", llm_config=cfg),
        Node(id="n2", role="r2", llm_config=cfg),
    )
    v_edges = (Edge(from_id="n0", to_id="n1"),)  # edges don't matter for ablation
    topo_v = Topology(
        name="toy_v", nodes=v_nodes, edges=v_edges,
        entrypoint="n3", terminal=("n3",),
    )

    # Jury queue: re-judgments will all return 1.0 (canonical_aligned).
    # 1 terminal substitution + 0 upstream (n0/n1/n2 ran AFTER terminal
    # in V's order → all skip_downstream) = 1 meaningful ablation.
    judge = MockJuryJudge(scores_queue=[1.0])
    fake_llm = FakeLLM()

    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir,
            run_id="ablate-001",
            triples=triples,
            by_hash={canonical_id: topo_canonical, v_id: topo_v},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge,
            max_cost_usd=None,
            llm_callable=fake_llm,
            dry_run=False,
        )
    )

    # Terminal substitution: 1 meaningful ablation, 0 LLM agent calls
    # for the terminal swap (we substitute directly). Upstream nodes
    # n0/n1/n2 were AFTER the terminal in V's order, so all three are
    # skip_downstream.
    assert summary.n_flipping_pairs == 1
    assert summary.n_ablation_points == 4
    assert summary.n_ablation_points_meaningful == 1
    assert len(fake_llm.calls) == 0  # terminal substitution, no re-exec
    terminal_results = [r for r in summary.results if r.kind == "substitute_terminal"]
    assert len(terminal_results) == 1
    assert terminal_results[0].ablated_node_id == "n3"
    assert terminal_results[0].new_v_score == pytest.approx(1.0)
    assert terminal_results[0].flip_closed is True
    skip = [r for r in summary.results if r.kind == "skip_downstream"]
    assert len(skip) == 3


def test_ablate_async_reexecutes_terminal_with_hybrid_prior(tmp_path):
    """Non-terminal ablation: build hybrid prior with canonical's output
    for N substituted into V's prior slice, re-execute terminal, re-judge.

    Mock jury returns 1.0 for the upstream ablation → flip_closed=True."""
    canonical_outputs = [
        _make_node_output_dict("n0", response_text="CANONICAL n0 says APPROVE"),
        _make_node_output_dict("n1", response_text="CANONICAL n1 agrees"),
        _make_node_output_dict("n2", response_text="CANONICAL n2 concurs"),
        _make_node_output_dict("n3", response_text="CANONICAL terminal: APPROVE"),
    ]
    v_outputs = [
        _make_node_output_dict("n0", response_text="V n0 worried"),
        _make_node_output_dict("n1", response_text="V n1 alarmed"),
        _make_node_output_dict("n2", response_text="V n2 says risk"),
        _make_node_output_dict("n3", response_text="V terminal: DENY"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    # V's order matches canonical's (n0..n3 terminal-last) — so every
    # upstream node IS in V's prior slice and ablating it triggers
    # terminal re-execution.
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=tuple(
            Node(id=f"n{i}", role=f"r{i}", llm_config=cfg) for i in range(4)
        ),
        edges=(Edge(from_id="n0", to_id="n1"),),
        entrypoint="n0",
        terminal=("n3",),
    )

    # 3 upstream re-executions (n0, n1, n2) + 1 terminal sub (n3) =
    # 4 meaningful jury calls. Sequenced: 3 re-execs flip to APPROVE,
    # terminal sub also APPROVE.
    judge = MockJuryJudge(scores_queue=[1.0, 1.0, 1.0, 1.0])
    fake_llm = FakeLLM(response_text="HYBRID terminal: APPROVE")

    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir,
            run_id="ablate-001",
            triples=triples,
            by_hash={canonical_id: topo, v_id: topo},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge,
            max_cost_usd=None,
            llm_callable=fake_llm,
            dry_run=False,
        )
    )

    assert summary.n_ablation_points_meaningful == 4
    # Three upstream re-executions → exactly 3 mock LLM calls.
    assert len(fake_llm.calls) == 3
    reexec = [r for r in summary.results if r.kind == "reexecute_terminal"]
    assert len(reexec) == 3
    assert all(r.flip_closed is True for r in reexec)
    assert all(r.new_terminal_response_text.startswith("HYBRID") for r in reexec)
    sub = [r for r in summary.results if r.kind == "substitute_terminal"]
    assert len(sub) == 1
    assert sub[0].flip_closed is True

    # Attribution by_node: every node 100% (every ablation closed).
    # is_terminal flag carries the baseline marker.
    assert summary.by_node["n3"]["is_terminal"] is True
    assert summary.by_node["n3"]["attribution_pct"] == 100.0
    for nid in ("n0", "n1", "n2"):
        assert summary.by_node[nid]["is_terminal"] is False
        assert summary.by_node[nid]["attribution_pct"] == 100.0


def test_ablate_async_flip_does_not_close_when_terminal_repeats_v(tmp_path):
    """If the re-executed terminal produces a V-aligned verdict, the flip
    does NOT close — attribution_pct should reflect that."""
    canonical_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n3", response_text="CANONICAL APPROVE"),
    ]
    v_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n3", response_text="V DENY"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=(
            Node(id="n0", role="r0", llm_config=cfg),
            Node(id="n1", role="r1", llm_config=cfg),
            Node(id="n3", role="r3", llm_config=cfg),
        ),
        edges=(Edge(from_id="n0", to_id="n1"),),
        entrypoint="n0",
        terminal=("n3",),
    )
    # Jury: 2 upstream re-judgments DENY (0.0), terminal sub APPROVE (1.0).
    judge = MockJuryJudge(scores_queue=[0.0, 0.0, 1.0])
    fake_llm = FakeLLM()

    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir,
            run_id="ablate-001",
            triples=triples,
            by_hash={canonical_id: topo, v_id: topo},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge,
            max_cost_usd=None,
            llm_callable=fake_llm,
            dry_run=False,
        )
    )
    # Upstream ablations: V's DENY persists → flip did NOT close.
    upstream = [r for r in summary.results if r.kind == "reexecute_terminal"]
    assert all(r.flip_closed is False for r in upstream)
    assert summary.by_node["n0"]["attribution_pct"] == 0.0
    assert summary.by_node["n1"]["attribution_pct"] == 0.0
    # Terminal sub still closes (control column).
    assert summary.by_node["n3"]["attribution_pct"] == 100.0


def test_ablate_async_skip_no_canonical_output(tmp_path):
    """A node present in V but absent from canonical's outputs is
    recorded as skip_no_canonical_output, not failed."""
    canonical_outputs = [
        # canonical has only n0 + terminal
        _make_node_output_dict("n0"),
        _make_node_output_dict("n3", response_text="CANONICAL APPROVE"),
    ]
    v_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),  # canonical has no n1
        _make_node_output_dict("n3", response_text="V DENY"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=(
            Node(id="n0", role="r0", llm_config=cfg),
            Node(id="n1", role="r1", llm_config=cfg),
            Node(id="n3", role="r3", llm_config=cfg),
        ),
        edges=(Edge(from_id="n0", to_id="n1"),),
        entrypoint="n0",
        terminal=("n3",),
    )
    judge = MockJuryJudge(scores_queue=[1.0, 1.0])  # n0 re-exec + n3 sub
    fake_llm = FakeLLM()

    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir,
            run_id="ablate-001",
            triples=triples,
            by_hash={canonical_id: topo, v_id: topo},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge,
            max_cost_usd=None,
            llm_callable=fake_llm,
            dry_run=False,
        )
    )
    skipped = [r for r in summary.results if r.kind == "skip_no_canonical_output"]
    assert len(skipped) == 1
    assert skipped[0].ablated_node_id == "n1"
    assert summary.n_ablation_points_meaningful == 2  # n0 re-exec + n3 sub


def test_ablate_async_terminal_reexec_failure_is_recorded(tmp_path):
    """A terminal LLM call that raises an exception lands as kind='failed',
    not as a crashed pass."""
    canonical_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n3", response_text="CANONICAL APPROVE"),
    ]
    v_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n3", response_text="V DENY"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=(
            Node(id="n0", role="r0", llm_config=cfg),
            Node(id="n3", role="r3", llm_config=cfg),
        ),
        edges=(Edge(from_id="n0", to_id="n3"),),
        entrypoint="n0",
        terminal=("n3",),
    )

    async def boom_llm(**kwargs):
        raise RuntimeError("network down")

    # Terminal substitution still fires (no LLM call) → 1 jury call.
    judge = MockJuryJudge(scores_queue=[1.0])
    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir,
            run_id="ablate-001",
            triples=triples,
            by_hash={canonical_id: topo, v_id: topo},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge,
            max_cost_usd=None,
            llm_callable=boom_llm,
            dry_run=False,
        )
    )
    failed = [r for r in summary.results if r.kind == "failed"]
    assert len(failed) == 1
    assert failed[0].ablated_node_id == "n0"
    assert failed[0].new_v_score is None
    assert failed[0].flip_closed is None
    # Pass did NOT crash; terminal substitution still ran.
    sub = [r for r in summary.results if r.kind == "substitute_terminal"]
    assert len(sub) == 1


def test_ablate_async_cost_cap_hard_stop_writes_partial(tmp_path):
    """Cost-cap hard-stop persists the partial sidecar with partial=True."""
    canonical_outputs = [
        _make_node_output_dict(f"n{i}", response_text=f"CANONICAL n{i}")
        for i in range(4)
    ]
    v_outputs = [
        _make_node_output_dict(f"n{i}", response_text=f"V n{i}")
        for i in range(4)
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=tuple(
            Node(id=f"n{i}", role=f"r{i}", llm_config=cfg) for i in range(4)
        ),
        edges=(Edge(from_id="n0", to_id="n1"),),
        entrypoint="n0",
        terminal=("n3",),
    )

    # Each terminal re-exec bills ~$0.0001 via cost_for_call's gemini-
    # flash rate × 5 output tokens. Set cap so it trips after 2 ablations.
    judge = MockJuryJudge(scores_queue=[1.0] * 10)
    fake_llm = FakeLLM()

    # Force a non-trivial per-call cost so the cap actually fires.
    # Substitute a llm_callable that reports large usage.
    async def expensive_llm(**kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "model": kwargs.get("model", ""),
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        }

    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir,
            run_id="ablate-001",
            triples=triples,
            by_hash={canonical_id: topo, v_id: topo},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge,
            max_cost_usd=0.05,  # very small cap
            llm_callable=expensive_llm,
            dry_run=False,
        )
    )
    assert summary.partial is True
    # At least one ablation completed before the cap tripped.
    assert summary.n_ablation_points >= 1


# ---------------------------------------------------------------------------
# JuryJudge reconstruction
# ---------------------------------------------------------------------------


def test_build_jury_from_resolved_dict(tmp_path):
    """`jury_models_resolved` (post-parse) takes precedence."""
    rubric = tmp_path / "rubric.txt"
    rubric.write_text(
        "Score: 5\nRationale: stub\n", encoding="utf-8"
    )
    cli_args = {
        "rubric": str(rubric),
        "jury_of_3": True,
        "jury_models_resolved": {
            "anthropic": "anthropic/claude-opus-4-7",
            "openai": "openai/gpt-5",
            "gemini": "google/gemini-2.5-pro",
        },
    }
    judge = _build_jury(cli_args, llm_callable=None, dry_run=True)
    # Vendor names sorted (per JuryJudge construction-order property).
    assert set(judge.vendors) == {"anthropic", "openai", "gemini"}
    assert judge.dry_run is True


def test_build_jury_falls_back_to_verbatim_string(tmp_path):
    """When jury_models_resolved is missing, fall back to re-parsing the
    verbatim ``--jury-models`` string."""
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("rubric\n", encoding="utf-8")
    cli_args = {
        "rubric": str(rubric),
        "jury_of_3": True,
        "jury_models": "google/gemini-2.5-flash,google/gemini-2.5-pro,deepseek/deepseek-chat",
    }
    judge = _build_jury(cli_args, llm_callable=None, dry_run=True)
    # The CLI's _parse_jury_models normalizes by case-folded sort then
    # derives provider-prefix vendor names. With one duplicate "google"
    # prefix the suffix walker produces google + google_2 + deepseek.
    assert "google" in judge.vendors
    assert "google_2" in judge.vendors
    assert "deepseek" in judge.vendors


def test_build_jury_uses_defaults_when_no_jury_models(tmp_path):
    """Missing jury_models AND jury_models_resolved → use BLUEPRINT default."""
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("rubric\n", encoding="utf-8")
    cli_args = {"rubric": str(rubric), "jury_of_3": True}
    judge = _build_jury(cli_args, llm_callable=None, dry_run=True)
    assert set(judge.vendors) == {"anthropic", "openai", "gemini"}


# ---------------------------------------------------------------------------
# Misc unit tests
# ---------------------------------------------------------------------------


def test_truncate_response_codepoint_safe():
    """Truncation never produces invalid UTF-8 mid-codepoint."""
    s = "x" * 10_000 + "é" * 500  # 'é' = 2 bytes UTF-8
    truncated = _truncate_response(s)
    # Round-trip through UTF-8: must not raise.
    assert truncated is not None
    truncated.encode("utf-8").decode("utf-8")
    assert "[truncated]" in truncated


def test_truncate_response_below_cap_is_passthrough():
    s = "hello"
    assert _truncate_response(s) == s


def test_truncate_response_none_passthrough():
    assert _truncate_response(None) is None


def test_dicts_to_node_outputs_round_trip():
    """JSON-backed dicts hydrate back into NodeOutput dataclasses."""
    dicts = [
        _make_node_output_dict("n0", response_text="hello"),
        _make_node_output_dict("n1", response_text="world"),
    ]
    outputs = _dicts_to_node_outputs(dicts)
    assert len(outputs) == 2
    assert outputs[0].node_id == "n0"
    assert outputs[0].response_text == "hello"
    assert outputs[1].role == "role-n1"


def test_ablation_summary_dataclass_shape():
    """AblationSummary fields are stable — sidecar consumers depend on these."""
    summary = AblationSummary(
        n_flipping_pairs=2,
        n_ablation_points=8,
        n_ablation_points_meaningful=6,
        total_cost_usd=0.42,
        partial=False,
        by_node={
            "n0": {
                "flips_closed": 0, "total_meaningful": 2,
                "attribution_pct": 0.0, "is_terminal": False,
            },
        },
        results=(),
    )
    assert summary.n_flipping_pairs == 2
    assert summary.by_node["n0"]["attribution_pct"] == 0.0


def test_ablation_sidecar_write_round_trip(tmp_path):
    """A summary writes + reads back identically."""
    from topofuzz.ablation import _write_ablation_sidecar

    result = AblationResult(
        canonical_variant_id="c" * 64,
        flipping_variant_id="v" * 64,
        eval_id="eval_001",
        ablated_node_id="n0",
        kind="reexecute_terminal",
        original_v_score=0.0,
        canonical_score=1.0,
        new_v_score=1.0,
        flip_closed=True,
        cost_usd=0.03,
        new_terminal_response_text="HYBRID",
        jury_votes=(
            {"vendor": "google", "score": 1.0, "rationale": "approve"},
        ),
    )
    summary = AblationSummary(
        n_flipping_pairs=1,
        n_ablation_points=1,
        n_ablation_points_meaningful=1,
        total_cost_usd=0.03,
        partial=False,
        by_node={
            "n0": {
                "flips_closed": 1, "total_meaningful": 1,
                "attribution_pct": 100.0, "is_terminal": False,
            },
        },
        results=(result,),
    )
    _write_ablation_sidecar(tmp_path, summary)
    data = read_ablation_sidecar(tmp_path)
    assert data is not None
    assert data["schema_version"] == 1
    assert data["results"][0]["ablated_node_id"] == "n0"
    assert data["results"][0]["flip_closed"] is True
    assert data["by_node"]["n0"]["attribution_pct"] == 100.0


# ---------------------------------------------------------------------------
# Pass-1 fix regression tests
# ---------------------------------------------------------------------------


def test_atomic_sidecar_write_preserves_prior_on_crash(tmp_path, monkeypatch):
    """RT BLOCKER B1: a SIGKILL / disk-full mid-write must NOT destroy the
    previously-good sidecar. The atomic temp+replace pattern means the
    original survives byte-for-byte until ``os.replace`` completes."""
    from topofuzz.ablation import _write_ablation_sidecar

    summary_good = AblationSummary(
        n_flipping_pairs=1, n_ablation_points=1, n_ablation_points_meaningful=1,
        total_cost_usd=0.0, partial=False,
        by_node={"n0": {"flips_closed": 1, "total_meaningful": 1,
                        "attribution_pct": 100.0, "is_terminal": False}},
        results=(),
    )
    _write_ablation_sidecar(tmp_path, summary_good)
    pre_bytes = (tmp_path / "ablation.json").read_bytes()
    assert pre_bytes, "prior sidecar must land before the crash test"

    # Monkeypatch os.fdopen to write some bytes then raise mid-write.
    import os as _os
    real_fdopen = _os.fdopen
    raise_token = RuntimeError("disk full")

    def boom_fdopen(fd, mode, *args, **kwargs):
        f = real_fdopen(fd, mode, *args, **kwargs)
        original_write = f.write

        def _boom(*a, **k):
            original_write("partial bytes")
            raise raise_token

        f.write = _boom
        return f

    monkeypatch.setattr("os.fdopen", boom_fdopen)
    summary_new = AblationSummary(
        n_flipping_pairs=99, n_ablation_points=99,
        n_ablation_points_meaningful=99, total_cost_usd=99.0, partial=True,
        by_node={}, results=(),
    )
    with pytest.raises(RuntimeError, match="disk full"):
        _write_ablation_sidecar(tmp_path, summary_new)

    # The prior good sidecar must still be on disk, untouched.
    post_bytes = (tmp_path / "ablation.json").read_bytes()
    assert post_bytes == pre_bytes
    # No temp leak in the run dir.
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".ablation")]
    assert leftover == [], f"temp file leaked: {leftover}"


def test_load_judge_score_prefers_score_value_over_passed(tmp_path):
    """RT BLOCKER B2: a binary-judge cell with score_value=0.72, passed=0
    must return 0.72 in AblationResult.canonical_score, not 0.0 (the
    `passed`-only lossy projection). The field name promises the score."""
    from topofuzz.ablation import _load_judge_score

    A = "a" * 64
    rows = [
        ("synth", A, "eval_001", "continuous", 0.72, 0, None, None, None),
    ]
    run_dir = _write_synthetic_run(tmp_path, judge_rows=rows, hashes=[A])
    assert _load_judge_score(run_dir, A, "eval_001") == pytest.approx(0.72)


def test_load_judge_score_falls_back_to_passed_when_score_null(tmp_path):
    """If score_value is NULL but passed=1, return 1.0 — preserves binary
    cells whose score column was never populated."""
    from topofuzz.ablation import _load_judge_score

    A = "a" * 64
    rows = [
        ("synth", A, "eval_001", "binary", None, 1, None, None, None),
    ]
    run_dir = _write_synthetic_run(tmp_path, judge_rows=rows, hashes=[A])
    assert _load_judge_score(run_dir, A, "eval_001") == 1.0


def test_estimate_jury_cost_returns_floor_on_reflection_failure(tmp_path):
    """RT MAJOR M1 / Fuzzer MAJOR M2: a JuryJudge with a broken vendors
    property must NOT silently return $0 — that would let an unbounded
    jury pass sneak past the cap. The fallback floor is non-zero."""
    from topofuzz.ablation import _JURY_COST_FALLBACK_FLOOR_USD, _estimate_jury_cost

    class BrokenJury:
        @property
        def vendors(self):
            raise AttributeError("vendors not available on this subclass")

    cost = _estimate_jury_cost(BrokenJury())
    assert cost >= _JURY_COST_FALLBACK_FLOOR_USD
    assert cost > 0.0


def test_estimate_jury_cost_returns_floor_on_empty_vendors(tmp_path):
    """A judge with empty vendors tuple must still produce the floor —
    a zero estimate would let cap accounting under-bill."""
    from topofuzz.ablation import _JURY_COST_FALLBACK_FLOOR_USD, _estimate_jury_cost

    @dataclass
    class EmptyJury:
        vendors: tuple = ()

    cost = _estimate_jury_cost(EmptyJury())
    assert cost >= _JURY_COST_FALLBACK_FLOOR_USD


def test_consecutive_failures_abort_the_pass(tmp_path):
    """RT MAJOR M2 / Fuzzer parallel: a programmer bug producing back-
    to-back failures aborts the pass rather than silently producing
    N kind=failed rows. Real vendor blips remain demoted to kind=failed."""
    canonical_outputs = [
        _make_node_output_dict(f"n{i}") for i in range(4)
    ]
    v_outputs = [
        _make_node_output_dict(f"n{i}") for i in range(4)
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=tuple(Node(id=f"n{i}", role=f"r{i}", llm_config=cfg) for i in range(4)),
        edges=(Edge(from_id="n0", to_id="n1"),),
        entrypoint="n0",
        terminal=("n3",),
    )

    async def always_boom(**kwargs):
        raise TypeError("simulated signature drift")

    judge = MockJuryJudge(scores_queue=[1.0] * 20)
    with pytest.raises(AblationError, match="consecutive"):
        asyncio.run(
            _ablate_async(
                run_dir=run_dir, run_id="ablate-001", triples=triples,
                by_hash={canonical_id: topo, v_id: topo},
                eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
                judge=judge, max_cost_usd=None, llm_callable=always_boom,
                dry_run=False,
            )
        )


def test_duplicate_node_ids_skip_with_warning(tmp_path, caplog):
    """RT MAJOR M3: duplicate node_id entries in node_outputs_json cause
    the per-node dict comprehension to silently keep only the last
    snapshot. Ablation refuses to guess; records skip_no_canonical_output
    and emits a WARNING log."""
    canonical_outputs = [
        _make_node_output_dict("n0", response_text="first"),
        _make_node_output_dict("n0", response_text="dup-snapshot"),  # duplicate
        _make_node_output_dict("n3", response_text="canonical-terminal"),
    ]
    v_outputs = [
        _make_node_output_dict("n0", response_text="v-first"),
        _make_node_output_dict("n3", response_text="v-terminal"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=(
            Node(id="n0", role="r0", llm_config=cfg),
            Node(id="n3", role="r3", llm_config=cfg),
        ),
        edges=(Edge(from_id="n0", to_id="n3"),),
        entrypoint="n0",
        terminal=("n3",),
    )
    judge = MockJuryJudge(scores_queue=[1.0, 1.0])
    fake_llm = FakeLLM()
    import logging as _logging
    with caplog.at_level(_logging.WARNING, logger="topofuzz.ablation"):
        summary = asyncio.run(
            _ablate_async(
                run_dir=run_dir, run_id="ablate-001", triples=triples,
                by_hash={canonical_id: topo, v_id: topo},
                eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
                judge=judge, max_cost_usd=None, llm_callable=fake_llm,
                dry_run=False,
            )
        )
    # The duplicate-bearing node_id (n0) lands as skip, not as a
    # silently-using-wrong-snapshot reexecute.
    n0_results = [r for r in summary.results if r.ablated_node_id == "n0"]
    assert all(r.kind == "skip_no_canonical_output" for r in n0_results)
    assert any("duplicate node_outputs" in m for m in caplog.messages)


def test_jury_models_resolved_disagreement_raises(tmp_path):
    """RT MAJOR M4: when manifest carries BOTH jury_models_resolved AND
    jury_models AND they don't match, refuse the pass — silently
    preferring the dict would re-judge under a different jury."""
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("rubric\n", encoding="utf-8")
    # Resolved dict says (anthropic, openai, gemini) but verbatim says
    # all-google trio. Disagree → must raise.
    cli_args = {
        "rubric": str(rubric),
        "jury_of_3": True,
        "jury_models": "google/x,google/y,google/z",
        "jury_models_resolved": {
            "anthropic": "anthropic/claude-opus-4-7",
            "openai": "openai/gpt-5",
            "gemini": "google/gemini-2.5-pro",
        },
    }
    with pytest.raises(AblationError, match="disagree"):
        _build_jury(cli_args, llm_callable=None, dry_run=True)


def test_run_ablation_rejects_custom_prompt_builder(tmp_path):
    """Fuzzer MAJOR M4: a manifest with a custom prompt_builder field
    must be rejected — F4's terminal re-exec would use the default
    builder, contaminating the measurement."""
    A, B = "a" * 64, "b" * 64
    rows = [
        ("synth", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth", B, "eval_001", "continuous", 0.1, None, None, None),
    ]
    run_dir = _write_synthetic_run(
        tmp_path, judge_rows=rows, hashes=[A, B],
        cli_args={"prompt_builder": "custom_module:build"},
    )
    with pytest.raises(AblationError, match="custom prompt_builder"):
        run_ablation_pass(run_dir)


def test_run_ablation_rejects_custom_jury_kwargs(tmp_path):
    """Fuzzer MAJOR M3: a manifest with jury-construction knobs F4
    doesn't yet plumb through _build_jury must be refused — silently
    re-judging under different kwargs would contaminate attribution."""
    A, B = "a" * 64, "b" * 64
    rows = [
        ("synth", A, "eval_001", "continuous", 0.9, None, None, None),
        ("synth", B, "eval_001", "continuous", 0.1, None, None, None),
    ]
    run_dir = _write_synthetic_run(
        tmp_path, judge_rows=rows, hashes=[A, B],
        cli_args={"jury_temperature": 0.2},
    )
    with pytest.raises(AblationError, match="jury-construction"):
        run_ablation_pass(run_dir)


def test_inspect_view_skip_rows_have_has_measurement_false():
    """Fuzzer BLOCKER B2: skip kinds produce has_measurement=False so the
    template renders em-dash (not a 0% bar). A 0% bar reads as 'this
    agent contributes nothing' — false for a structurally-skipped node."""
    from unittest.mock import MagicMock

    from topofuzz.inspect import _ablation_view

    sidecar = {
        "schema_version": 1,
        "results": [
            {
                "canonical_variant_id": "c" * 64,
                "flipping_variant_id": "v" * 64,
                "eval_id": "eval_001",
                "ablated_node_id": "n0",
                "kind": "skip_downstream",
                "flip_closed": None,
                "new_terminal_response_text": None,
            },
            {
                "canonical_variant_id": "c" * 64,
                "flipping_variant_id": "v" * 64,
                "eval_id": "eval_001",
                "ablated_node_id": "n3",
                "kind": "substitute_terminal",
                "flip_closed": True,
                "new_terminal_response_text": "APPROVE",
            },
        ],
    }
    # Real Jinja2 Markup/escape for accuracy.
    from markupsafe import Markup, escape

    view = _ablation_view(
        sidecar,
        canonical_hash="c" * 64,
        flipping_hash="v" * 64,
        eval_id="eval_001",
        escape=escape,
        Markup=Markup,
    )
    assert view is not None
    rows_by_id = {r["node_id"]: r for r in view["rows"]}
    assert rows_by_id["n0"]["has_measurement"] is False
    assert rows_by_id["n3"]["has_measurement"] is True
    # Headline reflects the structural-cause finding.
    assert (
        "after the terminal" in view["headline"]
        or "no upstream" in view["headline"].lower()
    )


def test_inspect_view_scope_note_present():
    """Fuzzer MAJOR M1: the per-flip view's scope_note must unambiguously
    state that the attribution is for THIS flip (not the run-level)."""
    from markupsafe import Markup, escape
    from topofuzz.inspect import _ablation_view

    sidecar = {
        "schema_version": 1,
        "results": [{
            "canonical_variant_id": "c" * 64,
            "flipping_variant_id": "v" * 64,
            "eval_id": "eval_001",
            "ablated_node_id": "n3",
            "kind": "substitute_terminal",
            "flip_closed": True,
            "new_terminal_response_text": "x",
        }],
    }
    view = _ablation_view(
        sidecar, canonical_hash="c" * 64, flipping_hash="v" * 64,
        eval_id="eval_001", escape=escape, Markup=Markup,
    )
    assert view is not None
    assert "scope_note" in view
    assert "this flip only" in view["scope_note"].lower()


def test_failed_jury_records_cost_estimate_not_zero(tmp_path):
    """Pass-1 RT Mi6: a failed jury call must still count its estimated
    cost against the cap, so a pass with many failed jurys doesn't
    silently exceed the budget."""
    canonical_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n3", response_text="canonical"),
    ]
    v_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n3", response_text="v"),
    ]
    run_dir, canonical_id, v_id = _build_run_dir_for_ablate(
        tmp_path,
        canonical_outputs=canonical_outputs,
        v_outputs=v_outputs,
    )
    triples = find_all_flipping_triples(run_dir)
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    topo = Topology(
        name="toy",
        nodes=(
            Node(id="n0", role="r0", llm_config=cfg),
            Node(id="n3", role="r3", llm_config=cfg),
        ),
        edges=(Edge(from_id="n0", to_id="n3"),),
        entrypoint="n0",
        terminal=("n3",),
    )

    @dataclass
    class FlakyJudge:
        calls: int = 0
        vendors: tuple = ("a", "b", "c")
        def score(self, payload):
            self.calls += 1
            # First call (terminal substitution for n3) succeeds.
            # Second call (n0 re-judge) fails.
            if self.calls == 1:
                return ContinuousScore(value=0.0, rationale="m", votes=())
            raise RuntimeError("simulated jury failure")

    judge = FlakyJudge()
    fake_llm = FakeLLM()

    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir, run_id="ablate-001", triples=triples,
            by_hash={canonical_id: topo, v_id: topo},
            eval_rows_by_id={"eval_001": {"id": "eval_001", "input": "x"}},
            judge=judge, max_cost_usd=None, llm_callable=fake_llm,
            dry_run=False,
        )
    )
    failed = [r for r in summary.results if r.kind == "failed"]
    assert len(failed) == 1
    # Failed row's cost_usd is exec_cost + jury floor (>= floor alone).
    assert failed[0].cost_usd > 0.0


def test_inspect_view_failed_row_renders_explicit_message(tmp_path):
    """Pass-1 Fuzzer MINOR m3: a failed-row's new_terminal_response_html
    must read as 'failed', not as a leftover terminal text the buyer
    might infer is the post-ablation output."""
    from markupsafe import Markup, escape
    from topofuzz.inspect import _ablation_view

    sidecar = {
        "schema_version": 1,
        "results": [{
            "canonical_variant_id": "c" * 64,
            "flipping_variant_id": "v" * 64,
            "eval_id": "eval_001",
            "ablated_node_id": "n0",
            "kind": "failed",
            "flip_closed": None,
            "new_terminal_response_text": "leftover text",
        }],
    }
    view = _ablation_view(
        sidecar, canonical_hash="c" * 64, flipping_hash="v" * 64,
        eval_id="eval_001", escape=escape, Markup=Markup,
    )
    assert view is not None
    row = view["rows"][0]
    html = str(row["new_terminal_response_html"])
    assert "re-execution failed" in html
    # The original leftover MUST NOT appear in the rendered HTML.
    assert "leftover" not in html
