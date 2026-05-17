# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``topofuzz.counterfactual_audit`` — F7 eval-axis audit driver.

The audit driver mirrors F4's ablation pass on the INPUT axis: it swaps
a demographic token in the canonical eval, re-runs the variant, and
attributes any verdict flip to the upstream agent whose response was
load-bearing. These tests exercise the driver's gates + math without
hitting the network.

Layered into:

* **Refusal gates** — non-jury source runs / custom prompt builders /
  unrecognised jury kwargs / unknown dim strings all raise eagerly.
* **Sidecar I/O** — round-trip; absent file returns None; corrupt JSON
  raises a clear AuditError; ``_swap_pair_key`` rejects pipe chars.
* **Helpers** — ``_truncate_response`` is UTF-8 codepoint-safe at the
  boundary; ``_atomic_write_text`` survives mid-write crashes.
* **Driver math** — by_node attribution percentage, ``max_variants`` +
  ``max_canonical_evals`` truncation, dry-run no-LLM contract, flipping
  vs non-flipping triple bookkeeping, cost-cap soft-warn + hard-stop,
  consecutive-failure abort, cancellation propagation.
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

from topofuzz.counterfactual_audit import (  # noqa: E402
    AUDIT_SIDECAR_FILENAME,
    AuditError,
    CounterfactualAuditResult,
    CounterfactualAuditSummary,
    _MAX_CONSECUTIVE_FAILURES,
    _RESPONSE_TEXT_CAP,
    _aggregate,
    _atomic_write_text,
    _audit_async,
    _swap_pair_key,
    _truncate_response,
    _write_audit_sidecar,
    read_audit_sidecar,
    run_audit_pass,
)
from topofuzz.ir import Edge, LLMConfig, Node, Topology  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class MockJuryJudge:
    """Mirrors test_ablation.MockJuryJudge: queue-driven score returns."""

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
    response_text: str = "[MOCK terminal output]"
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": self.response_text}}],
            "model": kwargs.get("model", ""),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _toy_topology(n_nodes: int = 3) -> Topology:
    cfg = LLMConfig(model="google/gemini-2.5-flash", temperature=0.0)
    nodes = tuple(
        Node(id=f"n{i}", role=f"r{i}", llm_config=cfg) for i in range(n_nodes)
    )
    edges = tuple(
        Edge(from_id=f"n{i}", to_id=f"n{i+1}") for i in range(n_nodes - 1)
    )
    return Topology(
        name="toy_lin",
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


def _write_manifest_run(
    tmp_path: Path,
    *,
    hashes: list[str],
    cli_args: dict[str, Any] | None = None,
    eval_ids: list[str] | None = None,
) -> Path:
    """Lay down a manifest + bare run.sqlite (no cells/judge rows)."""
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    manifest_cli = {
        "fixture": "x.py:build_graph",
        "eval": "e.csv",
        "rng_seed": 0,
        "variants": len(hashes),
        "jury_of_3": True,
        "rubric": "rubric.txt",
    }
    if cli_args:
        manifest_cli.update(cli_args)
    manifest = {
        "run_id": "audit-001",
        "variant_canonical_hashes": hashes,
        "variant_orderings": [["n0", "n1", "n2"]] * len(hashes),
        "eval_ids": eval_ids or ["eval_001"],
        "cli_args": manifest_cli,
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
    db.commit()
    db.close()
    return run_dir


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------


def test_read_audit_sidecar_returns_none_when_absent(tmp_path):
    assert read_audit_sidecar(tmp_path) is None


def test_read_audit_sidecar_raises_on_corrupt_json(tmp_path):
    (tmp_path / AUDIT_SIDECAR_FILENAME).write_text("{ malformed", encoding="utf-8")
    with pytest.raises(AuditError, match="unreadable / not valid JSON"):
        read_audit_sidecar(tmp_path)


def test_read_audit_sidecar_raises_on_non_object_root(tmp_path):
    (tmp_path / AUDIT_SIDECAR_FILENAME).write_text("[]", encoding="utf-8")
    with pytest.raises(AuditError, match="not a JSON object"):
        read_audit_sidecar(tmp_path)


def test_audit_sidecar_round_trip(tmp_path):
    """A summary writes + reads back identically; key fields preserved."""
    result = CounterfactualAuditResult(
        variant_id="v" * 64,
        canonical_eval_id="eval_001",
        counterfactual_eval_id="eval_001__cf_name_jamal_jones",
        dim="name",
        from_token="Greg Walsh",
        to_token="Jamal Jones",
        ablated_node_id="risk",
        kind="reexecute_terminal",
        canonical_score=1.0,
        original_cf_score=0.0,
        new_cf_score=1.0,
        flip_closed=True,
        cost_usd=0.04,
        new_terminal_response_text="APPROVE",
        jury_votes=({"vendor": "g", "score": 1.0, "rationale": "ok"},),
    )
    summary = CounterfactualAuditSummary(
        n_canonical_pairs=1,
        n_flipping_pairs=1,
        n_ablation_points=1,
        n_ablation_points_meaningful=1,
        dims=("name",),
        total_cost_usd=0.04,
        partial=False,
        by_dim={
            "name": {
                "flipping_pairs": 1, "total_pairs": 1,
                "flip_rate_pct": 100.0, "by_node": {},
            }
        },
        by_swap_pair={
            "name|Greg Walsh|Jamal Jones": {
                "dim": "name", "from_token": "Greg Walsh",
                "to_token": "Jamal Jones",
                "flipping_pairs": 1, "total_pairs": 1, "by_node": {},
            }
        },
        by_node={
            "risk": {
                "flips_closed": 1, "total_meaningful": 1,
                "attribution_pct": 100.0, "is_terminal": False,
            }
        },
        unpriced_call_count=0,
        unpriced_models=(),
        results=(result,),
    )
    _write_audit_sidecar(tmp_path, summary)
    data = read_audit_sidecar(tmp_path)
    assert data is not None
    # schema_version bumped 1→2 by Pass-1 BLOCKER B1+B4 fix.
    assert data["schema_version"] == 2
    assert data["n_flipping_pairs"] == 1
    # B4 fields are present in the sidecar.
    assert data["unpriced_call_count"] == 0
    assert data["unpriced_models"] == []
    assert data["by_node"]["risk"]["attribution_pct"] == 100.0
    assert data["results"][0]["dim"] == "name"
    assert data["results"][0]["from_token"] == "Greg Walsh"
    assert data["dims"] == ["name"]


def test_audit_sidecar_atomic_write_preserves_prior_on_crash(tmp_path, monkeypatch):
    """A SIGKILL / disk-full mid-write must NOT clobber an existing sidecar.

    The atomic temp+replace pattern means the original survives byte-
    for-byte until ``os.replace`` completes.
    """
    summary_good = CounterfactualAuditSummary(
        n_canonical_pairs=1, n_flipping_pairs=0, n_ablation_points=0,
        n_ablation_points_meaningful=0, dims=("name",),
        total_cost_usd=0.0, partial=False,
        by_dim={}, by_swap_pair={}, by_node={},
        unpriced_call_count=0, unpriced_models=(),
        results=(),
    )
    _write_audit_sidecar(tmp_path, summary_good)
    pre_bytes = (tmp_path / AUDIT_SIDECAR_FILENAME).read_bytes()
    assert pre_bytes

    import os as _os
    real_fdopen = _os.fdopen

    def boom_fdopen(fd, mode, *args, **kwargs):
        f = real_fdopen(fd, mode, *args, **kwargs)
        original_write = f.write

        def _boom(*a, **k):
            original_write("partial")
            raise RuntimeError("disk full")

        f.write = _boom
        return f

    monkeypatch.setattr("os.fdopen", boom_fdopen)
    summary_new = CounterfactualAuditSummary(
        n_canonical_pairs=99, n_flipping_pairs=99, n_ablation_points=99,
        n_ablation_points_meaningful=99, dims=("name",),
        total_cost_usd=99.0, partial=True,
        by_dim={}, by_swap_pair={}, by_node={},
        unpriced_call_count=0, unpriced_models=(),
        results=(),
    )
    with pytest.raises(RuntimeError, match="disk full"):
        _write_audit_sidecar(tmp_path, summary_new)
    # Prior good sidecar survives untouched.
    assert (tmp_path / AUDIT_SIDECAR_FILENAME).read_bytes() == pre_bytes
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(f".{AUDIT_SIDECAR_FILENAME}")]
    assert leftover == [], f"temp file leaked: {leftover}"


# ---------------------------------------------------------------------------
# _swap_pair_key
# ---------------------------------------------------------------------------


def test_swap_pair_key_flattened_shape():
    assert _swap_pair_key("name", "Greg Walsh", "Jamal Jones") == "name|Greg Walsh|Jamal Jones"


def test_swap_pair_key_rejects_pipe_in_dim():
    with pytest.raises(AuditError, match="pipe character"):
        _swap_pair_key("na|me", "a", "b")


def test_swap_pair_key_rejects_pipe_in_from_token():
    with pytest.raises(AuditError, match="pipe character"):
        _swap_pair_key("name", "ev|il", "b")


def test_swap_pair_key_rejects_pipe_in_to_token():
    with pytest.raises(AuditError, match="pipe character"):
        _swap_pair_key("name", "a", "ev|il")


# ---------------------------------------------------------------------------
# _truncate_response — UTF-8 codepoint safety
# ---------------------------------------------------------------------------


def test_truncate_response_none_passthrough():
    assert _truncate_response(None) is None


def test_truncate_response_below_cap_unchanged():
    assert _truncate_response("hello") == "hello"


def test_truncate_response_above_cap_truncated():
    s = "x" * (_RESPONSE_TEXT_CAP + 100)
    out = _truncate_response(s)
    assert out is not None
    assert "[truncated]" in out


def test_truncate_response_codepoint_safe_at_4_byte_utf8_boundary():
    """Feed a 4-byte UTF-8 char straddling the cap byte and verify the
    output is valid UTF-8 — no mid-codepoint cut.

    'é' is 2 bytes; we use 𠮷 (U+20BB7, 4 bytes UTF-8) so we cover the
    worst case where 3 bytes of a 4-byte sequence land before the cap.
    """
    # Fill up to one byte before the cap, then place a 4-byte char
    # straddling the boundary.
    prefix = "x" * (_RESPONSE_TEXT_CAP - 1)
    s = prefix + "𠮷" + "y" * 50
    out = _truncate_response(s)
    assert out is not None
    # Round-trip must succeed without UnicodeDecodeError.
    out.encode("utf-8").decode("utf-8")
    # 4-byte char straddled the boundary so it's dropped, not corrupted.
    assert "[truncated]" in out


def test_truncate_response_codepoint_safe_at_2_byte_utf8_boundary():
    s = "x" * (_RESPONSE_TEXT_CAP - 1) + "é" + "y" * 100
    out = _truncate_response(s)
    assert out is not None
    out.encode("utf-8").decode("utf-8")


# ---------------------------------------------------------------------------
# run_audit_pass — gate refusals
# ---------------------------------------------------------------------------


def test_run_audit_rejects_single_judge_source(tmp_path):
    """A jury_of_3=False source must be refused — single-judge runs
    inherit ~26% noise that drowns the attribution signal."""
    run_dir = _write_manifest_run(
        tmp_path, hashes=["a" * 64], cli_args={"jury_of_3": False}
    )
    with pytest.raises(AuditError, match="jury_of_3"):
        run_audit_pass(run_dir)


def test_run_audit_rejects_custom_prompt_builder(tmp_path):
    run_dir = _write_manifest_run(
        tmp_path, hashes=["a" * 64],
        cli_args={"prompt_builder": "custom_module:build"},
    )
    with pytest.raises(AuditError, match="custom prompt_builder"):
        run_audit_pass(run_dir)


def test_run_audit_rejects_custom_prompt_builder_module(tmp_path):
    run_dir = _write_manifest_run(
        tmp_path, hashes=["a" * 64],
        cli_args={"prompt_builder_module": "x"},
    )
    with pytest.raises(AuditError, match="custom prompt_builder"):
        run_audit_pass(run_dir)


def test_run_audit_rejects_jury_temperature_field(tmp_path):
    run_dir = _write_manifest_run(
        tmp_path, hashes=["a" * 64],
        cli_args={"jury_temperature": 0.2},
    )
    with pytest.raises(AuditError, match="jury-construction"):
        run_audit_pass(run_dir)


def test_run_audit_rejects_jury_max_tokens_field(tmp_path):
    run_dir = _write_manifest_run(
        tmp_path, hashes=["a" * 64],
        cli_args={"jury_max_tokens": 1024},
    )
    with pytest.raises(AuditError, match="jury-construction"):
        run_audit_pass(run_dir)


def test_run_audit_rejects_unknown_dim_before_llm(tmp_path):
    """An unknown dim string is rejected at the manifest-validation gate
    before the LLM is ever touched."""
    run_dir = _write_manifest_run(tmp_path, hashes=["a" * 64])
    with pytest.raises(AuditError, match="Unknown counterfactual dim"):
        run_audit_pass(run_dir, dims=("name", "race"))  # race is not a dim


def test_run_audit_rejects_missing_manifest(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AuditError, match="manifest.json missing"):
        run_audit_pass(empty)


def test_run_audit_rejects_corrupt_manifest(tmp_path):
    run_dir = tmp_path / "corrupt"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{ malformed", encoding="utf-8")
    with pytest.raises(AuditError, match="not valid JSON"):
        run_audit_pass(run_dir)


def test_run_audit_rejects_empty_variant_hashes(tmp_path, monkeypatch):
    """A manifest with zero variants reaches the recon path then refuses."""
    run_dir = _write_manifest_run(tmp_path, hashes=[])
    # Bypass _reconstruct_variants to avoid the fixture load (no fixture file on disk).
    monkeypatch.setattr(
        "topofuzz.confirm._reconstruct_variants", lambda m, **kw: {}
    )
    monkeypatch.setattr(
        "topofuzz.confirm._load_eval_rows_by_id", lambda c, r: {}
    )
    with pytest.raises(AuditError, match="empty"):
        run_audit_pass(run_dir)


# ---------------------------------------------------------------------------
# _audit_async driver math
# ---------------------------------------------------------------------------


def test_audit_async_flipping_triple_runs_leave_one_out(tmp_path):
    """A canonical/cf pair whose verdicts disagree triggers per-node
    leave-one-out; the by_node aggregate should reflect each node."""
    canonical_outputs = [
        _make_node_output_dict("n0", response_text="canonical n0"),
        _make_node_output_dict("n1", response_text="canonical n1"),
        _make_node_output_dict("n2", response_text="canonical APPROVE"),
    ]
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "audit-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
        "INSERT INTO judge_results VALUES ('audit-001', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id,)
    )
    db.execute(
        "INSERT INTO cells VALUES ('audit-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(canonical_outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    # eval row that triggers a name swap
    canonical_row = {"id": "eval_001", "input": "Greg Walsh applied at 11211."}

    # Sequenced jury returns:
    # phase-1 cell judge: 0.0 (CF flips to DENY).
    # phase-2 leave-one-out: 3 re-judgments — n0 closes (1.0),
    #   n1 doesn't (0.0), n2 substitution closes (1.0).
    # Wait: terminal is n2, so n2 is substitute (no LLM, 1 jury call).
    # n0, n1 are upstream → reexecute (each = 1 LLM call + 1 jury call).
    judge = MockJuryJudge(scores_queue=[0.0, 1.0, 0.0, 1.0])
    fake_llm = FakeLLM(response_text="HYBRID APPROVE")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role,
                response_text=f"cf_{n.id}_text",
                model_served_version="google/gemini-2.5-flash",
                cache_hit=False, latency_ms=1,
                input_tokens=1, output_tokens=1, cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": variant_ir.canonical_hash(),
                          "eval_row": eval_row,
                          "agent_output": nodes[-1].response_text})
        return Cell(
            variant_id=variant_ir.canonical_hash(),
            eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes),
            judge_result=jr,
            judge_failure=False,
            judge_failure_reason=None,
            error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir,
            run_id="audit-001",
            variant_ids=[canonical_id],
            by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={"eval_001": canonical_row},
            dims=("name",),
            max_cost_usd=None,
            max_canonical_evals=None,
            judge=judge,
            llm_callable=fake_llm,
            dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    assert summary.n_canonical_pairs == 1
    assert summary.n_flipping_pairs == 1
    # n0 + n1 re-exec + n2 substitute = 3 meaningful experiments.
    assert summary.n_ablation_points_meaningful == 3
    # 2 upstream re-exec produced 2 LLM calls (n2 is terminal substitution).
    assert len(fake_llm.calls) == 2
    # by_node math: n0 closed (1/1=100%), n1 didn't (0/1=0%),
    # n2 substitute closed (1/1=100%).
    assert summary.by_node["n0"]["attribution_pct"] == 100.0
    assert summary.by_node["n1"]["attribution_pct"] == 0.0
    assert summary.by_node["n2"]["attribution_pct"] == 100.0
    assert summary.by_node["n2"]["is_terminal"] is True
    # Sidecar was written.
    sc = read_audit_sidecar(run_dir)
    assert sc is not None
    assert sc["n_flipping_pairs"] == 1


def test_audit_async_non_flipping_triple_skips_leave_one_out(tmp_path):
    """When canonical & cf verdicts agree, no leave-one-out fires."""
    canonical_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2", response_text="APPROVE"),
    ]
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "audit-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
        "INSERT INTO judge_results VALUES ('audit-001', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id,)
    )
    db.execute(
        "INSERT INTO cells VALUES ('audit-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(canonical_outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    canonical_row = {"id": "eval_001", "input": "Greg Walsh applied."}
    # cf cell judge also returns APPROVE (no flip).
    judge = MockJuryJudge(scores_queue=[0.9])
    fake_llm = FakeLLM()

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="audit-001",
            variant_ids=[canonical_id],
            by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={"eval_001": canonical_row},
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    assert summary.n_canonical_pairs == 1
    assert summary.n_flipping_pairs == 0
    assert summary.n_ablation_points == 0


def test_audit_async_max_canonical_evals_truncates(tmp_path):
    """max_canonical_evals caps the per-dim iteration."""
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    eval_ids = ["eval_001", "eval_002", "eval_003"]
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "audit-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": eval_ids,
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2"),
    ]
    for eid in eval_ids:
        db.execute(
            "INSERT INTO judge_results VALUES ('audit-001', ?, ?, "
            "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id, eid)
        )
        db.execute(
            "INSERT INTO cells VALUES ('audit-001', ?, ?, '', ?, NULL, NULL)",
            (canonical_id, eid, json.dumps(outputs)),
        )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    rows = {eid: {"id": eid, "input": "Greg Walsh applied."} for eid in eval_ids}
    judge = MockJuryJudge(scores_queue=[0.9, 0.9, 0.9, 0.9])
    fake_llm = FakeLLM()

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row,
                          "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="audit-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id=rows,
            dims=("name",), max_cost_usd=None,
            max_canonical_evals=2,  # cap below the 3 evals
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    # Only 2 canonical pairs ran (the third eval was truncated).
    assert summary.n_canonical_pairs == 2


def test_audit_async_dry_run_writes_sidecar_without_llm(tmp_path):
    """Dry-run completes + writes a syntactically valid audit.json with
    zero spend; counterfactual cell judging produces same outputs as
    canonical → no flips (the documented dry-run contract)."""
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "audit-dry",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2"),
    ]
    db.execute(
        "INSERT INTO judge_results VALUES ('audit-dry', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id,)
    )
    db.execute(
        "INSERT INTO cells VALUES ('audit-dry', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    judge = MockJuryJudge(scores_queue=[0.9])

    llm_was_called = False

    async def explode_llm(**kwargs):
        nonlocal llm_was_called
        llm_was_called = True
        raise AssertionError("dry-run path must not call LLM")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        # Dry-run: caller passes our explode_llm; we just don't invoke
        # it. Produce canonical-aligned outputs + a 0.9 judge score so
        # no flip triggers.
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="dry",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.0,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="audit-dry",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={
                "eval_001": {"id": "eval_001", "input": "Greg Walsh applied."}
            },
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=explode_llm, dry_run=True,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    assert not llm_was_called
    assert summary.total_cost_usd == 0.0
    # Sidecar exists and reads back as a JSON object.
    data = read_audit_sidecar(run_dir)
    assert data is not None
    assert isinstance(data, dict)
    # schema_version 2 after Pass-1 BLOCKER B1+B4 fold.
    assert data["schema_version"] == 2


def test_audit_async_consecutive_failures_abort(tmp_path):
    """N back-to-back terminal re-exec failures abort the pass with AuditError.

    We use a 5-node linear chain so there are 4 upstream nodes available
    for the re-exec path — well above _MAX_CONSECUTIVE_FAILURES=3.
    """
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "audit-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [[f"n{i}" for i in range(5)]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [_make_node_output_dict(f"n{i}") for i in range(5)]
    db.execute(
        "INSERT INTO judge_results VALUES ('audit-001', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id,)
    )
    db.execute(
        "INSERT INTO cells VALUES ('audit-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=5)
    judge = MockJuryJudge(scores_queue=[0.0])  # cf flip → phase-2 kicks in

    async def always_boom(**kwargs):
        raise RuntimeError("simulated programmer bug")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        with pytest.raises(AuditError, match="consecutive"):
            asyncio.run(_audit_async(
                run_dir=run_dir, run_id="audit-001",
                variant_ids=[canonical_id], by_hash={canonical_id: topo},
                canonical_eval_rows_by_id={
                    "eval_001": {"id": "eval_001",
                                 "input": "Greg Walsh applied."}
                },
                dims=("name",), max_cost_usd=None, max_canonical_evals=None,
                judge=judge, llm_callable=always_boom, dry_run=False,
            ))
    finally:
        runner_mod.run_variant = real_run_variant


def test_audit_async_cancellation_propagates(tmp_path):
    """KeyboardInterrupt / asyncio.CancelledError must propagate, not
    get swallowed by the broad-exception handler."""
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "audit-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2"),
    ]
    db.execute(
        "INSERT INTO judge_results VALUES ('audit-001', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id,)
    )
    db.execute(
        "INSERT INTO cells VALUES ('audit-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    judge = MockJuryJudge(scores_queue=[0.0])

    async def cancel_llm(**kwargs):
        raise asyncio.CancelledError()

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_audit_async(
                run_dir=run_dir, run_id="audit-001",
                variant_ids=[canonical_id], by_hash={canonical_id: topo},
                canonical_eval_rows_by_id={
                    "eval_001": {"id": "eval_001",
                                 "input": "Greg Walsh applied."}
                },
                dims=("name",), max_cost_usd=None, max_canonical_evals=None,
                judge=judge, llm_callable=cancel_llm, dry_run=False,
            ))
    finally:
        runner_mod.run_variant = real_run_variant


def test_audit_async_cost_cap_soft_warn_and_hard_stop(tmp_path, caplog):
    """Cost-cap soft-warn fires at ≥80% and hard-stop at ≥100%, marking
    the summary partial."""
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    eval_ids = [f"eval_{i:03d}" for i in range(5)]
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "audit-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": eval_ids,
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2"),
    ]
    for eid in eval_ids:
        db.execute(
            "INSERT INTO judge_results VALUES ('audit-001', ?, ?, "
            "'continuous', 0.9, NULL, NULL, NULL, NULL)",
            (canonical_id, eid)
        )
        db.execute(
            "INSERT INTO cells VALUES ('audit-001', ?, ?, '', ?, NULL, NULL)",
            (canonical_id, eid, json.dumps(outputs)),
        )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    rows = {
        eid: {"id": eid, "input": "Greg Walsh applied."}
        for eid in eval_ids
    }
    # Need 5 phase-1 judgments. All non-flipping (so no phase-2).
    judge = MockJuryJudge(scores_queue=[0.9] * 5)

    async def expensive_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        # Each node reports a high cost so cap trips quickly.
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.5,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    import logging as _logging
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = expensive_run_variant
    try:
        with caplog.at_level(_logging.WARNING, logger="topofuzz.counterfactual_audit"):
            summary = asyncio.run(_audit_async(
                run_dir=run_dir, run_id="audit-001",
                variant_ids=[canonical_id], by_hash={canonical_id: topo},
                canonical_eval_rows_by_id=rows,
                dims=("name",),
                max_cost_usd=2.0,  # 3 nodes × $0.5 = $1.5 per pair → trips after 2.
                max_canonical_evals=None,
                judge=judge, llm_callable=FakeLLM(), dry_run=False,
            ))
    finally:
        runner_mod.run_variant = real_run_variant

    assert summary.partial is True
    # The soft-warn at 80% must have fired before hard-stop.
    assert any("soft-warn" in rec.message for rec in caplog.records)


def test_audit_summary_dataclass_shape():
    """CounterfactualAuditSummary fields are stable — consumers depend on these."""
    summary = CounterfactualAuditSummary(
        n_canonical_pairs=1, n_flipping_pairs=0, n_ablation_points=0,
        n_ablation_points_meaningful=0, dims=("name",),
        total_cost_usd=0.0, partial=False,
        by_dim={}, by_swap_pair={}, by_node={},
        unpriced_call_count=0, unpriced_models=(),
        results=(),
    )
    assert summary.n_canonical_pairs == 1
    assert summary.dims == ("name",)
    assert summary.results == ()
    # Pass-1 BLOCKER B4 fields present on the dataclass.
    assert summary.unpriced_call_count == 0
    assert summary.unpriced_models == ()


def test_audit_result_dataclass_shape():
    """CounterfactualAuditResult mirrors the sidecar's per-row schema."""
    r = CounterfactualAuditResult(
        variant_id="v", canonical_eval_id="e",
        counterfactual_eval_id="e__cf_name_x",
        dim="name", from_token="A", to_token="B",
        ablated_node_id="n0", kind="reexecute_terminal",
        canonical_score=1.0, original_cf_score=0.0, new_cf_score=1.0,
        flip_closed=True, cost_usd=0.01,
        new_terminal_response_text="APPROVE", jury_votes=(),
    )
    assert r.dim == "name"
    assert r.flip_closed is True


# ---------------------------------------------------------------------------
# _aggregate — by_node math
# ---------------------------------------------------------------------------


def test_aggregate_by_node_attribution_arithmetic():
    """4 meaningful experiments, 2 flips closed on a node → 50.0%.

    Verifies the attribution_pct computation directly without spinning up
    the async driver.
    """
    results = (
        # n0: 2 flips of 4 → 50.0
        CounterfactualAuditResult(
            variant_id="v1", canonical_eval_id="e1",
            counterfactual_eval_id="e1__cf",
            dim="name", from_token="A", to_token="B",
            ablated_node_id="n0", kind="reexecute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=1.0, flip_closed=True, cost_usd=0.01,
            new_terminal_response_text=None, jury_votes=(),
        ),
        CounterfactualAuditResult(
            variant_id="v1", canonical_eval_id="e2",
            counterfactual_eval_id="e2__cf",
            dim="name", from_token="A", to_token="B",
            ablated_node_id="n0", kind="reexecute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=1.0, flip_closed=True, cost_usd=0.01,
            new_terminal_response_text=None, jury_votes=(),
        ),
        CounterfactualAuditResult(
            variant_id="v1", canonical_eval_id="e3",
            counterfactual_eval_id="e3__cf",
            dim="name", from_token="A", to_token="B",
            ablated_node_id="n0", kind="reexecute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=0.0, flip_closed=False, cost_usd=0.01,
            new_terminal_response_text=None, jury_votes=(),
        ),
        CounterfactualAuditResult(
            variant_id="v1", canonical_eval_id="e4",
            counterfactual_eval_id="e4__cf",
            dim="name", from_token="A", to_token="B",
            ablated_node_id="n0", kind="reexecute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=0.0, flip_closed=False, cost_usd=0.01,
            new_terminal_response_text=None, jury_votes=(),
        ),
    )
    summary = _aggregate(
        results=results, dims=("name",),
        n_canonical_pairs=4, n_flipping_pairs=4,
        total_cost=0.04, partial=False,
    )
    assert summary.by_node["n0"]["flips_closed"] == 2
    assert summary.by_node["n0"]["total_meaningful"] == 4
    assert summary.by_node["n0"]["attribution_pct"] == 50.0


def test_aggregate_by_swap_pair_flattened_key():
    """The JSON-shaped by_swap_pair uses `dim|from|to` as the flat key."""
    results = (
        CounterfactualAuditResult(
            variant_id="v", canonical_eval_id="e",
            counterfactual_eval_id="e__cf",
            dim="name", from_token="Greg Walsh", to_token="Jamal Jones",
            ablated_node_id="n0", kind="substitute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=1.0, flip_closed=True, cost_usd=0.0,
            new_terminal_response_text=None, jury_votes=(),
        ),
    )
    summary = _aggregate(
        results=results, dims=("name",),
        n_canonical_pairs=1, n_flipping_pairs=1,
        total_cost=0.0, partial=False,
    )
    assert "name|Greg Walsh|Jamal Jones" in summary.by_swap_pair


def test_aggregate_skip_kinds_excluded_from_attribution():
    """skip_* kinds don't count toward attribution_pct (only meaningful kinds)."""
    results = (
        CounterfactualAuditResult(
            variant_id="v", canonical_eval_id="e",
            counterfactual_eval_id="e__cf",
            dim="name", from_token="A", to_token="B",
            ablated_node_id="n0", kind="skip_downstream",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=None, flip_closed=None, cost_usd=0.0,
            new_terminal_response_text=None, jury_votes=(),
        ),
        CounterfactualAuditResult(
            variant_id="v", canonical_eval_id="e",
            counterfactual_eval_id="e__cf",
            dim="name", from_token="A", to_token="B",
            ablated_node_id="n1", kind="substitute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=1.0, flip_closed=True, cost_usd=0.0,
            new_terminal_response_text=None, jury_votes=(),
        ),
    )
    summary = _aggregate(
        results=results, dims=("name",),
        n_canonical_pairs=1, n_flipping_pairs=1,
        total_cost=0.0, partial=False,
    )
    # n0 was skip_downstream → not in by_node at all.
    assert "n0" not in summary.by_node
    # n1 substitute_terminal counts.
    assert summary.by_node["n1"]["attribution_pct"] == 100.0


def test_aggregate_empty_results_returns_zero_summary():
    summary = _aggregate(
        results=(), dims=("name",),
        n_canonical_pairs=0, n_flipping_pairs=0,
        total_cost=0.0, partial=False,
    )
    assert summary.n_ablation_points == 0
    assert summary.n_ablation_points_meaningful == 0
    assert summary.by_node == {}


def test_aggregate_by_dim_aggregates_across_swap_pairs():
    """by_dim sums flipping_pairs + total_pairs across swap-pairs in dim."""
    results = (
        CounterfactualAuditResult(
            variant_id="v", canonical_eval_id="e1",
            counterfactual_eval_id="e1__cf_A",
            dim="name", from_token="A", to_token="B",
            ablated_node_id="n0", kind="substitute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=1.0, flip_closed=True, cost_usd=0.0,
            new_terminal_response_text=None, jury_votes=(),
        ),
        CounterfactualAuditResult(
            variant_id="v", canonical_eval_id="e2",
            counterfactual_eval_id="e2__cf_C",
            dim="name", from_token="C", to_token="D",
            ablated_node_id="n0", kind="substitute_terminal",
            canonical_score=1.0, original_cf_score=0.0,
            new_cf_score=1.0, flip_closed=True, cost_usd=0.0,
            new_terminal_response_text=None, jury_votes=(),
        ),
    )
    summary = _aggregate(
        results=results, dims=("name",),
        n_canonical_pairs=2, n_flipping_pairs=2,
        total_cost=0.0, partial=False,
    )
    assert summary.by_dim["name"]["flipping_pairs"] == 2
    assert summary.by_dim["name"]["total_pairs"] == 2


# ---------------------------------------------------------------------------
# _atomic_write_text
# ---------------------------------------------------------------------------


def test_atomic_write_text_writes_file(tmp_path):
    target = tmp_path / "out.json"
    _atomic_write_text(target, '{"x": 1}\n')
    assert target.read_text(encoding="utf-8") == '{"x": 1}\n'
    # No temp file leaks.
    leaks = [p for p in tmp_path.iterdir() if p.name.startswith(".out.json")]
    assert leaks == []


def test_atomic_write_text_overwrites_existing(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")
    _atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_max_consecutive_failures_is_three():
    """The abort threshold matches ablation's vocabulary — 3."""
    assert _MAX_CONSECUTIVE_FAILURES == 3


def test_response_text_cap_is_8192():
    assert _RESPONSE_TEXT_CAP == 8192


# ---------------------------------------------------------------------------
# Pass-1 dual-audit regression tests
# ---------------------------------------------------------------------------


def _setup_run_dir_for_blocker_tests(
    tmp_path: Path,
    *,
    n_nodes: int = 5,
    canonical_id: str | None = None,
    eval_id: str = "eval_001",
) -> tuple[Path, str]:
    """Lay down the minimum run-dir shape the BLOCKER tests need.

    Returns (run_dir, canonical_variant_id). The DB carries one
    canonical cell + judge result so Phase-1 can run; downstream tests
    drive the actual Phase-2 path via _audit_async.
    """
    if canonical_id is None:
        canonical_id = "c" * 64
    run_dir = tmp_path / "blk-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "blk-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [[f"n{i}" for i in range(n_nodes)]],
        "eval_ids": [eval_id],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [_make_node_output_dict(f"n{i}") for i in range(n_nodes)]
    db.execute(
        "INSERT INTO judge_results VALUES ('blk-001', ?, ?, "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)",
        (canonical_id, eval_id),
    )
    db.execute(
        "INSERT INTO cells VALUES ('blk-001', ?, ?, '', ?, NULL, NULL)",
        (canonical_id, eval_id, json.dumps(outputs)),
    )
    db.commit()
    db.close()
    return run_dir, canonical_id


def test_pass1_b1_flip_rate_denominator_uses_phase1_attempts(tmp_path):
    """Pass-1 BLOCKER B1 regression: 10 Phase-1 cells, only 2 flip.
    The by_dim / by_swap_pair denominators must report 10 attempts,
    not 2. Prior bug: triple_seen was built from results[] only, so
    total_pairs collapsed to the flipping count and flip_rate_pct
    landed at ~100%.
    """
    canonical_id = "c" * 64
    run_dir = tmp_path / "blk-b1-run"
    run_dir.mkdir()
    eval_ids = [f"eval_{i:03d}" for i in range(10)]
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "blk-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": eval_ids,
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2"),
    ]
    for eid in eval_ids:
        db.execute(
            "INSERT INTO judge_results VALUES ('blk-001', ?, ?, "
            "'continuous', 0.9, NULL, NULL, NULL, NULL)",
            (canonical_id, eid),
        )
        db.execute(
            "INSERT INTO cells VALUES ('blk-001', ?, ?, '', ?, NULL, NULL)",
            (canonical_id, eid, json.dumps(outputs)),
        )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    rows = {
        eid: {"id": eid, "input": "Greg Walsh applied."}
        for eid in eval_ids
    }
    # 10 Phase-1 judge calls: 2 flip (score 0.0 — DENY); 8 don't (0.9).
    # Each flipping triple needs Phase-2 jury re-judgments: 2 upstream
    # nodes (n0, n1) × re-exec + 1 terminal substitute (M8: no jury).
    # That's 2 upstream re-judgments per flip → 2 flips × 2 = 4 extra
    # jury calls, all returning 1.0 (close the flip).
    judge = MockJuryJudge(
        scores_queue=(
            [0.9, 0.9, 0.0, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.9]
            + [1.0, 1.0, 1.0, 1.0]
        )
    )
    fake_llm = FakeLLM(response_text="HYBRID")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="blk-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id=rows,
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    # 10 Phase-1 attempts, 2 flipping triples.
    assert summary.n_canonical_pairs == 10
    assert summary.n_flipping_pairs == 2
    # B1 fix: by_dim total_pairs is the Phase-1 attempt count (10),
    # NOT 2. flipping_pairs is 2, flip_rate is 20%.
    assert summary.by_dim["name"]["total_pairs"] == 10
    assert summary.by_dim["name"]["flipping_pairs"] == 2
    assert summary.by_dim["name"]["flip_rate_pct"] == 20.0
    # B1 fix: by_swap_pair total_pairs is the Phase-1 attempt count
    # for the specific (dim, from, to) swap (10 attempts of Greg→Jamal).
    swap_key = "name|Greg Walsh|Jamal Jones"
    assert swap_key in summary.by_swap_pair
    assert summary.by_swap_pair[swap_key]["total_pairs"] == 10
    assert summary.by_swap_pair[swap_key]["flipping_pairs"] == 2


def test_pass1_b2_cost_cap_trips_mid_triple(tmp_path):
    """Pass-1 BLOCKER B2 regression: a 5-node topology with one
    flipping triple and a $0.30 cap MUST honor the cap after
    accumulating ~$0.30 of node-by-node cost, not blow past it because
    cost_tracker only updated post-triple.

    Mock LLM stub returns nothing (we assert via the partial flag and
    a real-LLM exec cost path of $0.10/call via the cost_for_call
    monkeypatch).
    """
    run_dir, canonical_id = _setup_run_dir_for_blocker_tests(
        tmp_path, n_nodes=5
    )
    topo = _toy_topology(n_nodes=5)
    # Phase-1: 1 cell, flips. Phase-2: try 4 upstream re-execs + 1
    # terminal substitute. Each re-exec costs $0.10 via cost_for_call.
    # Cap = $0.30 → should trip after ~3 re-execs.
    judge = MockJuryJudge(scores_queue=[0.0, 1.0, 1.0, 1.0, 1.0])
    fake_llm = FakeLLM(response_text="HYBRID")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    # Monkeypatch cost_for_call so each re-exec costs $0.10.
    import fuzzeval.costest as costest_mod
    real_cost_for_call = costest_mod.cost_for_call
    costest_mod.cost_for_call = lambda model, inp, out: 0.10  # type: ignore[assignment]

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="blk-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={
                "eval_001": {"id": "eval_001",
                             "input": "Greg Walsh applied."},
            },
            dims=("name",), max_cost_usd=0.30,
            max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant
        costest_mod.cost_for_call = real_cost_for_call

    # B2 fix: the cap actually trips mid-triple. With per-cell cost
    # $0.13 and cap $0.30, the cap-check at the top of each node
    # iteration sees the LIVE total after the prior node's update;
    # the loop breaks once total >= $0.30. Without the fix, the cap
    # would never trip during the single triple — total_cost would
    # be updated only AFTER the whole triple finished, so all 4
    # upstream re-execs would run → ~$0.55 of spend before any check.
    # The signal that the fix is working: total is strictly less than
    # the no-fix bound, AND fewer than all 4 upstream re-execs ran,
    # AND partial==True.
    # Pass-2 MINOR Mi1 fix: tightened 0.55 → 0.45. The no-fix shape
    # accumulates roughly $0.55 (4 × $0.10 re-execs + jury + Phase-1
    # cells); the prior 0.55 ceiling sat at the no-fix boundary, so a
    # regression that pushed cost by even a cent would slip past.
    # 0.45 gives ~$0.10 of guard band without false-positiving on the
    # fix's actual ~$0.30-cap-honoring shape.
    assert summary.total_cost_usd < 0.45, (
        f"B2 regression: total_cost_usd={summary.total_cost_usd:.4f} "
        "matches the no-fix shape; cap did not trip mid-triple."
    )
    # At least one upstream node must have been skipped by the cap.
    upstream_meaningful = sum(
        1 for r in summary.results
        if r.kind == "reexecute_terminal"
    )
    assert upstream_meaningful < 4, (
        f"B2 regression: {upstream_meaningful}/4 upstream nodes were "
        "re-executed; cap did not trip mid-triple."
    )
    assert summary.partial is True, (
        "B2 fix: partial flag must be set when the cap trips mid-triple."
    )


def test_pass1_b3_phase1_jury_cost_charged(tmp_path):
    """Pass-1 BLOCKER B3 regression: when _run_counterfactual_cell
    returns a Cell with judge_result populated, the cost MUST include
    the Phase-1 jury floor on top of per-node costs.

    Direct test of the helper: 3 nodes × $0.001 + jury floor $0.03 →
    at least $0.033.
    """
    from topofuzz.counterfactual_audit import (
        _JURY_COST_FALLBACK_FLOOR_USD,
        _run_counterfactual_cell,
    )
    from topofuzz.runner import Cell, NodeOutput
    from fuzzeval.judge import ContinuousScore

    cf_row = {"id": "x", "input": "Greg Walsh applied."}
    topo = _toy_topology(n_nodes=3)

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        nodes = tuple(
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        )
        return Cell(
            variant_id=variant_ir.canonical_hash(),
            eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=nodes,
            judge_result=ContinuousScore(value=0.5, rationale="ok"),
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    cell, total_cost = asyncio.run(_run_counterfactual_cell(
        variant_topology=topo,
        cf_row=cf_row,
        judge=None,
        llm_callable=None,
        dry_run=False,
        run_variant=fake_run_variant,
    ))
    # 3 × $0.001 (per-node) + jury floor ($0.03)
    assert total_cost >= 3 * 0.001 + _JURY_COST_FALLBACK_FLOOR_USD, (
        f"B3 regression: phase-1 jury cost dropped. total_cost="
        f"{total_cost:.4f}, expected >= "
        f"{3 * 0.001 + _JURY_COST_FALLBACK_FLOOR_USD:.4f}"
    )


def test_pass1_b4_unpriced_call_tracked_and_floor_applied(tmp_path):
    """Pass-1 BLOCKER B4 regression: when cost_for_call returns None
    (model missing from rate table OR usage tokens absent), the audit
    must increment unpriced_call_count, add the model to
    unpriced_models, and apply the fallback floor instead of silently
    coercing to $0.
    """
    run_dir, canonical_id = _setup_run_dir_for_blocker_tests(
        tmp_path, n_nodes=3
    )
    topo = _toy_topology(n_nodes=3)
    judge = MockJuryJudge(scores_queue=[0.0, 1.0, 1.0])
    fake_llm = FakeLLM(response_text="HYBRID")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    # Monkeypatch cost_for_call to always return None (unpriced).
    import fuzzeval.costest as costest_mod
    real_cost_for_call = costest_mod.cost_for_call
    costest_mod.cost_for_call = lambda model, inp, out: None  # type: ignore[assignment]

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="blk-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={
                "eval_001": {"id": "eval_001",
                             "input": "Greg Walsh applied."},
            },
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant
        costest_mod.cost_for_call = real_cost_for_call

    # B4 fix: unpriced count tracks each Phase-2 re-execution call.
    # n0 + n1 are upstream → 2 re-exec calls; n2 is terminal substitute
    # (M8 fix: no jury call, no exec) → not counted.
    assert summary.unpriced_call_count >= 1, (
        f"B4 regression: unpriced_call_count={summary.unpriced_call_count}, "
        "expected >= 1 when cost_for_call returns None."
    )
    # The toy topology uses model="google/gemini-2.5-flash"; that model
    # name should land in unpriced_models.
    assert any(
        "gemini" in m.lower() for m in summary.unpriced_models
    ), (
        f"B4 regression: unpriced_models={summary.unpriced_models}, "
        "expected to contain the toy topology model."
    )
    # B4 fix: total_cost_usd > 0 even though every cost_for_call
    # returned None — the fallback floor kept the cap accounting honest.
    assert summary.total_cost_usd > 0.0
    # Sidecar carries the new fields.
    sc = read_audit_sidecar(run_dir)
    assert sc is not None
    assert sc["unpriced_call_count"] == summary.unpriced_call_count
    assert sc["unpriced_models"] == list(summary.unpriced_models)


def test_pass1_m1_load_cell_outputs_rejects_non_dict_element(tmp_path):
    """Pass-1 MAJOR M1 regression: _load_cell_outputs raises with the
    offending index when a node_outputs_json array contains a non-dict
    element (string, number, null)."""
    from topofuzz.counterfactual_audit import _load_cell_outputs

    run_dir = tmp_path / "m1-run"
    run_dir.mkdir()
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE cells ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, "
        "ir_canonical_hash TEXT, node_outputs_json TEXT, error TEXT, "
        "cell_cost_usd REAL)"
    )
    # Mixed array: dict, then a string at index 1.
    mixed = [{"node_id": "n0"}, "not a dict", {"node_id": "n2"}]
    db.execute(
        "INSERT INTO cells VALUES ('r', 'v', 'e', '', ?, NULL, NULL)",
        (json.dumps(mixed),),
    )
    db.commit()
    db.close()
    with pytest.raises(AuditError, match=r"node_outputs_json\[1\]"):
        _load_cell_outputs(run_dir, "v", "e")


def test_pass1_m2_duplicate_node_id_skipped(tmp_path):
    """Pass-1 MAJOR M2 regression: when canonical OR counterfactual
    outputs carry duplicate node_id entries, the ablation routes the
    duplicate through the existing skip path rather than silently
    using the last snapshot. Mirrors ablation.py behaviour.
    """
    from topofuzz.counterfactual_audit import _duplicate_node_ids

    outputs = [
        {"node_id": "n0"},
        {"node_id": "n1"},
        {"node_id": "n0"},  # duplicate
    ]
    dupes = _duplicate_node_ids(outputs)
    assert dupes == {"n0"}


def test_pass1_m8_terminal_substitute_skips_jury(tmp_path):
    """Pass-1 MAJOR M8 regression: substitute_terminal must not call
    the jury (saves cost + makes the control truly 100% by
    construction). After the fix, the jury queue stays unconsumed for
    the terminal row.
    """
    canonical_outputs = [
        _make_node_output_dict("n0", response_text="canonical n0"),
        _make_node_output_dict("n1", response_text="canonical APPROVE"),
    ]
    canonical_id = "c" * 64
    run_dir = tmp_path / "m8-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "m8-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
        "INSERT INTO judge_results VALUES ('m8-001', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)",
        (canonical_id,),
    )
    db.execute(
        "INSERT INTO cells VALUES ('m8-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(canonical_outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=2)
    # Phase-1: 1 cell, flips (0.0). Phase-2: 1 upstream node (n0)
    # re-exec + jury (1.0); terminal n1 substitute MUST NOT consume a
    # jury vote under M8.
    judge = MockJuryJudge(scores_queue=[0.0, 1.0])
    fake_llm = FakeLLM(response_text="HYBRID")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role,
                response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="m8-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={
                "eval_001": {"id": "eval_001",
                             "input": "Greg Walsh applied."},
            },
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    # 2 judge.score calls total: 1 phase-1 + 1 phase-2 re-exec re-judge.
    # No third call for the terminal substitute (M8 fix).
    assert len(judge.calls) == 2, (
        f"M8 regression: judge.calls={len(judge.calls)}, "
        "expected 2 (phase-1 + n0 re-judge). Terminal substitute must "
        "skip the jury call."
    )
    # n1 row exists in summary.by_node with attribution_pct == 100.0
    # (substitute_terminal is 100% by construction).
    assert "n1" in summary.by_node
    assert summary.by_node["n1"]["attribution_pct"] == 100.0
    assert summary.by_node["n1"]["is_terminal"] is True


def test_pass1_m9_gender_sentinel_collision_refused():
    """Pass-1 MAJOR M9 regression: an adversarial input containing the
    literal sentinel string ``__CFSGENDER0__`` must be refused, not
    silently corrupted by the second-phase substitution."""
    from topofuzz.counterfactual import generate_gender_swaps

    eval_row = {
        "id": "evil",
        "input": "he applied. __CFSGENDER0__ is a sentinel injection.",
    }
    with pytest.raises(ValueError, match="__CFSGENDER0__"):
        generate_gender_swaps(eval_row)


def test_pass1_m10_empty_canonical_response_skipped(tmp_path):
    """Pass-1 MAJOR M10 regression: a canonical run with blank
    response_text at a non-terminal node must produce a skip row
    rather than re-judging on blank input."""
    canonical_outputs = [
        _make_node_output_dict("n0", response_text=""),  # empty!
        _make_node_output_dict("n1", response_text="canonical n1"),
        _make_node_output_dict("n2", response_text="canonical APPROVE"),
    ]
    canonical_id = "c" * 64
    run_dir = tmp_path / "m10-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "m10-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
        "INSERT INTO judge_results VALUES ('m10-001', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)",
        (canonical_id,),
    )
    db.execute(
        "INSERT INTO cells VALUES ('m10-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(canonical_outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    # Phase-1 flips (0.0); Phase-2: n0 should skip (empty canonical
    # text), n1 re-exec, n2 substitute_terminal.
    judge = MockJuryJudge(scores_queue=[0.0, 1.0])
    fake_llm = FakeLLM(response_text="HYBRID")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="m10-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={
                "eval_001": {"id": "eval_001",
                             "input": "Greg Walsh applied."},
            },
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    # n0 must NOT have a meaningful result (it was skipped on empty
    # canonical text). n1 + n2 do.
    assert "n0" not in summary.by_node, (
        "M10 regression: n0 had empty canonical response_text and "
        "should have been skipped, not re-judged."
    )
    assert "n2" in summary.by_node  # terminal substitute survives.


# ---------------------------------------------------------------------------
# Pass-1 RT MINOR Mi4 — input substitution actually reaches the runner
# ---------------------------------------------------------------------------


def test_audit_async_substituted_input_reaches_run_variant(tmp_path):
    """Pass-1 RT MINOR Mi4 fix: the counterfactual eval_row's ``input``
    field must carry the SUBSTITUTED token (e.g. "Jamal Jones") when
    ``run_variant`` is invoked. If ``generate_swaps`` silently produced
    ``counterfactual_input == canonical_input``, the prior audit-flow
    tests would still pass green because their fake ``run_variant``
    ignored the eval_row's ``input``. Capture every eval_row passed and
    assert the swapped string is present.
    """
    canonical_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2", response_text="APPROVE"),
    ]
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "mi4-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
        "INSERT INTO judge_results VALUES ('mi4-001', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id,)
    )
    db.execute(
        "INSERT INTO cells VALUES ('mi4-001', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(canonical_outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    canonical_row = {"id": "eval_001", "input": "Greg Walsh applied."}

    judge = MockJuryJudge(scores_queue=[0.9])
    fake_llm = FakeLLM()
    # Capture every eval_row passed to run_variant. The CF row will be
    # one of these; we assert the swapped name shows up in its input.
    captured_eval_rows: list[dict[str, Any]] = []

    async def recording_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        captured_eval_rows.append(dict(eval_row))
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = recording_run_variant
    try:
        asyncio.run(_audit_async(
            run_dir=run_dir, run_id="mi4-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={"eval_001": canonical_row},
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    # At least one captured row carries the substituted name in input.
    cf_inputs = [
        r["input"] for r in captured_eval_rows
        if r.get("id", "").startswith("eval_001__cf_name_")
    ]
    assert cf_inputs, (
        "RT MINOR Mi4 regression: no eval_row with a cf_name_* eval_id "
        "was passed to run_variant. The audit driver may have dropped "
        "the CF row before the runner hop."
    )
    # The B-M pair table maps Greg Walsh ↔ Jamal Jones. Assert the
    # swapped name physically appears in the CF input and the original
    # name does NOT.
    for inp in cf_inputs:
        assert "Jamal Jones" in inp, (
            f"CF input did not carry the swapped token: {inp!r}"
        )
        assert "Greg Walsh" not in inp, (
            f"CF input still carries the canonical token: {inp!r}"
        )


def test_audit_async_substituted_input_byte_distinct_from_canonical(tmp_path):
    """Companion to ``..._reaches_run_variant``: assert the CF row's input
    is BYTE-DISTINCT from the canonical's input. A latent bug where
    ``generate_swaps`` returned a same-string CounterfactualSwap would
    pass the substring assertions above only if the canonical already
    contained the to_token — this test forecloses that loophole.
    """
    canonical_outputs = [
        _make_node_output_dict("n0"),
        _make_node_output_dict("n1"),
        _make_node_output_dict("n2", response_text="APPROVE"),
    ]
    canonical_id = "c" * 64
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "mi4-002",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
        "INSERT INTO judge_results VALUES ('mi4-002', ?, 'eval_001', "
        "'continuous', 0.9, NULL, NULL, NULL, NULL)", (canonical_id,)
    )
    db.execute(
        "INSERT INTO cells VALUES ('mi4-002', ?, 'eval_001', '', ?, NULL, NULL)",
        (canonical_id, json.dumps(canonical_outputs)),
    )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    canonical_input_text = "Greg Walsh applied at 11211."
    canonical_row = {"id": "eval_001", "input": canonical_input_text}

    judge = MockJuryJudge(scores_queue=[0.9])
    fake_llm = FakeLLM()
    captured_eval_rows: list[dict[str, Any]] = []

    async def recording_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        captured_eval_rows.append(dict(eval_row))
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_run_variant = runner_mod.run_variant
    runner_mod.run_variant = recording_run_variant
    try:
        asyncio.run(_audit_async(
            run_dir=run_dir, run_id="mi4-002",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={"eval_001": canonical_row},
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_run_variant

    cf_inputs = [
        r["input"] for r in captured_eval_rows
        if r.get("id", "").startswith("eval_001__cf_name_")
    ]
    assert cf_inputs
    for inp in cf_inputs:
        assert inp != canonical_input_text, (
            "RT MINOR Mi4 regression: CF input is byte-identical to "
            "canonical input — generate_swaps produced a no-op swap."
        )


# ---------------------------------------------------------------------------
# Pass-2 BLOCKER B1 + Pass-2 MAJOR M3 — schema_version enforcement
# ---------------------------------------------------------------------------


def test_pass2_b1_read_audit_sidecar_rejects_v1_schema(tmp_path):
    """Pass-2 BLOCKER B1 + MAJOR M3 regression: a hand-edited / cached
    audit.json carrying schema_version: 1 must be rejected at the read
    site. v1 semantics (Phase-2 flipping triples) differ from v2
    (Phase-1 attempts), so silently rendering a v1 sidecar with v2-
    shape code would produce wrong denominators in the PDF.
    """
    v1_payload = {
        "schema_version": 1,
        "n_canonical_pairs": 1,
        "n_flipping_pairs": 1,
        "n_ablation_points": 0,
        "n_ablation_points_meaningful": 0,
        "dims": ["name"],
        "total_cost_usd": 0.0,
        "partial": False,
        "by_dim": {},
        "by_swap_pair": {},
        "by_node": {},
        "results": [],
    }
    (tmp_path / AUDIT_SIDECAR_FILENAME).write_text(
        json.dumps(v1_payload), encoding="utf-8"
    )
    with pytest.raises(AuditError, match="schema_version is 1"):
        read_audit_sidecar(tmp_path)


def test_pass2_b1_read_audit_sidecar_rejects_future_schema(tmp_path):
    """Pass-2 BLOCKER B1 + MAJOR M3 regression: a future-version
    sidecar (schema_version: 99) must also be rejected; the audit
    consumer cannot promise back-compat for unread fields.
    """
    future_payload = {
        "schema_version": 99,
        "n_canonical_pairs": 1,
        "n_flipping_pairs": 1,
        "n_ablation_points": 0,
        "n_ablation_points_meaningful": 0,
        "dims": ["name"],
        "total_cost_usd": 0.0,
        "partial": False,
        "by_dim": {},
        "by_swap_pair": {},
        "by_node": {},
        "results": [],
    }
    (tmp_path / AUDIT_SIDECAR_FILENAME).write_text(
        json.dumps(future_payload), encoding="utf-8"
    )
    with pytest.raises(AuditError, match="schema_version is 99"):
        read_audit_sidecar(tmp_path)


# ---------------------------------------------------------------------------
# Pass-2 MAJOR M1 — ConfirmError → AuditError wrap
# ---------------------------------------------------------------------------


def test_pass2_m1_confirm_error_wrapped_as_audit_error(tmp_path, monkeypatch):
    """Pass-2 MAJOR M1 regression: run_audit_pass calls
    _reconstruct_variants which raises ConfirmError on fixture-
    resolution failures. ConfirmError is not an AuditError subclass,
    so it used to escape the CLI's `except AuditError` and print a
    raw Python traceback. The fix wraps the ConfirmError into
    AuditError, preserving the original via __cause__.
    """
    run_dir = _write_manifest_run(
        tmp_path, hashes=["d" * 64], eval_ids=["eval_001"],
    )
    # _load_eval_rows_by_id needs a CSV; satisfy the call site via a
    # patched loader so we don't hit the fixture path failure first.
    from topofuzz.confirm import ConfirmError as _ConfirmError

    def _boom(manifest, *, fixture_override=None):
        raise _ConfirmError("fixture not found: /tmp/missing.py")

    monkeypatch.setattr(
        "topofuzz.confirm._reconstruct_variants", _boom
    )
    with pytest.raises(AuditError, match="fixture not found"):
        run_audit_pass(run_dir)
    # __cause__ preserved so a debugger / log can recover the original.
    try:
        run_audit_pass(run_dir)
    except AuditError as e:
        assert isinstance(e.__cause__, _ConfirmError)
        assert "fixture not found" in str(e.__cause__)


# ---------------------------------------------------------------------------
# Pass-2 MAJOR M2 — hostile-row + per-cell vendor flake resilience
# ---------------------------------------------------------------------------


def test_pass2_m2a_generate_swaps_value_error_skips_only_offending_row(
    tmp_path, monkeypatch, caplog
):
    """Pass-2 MAJOR M2 part A regression: a single canonical row whose
    generator raises ValueError must skip ONLY that row's swap for
    that dim — the rest of the matrix (other rows + other dims for
    the same row) keep going.
    """
    canonical_id = "c" * 64
    run_dir = tmp_path / "m2a-run"
    run_dir.mkdir()
    eval_ids = ["bias_001", "bias_002", "bias_003"]
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "m2a-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": eval_ids,
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [_make_node_output_dict(f"n{i}") for i in range(3)]
    for eid in eval_ids:
        db.execute(
            "INSERT INTO judge_results VALUES ('m2a-001', ?, ?, "
            "'continuous', 0.9, NULL, NULL, NULL, NULL)",
            (canonical_id, eid),
        )
        db.execute(
            "INSERT INTO cells VALUES ('m2a-001', ?, ?, '', ?, NULL, NULL)",
            (canonical_id, eid, json.dumps(outputs)),
        )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    judge = MockJuryJudge(scores_queue=[0.9] * 20)
    fake_llm = FakeLLM()
    canonical_rows = {
        eid: {"id": eid, "input": "Greg Walsh applied."}
        for eid in eval_ids
    }

    from topofuzz import counterfactual_audit as ca_mod
    real_gs = ca_mod.generate_swaps

    def selective_boom(row, *, dims):
        if row.get("id") == "bias_002":
            raise ValueError("synthetic hostile row")
        return real_gs(row, dims=dims)

    monkeypatch.setattr(ca_mod, "generate_swaps", selective_boom)

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_rv = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        with caplog.at_level("WARNING"):
            summary = asyncio.run(_audit_async(
                run_dir=run_dir, run_id="m2a-001",
                variant_ids=[canonical_id], by_hash={canonical_id: topo},
                canonical_eval_rows_by_id=canonical_rows,
                dims=("name",), max_cost_usd=None,
                max_canonical_evals=None,
                judge=judge, llm_callable=fake_llm, dry_run=False,
            ))
    finally:
        runner_mod.run_variant = real_rv

    # M2a: audit ran to completion — the hostile bias_002 ValueError
    # did NOT abort the pass. n_canonical_pairs > 0 means at least
    # one OTHER row's Phase-1 cell ran.
    assert summary.n_canonical_pairs > 0
    # The skip is logged.
    assert any(
        "counterfactual generator" in rec.message
        and "bias_002" in rec.message
        for rec in caplog.records
    )


def test_pass2_m2b_run_counterfactual_cell_vendor_flake_recoverable(
    tmp_path, monkeypatch
):
    """Pass-2 MAJOR M2 part B regression: a vendor SDK exception
    raised inside _run_counterfactual_cell must not crash the audit
    when only a few cells fail in succession; the audit recovers
    after the failures clear.

    Setup: 4 eval rows, each yielding 1 swap → 4 Phase-1 cells. The
    first 2 calls flake; calls 3+ succeed. After the 2 flakes the
    audit must complete with 2 successful Phase-1 cells, NOT abort.
    """
    canonical_id = "c" * 64
    run_dir = tmp_path / "m2b-rec"
    run_dir.mkdir()
    eval_ids = [f"eval_{i:03d}" for i in range(4)]
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "m2b-001",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": eval_ids,
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [_make_node_output_dict(f"n{i}") for i in range(3)]
    for eid in eval_ids:
        db.execute(
            "INSERT INTO judge_results VALUES ('m2b-001', ?, ?, "
            "'continuous', 0.9, NULL, NULL, NULL, NULL)",
            (canonical_id, eid),
        )
        db.execute(
            "INSERT INTO cells VALUES ('m2b-001', ?, ?, '', ?, NULL, NULL)",
            (canonical_id, eid, json.dumps(outputs)),
        )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    judge = MockJuryJudge(scores_queue=[0.9] * 20)
    fake_llm = FakeLLM()
    canonical_rows = {
        eid: {"id": eid, "input": "Greg Walsh applied."}
        for eid in eval_ids
    }

    call_count = {"n": 0}

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError("vendor flake")
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text="x",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import topofuzz.runner as runner_mod
    real_rv = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="m2b-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id=canonical_rows,
            dims=("name",), max_cost_usd=None,
            max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_rv

    # At least one Phase-1 cell eventually ran (after 2 transient
    # failures, the 3rd+ calls succeed). The pass did not crash.
    assert summary.n_canonical_pairs >= 1


def test_pass2_m2b_run_counterfactual_cell_sustained_outage_aborts(
    tmp_path, monkeypatch
):
    """Pass-2 MAJOR M2 part B regression: when _run_counterfactual_cell
    keeps failing, the audit aborts with AuditError mentioning the
    "consecutive Phase-1 cell failures" pattern (same shape as Phase-2).

    Setup: 5 eval rows × 1 dim → 5 Phase-1 cells, each guaranteed to
    fail. After _MAX_CONSECUTIVE_FAILURES (3) the pass must abort.
    """
    canonical_id = "c" * 64
    run_dir = tmp_path / "m2b-out"
    run_dir.mkdir()
    eval_ids = [f"eval_{i:03d}" for i in range(5)]
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "m2b-002",
        "variant_canonical_hashes": [canonical_id],
        "variant_orderings": [["n0", "n1", "n2"]],
        "eval_ids": eval_ids,
        "cli_args": {
            "fixture": "x.py:build_graph", "eval": "e.csv",
            "rng_seed": 0, "variants": 1, "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }), encoding="utf-8")
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
    outputs = [_make_node_output_dict(f"n{i}") for i in range(3)]
    for eid in eval_ids:
        db.execute(
            "INSERT INTO judge_results VALUES ('m2b-002', ?, ?, "
            "'continuous', 0.9, NULL, NULL, NULL, NULL)",
            (canonical_id, eid),
        )
        db.execute(
            "INSERT INTO cells VALUES ('m2b-002', ?, ?, '', ?, NULL, NULL)",
            (canonical_id, eid, json.dumps(outputs)),
        )
    db.commit()
    db.close()

    topo = _toy_topology(n_nodes=3)
    judge = MockJuryJudge(scores_queue=[0.9] * 20)
    fake_llm = FakeLLM()
    canonical_rows = {
        eid: {"id": eid, "input": "Greg Walsh applied."}
        for eid in eval_ids
    }

    async def always_boom(variant_ir, eval_row, *, llm_callable, judge):
        raise RuntimeError("vendor outage")

    import topofuzz.runner as runner_mod
    real_rv = runner_mod.run_variant
    runner_mod.run_variant = always_boom
    try:
        with pytest.raises(
            AuditError, match="consecutive Phase-1 cell failures"
        ):
            asyncio.run(_audit_async(
                run_dir=run_dir, run_id="m2b-002",
                variant_ids=[canonical_id], by_hash={canonical_id: topo},
                canonical_eval_rows_by_id=canonical_rows,
                dims=("name",), max_cost_usd=None,
                max_canonical_evals=None,
                judge=judge, llm_callable=fake_llm, dry_run=False,
            ))
    finally:
        runner_mod.run_variant = real_rv


# ---------------------------------------------------------------------------
# Pass-2 RT MAJOR M3 — Phase-2 terminal re-exec fallback cost
# ---------------------------------------------------------------------------


def test_pass2_rt_m3_phase2_unpriced_uses_higher_fallback(tmp_path):
    """Pass-2 RT MAJOR M3 regression: when cost_for_call returns None
    for a Phase-2 terminal re-exec, the per-cell cost contribution
    must be _PHASE2_TERMINAL_REEXEC_FALLBACK_USD (0.05), NOT the
    Phase-1 floor (0.01). Premium verdict-node models are
    $0.03-$0.10/call; the shared Phase-1 floor under-reported by 3-10×.
    """
    from topofuzz.counterfactual_audit import (
        _JURY_COST_FALLBACK_FLOOR_USD,
        _PHASE1_CELL_COST_FALLBACK_USD,
        _PHASE2_TERMINAL_REEXEC_FALLBACK_USD,
    )

    assert _PHASE2_TERMINAL_REEXEC_FALLBACK_USD > _PHASE1_CELL_COST_FALLBACK_USD
    assert _PHASE2_TERMINAL_REEXEC_FALLBACK_USD == 0.05

    run_dir, canonical_id = _setup_run_dir_for_blocker_tests(
        tmp_path, n_nodes=3
    )
    topo = _toy_topology(n_nodes=3)
    # Phase-1: 1 cell, flips. Phase-2: 2 upstream re-execs hit the
    # unpriced path → each charges $0.05. Terminal-substitute charges $0.
    judge = MockJuryJudge(scores_queue=[0.0, 1.0, 1.0])
    fake_llm = FakeLLM(response_text="HYBRID")

    async def fake_run_variant(variant_ir, eval_row, *, llm_callable, judge):
        from topofuzz.runner import Cell, NodeOutput
        nodes = [
            NodeOutput(
                node_id=n.id, role=n.role, response_text=f"cf_{n.id}",
                model_served_version="m", cache_hit=False, latency_ms=1,
                cost_usd=0.001,
            )
            for n in variant_ir.nodes
        ]
        jr = judge.score({"variant_id": "", "eval_row": eval_row, "agent_output": "x"})
        return Cell(
            variant_id=variant_ir.canonical_hash(), eval_id=eval_row["id"],
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=tuple(nodes), judge_result=jr,
            judge_failure=False, judge_failure_reason=None, error=None,
        )

    import fuzzeval.costest as costest_mod
    real_cost_for_call = costest_mod.cost_for_call
    costest_mod.cost_for_call = lambda model, inp, out: None  # type: ignore[assignment]

    import topofuzz.runner as runner_mod
    real_rv = runner_mod.run_variant
    runner_mod.run_variant = fake_run_variant
    try:
        summary = asyncio.run(_audit_async(
            run_dir=run_dir, run_id="blk-001",
            variant_ids=[canonical_id], by_hash={canonical_id: topo},
            canonical_eval_rows_by_id={
                "eval_001": {"id": "eval_001",
                             "input": "Greg Walsh applied."},
            },
            dims=("name",), max_cost_usd=None, max_canonical_evals=None,
            judge=judge, llm_callable=fake_llm, dry_run=False,
        ))
    finally:
        runner_mod.run_variant = real_rv
        costest_mod.cost_for_call = real_cost_for_call

    # Find the reexecute_terminal rows. Each one's cost_usd MUST be
    # _PHASE2_TERMINAL_REEXEC_FALLBACK_USD + jury floor (0.03), NOT
    # the Phase-1 0.01 fallback + 0.03 jury.
    reexec_rows = [
        r for r in summary.results if r.kind == "reexecute_terminal"
    ]
    assert reexec_rows, "test setup wrong: no reexecute_terminal rows"
    for r in reexec_rows:
        # exec_cost ($0.05) + jury floor ($0.03) = $0.08 per re-exec.
        assert r.cost_usd == pytest.approx(
            _PHASE2_TERMINAL_REEXEC_FALLBACK_USD
            + _JURY_COST_FALLBACK_FLOOR_USD
        ), (
            f"RT MAJOR M3 regression: unpriced re-exec cost_usd="
            f"{r.cost_usd:.4f}; expected "
            f"{_PHASE2_TERMINAL_REEXEC_FALLBACK_USD + _JURY_COST_FALLBACK_FLOOR_USD:.4f} "
            f"(Phase-2 fallback $0.05 + jury floor $0.03)."
        )
