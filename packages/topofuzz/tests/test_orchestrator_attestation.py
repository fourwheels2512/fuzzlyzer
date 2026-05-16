# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 10a: orchestrator-side ``_maybe_emit_attestation`` glue tests.

The full ``execute_run`` integration is exercised in
:mod:`test_orchestrator`. These tests target the private
``_maybe_emit_attestation`` helper directly so we can pin its three
failure modes (no key / load failure / emit failure) without spinning
up the full async run loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("diskcache")

from topofuzz.ir import Edge, LLMConfig, Node, Topology  # noqa: E402
from topofuzz.orchestrator import RunPlan, _maybe_emit_attestation  # noqa: E402


def _toy_topology() -> Topology:
    cfg = LLMConfig(model="google/gemini-2.5-pro", temperature=0.0)
    return Topology(
        name="toy",
        nodes=(Node(id="n0", role="r0", llm_config=cfg),),
        edges=(),
        entrypoint="n0",
        terminal=("n0",),
    )


def _toy_plan(run_dir: Path, run_id: str = "r1") -> RunPlan:
    topo = _toy_topology()
    return RunPlan(
        topology=topo,
        variants=(topo,),
        eval_rows=({"id": "e1"},),
        run_id=run_id,
        run_dir=run_dir,
        cache_dir=None,
        max_cost_usd=None,
        judge_model="google/gemini-2.5-pro",
    )


def _kickoff_manifest(run_dir: Path, plan: RunPlan) -> None:
    """Write a minimal manifest.json that satisfies the emit pipeline."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": plan.run_id,
                "topology_canonical_hash": plan.topology.canonical_hash(),
                "topology_name": plan.topology.name,
                "tds_version": plan.topology.tds_version,
                "n_variants": len(plan.variants),
                "n_eval_rows": len(plan.eval_rows),
            }
        ),
        encoding="utf-8",
    )


def test_maybe_emit_skips_silently_when_no_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
):
    """When the dev key is missing, we log INFO and skip — never warn,
    never crash, no attestation.json written."""
    # Redirect default_dev_priv_path at the module the orchestrator
    # imports inside _maybe_emit_attestation.
    monkeypatch.setattr(
        "fuzzeval.keys.default_dev_priv_path",
        lambda: tmp_path / "does-not-exist" / "dev.pem",
    )
    run_dir = tmp_path / "run-1"
    plan = _toy_plan(run_dir)
    _kickoff_manifest(run_dir, plan)
    with caplog.at_level(logging.INFO, logger="topofuzz.orchestrator"):
        _maybe_emit_attestation(
            plan,
            status="succeeded",
            successful_cells=1,
            total_cells=1,
            total_cost_usd=0.0,
            elapsed_s=0.1,
        )
    assert not (run_dir / "attestation.json").exists()
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("no dev key" in r.getMessage() for r in info_records)


def test_maybe_emit_succeeds_when_key_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Happy path: key exists → attestation.json appears and is valid."""
    from fuzzeval.keys import generate_dev_keys

    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    monkeypatch.setattr(
        "fuzzeval.keys.default_dev_priv_path", lambda: priv,
    )
    run_dir = tmp_path / "run-1"
    plan = _toy_plan(run_dir)
    _kickoff_manifest(run_dir, plan)
    _maybe_emit_attestation(
        plan,
        status="succeeded",
        successful_cells=1,
        total_cells=1,
        total_cost_usd=0.0,
        elapsed_s=0.1,
    )
    att = run_dir / "attestation.json"
    assert att.exists()
    # Verifies cleanly with the matching public key.
    from topofuzz.attest import load_and_verify_attestation

    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub.read_bytes(),
    )
    assert outcome.ok, outcome.checks


def test_maybe_emit_warns_on_unreadable_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
):
    """If manifest.json is unreadable (corrupted / missing), we log
    WARNING but still don't raise."""
    from fuzzeval.keys import generate_dev_keys

    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    monkeypatch.setattr(
        "fuzzeval.keys.default_dev_priv_path", lambda: priv,
    )
    run_dir = tmp_path / "run-1"
    plan = _toy_plan(run_dir)
    run_dir.mkdir()
    # Corrupted manifest — invalid JSON.
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="topofuzz.orchestrator"):
        _maybe_emit_attestation(
            plan,
            status="succeeded",
            successful_cells=0,
            total_cells=0,
            total_cost_usd=0.0,
            elapsed_s=0.0,
        )
    assert not (run_dir / "attestation.json").exists()
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("manifest.json read failed" in r.getMessage() for r in warns)


def test_maybe_emit_never_raises_on_malformed_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
):
    """A garbage PEM at the key path → WARNING + skip, never propagate."""
    priv = tmp_path / "dev.pem"
    priv.write_bytes(b"not a PEM")
    monkeypatch.setattr(
        "fuzzeval.keys.default_dev_priv_path", lambda: priv,
    )
    run_dir = tmp_path / "run-1"
    plan = _toy_plan(run_dir)
    _kickoff_manifest(run_dir, plan)
    with caplog.at_level(logging.WARNING, logger="topofuzz.orchestrator"):
        _maybe_emit_attestation(
            plan,
            status="succeeded",
            successful_cells=1,
            total_cells=1,
            total_cost_usd=0.0,
            elapsed_s=0.1,
        )
    assert not (run_dir / "attestation.json").exists()
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("could not be loaded" in r.getMessage() for r in warns)
