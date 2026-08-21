# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``topofuzz run --simulate`` / ``--offline`` (FIX 4).

``--simulate`` stubs BOTH the agent rollouts AND the judge with
deterministic responses, so a whole run makes ZERO network calls and
cannot bill — provable by patching ``litellm.acompletion`` to raise if
it is ever reached. The agent-rollout stub is the SAME
``_dry_run_llm_callable`` that ``topofuzz audit --dry-run`` already
relies on for the identical contract (see
``topofuzz.counterfactual_audit``'s module docstring); ``--simulate``
reuses it rather than inventing a second stub.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("diskcache")
pytest.importorskip("langgraph")

from typer.testing import CliRunner  # noqa: E402

from topofuzz.cli import app  # noqa: E402
from topofuzz.counterfactual_audit import (  # noqa: E402
    _DRY_RUN_STUB_RESPONSE_TEXT,
    _dry_run_llm_callable,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GALLERY_DIR = _REPO_ROOT / "examples" / "gallery" / "langgraph-lending-4role"
_FIXTURE = _GALLERY_DIR / "fixture.py"
_EVAL = _GALLERY_DIR / "eval.csv"


def test_offline_stub_llm_callable_is_litellm_shaped() -> None:
    """(2) Unit coverage of the exact stub ``--simulate`` injects: a
    LiteLLM-``ModelResponse``-shaped dict with zero usage tokens (so
    cost accounting stays at $0 even before any dry_run cost overwrite)."""
    resp = asyncio.run(
        _dry_run_llm_callable(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.0,
            max_tokens=128,
        )
    )
    assert resp["choices"][0]["message"]["content"] == _DRY_RUN_STUB_RESPONSE_TEXT
    assert resp["model"] == "google/gemini-2.5-flash"
    assert resp["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


def test_run_simulate_makes_zero_real_llm_calls(tmp_path, monkeypatch) -> None:
    """(1) End-to-end: ``run --simulate`` on the real
    ``langgraph-lending-4role`` gallery fixture never reaches
    ``litellm.acompletion`` — patched here to raise if called — and
    still completes successfully using the stub responses."""
    assert _FIXTURE.exists(), f"gallery fixture missing: {_FIXTURE}"
    assert _EVAL.exists(), f"gallery eval missing: {_EVAL}"

    import litellm

    async def _boom(*args, **kwargs):
        raise AssertionError(
            "litellm.acompletion was called under --simulate — offline "
            "mode must never touch the network."
        )

    monkeypatch.setattr(litellm, "acompletion", _boom)

    out_dir = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{_FIXTURE}:build_graph",
            "--eval", str(_EVAL),
            "--n", "1",
            "--variants", "1",
            "--out", str(out_dir),
            "--no-cache",
            "--simulate",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[SIMULATE]" in result.output

    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cli_args"]["simulate"] is True
    # --simulate implies the judge's --dry-run stub too.
    assert manifest["cli_args"]["dry_run"] is True

    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        cells = db.execute(
            "SELECT eval_id, error, cell_cost_usd FROM cells "
            "WHERE run_id = ?", (manifest["run_id"],)
        ).fetchall()
    finally:
        db.close()
    assert len(cells) == 1  # 1 variant x 1 eval row
    eval_id, error, cost = cells[0]
    assert error is None
    # The offline stub reports zero usage tokens -> zero cost.
    assert cost in (0.0, None)


def test_run_offline_alias_is_equivalent_to_simulate(tmp_path, monkeypatch) -> None:
    """``--offline`` is documented as an alias for ``--simulate``."""
    import litellm

    async def _boom(*args, **kwargs):
        raise AssertionError("litellm.acompletion was called under --offline")

    monkeypatch.setattr(litellm, "acompletion", _boom)

    out_dir = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{_FIXTURE}:build_graph",
            "--eval", str(_EVAL),
            "--n", "1",
            "--variants", "1",
            "--out", str(out_dir),
            "--no-cache",
            "--offline",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    manifest = json.loads(
        (run_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["cli_args"]["simulate"] is True
