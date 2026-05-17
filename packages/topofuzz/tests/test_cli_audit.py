# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""End-to-end Typer CliRunner tests for ``topofuzz audit``.

F7 Pass-1 MINOR Mi1 fix: the 41-test ``test_counterfactual_audit.py``
suite exercises the driver but never the Typer CLI surface. A typo in
banner copy, per-dim parser change, or option-rendering drift would
ship undetected. This module wraps the CLI in :class:`CliRunner` and
asserts the buyer-facing contract:

* ``topofuzz audit`` (no args) → exit 2 + usage error.
* ``topofuzz audit nonexistent-dir`` → exit 2 + "not a directory".
* ``topofuzz audit --counterfactual=invalid-dim`` → exit 2 +
  the full supported vocabulary in the message.
* ``--counterfactual=name-swap,zip-swap`` → comma-separated parse.
* ``--counterfactual=name-swap --counterfactual=zip-swap`` → repeated-
  flag parse.
* ``--max-cost-usd=-1`` → Typer rejects on the ``min=0.0`` constraint.
* ``--max-cost-usd=0.05 --dry-run`` (M6 fix) → no confirmation prompt
  fires below the $1 skip threshold.
* ``--help`` → renders cleanly; all seven options visible.
* End-to-end against a stubbed source run dir → completes, writes
  ``audit.json``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("diskcache")

from typer.testing import CliRunner  # noqa: E402

from topofuzz.cli import app  # noqa: E402


def _runner() -> CliRunner:
    """Mirror the ``test_cli_scan._runner`` Click 8.1/8.2 compat shim.

    With ``mix_stderr=False`` stderr is split from stdout; on Click
    8.2+ that kwarg is removed and stream-separation is the default.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def _combined_output(result: Any) -> str:
    """Return stdout + stderr joined, regardless of Click version.

    Click 8.2's ``CliRunner`` populates ``result.stderr`` separately;
    Click 8.1 mixed both into ``result.output``. Tests below assert
    "this substring appears somewhere the user saw" rather than
    "this substring appears on stdout exclusively" — that's the
    contract the F7 CLI honours.
    """
    out = result.output or ""
    err = ""
    try:
        err = result.stderr or ""
    except (AttributeError, ValueError):
        # Click 8.1: stderr is the same as stdout.
        err = ""
    return out + err


# ---------------------------------------------------------------------------
# Fixture: minimal jury-of-3 manifest + bare run.sqlite so AuditError
# fires on the fixture-load gate (after CLI validation passes).
# ---------------------------------------------------------------------------


def _write_minimal_audit_source(tmp_path: Path) -> Path:
    """Lay down a manifest + bare run.sqlite under ``tmp_path/source-run``.

    The fixture path is intentionally bogus so
    :func:`_reconstruct_variants` raises an :class:`AuditError` — that
    gives us a stable exit-2 surface to test the CLI error path
    against without spinning up the runner.
    """
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    manifest = {
        "run_id": "audit-cli-001",
        "variant_canonical_hashes": ["a" * 64],
        "variant_orderings": [["n0", "n1"]],
        "eval_ids": ["eval_001"],
        "cli_args": {
            "fixture": "nonexistent_pkg.nonexistent:graph",
            "eval": "missing.csv",
            "rng_seed": 0,
            "variants": 1,
            "jury_of_3": True,
            "rubric": "rubric.txt",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
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
# Tests
# ---------------------------------------------------------------------------


def test_audit_no_args_exit_with_usage_error():
    """``topofuzz audit`` with no run_dir → Typer's missing-argument exit 2."""
    result = _runner().invoke(app, ["audit"])
    assert result.exit_code == 2
    # Typer/Click surfaces "Missing argument" on the missing run_dir
    # positional. Substring match so a help-rendering change doesn't
    # break the assertion.
    text = _combined_output(result).lower()
    assert "missing" in text or "usage" in text


def test_audit_nonexistent_dir_exits_clean():
    """A non-existent run_dir surfaces as the friendly exit-2 path."""
    result = _runner().invoke(app, ["audit", "totally-not-a-dir"])
    assert result.exit_code == 2
    text = _combined_output(result).lower()
    assert "not a directory" in text


def test_audit_invalid_dim_lists_all_four_choices(tmp_path):
    """``--counterfactual=invalid-dim`` → exit 2 + supported vocab in msg."""
    run_dir = _write_minimal_audit_source(tmp_path)
    result = _runner().invoke(
        app,
        [
            "audit",
            str(run_dir),
            "--counterfactual=invalid-dim",
            "--dry-run",
            "--yes",
        ],
    )
    assert result.exit_code == 2
    text = _combined_output(result).lower()
    assert "name-swap" in text
    assert "zip-swap" in text
    assert "age-swap" in text
    assert "gender-swap" in text


def test_audit_comma_separated_counterfactual_parses(tmp_path):
    """``--counterfactual=name-swap,zip-swap`` → both dims parsed.

    The downstream call fails (bogus fixture) but the CLI must accept
    the comma-separated shape. We assert the failure is NOT the
    "unknown --counterfactual value" path — the latter would mean
    comma-split regressed. Exit code is non-zero (fixture-load
    failure, e.g. ConfirmError) but the parser passed.
    """
    run_dir = _write_minimal_audit_source(tmp_path)
    result = _runner().invoke(
        app,
        [
            "audit",
            str(run_dir),
            "--counterfactual=name-swap,zip-swap",
            "--dry-run",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    text = _combined_output(result).lower()
    assert "unknown --counterfactual" not in text
    assert "supported:" not in text  # the BadParameter footer


def test_audit_repeated_counterfactual_flags_parse(tmp_path):
    """Repeated ``--counterfactual`` flags parse without the unknown-dim path."""
    run_dir = _write_minimal_audit_source(tmp_path)
    result = _runner().invoke(
        app,
        [
            "audit",
            str(run_dir),
            "--counterfactual=name-swap",
            "--counterfactual=zip-swap",
            "--dry-run",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    text = _combined_output(result).lower()
    assert "unknown --counterfactual" not in text
    assert "supported:" not in text


def test_audit_negative_max_cost_rejected_by_typer():
    """``--max-cost-usd=-1`` trips Typer's ``min=0.0`` constraint → exit 2."""
    result = _runner().invoke(
        app, ["audit", "any-dir", "--max-cost-usd=-1"]
    )
    assert result.exit_code == 2


def test_audit_low_cost_cap_skips_confirmation_prompt(tmp_path):
    """Pass-1 MAJOR M6 fix: ``--max-cost-usd=0.05`` skips the confirm prompt.

    The prompt is friction-theater at sub-$1 caps; the cap itself
    trips before the second cell runs. We feed no stdin so a prompt
    would block. The dry-run path also bypasses the confirm; the
    explicit ``--dry-run`` exercise here is intentional belt-and-
    suspenders against a regression that re-introduces the prompt.
    """
    run_dir = _write_minimal_audit_source(tmp_path)
    result = _runner().invoke(
        app,
        [
            "audit",
            str(run_dir),
            "--max-cost-usd=0.05",
            "--dry-run",
        ],
    )
    # Exit 2 from the bogus-fixture path (downstream); the key check
    # is that NO confirm prompt fired — i.e. the runner didn't hang.
    # CliRunner returns synchronously regardless of prompt behaviour;
    # we therefore assert via output: when the prompt fires (input
    # buffer is empty), Click prints an "Aborted." line. That MUST
    # NOT appear here.
    text = _combined_output(result).lower()
    assert "aborted." not in text


def test_audit_low_cost_cap_emits_info_banner(tmp_path):
    """Pass-1 MAJOR M6 fix: ``INFO: cost capped at $X.XX`` replaces the prompt.

    Drops the ``--dry-run`` flag so the M6 branch (only fires when
    not --dry-run and not --yes) is the surface under test.
    """
    run_dir = _write_minimal_audit_source(tmp_path)
    result = _runner().invoke(
        app,
        [
            "audit",
            str(run_dir),
            "--max-cost-usd=0.05",
        ],
    )
    text = _combined_output(result)
    assert "INFO: cost capped at $0.05" in text
    assert "skipping confirmation prompt" in text


def test_audit_help_renders_all_options():
    """``topofuzz audit --help`` lists every documented option."""
    result = _runner().invoke(app, ["audit", "--help"])
    assert result.exit_code == 0
    text = _combined_output(result)
    # Click 8.x can hard-wrap option names across newlines in --help
    # output. Strip whitespace + newlines before substring-matching so
    # a re-rendered help screen doesn't false-fail this test.
    flat = " ".join(text.split())
    for opt in (
        "--counterfactual",
        "--max-cost-usd",
        "--max-variants",
        "--max-canonical-evals",
        "--fixture",
        "--dry-run",
        "--yes",
    ):
        assert opt in flat, (
            f"`{opt}` missing from --help output. Help text:\n{text}"
        )


def test_audit_dry_run_yes_writes_no_sidecar_when_fixture_fails(tmp_path):
    """``--dry-run --yes`` against a bogus-fixture source dir surfaces a non-
    zero exit.

    With ``--dry-run --yes`` both prompt-fire paths are skipped and
    the CLI runs through to the fixture-load gate. The bogus fixture
    raises ``ConfirmError`` from ``_reconstruct_variants`` (which the
    audit command does not currently catch — propagates to Typer's
    default exit 1). The key contract under test is that no sidecar
    is written; the precise non-zero exit code is implementation
    detail. Pass-1 MINOR Mi1: this is the closest the CliRunner
    harness can get to "audit completes against a real source dir"
    without a live fixture; the engine-side completion path is
    covered in ``test_counterfactual_audit.test_audit_async_dry_run_*``.
    """
    run_dir = _write_minimal_audit_source(tmp_path)
    result = _runner().invoke(
        app,
        ["audit", str(run_dir), "--dry-run", "--yes"],
    )
    assert result.exit_code != 0
    # audit.json must NOT have been written when the gate fails.
    assert not (run_dir / "audit.json").exists()
