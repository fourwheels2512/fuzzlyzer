"""Smoke tests for the topofuzz CLI.

The four subcommands (scan/run/report/verify) are stubs until later weekends.
This file verifies they exit cleanly with a friendly message rather than
raising a traceback (which the audit flagged as bad first-user UX).
"""

from __future__ import annotations

from typer.testing import CliRunner

from topofuzz.cli import RESERVED_FLAGS, app


def test_help_succeeds():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "topofuzz" in result.output.lower()


def test_version_prints_and_exits_cleanly():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "topofuzz" in result.output


def test_scan_stub_exits_with_friendly_message():
    result = CliRunner().invoke(app, ["scan", "some-fixture"])
    assert result.exit_code == 2
    assert "not yet implemented" in (result.output or "").lower()


def test_run_stub_exits_with_friendly_message():
    result = CliRunner().invoke(app, ["run", "some-fixture"])
    assert result.exit_code == 2
    assert "not yet implemented" in (result.output or "").lower()


def test_report_stub_exits_with_friendly_message():
    result = CliRunner().invoke(app, ["report", "some-dir"])
    assert result.exit_code == 2


def test_verify_stub_exits_with_friendly_message():
    result = CliRunner().invoke(app, ["verify", "some-dir"])
    assert result.exit_code == 2


def test_reserved_flags_constant_matches_blueprint_section_16():
    """BLUEPRINT §16 #2 names 9 reserved flags. Keep the constant in sync."""
    expected = {
        "--conformance", "--signoff", "--judge-pool",
        "--sandbox", "--key-uri", "--kms",
        "--allow-side-effects", "--tenant-id", "--per-cell-entropy",
    }
    assert set(RESERVED_FLAGS) == expected
