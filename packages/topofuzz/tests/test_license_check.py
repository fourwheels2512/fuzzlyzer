# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``topofuzz.license_check`` + CLI click-through wiring.

BACKLOG entry "MAJOR — First-run CLI license click-through acknowledgment"
acceptance criteria:
- First-run prompts and writes acceptance file on ``y``.
- Subsequent runs do not prompt.
- ``--accept-license`` flag bypasses the prompt.
- Non-interactive stdin refuses without writing.
- Version mismatch in stored acceptance triggers re-prompt.
- File permissions are 0o600 on POSIX.
- ``topofuzz verify`` runs without triggering the prompt.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topofuzz import __version__
from topofuzz.cli import _LICENSE_EXEMPT_COMMANDS, app
from topofuzz.license_check import (
    LICENSE_REFUSED_EXIT_CODE,
    LICENSE_SUMMARY,
    LICENSE_TAG,
    PROMPT,
    HomeUnresolvableError,
    acceptance_path,
    prompt_and_record,
)


# Sentinel fixture spec — never resolves to a real file. Used in CLI
# tests that exercise the license gate (the gate fires before the
# command body, so the fixture file does not need to exist).
_NEVER_RESOLVED_FIXTURE = "nonexistent-fixture-path.py:graph"


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to ``tmp_path`` for the duration of the
    test. Both ``acceptance_path()`` and the CLI wiring resolve to
    ``<tmp_path>/.topofuzz/license-accepted``.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``sys.stdin.isatty()`` return True for tests that exercise
    the interactive path under CliRunner (where stdin is a pipe by
    default)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)


# ---------------------------------------------------------------------------
# Library-level (``prompt_and_record`` direct) tests
# ---------------------------------------------------------------------------


def test_acceptance_path_under_home(fake_home: Path) -> None:
    assert acceptance_path() == fake_home / ".topofuzz" / "license-accepted"


def test_acceptance_path_raises_homeunresolvable_when_home_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If neither HOME nor USERPROFILE is set (some Docker entrypoints),
    Path.home() raises RuntimeError. We wrap it as
    HomeUnresolvableError so the CLI can emit a clean diagnostic."""
    def _raise() -> Path:
        raise RuntimeError("home not set")
    monkeypatch.setattr(Path, "home", _raise)
    with pytest.raises(HomeUnresolvableError, match="home not set"):
        acceptance_path()


def test_valid_acceptance_skips_prompt_without_input(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-crafted valid acceptance file (matching version + tag)
    must NOT cause prompt_and_record to call input(). This is the
    positive-version-match test that proves the steady-state silent
    path, decoupled from the writer."""
    path = acceptance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": __version__,
                "accepted_at": "2026-05-16T00:00:00.000000Z",
                "license": LICENSE_TAG,
            }
        ),
        encoding="utf-8",
    )

    def _should_not_be_called() -> str:
        raise AssertionError("input() called despite valid acceptance file")
    monkeypatch.setattr("builtins.input", _should_not_be_called)
    assert prompt_and_record() is True


def test_first_run_writes_acceptance_on_y(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert prompt_and_record() is True
    path = acceptance_path()
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == __version__
    assert data["license"] == "BUSL-1.1"
    assert "accepted_at" in data
    # ISO 8601 UTC timestamp
    assert data["accepted_at"].endswith("Z")


def test_first_run_refuses_on_n_does_not_write(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda: "n")
    assert prompt_and_record() is False
    assert not acceptance_path().exists()


def test_first_run_refuses_on_empty_does_not_write(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty answer ``[y/N]`` defaults to N — no acceptance written."""
    monkeypatch.setattr("builtins.input", lambda: "")
    assert prompt_and_record() is False
    assert not acceptance_path().exists()


def test_first_run_eof_refuses(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _eof() -> str:
        raise EOFError
    monkeypatch.setattr("builtins.input", _eof)
    assert prompt_and_record() is False
    assert not acceptance_path().exists()


def test_yes_word_accepted(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda: "yes")
    assert prompt_and_record() is True


def test_uppercase_y_accepted(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda: "Y")
    assert prompt_and_record() is True


def test_subsequent_run_does_not_prompt(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After acceptance, ``prompt_and_record`` returns True without
    calling ``input``."""
    monkeypatch.setattr("builtins.input", lambda: "y")
    prompt_and_record()

    def _should_not_be_called() -> str:
        raise AssertionError("input() was called on a subsequent run")
    monkeypatch.setattr("builtins.input", _should_not_be_called)
    assert prompt_and_record() is True


def test_force_bypasses_prompt(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _should_not_be_called() -> str:
        raise AssertionError("input() was called with force=True")
    monkeypatch.setattr("builtins.input", _should_not_be_called)
    assert prompt_and_record(force=True) is True
    assert acceptance_path().exists()


def test_non_interactive_refuses_without_writing(fake_home: Path) -> None:
    assert prompt_and_record(non_interactive=True) is False
    assert not acceptance_path().exists()


def test_non_interactive_with_force_writes(fake_home: Path) -> None:
    """force=True takes precedence over non_interactive — that's the CI
    path: ``--accept-license`` set on a piped invocation."""
    assert prompt_and_record(force=True, non_interactive=True) is True
    assert acceptance_path().exists()


def test_version_mismatch_reprompts(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored acceptance from an older version is ignored — the user
    must re-accept under the new (potentially modified) license."""
    path = acceptance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "0.0.1-ancient",
                "accepted_at": "1999-01-01T00:00:00Z",
                "license": "BUSL-1.1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert prompt_and_record() is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == __version__


def test_license_tag_mismatch_reprompts(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a future re-license changes the BSL tag (e.g. to a custom
    EULA tag), the prior acceptance does not carry over."""
    path = acceptance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": __version__,
                "accepted_at": "2026-05-16T00:00:00Z",
                "license": "Apache-2.0",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert prompt_and_record() is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["license"] == "BUSL-1.1"


def test_corrupt_json_reprompts(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = acceptance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {{{{", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert prompt_and_record() is True


def test_non_dict_json_reprompts(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON file that decodes to a non-dict (list, str, number)
    must be rejected — defensive against accidental file clobbering."""
    path = acceptance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[\"not a dict\"]", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert prompt_and_record() is True


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod 0o600 is a no-op on Windows (NTFS ACL inheritance)",
)
def test_acceptance_file_chmod_0600_on_posix(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda: "y")
    prompt_and_record()
    path = acceptance_path()
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_acceptance_file_is_valid_json_with_required_keys(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert the required keys are present. We deliberately do NOT
    assert exact key equality so a future schema bump (e.g. adding
    ``hostname`` or ``user``) doesn't break this test — and
    ``_is_accepted`` is lenient on extra keys by design."""
    monkeypatch.setattr("builtins.input", lambda: "y")
    prompt_and_record()
    data = json.loads(acceptance_path().read_text(encoding="utf-8"))
    required = {"version", "accepted_at", "license"}
    assert required <= set(data.keys())


def test_prompt_summary_contains_key_terms(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression-protect the key load-bearing clauses of the BSL
    summary (threshold, conversion timeline, commercial contact)."""
    monkeypatch.setattr("builtins.input", lambda: "n")
    prompt_and_record()
    captured = capsys.readouterr()
    text = captured.err
    assert "Business Source License" in text
    assert "USD $100,000" in text
    assert "four years" in text
    assert "kiran@modelbrew.ai" in text
    assert PROMPT in text


def test_summary_text_is_exported_for_reuse() -> None:
    """SECURITY.md / PRIVACY.md may need to reference the same summary;
    keep it importable as a constant."""
    assert "Business Source License" in LICENSE_SUMMARY
    assert "USD $100,000" in LICENSE_SUMMARY


# ---------------------------------------------------------------------------
# CLI-level wiring tests (verify is exempt, other commands gated)
# ---------------------------------------------------------------------------


def test_verify_skips_license_check_via_prompt_mock(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``topofuzz verify`` MUST bypass the license gate code path
    entirely. Strong assertion (vs string-matching the error message):
    monkeypatch ``prompt_and_record`` to raise — verify must not raise,
    proving the gate was never invoked for this command.
    """
    def _fail_if_called(*_: object, **__: object) -> bool:
        raise AssertionError(
            "prompt_and_record was called for `topofuzz verify` — exemption broke"
        )
    monkeypatch.setattr("topofuzz.cli.prompt_and_record", _fail_if_called)
    runner = CliRunner()
    # Nonexistent run_dir → verify will still exit on its own argument
    # validation (exit 2 with a "not a directory" message), but the
    # gate path must not have been touched.
    result = runner.invoke(app, ["verify", str(fake_home / "no-such-run")])
    # Just ensure we got through to verify's body — it printed its own
    # diagnostic, not the gate's.
    combined = (result.stdout or "") + (result.stderr or "")
    assert "license acceptance required" not in combined
    assert not acceptance_path().exists()


def test_scan_refuses_when_non_interactive_and_no_acceptance(
    fake_home: Path,
) -> None:
    """Under CliRunner stdin is not a TTY → non_interactive=True → the
    gate refuses with exit code 2 and an actionable message."""
    runner = CliRunner()
    result = runner.invoke(app, ["scan", _NEVER_RESOLVED_FIXTURE])
    assert result.exit_code == LICENSE_REFUSED_EXIT_CODE
    combined = (result.stdout or "") + (result.stderr or "")
    assert "license acceptance required" in combined
    assert not acceptance_path().exists()


def test_accept_license_flag_bypasses_gate(fake_home: Path) -> None:
    """``topofuzz --accept-license <cmd>`` writes the acceptance file
    and proceeds into the command. The command itself may then fail
    on its own arguments — we assert on the gate, not the command."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["--accept-license", "scan", _NEVER_RESOLVED_FIXTURE]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "license acceptance required" not in combined
    assert acceptance_path().exists()
    data = json.loads(acceptance_path().read_text(encoding="utf-8"))
    assert data["version"] == __version__


def test_accept_license_flag_after_subcommand_gives_placement_hint(
    fake_home: Path,
) -> None:
    """``topofuzz scan ... --accept-license`` (flag after subcommand)
    is a common UX mistake. The CLI must surface a placement hint
    rather than the generic "unknown option" message.

    Pre-write a valid acceptance directly so the gate does NOT fire
    on this invocation — we want to exercise scan's
    ``_reject_unknown_extras`` branch, not the gate.
    """
    accept_path = acceptance_path()
    accept_path.parent.mkdir(parents=True, exist_ok=True)
    accept_path.write_text(
        json.dumps(
            {
                "version": __version__,
                "accepted_at": "2026-05-16T00:00:00.000000Z",
                "license": LICENSE_TAG,
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app, ["scan", _NEVER_RESOLVED_FIXTURE, "--accept-license"]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "must come BEFORE the subcommand" in combined


def test_accept_license_flag_after_subcommand_includes_co_typos(
    fake_home: Path,
) -> None:
    """Pass-2 RT M1: the placement hint must NOT swallow other
    unknown flags. If the operator typed both
    ``--variantss=8 --accept-license``, they need to see both
    problems, not just the placement one."""
    accept_path = acceptance_path()
    accept_path.parent.mkdir(parents=True, exist_ok=True)
    accept_path.write_text(
        json.dumps(
            {
                "version": __version__,
                "accepted_at": "2026-05-16T00:00:00.000000Z",
                "license": LICENSE_TAG,
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scan",
            _NEVER_RESOLVED_FIXTURE,
            "--variantss=8",
            "--accept-license",
        ],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "must come BEFORE the subcommand" in combined
    assert "Also unknown option(s)" in combined
    assert "--variantss" in combined


def test_accept_license_persists_across_invocations(fake_home: Path) -> None:
    """Once accepted, subsequent runs without --accept-license must
    still proceed past the gate."""
    runner = CliRunner()
    # First run with the flag.
    runner.invoke(app, ["--accept-license", "scan", _NEVER_RESOLVED_FIXTURE])
    assert acceptance_path().exists()
    # Second run without the flag — should NOT hit the gate.
    result = runner.invoke(app, ["scan", _NEVER_RESOLVED_FIXTURE])
    combined = (result.stdout or "") + (result.stderr or "")
    assert "license acceptance required" not in combined


def test_version_flag_works_without_acceptance(fake_home: Path) -> None:
    """``topofuzz --version`` is handled in the root callback before
    the license gate would fire (the gate runs only for invoked
    subcommands)."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert not acceptance_path().exists()


def test_help_works_without_acceptance(fake_home: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert not acceptance_path().exists()


def test_no_subcommand_does_not_create_acceptance(fake_home: Path) -> None:
    """``topofuzz`` (no subcommand) prints help and exits — the gate is
    only for invoked subcommands, so no acceptance file is created.

    With ``no_args_is_help=True``, Click returns exit code 2 (the
    classic "no command" code). What matters for the gate is that the
    acceptance file is NOT created, not the specific exit code.
    """
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert not acceptance_path().exists()


# ---------------------------------------------------------------------------
# Drift guards
# ---------------------------------------------------------------------------


def test_license_exempt_set_only_contains_registered_commands() -> None:
    """Every entry in ``_LICENSE_EXEMPT_COMMANDS`` MUST be a real
    user-invocable Typer subcommand name. Without this guard, a
    rename of ``verify`` (e.g. to ``verify-bundle`` via ``name=``
    kwarg, with callback still named ``verify``) would silently gate
    it — defeating the buyer-side audit flow contract.

    We assert against the OPERATOR-FACING name only (``cmd.name`` if
    set via ``@app.command(name="...")``, else the callback function
    name) — NOT the union with the callback name. Otherwise the guard
    is too lenient: a renamed command's callback would still be in
    the union and silently match the stale exempt entry.
    """
    user_invocable_names = {
        cmd.name if cmd.name is not None else cmd.callback.__name__
        for cmd in app.registered_commands
    }
    missing = _LICENSE_EXEMPT_COMMANDS - user_invocable_names
    assert not missing, (
        f"_LICENSE_EXEMPT_COMMANDS references unregistered command(s) "
        f"{missing!r}; user-invocable command names = "
        f"{user_invocable_names!r}"
    )


def test_gate_fires_when_autouse_acceptance_removed(
    fake_home: Path,
) -> None:
    """Production-gate regression test. The conftest autouse fixture
    pre-creates a valid acceptance file for every CLI test (so the
    1500+ existing CLI tests don't all fail on the gate). This test
    DISABLES that pre-acceptance (by using the test-module-local
    ``fake_home`` fixture which monkeypatches Path.home to a FRESH tmp
    dir with no acceptance file) and verifies that a representative
    non-exempt subcommand DOES refuse without prior acceptance.

    If a future commit makes the gate a no-op (e.g. inverted condition,
    typo'd exempt set), this test fails — even though no other CLI
    test would notice.
    """
    # Confirm the per-test fake_home is "fresh" (no acceptance).
    assert not acceptance_path().exists()
    runner = CliRunner()
    result = runner.invoke(app, ["scan", _NEVER_RESOLVED_FIXTURE])
    assert result.exit_code == LICENSE_REFUSED_EXIT_CODE
    combined = (result.stdout or "") + (result.stderr or "")
    assert "license acceptance required" in combined
