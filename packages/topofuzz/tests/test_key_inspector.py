# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``topofuzz.key_inspector`` + ``topofuzz keys`` CLI.

BACKLOG entry "MINOR — `topofuzz keys` CLI subcommand" required
tests (spec lines 1372-1380):
- All keys present → all status="present", masked correctly.
- All keys absent → all status="missing", exit code 1.
- Mix of present/absent → correct per-row status, exit code 0.
- Short key (<8 chars) → status="too_short", empty masked_value.
- Key with exactly 8 chars → status="present", masked shows 4+4
  with NO middle ellipsis (boundary case).
- The ``env`` parameter override works (test isolation).
- The CLI never prints the full key (golden-file test of stdout).
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topofuzz.cli import app
from topofuzz.key_inspector import (
    _MASK_ELLIPSIS,
    _MIN_REAL_KEY_LENGTH,
    KeyInfo,
    at_least_one_present,
    format_table,
    inspect_keys,
)


def _combined_output(result) -> str:
    """Stdout + stderr joined, regardless of Click version (mirrors
    ``test_cli_audit._combined_output``). Click 8.2's ``CliRunner``
    captures stderr separately; Click 8.1's default runner mixes it
    into ``result.output`` and RAISES ``ValueError`` on ``.stderr``.
    """
    out = result.output or ""
    try:
        err = result.stderr or ""
    except (AttributeError, ValueError):
        err = ""  # Click 8.1: stderr already mixed into output.
    return out + err


@pytest.fixture(autouse=True)
def _clear_vendor_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Click's CliRunner ``env=`` parameter MERGES into ``os.environ``
    rather than replacing it. Any real GEMINI_API_KEY / OPENAI_API_KEY
    / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY exported by the developer
    leaks through and falsifies test results.

    Autouse-clear all four BEFORE each test runs; tests opt into
    specific values by passing ``env={...}`` to CliRunner.
    """
    for var in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Library-level (``inspect_keys`` direct) tests
# ---------------------------------------------------------------------------


def test_all_present_returns_present_status_and_masks() -> None:
    env = {
        "GEMINI_API_KEY": "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
        "OPENAI_API_KEY": "sk-abc1234567890ABCDEFXYZ0987654321xyz",
        "ANTHROPIC_API_KEY": "sk-ant-api03-12345678901234567890",
        "DEEPSEEK_API_KEY": "sk-deepseek-abcdefghij1234567890",
    }
    rows = inspect_keys(env=env)
    assert [r.status for r in rows] == ["present"] * 4
    # Vendor + env_var ordering is load-bearing for the CLI table —
    # confirm it stays stable.
    assert [r.vendor for r in rows] == [
        "google",
        "openai",
        "anthropic",
        "deepseek",
    ]
    assert [r.env_var for r in rows] == [
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
    ]
    # Masking shows first 4 + ellipsis + last 4
    for row in rows:
        raw = env[row.env_var]
        assert row.masked_value.startswith(raw[:4])
        assert row.masked_value.endswith(raw[-4:])
        assert _MASK_ELLIPSIS in row.masked_value
        # The full raw value MUST NEVER appear in the masked form.
        assert raw not in row.masked_value


def test_all_absent_returns_missing_status() -> None:
    rows = inspect_keys(env={})
    assert [r.status for r in rows] == ["missing"] * 4
    assert all(r.masked_value == "" for r in rows)
    assert not at_least_one_present(rows)


def test_mixed_present_absent_per_row_status() -> None:
    env = {
        "GEMINI_API_KEY": "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
        # OPENAI_API_KEY absent
        "ANTHROPIC_API_KEY": "sk-ant-api03-12345678901234567890",
        # DEEPSEEK_API_KEY absent
    }
    rows = inspect_keys(env=env)
    by_vendor = {r.vendor: r for r in rows}
    assert by_vendor["google"].status == "present"
    assert by_vendor["openai"].status == "missing"
    assert by_vendor["anthropic"].status == "present"
    assert by_vendor["deepseek"].status == "missing"
    assert at_least_one_present(rows)


def test_short_key_marked_too_short_and_masks_empty() -> None:
    env = {"GEMINI_API_KEY": "abc"}
    rows = inspect_keys(env=env)
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "too_short"
    # A short value's prefix/suffix must NOT be printed — could be the
    # whole key.
    assert google_row.masked_value == ""
    # No part of "abc" appears anywhere in the row representation.
    assert "abc" not in google_row.masked_value


def test_below_floor_marked_too_short() -> None:
    """Pass-1 RT M1: spec line 1376 suggested an 8-char value renders
    as 4+4 with no ellipsis — which prints the WHOLE 8-char key
    verbatim. We bumped ``_MIN_REAL_KEY_LENGTH`` to 12 so the mask
    always hides at least 4 middle chars.

    An 8-char value is now ``too_short``, NOT ``present``. This test
    locks the spec deviation in place — if a future agent reverts the
    threshold to 8, this test fires."""
    env = {"GEMINI_API_KEY": "AIzaWXYZ"}  # 8 chars — below 12-char floor
    rows = inspect_keys(env=env)
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "too_short"
    assert google_row.masked_value == ""
    # Critical: the 8-char value must NOT leak verbatim anywhere.
    assert "AIzaWXYZ" not in google_row.masked_value


def test_at_floor_present_with_ellipsis_in_middle() -> None:
    """Boundary partner: at exactly ``_MIN_REAL_KEY_LENGTH`` chars,
    the value is ``present`` and the mask hides the middle chars."""
    raw = "A" * _MIN_REAL_KEY_LENGTH  # exactly at floor
    raw = raw[:4] + "MIDDLE" + raw[-4:]  # 14 chars
    raw = raw[:_MIN_REAL_KEY_LENGTH]  # truncate back to exactly 12
    env = {"GEMINI_API_KEY": raw}
    rows = inspect_keys(env=env)
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "present"
    assert _MASK_ELLIPSIS in google_row.masked_value
    # Middle chars must NOT appear unmasked.
    middle = raw[4:-4]
    assert middle not in google_row.masked_value
    assert google_row.masked_value == raw[:4] + _MASK_ELLIPSIS + raw[-4:]


def test_whitespace_only_value_treated_as_missing() -> None:
    """Pass-1 RT M3: ``export GEMINI_API_KEY="   "`` is operator-intent
    "unset" (shell trailing-space typo, copy/paste accident), NOT a
    3-char key. Strip-then-check; render as ``missing``."""
    env = {"GEMINI_API_KEY": "   "}
    rows = inspect_keys(env=env)
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "missing"
    assert google_row.masked_value == ""


def test_padded_real_key_strips_whitespace_before_present() -> None:
    """Pass-1 RT M3: a real key with leading/trailing whitespace
    should still render as ``present`` after stripping."""
    raw = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345678"
    env = {"GEMINI_API_KEY": f"  {raw}  "}
    rows = inspect_keys(env=env)
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "present"
    assert google_row.masked_value == raw[:4] + _MASK_ELLIPSIS + raw[-4:]


def test_mask_helper_raises_below_floor() -> None:
    """Pass-1 RT P1 / CR-m1: ``_mask`` must guard with ``ValueError``
    (not ``assert``) so ``python -O`` cannot strip the check and leak
    short values."""
    from topofuzz.key_inspector import _mask
    with pytest.raises(ValueError, match="length >="):
        _mask("short")


def test_empty_string_value_treated_as_missing() -> None:
    """Empty env var (``export GEMINI_API_KEY=``) is functionally
    'unset' — distinguish present vs missing by truthiness, not by
    just .get()."""
    env = {"GEMINI_API_KEY": ""}
    rows = inspect_keys(env=env)
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "missing"
    assert google_row.masked_value == ""


def test_env_parameter_override_isolates_from_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When an explicit env is passed, os.environ is NOT consulted.
    Critical for test isolation — without it, a developer running
    pytest with a real GEMINI_API_KEY set would see misleading test
    failures."""
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-show-up")
    rows = inspect_keys(env={})
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "missing"
    assert "should-not-show-up" not in google_row.masked_value


def test_default_env_reads_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When env=None (default), reads os.environ at call time."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTESTKEY1234567890")
    rows = inspect_keys()  # default None → reads os.environ
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "present"


def test_runs_under_1_second() -> None:
    """BACKLOG acceptance criterion line 1390: '<100ms with no vendor
    calls'. The function makes zero network requests by construction
    (only reads dict); the test budget is relaxed to 1s/100 iterations
    so slow CI runners (Windows GC pauses, throttled VMs) don't flake.
    A real timing regression would still surface as orders-of-magnitude
    slower, not 2x."""
    env = {
        "GEMINI_API_KEY": "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
        "OPENAI_API_KEY": "sk-1234567890ABCDEFXYZ",
    }
    start = time.perf_counter()
    for _ in range(100):
        inspect_keys(env=env)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"inspect_keys averaged > 10ms/call: {elapsed/100*1000:.2f}ms"


def test_keyinfo_is_frozen() -> None:
    """KeyInfo is a frozen dataclass — pass-1 CR-m2 fix: narrow
    exception class so the test actually verifies frozen semantics."""
    info = KeyInfo(
        vendor="google",
        env_var="GEMINI_API_KEY",
        status="present",
        masked_value=f"AIza{_MASK_ELLIPSIS}1234",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.status = "missing"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# format_table tests
# ---------------------------------------------------------------------------


def test_format_table_includes_header_and_separator() -> None:
    rows = inspect_keys(env={})
    table = format_table(rows)
    lines = table.split("\n")
    assert lines[0].startswith("vendor")
    assert "env var" in lines[0]
    assert "status" in lines[0]
    assert set(lines[1]) == {"-"}, "expected dash separator on line 2"
    # One line per vendor.
    assert len(lines) == 2 + 4


def test_format_table_masks_present_values() -> None:
    env = {"GEMINI_API_KEY": "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    rows = inspect_keys(env=env)
    table = format_table(rows)
    # The full key MUST NOT appear in the table.
    assert env["GEMINI_API_KEY"] not in table
    # The masked form DOES appear.
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.masked_value in table


def test_format_table_short_key_does_not_leak_value() -> None:
    env = {"GEMINI_API_KEY": "abc"}
    rows = inspect_keys(env=env)
    table = format_table(rows)
    assert "abc" not in table
    assert "too_short" in table


def test_format_table_missing_rows_say_missing() -> None:
    rows = inspect_keys(env={})
    table = format_table(rows)
    # 4 vendors all missing → 4 occurrences (one per row).
    assert table.count("missing") == 4


def test_format_table_golden_byte_stable() -> None:
    """Pin the exact bytes for one mixed-state row set so future
    refactors don't silently shift the column layout (procurement
    snapshots / docs would drift)."""
    env = {
        "GEMINI_API_KEY": "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
        "OPENAI_API_KEY": "   ",  # whitespace → missing
        "ANTHROPIC_API_KEY": "sk-ant-api03-1234567890",
        # DEEPSEEK_API_KEY absent
    }
    rows = inspect_keys(env=env)
    table = format_table(rows)
    expected = (
        "vendor      env var               status\n"
        "----------------------------------------\n"
        "google      GEMINI_API_KEY        present  AIza..2345\n"
        "openai      OPENAI_API_KEY        missing\n"
        "anthropic   ANTHROPIC_API_KEY     present  sk-a..7890\n"
        "deepseek    DEEPSEEK_API_KEY      missing"
    )
    assert table == expected


# ---------------------------------------------------------------------------
# Drift / cross-file regression tests
# ---------------------------------------------------------------------------


def test_vendor_list_matches_security_md_network_egress() -> None:
    """BACKLOG line 1369-1371: adding a new vendor here MUST include
    a SECURITY.md "Network Egress" update in the same PR. This
    regression test enforces the link by asserting every vendor env
    var in ``VENDORS`` is referenced in SECURITY.md, AND every
    ``*_API_KEY`` token in SECURITY.md has a corresponding ``VENDORS``
    row (pass-2 RT m4: reverse drift check).

    If you're triggering this test, EITHER you added a vendor here
    without updating SECURITY.md (fix SECURITY.md), OR you added a
    vendor to SECURITY.md without adding it here (add to ``VENDORS``).
    """
    import re

    from topofuzz.key_inspector import VENDORS

    # Find SECURITY.md from repo root. Tests run from /SD or
    # /SD/packages/topofuzz, so walk up. Pass-2 M3: fail loud if not
    # found (a future restructure must NOT silently skip the drift
    # guard).
    here = Path(__file__).resolve()
    candidate = here
    for _ in range(8):
        if (candidate / "SECURITY.md").exists():
            break
        candidate = candidate.parent
    else:
        pytest.fail(
            f"SECURITY.md not found within 8 parents of {here}. "
            "If the repo layout changed, update this walk."
        )
    security_md = (candidate / "SECURITY.md").read_text(encoding="utf-8")

    # Forward: every VENDORS row's env var appears in SECURITY.md.
    for vendor, env_var in VENDORS:
        assert env_var in security_md, (
            f"VENDORS has {vendor!r} ({env_var}) but SECURITY.md does "
            "not reference that env var — update SECURITY.md or remove "
            "the vendor here."
        )

    # Reverse: every `*_API_KEY` token in SECURITY.md is in VENDORS,
    # so the two cannot drift in the opposite direction (a doc-only
    # vendor reference without an implementation).
    api_key_tokens = set(re.findall(r"\b[A-Z_]+_API_KEY\b", security_md))
    vendor_env_vars = {env_var for _, env_var in VENDORS}
    extra_in_security = api_key_tokens - vendor_env_vars
    assert not extra_in_security, (
        f"SECURITY.md references {extra_in_security!r} but those env "
        "vars are NOT in topofuzz.key_inspector.VENDORS — either add "
        "them to VENDORS or remove from SECURITY.md."
    )


def test_internal_whitespace_preserved_but_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass-2 RT m1: internal whitespace inside a value is NOT
    stripped (only leading/trailing). Verify masking still hides
    the middle, so internal whitespace is not leaked."""
    raw = "AIza Sy ABCDEFGHIJK MNOP"  # 24 chars, internal spaces
    rows = inspect_keys(env={"GEMINI_API_KEY": raw})
    google_row = next(r for r in rows if r.vendor == "google")
    assert google_row.status == "present"
    # Masked form omits the internal spaces (which live in the
    # middle of the value).
    assert google_row.masked_value == raw[:4] + _MASK_ELLIPSIS + raw[-4:]
    # The full raw value with internal whitespace MUST NOT appear.
    assert raw not in google_row.masked_value


# ---------------------------------------------------------------------------
# CLI-level tests (``topofuzz keys``)
# ---------------------------------------------------------------------------


def test_cli_keys_no_keys_exits_1() -> None:
    """The ``_clear_vendor_keys`` autouse fixture removes any real
    vendor keys; this test verifies the empty-env exit code is 1."""
    runner = CliRunner()
    result = runner.invoke(app, ["keys"])
    assert result.exit_code == 1
    assert "missing" in result.output


def test_cli_keys_at_least_one_present_exits_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    env = {
        "GEMINI_API_KEY": "AIzaSyTESTKEY01234567890",
        # other keys unset
    }
    result = runner.invoke(app, ["keys"], env=env)
    assert result.exit_code == 0
    assert "present" in result.output


def test_cli_keys_never_prints_full_key() -> None:
    """Golden-file-style assertion: the value passed in MUST NOT
    appear verbatim in stdout. Strongest protection against accidental
    leak of a real key in a screen-shared support call."""
    runner = CliRunner()
    secret = "AIzaSyTHIS_IS_A_FAKE_LOOKING_SECRET_VALUE"
    env = {"GEMINI_API_KEY": secret}
    result = runner.invoke(app, ["keys"], env=env)
    assert result.exit_code == 0
    assert secret not in result.output
    # Masked prefix + suffix DO appear.
    assert secret[:4] in result.output
    assert secret[-4:] in result.output


def test_cli_keys_short_key_does_not_leak() -> None:
    runner = CliRunner()
    env = {"GEMINI_API_KEY": "abc"}
    result = runner.invoke(app, ["keys"], env=env)
    # too_short with 1 present elsewhere would be exit 0; here all
    # other keys are missing AND this one is too_short → no "present"
    # → exit 1.
    assert "abc" not in result.output
    assert "too_short" in result.output
    assert result.exit_code == 1


def test_cli_keys_rejects_reserved_flag() -> None:
    runner = CliRunner()
    env = {"GEMINI_API_KEY": "AIzaSyTESTKEY01234567890"}
    result = runner.invoke(app, ["keys", "--kms"], env=env)
    assert result.exit_code == 2
    combined = _combined_output(result)
    assert "--kms" in combined


def test_cli_keys_rejects_unknown_flag() -> None:
    runner = CliRunner()
    env = {"GEMINI_API_KEY": "AIzaSyTESTKEY01234567890"}
    result = runner.invoke(app, ["keys", "--bogus"], env=env)
    assert result.exit_code == 2


def test_cli_keys_help_lists_vendors() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["keys", "--help"])
    assert result.exit_code == 0
    # All four vendors should be findable in the help text so an
    # operator running --help sees what's checked.
    assert "google" in result.output.lower()
    assert "openai" in result.output.lower()
    assert "anthropic" in result.output.lower()
    assert "deepseek" in result.output.lower()


def test_cli_keys_is_license_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per BACKLOG spec, `topofuzz keys` is NOT in the exempt set —
    the license gate fires before the body. This test removes the
    pre-accepted license file and verifies the gate runs.

    Documented spec deviation if a future product call exempts
    `keys`: this test would be inverted (proves NOT gated).
    """
    from topofuzz.license_check import acceptance_path
    # The autouse conftest pre-creates the acceptance file in the
    # per-test fake home; unlink it so this test exercises the gate.
    accept = acceptance_path()
    # Pass-2 M4 fix: lock the conftest contract — if a future change
    # stops pre-creating the file, this assert fires (rather than the
    # test silently passing because the gate fires for a different
    # reason).
    assert accept.exists(), (
        "conftest invariant: _preaccept_license autouse fixture must "
        "pre-create the acceptance file before this test runs"
    )
    accept.unlink()
    runner = CliRunner()
    env = {"GEMINI_API_KEY": "AIzaSyTESTKEY01234567890"}
    result = runner.invoke(app, ["keys"], env=env)
    # License gate fires (exit 2 with "license acceptance required").
    assert result.exit_code == 2
    combined = _combined_output(result)
    assert "license acceptance required" in combined
