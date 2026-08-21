# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for the CLI ``.env`` auto-load hook (ops fix, 2026-06-11).

The failure mode this closes: LiteLLM reads vendor keys from the process
environment only, so a key parked in the project's ``.env`` was silently
ignored unless the operator ``export``-ed it in every shell — and the
resulting API_KEY_INVALID surfaced from the vendor only AFTER the license
prompt and cost estimate (observed 2026-05-30).

Threat model under test (RT 2026-06-11): the loader fills ONLY the
documented vendor keys (:data:`topofuzz.key_inspector.VENDORS`) — an
unrestricted loader would let a hostile checkout's ``.env`` set
absent-by-default vars like ``OPENAI_API_BASE`` (vendor-traffic/key
redirect) or ``PYTHONPATH`` (arbitrary import in UI subprocess children).

The suite-wide ``_no_dotenv`` conftest fixture sets ``TOPOFUZZ_NO_DOTENV=1``
so the developer's real repo-root ``.env`` can never leak into other CLI
tests; the tests below delete that variable explicitly and operate on tmp
directories with fake key values, popping anything the loader set in a
``finally`` so no fake key outlives its test.
"""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from topofuzz.cli import (
    _DOTENV_ALLOWED_KEYS,
    _DOTENV_DISABLE_ENV,
    _load_dotenv_if_present,
    app,
)

# Real vendor key NAMES (required by the allowlist) with obviously fake
# values. ≥12 chars so `topofuzz keys` counts them as present rather
# than too_short.
_GEMINI = "GEMINI_API_KEY"
_DEEPSEEK = "DEEPSEEK_API_KEY"
_OPENAI = "OPENAI_API_KEY"


def _runner() -> CliRunner:
    """Click 8.1/8.2 compat shim — mirrors ``test_cli_scan._runner``."""
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def _enable_dotenv(monkeypatch, *absent_keys: str) -> None:
    """Undo the suite-wide ``TOPOFUZZ_NO_DOTENV`` guard and guarantee
    ``absent_keys`` are not present in the ambient environment (so the
    loader's fill path is exercised deterministically on machines that
    DO export real keys)."""
    monkeypatch.delenv(_DOTENV_DISABLE_ENV, raising=False)
    for key in absent_keys:
        monkeypatch.delenv(key, raising=False)


def test_loader_fills_missing_vendor_key(tmp_path: Path, monkeypatch) -> None:
    _enable_dotenv(monkeypatch, _GEMINI)
    (tmp_path / ".env").write_text(
        f"{_GEMINI}=fake-key-for-test-12345\n", encoding="utf-8"
    )
    try:
        assert _load_dotenv_if_present(tmp_path) == 1
        assert os.environ[_GEMINI] == "fake-key-for-test-12345"
    finally:
        os.environ.pop(_GEMINI, None)


def test_loader_ignores_non_vendor_keys(tmp_path: Path, monkeypatch) -> None:
    """Allowlist regression (RT 2026-06-11): a hostile ``.env`` must not
    be able to plant arbitrary env vars — only the documented vendor
    keys are ever filled."""
    _enable_dotenv(monkeypatch, _GEMINI)
    (tmp_path / ".env").write_text(
        "OPENAI_API_BASE=https://evil.example/v1\n"
        "PYTHONPATH=/tmp/evil\n"
        "HOME=/tmp/evil-home\n"
        f"{_GEMINI}=fake-key-for-test-12345\n",
        encoding="utf-8",
    )
    try:
        assert _load_dotenv_if_present(tmp_path) == 1  # only GEMINI
        assert os.environ[_GEMINI] == "fake-key-for-test-12345"
        assert os.environ.get("OPENAI_API_BASE") != "https://evil.example/v1"
        assert os.environ.get("PYTHONPATH") != "/tmp/evil"
        assert os.environ.get("HOME") != "/tmp/evil-home"
    finally:
        os.environ.pop(_GEMINI, None)
    # And the allowlist itself stays pinned to the documented vendors.
    assert _DOTENV_ALLOWED_KEYS == {
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
    }


def test_loader_never_overrides_existing_env(tmp_path: Path, monkeypatch) -> None:
    _enable_dotenv(monkeypatch)
    monkeypatch.setenv(_GEMINI, "from-real-env")
    (tmp_path / ".env").write_text(
        f"{_GEMINI}=from-dotenv\n", encoding="utf-8"
    )
    assert _load_dotenv_if_present(tmp_path) == 0
    assert os.environ[_GEMINI] == "from-real-env"


def test_loader_disabled_by_env_var(tmp_path: Path, monkeypatch) -> None:
    # conftest's _no_dotenv already sets the disable var; assert the
    # contract directly rather than relying on fixture ordering.
    monkeypatch.setenv(_DOTENV_DISABLE_ENV, "1")
    monkeypatch.delenv(_GEMINI, raising=False)
    (tmp_path / ".env").write_text(
        f"{_GEMINI}=fake-key-for-test-12345\n", encoding="utf-8"
    )
    assert _load_dotenv_if_present(tmp_path) == 0
    assert _GEMINI not in os.environ


def test_loader_missing_file_is_noop(tmp_path: Path, monkeypatch) -> None:
    _enable_dotenv(monkeypatch)
    assert _load_dotenv_if_present(tmp_path) == 0


def test_loader_parsing_rules(tmp_path: Path, monkeypatch) -> None:
    """export prefix, quotes, quoted-value + inline comment, unquoted
    inline comment, garbage lines, BOM."""
    _enable_dotenv(monkeypatch, _GEMINI, _DEEPSEEK, _OPENAI)
    content = (
        f"export {_GEMINI}='quoted fake key value'\n"
        "# full-line comment\n"
        "\n"
        f"{_DEEPSEEK}=plain-fake-key # trailing comment\n"
        f'{_OPENAI}="quoted-fake-key" # comment after quoted value\n'
        "NOT A KEY=skipped\n"
        "9LEADING_DIGIT=skipped\n"
        "novalue\n"
    )
    # PowerShell-style UTF-8 BOM prepended — the loader reads utf-8-sig.
    (tmp_path / ".env").write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    try:
        assert _load_dotenv_if_present(tmp_path) == 3
        assert os.environ[_GEMINI] == "quoted fake key value"
        assert os.environ[_DEEPSEEK] == "plain-fake-key"
        assert os.environ[_OPENAI] == "quoted-fake-key"  # quotes stripped
    finally:
        os.environ.pop(_GEMINI, None)
        os.environ.pop(_DEEPSEEK, None)
        os.environ.pop(_OPENAI, None)


def test_root_callback_loads_dotenv_and_notes_it(tmp_path: Path, monkeypatch) -> None:
    """End-to-end through ``_root``: a key-consuming subcommand invoked
    from a cwd carrying a ``.env`` fills the missing vendor key and says
    so on stderr — `topofuzz keys` then exits 0 BECAUSE of the .env key
    (it exits 1 when no key is visible at all)."""
    _enable_dotenv(monkeypatch, _GEMINI)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"{_GEMINI}=fake-key-for-test-12345\n", encoding="utf-8"
    )
    runner = _runner()
    try:
        result = runner.invoke(app, ["keys"])
        assert result.exit_code == 0, result.output
        assert os.environ.get(_GEMINI) == "fake-key-for-test-12345"
        assert "loaded 1 vendor key(s) from .env" in result.stderr
    finally:
        os.environ.pop(_GEMINI, None)


def test_root_callback_quiet_when_no_dotenv(tmp_path: Path, monkeypatch) -> None:
    """No ``.env`` in cwd → no stderr chatter about it."""
    _enable_dotenv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_GEMINI, "fake-key-for-test-12345")
    runner = _runner()
    result = runner.invoke(app, ["keys"])
    assert result.exit_code == 0, result.output
    assert ".env" not in result.stderr


def test_ui_command_still_loads_dotenv_despite_license_exemption(
    tmp_path: Path, monkeypatch
) -> None:
    """UX-1 made ``ui`` exempt from the BLOCKING license prompt, but it
    must still get the same ``.env`` vendor-key fill as any other
    command (unlike ``verify``, which is exempt from both) — otherwise a
    repo-root ``.env`` would silently stop reaching the browser
    keystore's ``os.environ`` fallback the moment ``ui`` became exempt.
    ``--help`` avoids actually starting the uvicorn server (Click
    dispatches ``--help`` on the subcommand only after the group
    callback ``_root`` — where the .env fill lives — has already run)."""
    _enable_dotenv(monkeypatch, _GEMINI)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"{_GEMINI}=fake-key-for-test-12345\n", encoding="utf-8"
    )
    runner = _runner()
    try:
        result = runner.invoke(app, ["ui", "--help"])
        assert result.exit_code == 0, result.output
        assert os.environ.get(_GEMINI) == "fake-key-for-test-12345"
        assert "loaded 1 vendor key(s) from .env" in result.stderr
    finally:
        os.environ.pop(_GEMINI, None)


# ---------------------------------------------------------------------------
# FIX 4(d): litellm's OWN dotenv autoload defeats TOPOFUZZ_NO_DOTENV
#
# litellm's ``__init__.py`` runs ``dotenv.load_dotenv()`` unconditionally at
# import time (LITELLM_MODE defaults to "DEV") — a SEPARATE loader from
# ``_load_dotenv_if_present`` above that knows nothing about
# TOPOFUZZ_NO_DOTENV. ``_harden_no_dotenv_against_litellm_autoload`` forces
# the (sys.modules-cached) litellm import under our control and strips any
# of the 4 vendor vars that appeared during it and were NOT present before.
#
# These tests never touch real litellm or the repo's real .env — they
# inject a fake "import" trigger that simulates the leak deterministically.
# ---------------------------------------------------------------------------


def test_harden_no_dotenv_strips_vendor_key_litellm_autoload_introduced(
    monkeypatch,
) -> None:
    from topofuzz.cli import _harden_no_dotenv_against_litellm_autoload

    monkeypatch.setenv(_DOTENV_DISABLE_ENV, "1")
    monkeypatch.delenv(_DEEPSEEK, raising=False)

    def _fake_litellm_import() -> None:
        # Simulates litellm's dotenv autoload silently filling in a
        # vendor key from the repo's real .env.
        os.environ[_DEEPSEEK] = "sk-leaked-from-dotenv-autoload"

    _harden_no_dotenv_against_litellm_autoload(_fake_litellm_import)
    assert _DEEPSEEK not in os.environ


def test_harden_no_dotenv_preserves_preexisting_vendor_key(monkeypatch) -> None:
    """A vendor var that was ALREADY present must never be deleted — only
    vars litellm's autoload newly introduced are stripped. (Real litellm/
    dotenv never overwrites an existing var — ``load_dotenv``'s default is
    ``override=False`` — so the fake trigger overwriting the value here is
    illustrative of the worst case, not a literal repro; the contract this
    pins is "never delete a var that was present beforehand.")"""
    from topofuzz.cli import _harden_no_dotenv_against_litellm_autoload

    monkeypatch.setenv(_DOTENV_DISABLE_ENV, "1")
    monkeypatch.setenv(_DEEPSEEK, "user-set-value")

    def _fake_litellm_import() -> None:
        os.environ[_DEEPSEEK] = "sk-leaked-from-dotenv-autoload"

    _harden_no_dotenv_against_litellm_autoload(_fake_litellm_import)
    assert _DEEPSEEK in os.environ


def test_harden_no_dotenv_is_noop_when_flag_unset(monkeypatch) -> None:
    """Do not change behavior when TOPOFUZZ_NO_DOTENV is unset — the
    (fake) litellm import is never even triggered."""
    from topofuzz.cli import _harden_no_dotenv_against_litellm_autoload

    monkeypatch.delenv(_DOTENV_DISABLE_ENV, raising=False)
    monkeypatch.delenv(_DEEPSEEK, raising=False)
    called = {"n": 0}

    def _fake_litellm_import() -> None:
        called["n"] += 1
        os.environ[_DEEPSEEK] = "sk-leaked-from-dotenv-autoload"

    _harden_no_dotenv_against_litellm_autoload(_fake_litellm_import)
    assert called["n"] == 0
    assert _DEEPSEEK not in os.environ


def test_harden_no_dotenv_default_trigger_is_real_litellm_import(
    monkeypatch,
) -> None:
    """No injected trigger → falls back to a real ``import litellm``.

    litellm is already an installed dependency in this environment; this
    only exercises the no-op branch (no vendor var was newly introduced
    by a real litellm import, because every real vendor var is already
    present as a placeholder per the test-suite ground rules), so no
    real key can leak."""
    from topofuzz.cli import _harden_no_dotenv_against_litellm_autoload

    monkeypatch.setenv(_DOTENV_DISABLE_ENV, "1")
    # Every vendor var is already set (placeholder) — assert none of them
    # get deleted (they were present beforehand, so they're never touched)
    # and the call doesn't raise even without an injected trigger.
    for name in (_GEMINI, _DEEPSEEK, _OPENAI, "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "placeholder")
    _harden_no_dotenv_against_litellm_autoload()
    for name in (_GEMINI, _DEEPSEEK, _OPENAI, "ANTHROPIC_API_KEY"):
        assert os.environ[name] == "placeholder"


def test_verify_subcommand_skips_dotenv(tmp_path: Path, monkeypatch) -> None:
    """``verify`` is the auditor-facing minimal-side-effect trust anchor:
    it must NOT read ``.env`` or mutate the environment (the loader runs
    only after the license-exempt early return)."""
    _enable_dotenv(monkeypatch, _GEMINI)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"{_GEMINI}=fake-key-for-test-12345\n", encoding="utf-8"
    )
    runner = _runner()
    try:
        # No run dir given → verify exits non-zero with a usage error,
        # which is fine: the assertion is purely that the loader never
        # fired for the exempt command.
        result = runner.invoke(app, ["verify", str(tmp_path / "missing")])
        assert _GEMINI not in os.environ
        assert ".env" not in (result.stderr or "")
    finally:
        os.environ.pop(_GEMINI, None)
