# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Topofuzz test fixtures.

Pass-2 MINOR-2 fix (step 10a): hermetic test isolation for the
attestation path. Without this fixture, any developer who runs
``topofuzz keygen`` on their machine sees every ``execute_run`` test
silently sign attestations with their real ``~/.topofuzz/dev.pem``
into the test's ``tmp_path``. That's not a key-leak (the key isn't
exfiltrated) but it breaks test hermeticity and could mask bugs in
``_maybe_emit_attestation``.

The autouse fixture forces ``default_dev_priv_path`` to a tmp path
that doesn't exist by default. Tests that DO want to exercise the
sign-path opt back in by overriding the monkeypatch (the
``test_orchestrator_attestation.py`` cases already do this
explicitly via ``monkeypatch.setattr``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from topofuzz import __version__


@pytest.fixture(autouse=True)
def _isolate_dev_key(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force ``fuzzeval.keys.default_dev_priv_path`` to a non-existent
    tmp path so any orchestrator code that reads it sees "no key" and
    skips attestation emit. Tests that EXPECT signing override this
    in-test with their own ``monkeypatch.setattr`` (which wins on the
    LIFO undo stack)."""
    fake_home = tmp_path_factory.mktemp("hermetic-home")
    monkeypatch.setattr(
        "fuzzeval.keys.default_dev_priv_path",
        lambda: fake_home / ".topofuzz" / "dev.pem",
    )


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI invocations hermetic against the developer's ``.env``.

    ``topofuzz.cli._root`` auto-loads ``$CWD/.env`` into ``os.environ``
    (ops fix, 2026-06-11), and the test suite typically runs with cwd at
    the repo root — which carries a real developer ``.env`` with live
    vendor keys. Without this guard, every CliRunner test would silently
    import those keys mid-suite, making key-presence assertions
    machine-dependent and leaking real keys into test state. The dotenv
    loader's own tests (``test_cli_dotenv.py``) delete this variable
    explicitly and chdir to a tmp cwd to exercise the load path.
    """
    monkeypatch.setenv("TOPOFUZZ_NO_DOTENV", "1")


@pytest.fixture(autouse=True)
def _preaccept_license(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Pre-accept the BSL license in a per-test fake home directory.

    The CLI's first-run click-through (``topofuzz.license_check``)
    refuses non-interactive invocations without ``--accept-license`` —
    and ``CliRunner`` is always non-interactive. Without this fixture,
    every existing CLI test would hit the gate and exit 2 with
    "license acceptance required."

    The fixture (a) redirects ``Path.home()`` to a per-test tmp dir so
    no real user state is mutated, and (b) writes a valid acceptance
    record there. ``test_license_check.py`` overrides ``Path.home`` with
    its own per-test ``fake_home`` fixture (monkeypatch undo is LIFO),
    so it controls whether the acceptance file is present.
    """
    fake_home = tmp_path_factory.mktemp("license-home")
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    accept_dir = fake_home / ".topofuzz"
    accept_dir.mkdir(parents=True, exist_ok=True)
    (accept_dir / "license-accepted").write_text(
        json.dumps(
            {
                "version": __version__,
                "accepted_at": "2026-05-16T00:00:00Z",
                "license": "BUSL-1.1",
            }
        ),
        encoding="utf-8",
    )
    return fake_home
