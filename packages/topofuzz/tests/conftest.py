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

import pytest


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
