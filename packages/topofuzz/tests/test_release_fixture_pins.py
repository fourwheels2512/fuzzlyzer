"""Release-fixture pin tests.

Pins the cryptographic identity of the committed v0.5 dev pubkey at
``keys/dev.pub.pem`` so a stray ``_regen.py`` run cannot silently
rotate the published key between v0.5.0rc1 and v0.5.0 GA.

The "rc1 → GA byte-identical" promise in ``SECURITY.md`` is otherwise
just prose; this test makes it a CI gate (audit RT-M-5 fix).

Updating the pin
----------------

The pin **should not be bumped** during the v0.5.0rc1 → v0.5.0 GA
promotion window. If you intentionally rotate the dev key in a later
release (v0.5.1+, or any forced rotation), update
``_EXPECTED_DEV_PUBKEY_SHA256`` to the new digest in the same commit
that re-runs ``_regen.py`` and re-blesses the signed fixture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# Repo root: this file lives at packages/topofuzz/tests/, walk three up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PUBKEY_PATH = _REPO_ROOT / "keys" / "dev.pub.pem"

# Pinned sha256 of the v0.5.0rc1 committed dev pubkey. Holds across the
# rc1 → 0.5.0 GA promotion. Any subsequent intentional rotation MUST
# bump this digest in the same commit that re-blesses the signed
# fixture under packages/topofuzz/tests/fixtures/signed/.
_EXPECTED_DEV_PUBKEY_SHA256 = (
    "0391d65e1059ba1e4a9161ae89605cf498399025b4a1d82b6b88742fadd411d9"
)


@pytest.mark.skipif(
    not _PUBKEY_PATH.is_file(),
    reason=(
        "keys/dev.pub.pem missing — running outside a full source checkout. "
        "This test is only meaningful when the committed fixture is present."
    ),
)
def test_committed_dev_pubkey_byte_identical_across_releases() -> None:
    """The v0.5 published dev pubkey is pinned by sha256.

    If this fails, somebody re-ran
    ``packages/topofuzz/tests/fixtures/signed/_regen.py`` which mints
    a fresh Ed25519 keypair every invocation. The published pubkey
    at the ``v0.5.0`` git tag would then diverge from the v0.5.0rc1
    pubkey, breaking every external verifier that pinned to rc1's
    canonical URL.

    Fix: either revert the regen, or — if the rotation is
    intentional — bump ``_EXPECTED_DEV_PUBKEY_SHA256`` in this test
    and call the rotation out in CHANGELOG.md + SECURITY.md.
    """
    content = _PUBKEY_PATH.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    assert actual == _EXPECTED_DEV_PUBKEY_SHA256, (
        f"keys/dev.pub.pem sha256 changed: expected "
        f"{_EXPECTED_DEV_PUBKEY_SHA256}, got {actual}. The rc1 → GA "
        f"byte-identical promise in SECURITY.md has been violated. "
        f"See this test's docstring for the rotation procedure."
    )
