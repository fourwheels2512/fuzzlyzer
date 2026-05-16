# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Regenerate the v0.5 signed fixture.

This is NOT a packaged tool — tools/regen_fixture.py was deferred to
v0.6 per the step-10b decision (the v1 payload schema is locked, so
the fixture only needs regeneration when the schema bumps). It lives
inside the fixture dir so a developer who wants to refresh the bundle
runs:

    $ uv run python packages/topofuzz/tests/fixtures/signed/_regen.py

The script writes:

* ``keys/dev.pub.pem`` at repo root (committed; the published pubkey
  per BLUEPRINT_v5plus.md §17a literal path).
* ``packages/topofuzz/tests/fixtures/signed/attestation.json``
  (the DSSE envelope, committed).
* ``packages/topofuzz/tests/fixtures/signed/manifest.json``
  (the on-disk manifest the envelope's ``manifest`` field mirrors;
  required for the in-place ``topofuzz verify`` path).
* ``packages/topofuzz/tests/fixtures/signed/run.sqlite``
  (3-byte placeholder; the envelope's ``run_sqlite_sha256`` is the
  hash of these bytes).
* ``packages/topofuzz/tests/fixtures/signed/dev.priv.pem``
  (private key, gitignored; needed only when regenerating the fixture).

The private key is intentionally written next to the fixture and
gitignored — committing it would let anyone forge envelopes claiming
to be the dev key. Regenerating produces a fresh keypair; the
committed ``keys/dev.pub.pem`` is the canonical published pubkey for
the v0.5 release tag.

Reproducibility model:

* Re-running this script generates a **fresh** Ed25519 keypair every
  time (line ``Ed25519PrivateKey.generate()`` below). Every regen
  therefore produces a different ``keys/dev.pub.pem`` AND a different
  ``attestation.json`` — they are mutually-dependent outputs and MUST
  be committed together.
* ``signed_at`` is pinned to a frozen UTC string so that *given the
  same priv key*, the envelope bytes are byte-stable across runs
  (Ed25519 is deterministic per RFC 8032 §5.1.6 — the signature is a
  pure function of (key, message), not a randomized scheme). This
  property matters only if you save off the priv key and want to
  re-derive the fixture; it does NOT mean re-running ``_regen.py`` is
  idempotent.

Atomic-write semantics (RT pass-1 M-2):

The four committed outputs (``run.sqlite``, ``manifest.json``,
``attestation.json``, ``keys/dev.pub.pem``) are each written via
tmp-file + ``os.replace``. The replace order is **pubkey last**
— matching the RT pass-1 spec "write ``attestation.json`` BEFORE
``keys/dev.pub.pem``". No single ordering eliminates the half-run
desync window (both pubkey-last AND envelope-last leave a crash
gap where the committed pair is mismatched and ``topofuzz verify``
fails), so the choice is by convention: ``keys/dev.pub.pem`` is the
externally-published artifact (per BLUEPRINT_v5plus.md §17a), so
publishing it last means the released bundle is never advertised
ahead of the matching envelope it certifies.

Crash recovery: if the process dies after any partial state, a
re-run of ``_regen.py`` overwrites every output deterministically.
The failure mode is loud — the in-place ``topofuzz verify`` reports
which check failed, not silent corruption — so a half-finished
regen is caught before commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fuzzeval.attestation import Ed25519Signer, dsse_sign
from topofuzz.attest import (
    PAYLOAD_SCHEMA_VERSION,
    PAYLOAD_TYPE,
    build_attestation_payload,
    compute_headline,
)


FIXTURE_DIR = Path(__file__).parent
REPO_ROOT = FIXTURE_DIR.parents[4]
KEYS_DIR = REPO_ROOT / "keys"


def _atomic_write_bytes(path: Path, data: bytes, *, posix_mode: int | None = None) -> None:
    """Write ``data`` to ``path`` via a sibling ``.tmp`` + ``os.replace``.

    ``os.replace`` is atomic on POSIX and on NTFS — either the new bytes
    are visible at ``path`` or the old contents remain. A crash mid-write
    leaves only the orphan ``.tmp`` (which the next regen overwrites).

    Windows caveat: ``os.replace`` raises ``PermissionError`` (WinError 5)
    if the destination file is held open by another process (e.g. an
    editor, an AV scanner, or a stale Python tail in another terminal).
    The dev script surfaces that as a stack trace; the operator closes
    the offending handle and re-runs.

    ``posix_mode``: when set, the ``.tmp`` is chmod-ed BEFORE rename so
    the final file never exists with a wider permission bit. No-op on
    Windows (where NTFS ACLs are inherited from the parent directory).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    if posix_mode is not None and os.name == "posix":
        os.chmod(tmp, posix_mode)
    os.replace(tmp, path)


def main() -> int:
    KEYS_DIR.mkdir(parents=True, exist_ok=True)

    # Tiny placeholder run.sqlite payload — three bytes so the SHA-256
    # digest is non-empty and the envelope's ``run_sqlite_sha256`` binds
    # the bundle to a concrete file. Tampering with these three bytes
    # breaks the in-place ``topofuzz verify`` path.
    sqlite_bytes = b"v05"
    sqlite_sha = hashlib.sha256(sqlite_bytes).hexdigest()

    manifest = {
        "run_id": "fixture-v0.5",
        "topology_canonical_hash": "fixture-topo-hash",
        "topology_name": "lending_4role",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": 1,
        "n_eval_rows": 1,
        "variant_canonical_hashes": ["fixture-variant-hash"],
        "eval_ids": ["fixture-eval-0"],
        "max_cost_usd": None,
        "judge_model": "google/gemini-2.5-pro",
        "rubric_hash": "fixture-rubric-hash",
        "semaphore_size": 8,
        "cli_args": {},
        "kickoff_ts": 1700000000.0,
    }
    # On-disk manifest.json format intentionally diverges from
    # ``orchestrator.write_manifest`` (which uses ``ensure_ascii=False`` and
    # no ``sort_keys``). The reason: this fixture is read by humans during
    # audits, so sorted-keys + ASCII makes the file diff-friendly. The
    # envelope's ``manifest_match`` check is JCS-based, so on-disk JSON
    # formatting choices do not affect ``topofuzz verify``.
    manifest_bytes = json.dumps(
        manifest, indent=2, sort_keys=True
    ).encode("utf-8")

    headline = compute_headline(
        status="succeeded",
        n_successful_cells=1,
        n_total_cells=1,
        n_variants=1,
        n_evals=1,
        total_cost_usd=0.0,
        elapsed_s=0.42,
        voo_per_eval=[{"eval_id": "fixture-eval-0", "point": 0.0}],
    )
    payload = build_attestation_payload(
        run_id="fixture-v0.5",
        manifest=manifest,
        headline=headline,
        run_sqlite_sha256=sqlite_sha,
        signed_at="2026-05-13T00:00:00.000000Z",
    )

    # Fresh Ed25519 keypair. Ed25519 signatures are deterministic per
    # RFC 8032 §5.1.6 (no randomization), so given the same priv key
    # the envelope bytes are byte-stable across runs. Re-running this
    # script generates a NEW keypair though, so every regen produces a
    # new pubkey + envelope pair that MUST be committed together.
    priv = Ed25519PrivateKey.generate()
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Sign the envelope in-memory; we control the on-disk write below
    # so the four committed outputs can be flushed in a deliberate order.
    signer = Ed25519Signer(priv)
    canonical = rfc8785.dumps(payload)
    envelope_bytes = dsse_sign(canonical, PAYLOAD_TYPE, signer)

    # Write order: envelope BEFORE pubkey (per RT pass-1 M-2 "write
    # attestation.json BEFORE keys/dev.pub.pem"). The published pubkey
    # is the last artifact to land so the externally-advertised file
    # never points ahead of the matching envelope it certifies. The
    # crash gap (envelope replaced, pubkey still old) is loud — verify
    # fails on signature mismatch, not silent corruption. The priv key
    # gets posix_mode=0o600 so the on-disk .tmp briefly between write
    # and rename is never world-readable; gitignored regardless.
    _atomic_write_bytes(FIXTURE_DIR / "run.sqlite", sqlite_bytes)
    _atomic_write_bytes(FIXTURE_DIR / "manifest.json", manifest_bytes)
    _atomic_write_bytes(
        FIXTURE_DIR / "dev.priv.pem", priv_pem, posix_mode=0o600
    )  # gitignored
    _atomic_write_bytes(FIXTURE_DIR / "attestation.json", envelope_bytes)
    _atomic_write_bytes(KEYS_DIR / "dev.pub.pem", pub_pem)  # last

    print(f"schema_version = {PAYLOAD_SCHEMA_VERSION}")
    print(f"signed_at      = {payload['signed_at']}")
    print(f"sqlite_sha256  = {sqlite_sha}")
    print(f"wrote sqlite   = {FIXTURE_DIR / 'run.sqlite'}")
    print(f"wrote manifest = {FIXTURE_DIR / 'manifest.json'}")
    print(f"wrote priv pem = {FIXTURE_DIR / 'dev.priv.pem'} (gitignored)")
    print(f"wrote envelope = {FIXTURE_DIR / 'attestation.json'}")
    print(f"wrote pubkey   = {KEYS_DIR / 'dev.pub.pem'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
