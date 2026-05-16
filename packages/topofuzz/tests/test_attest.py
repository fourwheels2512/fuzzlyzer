# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 10a: topofuzz.attest glue tests.

The fuzzeval-side primitives (PAE byte-exactness, Ed25519 sign/verify)
are exercised in ``packages/fuzzeval/tests/test_attestation.py``. These
tests cover the topofuzz-side glue: payload assembly, ``emit_attestation``
file writes, ``load_and_verify_attestation`` checklist, and the
``compute_voo_for_attestation`` graceful-degradation paths.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import rfc8785
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fuzzeval.attestation import Ed25519Signer

from topofuzz.attest import (
    ATTESTATION_FILENAME,
    PAYLOAD_SCHEMA_VERSION,
    PAYLOAD_TYPE,
    AttestationDiscoveryError,
    build_attestation_payload,
    compute_headline,
    compute_voo_for_attestation,
    emit_attestation,
    emit_run_attestation,
    load_and_verify_attestation,
    sha256_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signer_pair() -> tuple[Ed25519Signer, bytes]:
    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    return signer, signer.public_key_pem


def _minimal_manifest(run_id: str = "run-test") -> dict:
    """A manifest dict shaped like ``orchestrator.write_manifest``'s output."""
    return {
        "run_id": run_id,
        "topology_canonical_hash": "abc123",
        "topology_name": "lending_4role",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": 2,
        "n_eval_rows": 3,
        "variant_canonical_hashes": ["h1", "h2"],
        "eval_ids": ["e1", "e2", "e3"],
        "max_cost_usd": None,
        "judge_model": "google/gemini-2.5-pro",
        "rubric_hash": "rh",
        "semaphore_size": 8,
        "cli_args": {},
        "kickoff_ts": 1700000000.0,
    }


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path: Path):
    """Streaming SHA-256 matches one-shot hashlib for small files."""
    import hashlib

    payload = b"hello world" * 1000
    f = tmp_path / "x.bin"
    f.write_bytes(payload)
    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_handles_empty_file(tmp_path: Path):
    f = tmp_path / "empty"
    f.write_bytes(b"")
    # sha256 of empty bytes — pinned value.
    assert sha256_file(f) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_file_streams_in_chunks(tmp_path: Path):
    """Force the chunked-read path with a small chunk_size."""
    import hashlib

    f = tmp_path / "big.bin"
    f.write_bytes(b"abc" * 10_000)
    expected = hashlib.sha256(b"abc" * 10_000).hexdigest()
    assert sha256_file(f, chunk_size=128) == expected


# ---------------------------------------------------------------------------
# compute_headline / build_attestation_payload
# ---------------------------------------------------------------------------


def test_compute_headline_default_ecr_kappa_null():
    """ECR and κ are explicit-null in v0.5 — see module docstring."""
    h = compute_headline(
        status="succeeded",
        n_successful_cells=10,
        n_total_cells=10,
        n_variants=2,
        n_evals=5,
        total_cost_usd=0.42,
        elapsed_s=3.14,
    )
    assert h["ecr"] is None
    assert h["kappa"] is None
    assert h["voo_per_eval"] is None
    assert h["status"] == "succeeded"
    assert h["n_successful_cells"] == 10
    assert h["total_cost_usd"] == pytest.approx(0.42)


def test_compute_headline_coerces_numeric_types():
    """ints/floats get explicitly coerced so a numpy.int64 leak from
    upstream callers doesn't break rfc8785 (which only accepts native
    Python numeric types)."""
    h = compute_headline(
        status="degraded",
        n_successful_cells=True,  # bool is a subclass of int
        n_total_cells=10,
        n_variants=2,
        n_evals=5,
        total_cost_usd=0,  # int → float
        elapsed_s=1,
    )
    assert isinstance(h["n_successful_cells"], int)
    assert h["n_successful_cells"] == 1
    assert isinstance(h["total_cost_usd"], float)
    assert isinstance(h["elapsed_s"], float)


def test_build_attestation_payload_shape():
    payload = build_attestation_payload(
        run_id="r1",
        manifest={"k": "v"},
        headline={"status": "succeeded"},
        run_sqlite_sha256="deadbeef",
        signed_at="2026-01-01T00:00:00.000000Z",
    )
    assert payload["schema_version"] == PAYLOAD_SCHEMA_VERSION
    assert payload["run_id"] == "r1"
    assert payload["manifest"] == {"k": "v"}
    assert payload["headline"] == {"status": "succeeded"}
    assert payload["run_sqlite_sha256"] == "deadbeef"
    assert payload["signed_at"] == "2026-01-01T00:00:00.000000Z"


def test_build_attestation_payload_default_signed_at_is_iso8601_z():
    payload = build_attestation_payload(
        run_id="r1",
        manifest={},
        headline={},
        run_sqlite_sha256="",
    )
    # Default signed_at: "YYYY-MM-DDTHH:MM:SS.ffffffZ"
    signed_at = payload["signed_at"]
    assert signed_at.endswith("Z")
    assert "T" in signed_at
    # Microsecond field is exactly 6 digits.
    decimal_part = signed_at.split(".")[1].rstrip("Z")
    assert len(decimal_part) == 6 and decimal_part.isdigit()


# ---------------------------------------------------------------------------
# emit_attestation + sign-then-verify roundtrip
# ---------------------------------------------------------------------------


def test_emit_attestation_writes_valid_envelope(tmp_path: Path):
    signer, pub_pem = _signer_pair()
    payload = build_attestation_payload(
        run_id="r1",
        manifest=_minimal_manifest(),
        headline=compute_headline(
            status="succeeded",
            n_successful_cells=5,
            n_total_cells=5,
            n_variants=1,
            n_evals=5,
            total_cost_usd=0.01,
            elapsed_s=1.0,
        ),
        run_sqlite_sha256="cafe",
        signed_at="2026-01-01T00:00:00.000000Z",
    )
    out = emit_attestation(payload, signer, tmp_path / ATTESTATION_FILENAME)
    assert out.exists()
    env = json.loads(out.read_bytes())
    assert env["payloadType"] == PAYLOAD_TYPE
    raw = base64.b64decode(env["payload"], validate=True)
    # JCS round-trip: re-canonicalize and bytes must match.
    assert rfc8785.dumps(payload) == raw


def test_emit_attestation_creates_parent_dir(tmp_path: Path):
    signer, _ = _signer_pair()
    payload = build_attestation_payload(
        run_id="r1",
        manifest={},
        headline={},
        run_sqlite_sha256="",
        signed_at="X",
    )
    deep = tmp_path / "nested" / "deep" / "dir" / "attestation.json"
    emit_attestation(payload, signer, deep)
    assert deep.exists()


def test_load_and_verify_attestation_passes_for_signed_run(tmp_path: Path):
    signer, pub_pem = _signer_pair()
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = _minimal_manifest()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    # Empty run.sqlite — we just want a file present.
    (run_dir / "run.sqlite").write_bytes(b"")
    payload = build_attestation_payload(
        run_id="run-1",
        manifest=manifest,
        headline=compute_headline(
            status="succeeded",
            n_successful_cells=6,
            n_total_cells=6,
            n_variants=2,
            n_evals=3,
            total_cost_usd=0.0,
            elapsed_s=0.0,
        ),
        run_sqlite_sha256=sha256_file(run_dir / "run.sqlite"),
        signed_at="2026-01-01T00:00:00.000000Z",
    )
    emit_attestation(payload, signer, run_dir / ATTESTATION_FILENAME)
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem,
    )
    assert outcome.ok, outcome.checks
    assert outcome.watermark == "[DEV-KEY]"
    # All checks present in expected order.
    names = [c[0] for c in outcome.checks]
    assert names == [
        "discover",
        "envelope_signature",
        "payload_type",
        "jcs_round_trip",
        "schema_version",
        "manifest_match",
        "run_sqlite_sha256_match",
    ]


def test_load_and_verify_attestation_detects_tampered_byte(tmp_path: Path):
    """Flip one byte of the on-disk attestation.json → FAIL."""
    signer, pub_pem = _signer_pair()
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    payload = build_attestation_payload(
        run_id="run-1",
        manifest=_minimal_manifest(),
        headline=compute_headline(
            status="succeeded",
            n_successful_cells=1,
            n_total_cells=1,
            n_variants=1,
            n_evals=1,
            total_cost_usd=0.0,
            elapsed_s=0.0,
        ),
        run_sqlite_sha256="",
        signed_at="X",
    )
    att_path = run_dir / ATTESTATION_FILENAME
    emit_attestation(payload, signer, att_path)
    raw = bytearray(att_path.read_bytes())
    # Flip a byte well inside the payload base64 (which is the bulkiest
    # field and most likely to alter the signed bytes).
    flip_at = raw.index(b'"payload":"') + len(b'"payload":"') + 4
    raw[flip_at] = (raw[flip_at] ^ 0x01)
    att_path.write_bytes(bytes(raw))
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem,
    )
    assert not outcome.ok
    # The failing check is one of envelope_signature / envelope_decode
    # (depending on whether the flip lands in valid base64 or not).
    failed_names = [c[0] for c in outcome.checks if not c[1]]
    assert failed_names, outcome.checks


def test_load_and_verify_attestation_detects_manifest_drift(tmp_path: Path):
    """If on-disk manifest.json no longer matches the signed manifest,
    the verify pipeline catches it at manifest_match (post-signature)."""
    signer, pub_pem = _signer_pair()
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = _minimal_manifest()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    payload = build_attestation_payload(
        run_id="run-1",
        manifest=manifest,
        headline={},
        run_sqlite_sha256="",
        signed_at="X",
    )
    emit_attestation(payload, signer, run_dir / ATTESTATION_FILENAME)
    # Now tamper with the on-disk manifest (post-signing).
    tampered = dict(manifest)
    tampered["topology_name"] = "DIFFERENT"
    (run_dir / "manifest.json").write_text(
        json.dumps(tampered), encoding="utf-8",
    )
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem,
    )
    assert not outcome.ok
    failed = [c for c in outcome.checks if not c[1]]
    assert failed and failed[0][0] == "manifest_match"


def test_load_and_verify_attestation_detects_sqlite_drift(tmp_path: Path):
    signer, pub_pem = _signer_pair()
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = _minimal_manifest()
    # Step 10a B-3 fix: manifest.json is REQUIRED when verifying against
    # a run_dir, so write it here to exercise the sqlite-drift path
    # (otherwise we'd fail at manifest_match before getting to sqlite).
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    sqlite_path = run_dir / "run.sqlite"
    sqlite_path.write_bytes(b"original")
    signed_sha = sha256_file(sqlite_path)
    payload = build_attestation_payload(
        run_id="run-1",
        manifest=manifest,
        headline={},
        run_sqlite_sha256=signed_sha,
        signed_at="X",
    )
    emit_attestation(payload, signer, run_dir / ATTESTATION_FILENAME)
    # Tamper with run.sqlite post-signing.
    sqlite_path.write_bytes(b"changed")
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem,
    )
    assert not outcome.ok
    failed = [c for c in outcome.checks if not c[1]]
    assert failed and failed[0][0] == "run_sqlite_sha256_match"


def test_load_and_verify_attestation_replay_into_empty_run_dir_fails(
    tmp_path: Path,
):
    """Step 10a B-3: dropping a signed attestation.json into an EMPTY
    run dir (no manifest, no sqlite) must FAIL — defends against the
    orphan-attestation replay vector where an attacker exfiltrates one
    valid attestation and replays it in a different run's directory."""
    signer, pub_pem = _signer_pair()
    # Sign in run_dir_A.
    run_dir_a = tmp_path / "run-A"
    run_dir_a.mkdir()
    manifest_a = _minimal_manifest("run-A")
    (run_dir_a / "manifest.json").write_text(
        json.dumps(manifest_a), encoding="utf-8",
    )
    payload = build_attestation_payload(
        run_id="run-A",
        manifest=manifest_a,
        headline={},
        run_sqlite_sha256="",
        signed_at="X",
    )
    emit_attestation(payload, signer, run_dir_a / ATTESTATION_FILENAME)

    # Replay: copy ONLY attestation.json into a foreign empty run_dir.
    run_dir_b = tmp_path / "run-B"
    run_dir_b.mkdir()
    (run_dir_b / ATTESTATION_FILENAME).write_bytes(
        (run_dir_a / ATTESTATION_FILENAME).read_bytes()
    )
    outcome = load_and_verify_attestation(
        run_dir=run_dir_b, bundle=None, public_key_pem=pub_pem,
    )
    assert not outcome.ok
    failed = [c for c in outcome.checks if not c[1]]
    assert failed and failed[0][0] == "manifest_match"


def test_load_and_verify_attestation_detects_non_jcs_payload(tmp_path: Path):
    """Step 10a RT-audit M-1: ensure the jcs_round_trip check catches a
    signer that emitted non-JCS-canonical payload bytes. Without this
    test, a refactor that drops the JCS check entirely would pass."""
    from fuzzeval.attestation import dsse_sign

    signer, pub_pem = _signer_pair()
    bundle_path = tmp_path / "att.json"
    # Build a real payload, then sign the NON-JCS bytes (note the extra
    # space after ':' — rfc8785 strips that, raw json.dumps preserves
    # the default separator which is ", " (with whitespace, also non-JCS).
    payload = build_attestation_payload(
        run_id="run-1",
        manifest=_minimal_manifest(),
        headline={},
        run_sqlite_sha256="",
        signed_at="X",
    )
    non_jcs_bytes = json.dumps(payload).encode("utf-8")  # default separators
    envelope = dsse_sign(non_jcs_bytes, "application/vnd.topofuzz.run+json;v=0.5", signer)
    bundle_path.write_bytes(envelope)
    outcome = load_and_verify_attestation(
        run_dir=None, bundle=bundle_path, public_key_pem=pub_pem,
    )
    assert not outcome.ok
    failed = [c for c in outcome.checks if not c[1]]
    assert failed and failed[0][0] == "jcs_round_trip"


def test_load_and_verify_attestation_missing_bundle_fails_discover(tmp_path: Path):
    signer, pub_pem = _signer_pair()
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem,
    )
    assert not outcome.ok
    assert outcome.checks[0][0] == "discover"
    assert not outcome.checks[0][1]


def test_load_and_verify_attestation_bundle_override(tmp_path: Path):
    """``--bundle`` reads a standalone file; run_dir-based file checks
    are skipped when run_dir is None."""
    signer, pub_pem = _signer_pair()
    bundle_path = tmp_path / "att.json"
    payload = build_attestation_payload(
        run_id="run-1",
        manifest=_minimal_manifest(),
        headline={},
        run_sqlite_sha256="",
        signed_at="X",
    )
    emit_attestation(payload, signer, bundle_path)
    outcome = load_and_verify_attestation(
        run_dir=None, bundle=bundle_path, public_key_pem=pub_pem,
    )
    assert outcome.ok, outcome.checks
    # No manifest_match / run_sqlite_sha256_match checks when run_dir=None.
    names = [c[0] for c in outcome.checks]
    assert "manifest_match" not in names
    assert "run_sqlite_sha256_match" not in names


def test_load_and_verify_attestation_bundle_nonexistent_path(tmp_path: Path):
    signer, pub_pem = _signer_pair()
    outcome = load_and_verify_attestation(
        run_dir=None,
        bundle=tmp_path / "nope.json",
        public_key_pem=pub_pem,
    )
    assert not outcome.ok
    assert outcome.checks[0][0] == "discover"


def test_load_and_verify_attestation_no_run_dir_no_bundle_fails(tmp_path: Path):
    signer, pub_pem = _signer_pair()
    outcome = load_and_verify_attestation(
        run_dir=None, bundle=None, public_key_pem=pub_pem,
    )
    assert not outcome.ok
    assert outcome.checks[0][0] == "discover"


def test_load_and_verify_attestation_rejects_wrong_payload_type(tmp_path: Path):
    """If the envelope's payloadType doesn't match expected, fail."""
    signer, pub_pem = _signer_pair()
    bundle_path = tmp_path / "att.json"
    payload = build_attestation_payload(
        run_id="run-1",
        manifest={},
        headline={},
        run_sqlite_sha256="",
        signed_at="X",
    )
    emit_attestation(payload, signer, bundle_path)
    outcome = load_and_verify_attestation(
        run_dir=None,
        bundle=bundle_path,
        public_key_pem=pub_pem,
        payload_type_expected="application/something-else+json",
    )
    assert not outcome.ok
    failed = [c for c in outcome.checks if not c[1]]
    assert failed and failed[0][0] == "payload_type"


# ---------------------------------------------------------------------------
# compute_voo_for_attestation graceful degradation
# ---------------------------------------------------------------------------


def test_compute_voo_for_attestation_returns_none_for_missing_sqlite(tmp_path: Path):
    """No run.sqlite → None, not an exception."""
    out = compute_voo_for_attestation(tmp_path, run_id="nonexistent")
    assert out is None


def test_compute_voo_for_attestation_returns_none_for_empty_db(tmp_path: Path):
    """Empty / schemaless run.sqlite → None, not an exception."""
    (tmp_path / "run.sqlite").write_bytes(b"")
    out = compute_voo_for_attestation(tmp_path, run_id="anything")
    assert out is None


# ---------------------------------------------------------------------------
# emit_run_attestation end-to-end (with a real RunWriter-shaped DB)
# ---------------------------------------------------------------------------


def test_emit_run_attestation_no_sqlite(tmp_path: Path):
    """Happy-path emit when run.sqlite doesn't exist (degenerate / abort
    scenarios). voo is None, sqlite_sha256 is empty string, but the
    envelope verifies cleanly."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = _minimal_manifest()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    signer, pub_pem = _signer_pair()
    out_path = emit_run_attestation(
        run_dir=run_dir,
        run_id="r1",
        manifest=manifest,
        status="failed",
        successful_cells=0,
        total_cells=2,
        n_variants=1,
        n_evals=2,
        total_cost_usd=0.0,
        elapsed_s=0.5,
        signer=signer,
    )
    assert out_path == run_dir / ATTESTATION_FILENAME
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem,
    )
    assert outcome.ok, outcome.checks
    headline = outcome.payload["headline"]
    assert headline["status"] == "failed"
    assert headline["voo_per_eval"] is None
    assert outcome.payload["run_sqlite_sha256"] == ""


def test_emit_run_attestation_with_sqlite_digest(tmp_path: Path):
    """When run.sqlite exists (even if empty / unreadable as a DB), the
    emit step hashes it and the verify step's run_sqlite_sha256_match
    check passes."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = _minimal_manifest()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    # An empty-but-existing run.sqlite — sha256_file hashes whatever's
    # there. compute_voo gracefully returns None on a non-schema DB.
    (run_dir / "run.sqlite").write_bytes(b"")
    signer, pub_pem = _signer_pair()
    emit_run_attestation(
        run_dir=run_dir,
        run_id="r1",
        manifest=manifest,
        status="succeeded",
        successful_cells=0,
        total_cells=0,
        n_variants=0,
        n_evals=0,
        total_cost_usd=0.0,
        elapsed_s=0.0,
        signer=signer,
    )
    outcome = load_and_verify_attestation(
        run_dir=run_dir, bundle=None, public_key_pem=pub_pem,
    )
    assert outcome.ok, outcome.checks
    names = [c[0] for c in outcome.checks]
    assert "run_sqlite_sha256_match" in names
