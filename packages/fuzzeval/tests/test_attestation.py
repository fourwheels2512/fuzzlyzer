# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""DSSE v1.0 PAE byte-exact + envelope round-trip tests.

The hard byte-exact gate against ``secure-systems-lab/dsse`` v1.0.0
reference vectors lands in step 12a (per BLUEPRINT_v5plus.md §14
line 398). These tests pin the in-tree PAE construction so that future
refactors can't drift from the spec, and exercise the envelope shape
+ tamper-detection that step 10a's DoD calls out.
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fuzzeval.attestation import (
    DSSE_PAE_PREFIX,
    DSSEDecodeError,
    DSSESignatureError,
    DSSEVerifyError,
    Ed25519Signer,
    dsse_sign,
    dsse_verify,
    pae,
)

# ---------------------------------------------------------------------------
# PAE
# ---------------------------------------------------------------------------


def test_pae_prefix_is_literal_dssev1_space():
    """The literal prefix is the leading 7 bytes ``b"DSSEv1 "``."""
    assert DSSE_PAE_PREFIX == b"DSSEv1 "
    out = pae("application/test+json", b"hi")
    assert out.startswith(DSSE_PAE_PREFIX)


def test_pae_byte_layout_empty_payload():
    """PAE of an empty payload still emits ``len=0`` and the trailing SP."""
    out = pae("t", b"")
    # b"DSSEv1 " + b"1" + b" " + b"t" + b" " + b"0" + b" " + b""
    assert out == b"DSSEv1 1 t 0 "


def test_pae_byte_layout_ascii_payload():
    """Plain-ASCII payload: lengths reflect byte counts."""
    out = pae("application/vnd.topofuzz.run+json;v=0.5", b"hello")
    expected_type = b"application/vnd.topofuzz.run+json;v=0.5"
    expected = (
        b"DSSEv1 "
        + str(len(expected_type)).encode("ascii")
        + b" "
        + expected_type
        + b" "
        + b"5"
        + b" "
        + b"hello"
    )
    assert out == expected


def test_pae_byte_layout_utf8_payload():
    """Multi-byte UTF-8 in the payload: length is byte count, NOT codepoint count."""
    # "héllo" — 'é' is 2 bytes in UTF-8 — total 6 bytes.
    payload = "héllo".encode("utf-8")
    assert len(payload) == 6
    out = pae("t", payload)
    assert out == b"DSSEv1 1 t 6 " + payload


def test_pae_byte_layout_utf8_payload_type():
    """Non-ASCII payload_type: length is UTF-8 byte count."""
    # Use a contrived type with a non-ASCII char to exercise the encoder.
    ptype = "ä+json"
    ptype_bytes = ptype.encode("utf-8")
    assert len(ptype_bytes) == 7  # ä = 2 bytes
    out = pae(ptype, b"x")
    assert out == (
        b"DSSEv1 "
        + str(len(ptype_bytes)).encode("ascii")
        + b" "
        + ptype_bytes
        + b" 1 x"
    )


def test_pae_rejects_str_payload():
    """str payload would silently UTF-8 encode and break byte-exactness."""
    with pytest.raises(TypeError, match="payload must be bytes"):
        pae("t", "hello")  # type: ignore[arg-type]


def test_pae_rejects_bytes_payload_type():
    """payload_type is str by spec; bytes input is a programmer error."""
    with pytest.raises(TypeError, match="payload_type must be str"):
        pae(b"t", b"hello")  # type: ignore[arg-type]


def test_pae_accepts_bytearray_payload():
    """bytearray is bytes-like; mutability isn't a problem (we copy)."""
    ba = bytearray(b"data")
    out = pae("t", ba)
    assert out == b"DSSEv1 1 t 4 data"


def test_pae_length_field_is_byte_count_not_char_count():
    """Regression pin: length is len(bytes), not len(decoded-str)."""
    # 1 codepoint, 3 bytes (CJK char "漢" is U+6F22 → 0xE6 0xBC 0xA2).
    payload = "漢".encode("utf-8")
    assert len(payload) == 3
    out = pae("t", payload)
    # PAE bytes:  b"DSSEv1 1 t 3 <3-byte payload>"
    # split(b" "): [b"DSSEv1", b"1", b"t", b"3", <3-byte>] — index 3 holds
    # the payload-length field. (The 5-byte CJK case below would shift
    # this index if more SP-delimited fields were added; pin to 3 here.)
    parts = out.split(b" ")
    assert parts[3] == b"3"
    # And the payload trailer is exactly the byte triple.
    assert parts[4] == payload


# ---------------------------------------------------------------------------
# Envelope shape + sign/verify roundtrip
# ---------------------------------------------------------------------------


def _make_signer() -> Ed25519Signer:
    """Deterministic signer for tests. cryptography.generate() pulls from
    os.urandom — fine for tests since we don't rely on the key bits."""
    return Ed25519Signer(Ed25519PrivateKey.generate())


def test_dsse_sign_produces_dsse_v1_envelope_shape():
    signer = _make_signer()
    envelope = dsse_sign(b"hello", "application/test+json", signer)
    env = json.loads(envelope)
    assert set(env.keys()) == {"payload", "payloadType", "signatures"}
    assert env["payloadType"] == "application/test+json"
    # payload field is base64 of the raw payload bytes.
    assert base64.b64decode(env["payload"], validate=True) == b"hello"
    # signatures is a non-empty list of {keyid, sig}.
    assert isinstance(env["signatures"], list) and len(env["signatures"]) == 1
    sig_entry = env["signatures"][0]
    assert set(sig_entry.keys()) == {"keyid", "sig"}
    assert sig_entry["keyid"] == "topofuzz-dev-v0.5"
    # sig is base64 of 64-byte Ed25519 signature.
    assert len(base64.b64decode(sig_entry["sig"], validate=True)) == 64


def test_dsse_sign_envelope_is_compact_keysorted_json():
    """Envelope bytes are deterministic given payload + signer state.

    sign() with the same signer produces a different signature each
    call (Ed25519 is deterministic per RFC 8032, but our key is fresh
    per signer), so we check the json shape, not byte equality, here.
    """
    signer = _make_signer()
    envelope = dsse_sign(b"hello", "t", signer)
    # No leading/trailing whitespace, no indented pretty-printing.
    assert b"\n" not in envelope and b"  " not in envelope
    # Keys are sorted: payload < payloadType < signatures.
    parsed_pos_payload = envelope.index(b'"payload"')
    parsed_pos_type = envelope.index(b'"payloadType"')
    parsed_pos_sigs = envelope.index(b'"signatures"')
    assert parsed_pos_payload < parsed_pos_type < parsed_pos_sigs


def test_dsse_sign_then_verify_roundtrip():
    signer = _make_signer()
    envelope = dsse_sign(b'{"a":1}', "application/test+json", signer)
    verified = dsse_verify(envelope, signer.public_key_pem)
    assert verified.payload == {"a": 1}
    assert verified.payload_type == "application/test+json"
    # Step 10a B-1 fix: VerifiedPayload exposes the AUTHENTICATED bytes
    # so callers don't have to re-parse the envelope to do JCS-roundtrip
    # checks. The bytes equal the raw JSON that was PAE'd.
    assert verified.payload_bytes == b'{"a":1}'
    assert verified.signature_index == 0


def test_dsse_verify_detects_tampered_payload():
    """Flipping one byte of the payload base64 should fail signature verify."""
    signer = _make_signer()
    envelope = dsse_sign(b'{"a":1}', "application/test+json", signer)
    env = json.loads(envelope)
    # Tamper: alter the payload field. base64 of {"a":1} -> base64 of {"a":2}.
    env["payload"] = base64.b64encode(b'{"a":2}').decode("ascii")
    tampered = json.dumps(env).encode("utf-8")
    with pytest.raises(DSSESignatureError):
        dsse_verify(tampered, signer.public_key_pem)


def test_dsse_verify_detects_tampered_signature():
    """Flipping one signature byte should fail verify."""
    signer = _make_signer()
    envelope = dsse_sign(b'{"a":1}', "t", signer)
    env = json.loads(envelope)
    sig_bytes = bytearray(base64.b64decode(env["signatures"][0]["sig"]))
    sig_bytes[0] ^= 0x01
    env["signatures"][0]["sig"] = base64.b64encode(bytes(sig_bytes)).decode("ascii")
    tampered = json.dumps(env).encode("utf-8")
    with pytest.raises(DSSESignatureError):
        dsse_verify(tampered, signer.public_key_pem)


def test_dsse_verify_detects_tampered_payload_type():
    """Changing payloadType in the envelope changes the PAE preimage,
    so the signature no longer matches."""
    signer = _make_signer()
    envelope = dsse_sign(b'{"a":1}', "t1", signer)
    env = json.loads(envelope)
    env["payloadType"] = "t2"
    tampered = json.dumps(env).encode("utf-8")
    with pytest.raises(DSSESignatureError):
        dsse_verify(tampered, signer.public_key_pem)


def test_dsse_verify_rejects_wrong_public_key():
    signer = _make_signer()
    envelope = dsse_sign(b'{"a":1}', "t", signer)
    other = _make_signer()
    with pytest.raises(DSSESignatureError):
        dsse_verify(envelope, other.public_key_pem)


def test_dsse_verify_rejects_malformed_envelope_json():
    signer = _make_signer()
    with pytest.raises(DSSEDecodeError, match="not valid JSON"):
        dsse_verify(b"not json at all", signer.public_key_pem)


def test_dsse_verify_rejects_envelope_missing_fields():
    signer = _make_signer()
    bad = json.dumps({"payload": "aGk=", "payloadType": "t"}).encode("utf-8")
    with pytest.raises(DSSEDecodeError, match="missing required field 'signatures'"):
        dsse_verify(bad, signer.public_key_pem)


def test_dsse_verify_rejects_non_object_payload():
    """The contract is that the inner payload is always a JSON object.
    A bare array or string verifies cryptographically but our wrapper
    refuses because callers rely on the dict return type."""
    signer = _make_signer()
    envelope = dsse_sign(b"[1,2,3]", "t", signer)
    with pytest.raises(DSSEDecodeError, match="payload must be a JSON object"):
        dsse_verify(envelope, signer.public_key_pem)


def test_dsse_verify_rejects_non_pem_public_key():
    signer = _make_signer()
    envelope = dsse_sign(b"{}", "t", signer)
    with pytest.raises(DSSEDecodeError, match="public key PEM is malformed"):
        dsse_verify(envelope, b"not a PEM")


def test_dsse_verify_rejects_non_ed25519_public_key():
    """RSA public keys are valid PEMs but not Ed25519 — refuse cleanly."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_pub_pem = rsa_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = _make_signer()
    envelope = dsse_sign(b"{}", "t", signer)
    with pytest.raises(DSSEDecodeError, match="expected Ed25519"):
        dsse_verify(envelope, rsa_pub_pem)


def test_dsse_verify_error_hierarchy_allows_single_catch():
    """Both decode and signature errors subclass DSSEVerifyError so a
    CLI catch-all is straightforward."""
    assert issubclass(DSSEDecodeError, DSSEVerifyError)
    assert issubclass(DSSESignatureError, DSSEVerifyError)


def test_ed25519_signer_exposes_public_key_pem():
    """The Signer interface contract: public_key_pem returns a PEM
    SubjectPublicKeyInfo bytestring that load_pem_public_key accepts."""
    signer = _make_signer()
    pem = signer.public_key_pem
    assert pem.startswith(b"-----BEGIN PUBLIC KEY-----")
    pub = serialization.load_pem_public_key(pem)
    assert pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ) == signer.private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def test_ed25519_signer_default_keyid_is_pinned():
    """The keyid lands in the envelope; pinning the literal so a v0.6
    rename has to update this test (and confront the cross-version
    compat question)."""
    signer = _make_signer()
    assert signer.keyid == "topofuzz-dev-v0.5"


def test_envelope_with_first_signature_matching_verifies():
    """DSSE allows multiple signatures; first matching wins. We don't
    emit multi-sig in v0.5, but the verifier accepts it (forward-compat
    with v1.0 multi-party signing)."""
    signer = _make_signer()
    envelope_bytes = dsse_sign(b"{}", "t", signer)
    env = json.loads(envelope_bytes)
    # Prepend a junk signature (wrong bytes) and re-serialize.
    env["signatures"].insert(
        0,
        {"keyid": "junk", "sig": base64.b64encode(b"\x00" * 64).decode("ascii")},
    )
    multi = json.dumps(env).encode("utf-8")
    # The first sig fails; the second (real) sig matches → verify ok.
    verified = dsse_verify(multi, signer.public_key_pem)
    assert verified.payload == {}
    # The second signature (index 1) is the real one; the verifier
    # records which signature index matched.
    assert verified.signature_index == 1


def test_envelope_with_no_matching_signature_fails():
    """If NO signature in the list matches, verify raises."""
    signer = _make_signer()
    other = _make_signer()
    envelope_bytes = dsse_sign(b"{}", "t", other)  # signed by `other`
    with pytest.raises(DSSESignatureError):
        dsse_verify(envelope_bytes, signer.public_key_pem)
