# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""DSSE v1.0 signing / verification (PAE byte-exact).

This module is the §17a-frozen cryptographic primitive shared by
``topofuzz attest`` (signer) and the standalone reference verifier
in ``SECURITY.md`` (verifier). It is intentionally verify-path-safe:
no numpy, no sqlite, no LangGraph. The only external deps are
``cryptography`` (already pinned in ``fuzzeval`` base deps) and the
Python stdlib.

The PAE construction is the literal DSSE v1.0 formula:

    DSSEv1 SP len(type) SP type SP len(payload) SP payload

where ``len(...)`` is the ASCII decimal repr of the byte length, ``SP``
is a single 0x20 byte, ``type`` is the UTF-8 encoding of the
``payloadType`` string, and ``payload`` is the raw payload bytes
(NOT base64-wrapped — the envelope's ``payload`` field is base64, but
the PAE is computed over the decoded form).

See https://github.com/secure-systems-lab/dsse/blob/master/protocol.md
and BLUEPRINT_v5plus.md §11 line 304. The byte-exact test vectors land
in step 12a per the blueprint roadmap (`secure-systems-lab/dsse` v1.0.0
forked into ``topofuzz/test-vectors/dsse/``); this module's PAE
implementation is the production-bound code those vectors will exercise.

JCS round-trip stability: callers wanting "the bytes a verifier
recomputes match the bytes a signer signed" should JCS-canonicalize
the payload BEFORE signing (this module accepts already-canonical bytes
and does not re-canonicalize internally). :func:`dsse_verify` returns a
:class:`VerifiedPayload` carrying the authenticated bytes alongside the
parsed dict so callers can do that round-trip check WITHOUT re-parsing
the envelope (which would be a DSSE spec MUST violation — the spec
forbids re-extracting the payload from a second parse of the envelope).

Base64 encoding: DSSE v1.0 requires verifiers to accept BOTH standard
and URL-safe base64 encodings of payload + signatures. Signers MAY emit
either; this module emits standard at sign time but accepts both at
verify time.

§17a / naming note: BLUEPRINT_v5plus.md §17a line 556 names the local
signer class ``AgePEMSigner``. v0.5 ships :class:`Ed25519Signer` instead
because the user-locked step-10a decision dropped age-encryption (it
added a transitive ``keyring``/``age`` dep without v0.5 value; production
signing goes to Cloud KMS at v1.0 per §11 line 308). The blueprint name
will be updated alongside the v0.6 keyring/age work.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any, Final, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

__all__ = [
    "DSSE_PAE_PREFIX",
    "DSSEDecodeError",
    "DSSESignatureError",
    "DSSEVerifyError",
    "Ed25519Signer",
    "Signer",
    "VerifiedPayload",
    "dsse_sign",
    "dsse_verify",
    "pae",
]

# The literal first 7 bytes of every DSSE v1.0 PAE: "DSSEv1" + a space.
DSSE_PAE_PREFIX: Final[bytes] = b"DSSEv1 "


class DSSEVerifyError(ValueError):
    """Parent for any verify-time failure. Callers wanting a single
    ``except`` clause for both decode and signature failures catch this;
    callers wanting to discriminate use ``DSSEDecodeError`` /
    ``DSSESignatureError`` directly."""


class DSSEDecodeError(DSSEVerifyError):
    """Envelope JSON is malformed or missing required fields."""


class DSSESignatureError(DSSEVerifyError):
    """Ed25519 signature failed cryptographic verification."""


class Signer(Protocol):
    """The signer interface shared by Ed25519Signer and the v1.0 KMSSigner.

    The envelope shape is signer-agnostic (BLUEPRINT_v5plus.md §16 #3 unit
    test) — anything that produces 64-byte Ed25519 signatures and exposes
    its public key as a PEM SubjectPublicKeyInfo conforms. ``keyid`` is a
    short opaque string that lands in the envelope; verifiers ignore it
    cryptographically but tooling uses it to pick a public key.

    Declared as class-level annotations (not ``@property``) so a
    dataclass-field implementation (Ed25519Signer) and a property-based
    implementation (a future KMSSigner) both satisfy the protocol
    without the implementer having to mirror the decorator choice.
    """

    keyid: str
    public_key_pem: bytes

    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True)
class Ed25519Signer:
    """Local Ed25519 signer wrapping a ``cryptography`` private key.

    The DEV-KEY watermark and the threat model around unencrypted
    on-disk PEMs are documented in SECURITY.md (step 10b). This class
    intentionally has no awareness of the on-disk file format — key
    loading lives in :mod:`fuzzeval.keys`.

    The ``keyid`` literal is pinned at the minor version
    (``topofuzz-dev-v0.5``); a major / minor bump that breaks
    attestation cross-compat should rotate it. Patch-level bumps reuse.
    """

    private_key: Ed25519PrivateKey
    keyid: str = "topofuzz-dev-v0.5"

    @property
    def public_key_pem(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)


@dataclass(frozen=True)
class VerifiedPayload:
    """The output of :func:`dsse_verify`.

    Carries the EXACT bytes that were cryptographically authenticated
    (``payload_bytes``) alongside the parsed dict (``payload``) and the
    envelope's ``payloadType``. Returning the bytes alongside the dict
    is a deliberate DSSE-spec-compliance choice — callers needing the
    JCS round-trip invariant ("the bytes I signed equal the bytes I
    re-canonicalize from the dict") MUST NOT re-parse the envelope to
    pull out the payload field; they MUST use ``payload_bytes`` here.
    Per DSSE v1.0 spec: *"implementations MUST NOT re-parse the
    envelope after verification to pull out the payload."*
    """

    payload_bytes: bytes
    payload: dict[str, Any]
    payload_type: str
    signature_index: int


def pae(payload_type: str, payload: bytes) -> bytes:
    """Return the DSSE v1.0 pre-authentication encoding (byte-exact).

    Formula::

        b"DSSEv1 " + ascii(len(type_utf8)) + b" "
                   + type_utf8           + b" "
                   + ascii(len(payload)) + b" "
                   + payload

    Raises ``TypeError`` if ``payload`` is not bytes (a frequent
    integration-time mistake — accepting str would silently re-encode
    via implicit UTF-8 and break byte-exactness for any non-ASCII
    payload). ``payload_type`` is a str by spec and is UTF-8 encoded
    here.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(
            f"pae: payload must be bytes, got {type(payload).__name__}"
        )
    if not isinstance(payload_type, str):
        raise TypeError(
            f"pae: payload_type must be str, got {type(payload_type).__name__}"
        )
    type_bytes = payload_type.encode("utf-8")
    return (
        DSSE_PAE_PREFIX
        + str(len(type_bytes)).encode("ascii")
        + b" "
        + type_bytes
        + b" "
        + str(len(payload)).encode("ascii")
        + b" "
        + bytes(payload)
    )


def _b64_decode_accept_both(s: str) -> bytes:
    """Decode a base64 string accepting both standard and URL-safe alphabets.

    DSSE v1.0 spec: *"Either standard or URL-safe base64 encodings are
    allowed. Signers may use either, and verifiers MUST accept either."*
    Standard uses ``+``/``/``; URL-safe uses ``-``/``_``. Both may have
    ``=`` padding or omit it. We try standard first (the common case),
    then URL-safe with padding restored.

    Raises ``binascii.Error`` if neither decodes.
    """
    if not isinstance(s, str):
        raise binascii.Error(f"base64 input must be str, got {type(s).__name__}")
    # Cheap heuristic: if the string contains '-' or '_', it's URL-safe.
    # Otherwise try standard first.
    has_urlsafe_chars = "-" in s or "_" in s
    # Pad to a multiple of 4. b64decode tolerates missing padding only
    # with validate=False; we want validate=True so we restore padding
    # explicitly. (Adding padding to a string that already had it is a
    # parse error, so only pad if length % 4 != 0.)
    padded = s + "=" * (-len(s) % 4)
    if not has_urlsafe_chars:
        try:
            return base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            pass
    return base64.urlsafe_b64decode(padded)


def dsse_sign(payload: bytes, payload_type: str, signer: Signer) -> bytes:
    """Return a serialized DSSE envelope as UTF-8 JSON bytes.

    The caller is responsible for canonicalizing ``payload`` (typically
    via JCS / rfc8785) before passing it in — this function signs
    whatever bytes it gets. The envelope shape is DSSE v1.0:

        {"payload": b64, "payloadType": str, "signatures": [{"keyid","sig"}]}

    ``signer`` must conform to the :class:`Signer` Protocol —
    specifically it must expose a ``keyid: str`` and a
    ``public_key_pem: bytes`` attribute and a ``sign(bytes) -> bytes``
    method. We check ``keyid`` at sign-time so a buggy custom Signer
    that forgets the attribute fails here with a clear message rather
    than at envelope-decode time with a cryptic ``AttributeError``.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(
            f"dsse_sign: payload must be bytes, got {type(payload).__name__}"
        )
    if not hasattr(signer, "keyid") or not isinstance(signer.keyid, str):
        raise TypeError(
            "dsse_sign: signer must expose .keyid: str — "
            f"got {type(signer).__name__} without conforming keyid attr"
        )
    if not hasattr(signer, "public_key_pem"):
        raise TypeError(
            "dsse_sign: signer must expose .public_key_pem: bytes"
        )
    sig_bytes = signer.sign(pae(payload_type, payload))
    envelope = {
        "payload": base64.b64encode(bytes(payload)).decode("ascii"),
        "payloadType": payload_type,
        "signatures": [
            {
                "keyid": signer.keyid,
                "sig": base64.b64encode(sig_bytes).decode("ascii"),
            }
        ],
    }
    return json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def dsse_verify(envelope: bytes, public_key_pem: bytes) -> VerifiedPayload:
    """Verify ``envelope`` against ``public_key_pem`` and return a
    :class:`VerifiedPayload`.

    Verification covers:

    1. Envelope JSON shape (``payload``, ``payloadType``, ``signatures[]``).
    2. Base64 well-formedness of ``payload`` and each ``sig`` (accepts
       both standard AND URL-safe base64 per DSSE v1.0 MUST).
    3. Ed25519 signature over the PAE-reconstructed preimage.
    4. Payload parses as JSON object (the v0.5 contract for the inner
       payload is always a JSON object; this is checked so callers can
       trust the returned ``payload`` dict).

    Returns a :class:`VerifiedPayload` carrying the AUTHENTICATED
    payload bytes (not the envelope bytes — those went through PAE).
    Callers MUST use ``VerifiedPayload.payload_bytes`` for any
    downstream byte-level checks (e.g. JCS round-trip stability),
    NOT re-parse the envelope, per DSSE v1.0 spec.

    Raises ``DSSEDecodeError`` for shape/encoding failures and
    ``DSSESignatureError`` for cryptographic failures. When the envelope
    contains multiple signatures and none verify, the error message
    summarises ALL failures (not just the last attempt) so ops can
    distinguish "all sigs malformed" from "all sigs wrong key".
    """
    if not isinstance(envelope, (bytes, bytearray)):
        raise DSSEDecodeError(
            f"envelope must be bytes, got {type(envelope).__name__}"
        )
    try:
        env = json.loads(envelope)
    except json.JSONDecodeError as e:
        raise DSSEDecodeError(f"envelope is not valid JSON: {e}") from e
    if not isinstance(env, dict):
        raise DSSEDecodeError(
            f"envelope must be a JSON object, got {type(env).__name__}"
        )
    for key in ("payload", "payloadType", "signatures"):
        if key not in env:
            raise DSSEDecodeError(f"envelope missing required field {key!r}")
    payload_type = env["payloadType"]
    if not isinstance(payload_type, str):
        raise DSSEDecodeError("payloadType must be a string")
    if not isinstance(env["payload"], str):
        raise DSSEDecodeError("payload must be a base64 string")
    sigs = env["signatures"]
    if not isinstance(sigs, list) or not sigs:
        raise DSSEDecodeError("signatures must be a non-empty list")
    try:
        payload_bytes = _b64_decode_accept_both(env["payload"])
    except (binascii.Error, ValueError) as e:
        raise DSSEDecodeError(f"payload is not valid base64: {e}") from e

    preimage = pae(payload_type, payload_bytes)

    try:
        pub = serialization.load_pem_public_key(public_key_pem)
    except (ValueError, TypeError) as e:
        raise DSSEDecodeError(f"public key PEM is malformed: {e}") from e
    if not isinstance(pub, Ed25519PublicKey):
        raise DSSEDecodeError(
            f"expected Ed25519 public key, got {type(pub).__name__}"
        )

    # DSSE v1.0 envelopes carry a list of signatures so multi-party
    # signing slots in without a shape change. v0.5 only emits one
    # signature, but verify accepts ANY signature in the list that
    # matches — the first match wins. This mirrors `sigstore` / the
    # reference DSSE Go verifier.
    #
    # Accumulate per-signature failures so a multi-sig envelope where
    # none verify can report ALL failure modes, not just the last one
    # attempted. Decode failures (malformed shape, bad base64) are
    # reported before cryptographic failures because they're more
    # diagnostic — they tell ops the producer was broken, not the key.
    decode_failures: list[str] = []
    signature_failures: list[str] = []
    for idx, sig_entry in enumerate(sigs):
        if not isinstance(sig_entry, dict) or "sig" not in sig_entry:
            decode_failures.append(
                f"[{idx}] missing 'sig' field"
            )
            continue
        if not isinstance(sig_entry["sig"], str):
            decode_failures.append(
                f"[{idx}] sig is not a string"
            )
            continue
        try:
            sig_bytes = _b64_decode_accept_both(sig_entry["sig"])
        except (binascii.Error, ValueError) as e:
            decode_failures.append(f"[{idx}] sig not valid base64: {e}")
            continue
        try:
            pub.verify(sig_bytes, preimage)
        except InvalidSignature:
            signature_failures.append(
                f"[{idx}] does not verify against the provided public key"
            )
            continue
        # Match. Decode the payload and return.
        try:
            payload_obj = json.loads(payload_bytes)
        except json.JSONDecodeError as e:
            raise DSSEDecodeError(
                f"payload bytes verified but are not valid JSON: {e}"
            ) from e
        if not isinstance(payload_obj, dict):
            raise DSSEDecodeError(
                f"payload must be a JSON object, got "
                f"{type(payload_obj).__name__}"
            )
        return VerifiedPayload(
            payload_bytes=payload_bytes,
            payload=payload_obj,
            payload_type=payload_type,
            signature_index=idx,
        )

    # No signature matched. Report decode failures first (they're more
    # diagnostic), then crypto failures.
    msg_parts = []
    if decode_failures:
        msg_parts.append("decode failures: " + "; ".join(decode_failures))
    if signature_failures:
        msg_parts.append("signature failures: " + "; ".join(signature_failures))
    summary = (
        f"{len(sigs)} signature(s), none verified — " + " | ".join(msg_parts)
    )
    # Prefer DSSESignatureError when ALL failures are crypto (the common
    # case for tampered envelopes); DSSEDecodeError otherwise so ops can
    # tell "tampered" apart from "malformed producer".
    if signature_failures and not decode_failures:
        raise DSSESignatureError(summary)
    raise DSSEDecodeError(summary)
