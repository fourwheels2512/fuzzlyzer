# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""DSSE v1.0.0 reference-vector byte-compare gate.

This file is the §12a CI conformance gate. It loads each JSON case under
``tests/test-vectors/dsse/pae/`` and asserts that
``fuzzeval.attestation.pae`` produces the byte-exact expected PAE output
that the vector pins. Drift from the spec is caught here.

The companion file ``test-vectors/dsse/README.md`` documents provenance,
the v1.0.0 upstream SHA, and the per-vector source classification
(``official`` vs ``derived``). Adding a new vector is a JSON-only change:
drop a file in ``pae/`` with the four documented fields and the gate
picks it up via glob.

The in-tree :mod:`test_attestation` already covers PAE edge cases
(UTF-8, bytearray, error paths). This file is intentionally narrow —
**byte-exact** comparison against vendored reference outputs is the only
thing it does. That's what makes it a conformance gate rather than yet
another unit test.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from fuzzeval.attestation import DSSE_PAE_PREFIX, pae

# The vector directory lives next to this test file so the test stays
# locatable without a project-root-relative path (which would break
# when tests move under conftest changes). Resolve via __file__.
_VECTOR_DIR = Path(__file__).parent / "test-vectors" / "dsse" / "pae"

# Sort for deterministic test ordering — Path.glob iteration order is
# filesystem-dependent and we don't want test IDs to reshuffle between
# Windows (NTFS, case-insensitive) and Linux (ext4, case-sensitive).
_VECTOR_FILES = sorted(_VECTOR_DIR.glob("*.json"))


def _load_vector(path: Path) -> dict[str, object]:
    """Parse a vector JSON file and validate its shape.

    Raises ``AssertionError`` with the vector path in the message so a
    malformed vendor file surfaces in test output rather than as a
    cryptic ``KeyError`` from later access.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "source",
        "spec_reference",
        "payload_type",
        "payload_b64",
        "expected_pae_b64",
    }
    missing = required - set(data.keys())
    assert not missing, f"{path.name}: missing fields {sorted(missing)}"
    assert data["source"] in {"official", "derived"}, (
        f"{path.name}: source must be 'official' or 'derived', "
        f"got {data['source']!r}"
    )
    return data


def test_vector_directory_is_nonempty():
    """Guard against an empty `pae/` dir silently passing the gate.

    pytest.mark.parametrize over an empty list collects zero test items
    — no warning, no failure. Without this separate assertion, deleting
    every vector would not break the suite. Pin a floor here.
    """
    assert _VECTOR_FILES, (
        f"no DSSE PAE test vectors discovered under {_VECTOR_DIR}; "
        f"the byte-compare gate would silently pass — refusing"
    )


def test_at_least_one_official_vector_present():
    """At least one vector must be `source: official`.

    The whole point of vendoring is to anchor against an external
    spec; if every vector becomes ``derived`` (e.g. someone removes the
    HelloWorld file by mistake) the gate degrades to "compare our
    implementation against our own derivation" — circular.
    """
    sources = [_load_vector(p)["source"] for p in _VECTOR_FILES]
    assert "official" in sources, (
        "no `source: official` vectors found — refusing to ship a "
        "gate that only compares against derived vectors (circular)"
    )


@pytest.mark.parametrize(
    "vector_path",
    _VECTOR_FILES,
    ids=lambda p: p.stem,
)
def test_pae_matches_reference_bytes(vector_path: Path):
    """Byte-exact compare ``pae(type, payload)`` against the vendored PAE.

    Failure surface includes vector name + first divergence offset so a
    drift report tells the maintainer **where** the bytes diverged, not
    just that they did. The two byte sequences are kept short enough
    to print in full on mismatch (longest official vector is 54 bytes).
    """
    vector = _load_vector(vector_path)
    payload_type = vector["payload_type"]
    payload = base64.b64decode(vector["payload_b64"], validate=True)
    expected_pae = base64.b64decode(vector["expected_pae_b64"], validate=True)

    # Sanity-check the vendored bytes themselves: the spec requires the
    # output to start with the literal DSSE_PAE_PREFIX. Catching this at
    # the vector level (rather than letting it surface as "they differ")
    # gives a clearer diagnostic when someone hand-edits the JSON and
    # corrupts the base64.
    assert expected_pae.startswith(DSSE_PAE_PREFIX), (
        f"{vector_path.name}: vendored expected_pae does not start with "
        f"{DSSE_PAE_PREFIX!r}; the vector itself is malformed, not the "
        f"in-tree pae() implementation"
    )

    actual_pae = pae(payload_type, payload)

    if actual_pae != expected_pae:
        # Find the first divergent offset for a maintainer-friendly diff.
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(actual_pae, expected_pae))
             if a != b),
            min(len(actual_pae), len(expected_pae)),
        )
        pytest.fail(
            f"PAE byte-mismatch for vector {vector['name']!r} "
            f"({vector['source']}):\n"
            f"  expected ({len(expected_pae)} B): {expected_pae!r}\n"
            f"  actual   ({len(actual_pae)} B): {actual_pae!r}\n"
            f"  first divergence at offset {first_diff}"
        )


@pytest.mark.parametrize(
    "vector_path",
    _VECTOR_FILES,
    ids=lambda p: p.stem,
)
def test_pae_structure_pins_byte_count_and_trailing_sp(vector_path: Path):
    """Per-vector structural pin: length field is BYTE count and the
    payload bytes follow a literal SP — including when the payload is
    empty (the trailing SP still appears).

    The byte-compare test above is the primary gate; this test is the
    secondary diagnostic that tells a re-implementer in a different
    language *why* their bytes diverge. The two assertions pin the two
    drift modes most common across re-implementations:

    1. Using ``len(str)`` instead of ``len(bytes)`` for the length field.
    2. Stripping a "looks-trailing" SP when ``len(payload) == 0``.

    Both are caught by byte-compare for our impl, but a vendor reusing
    these vectors gets a clearer diagnostic from this test than a flat
    byte-mismatch.
    """
    vector = _load_vector(vector_path)
    payload_type = vector["payload_type"]
    payload = base64.b64decode(vector["payload_b64"], validate=True)
    expected_pae = base64.b64decode(vector["expected_pae_b64"], validate=True)

    # The format is: DSSEv1 SP <len(type)> SP <type> SP <len(payload)> SP <payload>
    # Reconstruct the prefix-up-to-and-including-the-second-SP-after-payload-len
    # without using ``pae()`` (to keep this test independent of the impl).
    type_bytes = payload_type.encode("utf-8")
    prefix = (
        DSSE_PAE_PREFIX
        + str(len(type_bytes)).encode("ascii")
        + b" "
        + type_bytes
        + b" "
        + str(len(payload)).encode("ascii")
        + b" "
    )
    assert expected_pae.startswith(prefix), (
        f"{vector['name']}: vendored PAE does not begin with the expected "
        f"prefix.\n  prefix expected: {prefix!r}\n  prefix found:    "
        f"{expected_pae[: len(prefix)]!r}"
    )
    # And the trailer must be EXACTLY the payload bytes — no extra SP,
    # no missing trailing byte. This is the part that pins the trailing
    # SP even when ``payload`` is empty: ``expected_pae[len(prefix):]``
    # must equal ``payload`` (zero-length for empty_payload).
    trailer = expected_pae[len(prefix):]
    assert trailer == payload, (
        f"{vector['name']}: PAE trailer does not equal the decoded payload.\n"
        f"  payload ({len(payload)} B): {payload!r}\n"
        f"  trailer ({len(trailer)} B): {trailer!r}"
    )
    # Total length sanity: prefix + payload exhausts the PAE bytes.
    assert len(expected_pae) == len(prefix) + len(payload), (
        f"{vector['name']}: unexpected extra bytes in PAE "
        f"(len={len(expected_pae)}, prefix+payload={len(prefix) + len(payload)})"
    )
