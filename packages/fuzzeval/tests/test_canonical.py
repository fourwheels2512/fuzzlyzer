"""Direct unit tests for the shared canonical-hash helper.

``fuzzeval.canonical.canonical_hash`` is the single source of truth for
RFC 8785 (JCS) + SHA-256 hashing across the codebase. The chain-
consistency tests in ``topofuzz.tests.test_ir`` pin that
``Topology.canonical_hash`` and ``LLMConfig.canonical_hash`` route
through this helper; this file pins the helper itself — recipe
algorithm (sha256, not sha3/md5), error semantics, and behaviour on
edge-case JSON values.
"""

from __future__ import annotations

import hashlib

import pytest

from fuzzeval.canonical import canonical_hash


class TestRecipePin:
    def test_returns_64_char_lowercase_hex(self):
        h = canonical_hash({"k": "v"})
        assert len(h) == 64
        int(h, 16)  # raises if not hex
        assert h == h.lower()

    def test_empty_dict_golden(self):
        # Direct sha256-of-JCS recipe pin: an algorithm swap (e.g. sha3,
        # md5) is caught by the length checks above, but a swap to a
        # *different 64-char hash* (sha3_256, blake2b_256) would slip
        # past those. This golden value comes from running
        # `sha256(b"{}").hexdigest()` — JCS canonicalisation of `{}`
        # is the two bytes `{}`.
        expected = hashlib.sha256(b"{}").hexdigest()
        assert canonical_hash({}) == expected
        # Cross-check that the expected value is the documented one.
        assert (
            canonical_hash({})
            == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        )

    def test_deterministic_across_calls(self):
        a = canonical_hash({"a": 1, "b": [1, 2, 3]})
        b = canonical_hash({"a": 1, "b": [1, 2, 3]})
        assert a == b

    def test_key_order_invariant(self):
        # JCS sorts object keys lexicographically — different insertion
        # order must hash identically.
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_distinguishes_value_change(self):
        assert canonical_hash({"k": 1}) != canonical_hash({"k": 2})

    def test_distinguishes_key_change(self):
        assert canonical_hash({"k": 1}) != canonical_hash({"K": 1})

    def test_negative_zero_normalised(self):
        # rfc8785 emits "0" for both 0.0 and -0.0. A future regression
        # in rfc8785 surfaces here — and we want it to, because the
        # variants/decoding_param_swap contract relies on this.
        assert canonical_hash({"t": 0.0}) == canonical_hash({"t": -0.0})


class TestUnserializableInputs:
    def test_bytes_value_rejected(self):
        with pytest.raises(TypeError, match="JSON / JCS serializable"):
            canonical_hash({"k": b"raw"})

    def test_set_value_rejected(self):
        with pytest.raises(TypeError, match="JSON / JCS serializable"):
            canonical_hash({"k": {1, 2, 3}})

    def test_nan_value_rejected(self):
        # NaN is a float but rfc8785 cannot represent it in JCS.
        with pytest.raises(TypeError, match="JSON / JCS serializable"):
            canonical_hash({"t": float("nan")})

    def test_inf_value_rejected(self):
        with pytest.raises(TypeError, match="JSON / JCS serializable"):
            canonical_hash({"t": float("inf")})


class TestPrimitivesAccepted:
    def test_string_accepted(self):
        canonical_hash("hello")

    def test_int_accepted(self):
        canonical_hash(42)

    def test_float_accepted(self):
        canonical_hash(3.14)

    def test_bool_accepted_as_json_bool(self):
        # bool IS valid JSON; the helper accepts it. (LLMConfig's
        # field_validator separately rejects bool *temperature* because
        # that field semantically demands a float — but the helper
        # itself is value-agnostic.)
        assert canonical_hash(True) != canonical_hash(False)
        assert canonical_hash(True) != canonical_hash(1)

    def test_none_accepted_as_json_null(self):
        canonical_hash(None)

    def test_nested_structures_accepted(self):
        canonical_hash({"a": [1, {"b": None, "c": [True, False]}]})
