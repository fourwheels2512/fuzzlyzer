# DSSE v1.0.0 PAE Test Vectors

This directory vendors byte-exact reference outputs for the DSSE v1.0
Pre-Authentication Encoding primitive defined in
`fuzzeval.attestation.pae`. It is the spec-conformance gate referenced in
`BLUEPRINT_v5plus.md` §12a line 398.

## Provenance

Upstream: <https://github.com/secure-systems-lab/dsse>
Tag: `v1.0.0`
Commit SHA: `cacf247ea07024437c91e69ae65f3f4a2df3c657`

The canonical `HelloWorld` vector is reproduced byte-exactly from the
upstream "Test Vectors" section of `protocol.md` at that tag:

> SERIALIZED_BODY: `hello world`
> PAYLOAD_TYPE:    `http://example.com/HelloWorld`
> PAE:             `DSSEv1 29 http://example.com/HelloWorld 11 hello world`

Source URL (verbatim quote):
<https://github.com/secure-systems-lab/dsse/blob/v1.0.0/protocol.md#test-vectors>

## What's vendored

Each `.json` file under `pae/` is a single test case carrying the four
fields the gate needs:

| field              | meaning                                                    |
|--------------------|------------------------------------------------------------|
| `name`             | Human-readable case name (matches the filename stem).      |
| `source`           | `official` (verbatim from spec) or `derived` (hand-derived from the PAE formula in `protocol.md` §4 with the same byte-counting rules). |
| `spec_reference`   | Either the upstream URL or the formula quote that derived the case. |
| `payload_type`     | The `payloadType` UTF-8 string fed into `pae()`.           |
| `payload_b64`      | Standard base64 of the raw `payload` bytes (no padding stripping). |
| `expected_pae_b64` | Standard base64 of the byte-exact PAE output the spec demands. |

`payload` and `expected_pae` are base64-encoded so binary or
non-printable bytes survive a `git diff` cleanly.

## Vectors

| file                          | source   | what it pins                                           |
|-------------------------------|----------|--------------------------------------------------------|
| `pae/helloworld.json`         | official | The canonical spec example. Drift-detection anchor.    |
| `pae/empty_payload.json`      | derived  | `len(payload) == 0`: trailing space + `0` length field.|
| `pae/utf8_payload.json`       | derived  | Multi-byte UTF-8 payload: length is byte-count, not codepoint count. |
| `pae/utf8_payload_type.json`  | derived  | Multi-byte UTF-8 payloadType: type-length is byte-count.|

## Adding new vectors

1. Drop a new `*.json` in `pae/` with the four fields above.
2. The parametrized gate in `tests/test_dsse_vectors.py` discovers files
   by glob — no Python edit needed.
3. Set `source: "official"` only when the bytes are verbatim from a
   tagged spec / reference implementation; otherwise `derived`.

## Regenerating expected_pae_b64 by hand

The PAE formula (DSSE v1.0 protocol.md §4) is:

```
DSSEv1 SP <ascii len(type_utf8)> SP <type_utf8> SP <ascii len(payload)> SP <payload>
```

where `SP` is a single 0x20 byte. There is no trailing newline. Lengths
are ASCII decimal of the **byte** count (not character / codepoint
count).

Even when `payload` is empty, the trailing SP separator before the
payload field remains in the output — the PAE for an empty-payload case
ends with a literal `0 ` (a zero, then a single 0x20 byte). The
`empty_payload.json` vector pins this directly. Re-derivers who omit
the trailing SP for empty payloads produce a 1-byte-short PAE that
the gate will (correctly) reject.
