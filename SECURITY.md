# Security

This document covers the v0.5 attestation threat model, the published
dev-key location, and a standalone reference verifier for DSSE-signed
topofuzz runs.

## Threat model (v0.5)

topofuzz v0.5 ships **developer-grade signing only**. Attestations are
emitted under a single Ed25519 dev key stored on disk in unencrypted
PKCS#8 PEM at `~/.topofuzz/dev.pem`. Every `topofuzz verify` PASS
banner reads `[DEV-KEY]` exactly so an operator never mistakes a
local-signed attestation for a production-grade one.

| Concern | v0.5 stance | v1.0 plan |
|---|---|---|
| Private key encryption | Unencrypted PEM | Age-encrypted PEM (v0.6); KMS (v1.0) |
| Key rotation | Manual `topofuzz keygen --force` | Rotation aliases + transparency log |
| Multi-party signing | Single signer; envelope shape is plural-ready | Jury-of-3 co-signing |
| Backing key store | Local filesystem | Cloud KMS adapters (`--kms` reserved) |
| Verifier integrity | This document is the reference | Sigstore transparency log |

Threats this signing scheme **does** mitigate:

* Drive-by tampering of `attestation.json`, `manifest.json`, or
  `run.sqlite` after a run finishes (signature, JCS round-trip, and
  digest checks all flag this).
* Orphan-attestation replay (dropping `attestation.json` from one run
  into another run dir): `topofuzz verify` requires `manifest.json`
  and the signed `run.sqlite` digest to be co-located when verifying
  in-place.

Threats this scheme **does not** mitigate:

* A local attacker with read access to `~/.topofuzz/dev.pem` can
  forge any attestation indistinguishably from the developer. This
  is by design for v0.5 and the reason the `[DEV-KEY]` watermark
  exists.
* Long-lived auditability: there is no transparency log in v0.5.
  Rotating the dev key invalidates every prior `topofuzz verify`
  against a run signed with the old key — keep the old `dev.pub.pem`
  around if you need to re-verify historical bundles.

## Published public key

The canonical v0.5 dev pubkey lives in this repository at:

```
keys/dev.pub.pem
```

For a tagged release the stable URL is:

```
https://github.com/kirannayudu/topofuzz/blob/v0.5.0rc1/keys/dev.pub.pem
```

The pubkey is byte-identical across the `v0.5.0rc1` → `v0.5.0` promotion
(same dev-key fixture); on GA promotion the canonical URL switches to
`/blob/v0.5.0/...` without re-signing the v0.5 fixture bundle. This
byte-identical promise is enforced by code-review discipline plus the
sha256-pin in
[`packages/topofuzz/tests/test_release_fixture_pins.py`](https://github.com/kirannayudu/topofuzz/blob/main/packages/topofuzz/tests/test_release_fixture_pins.py)
— the test MUST exist at every release tag; a rotation requires
bumping the pinned digest in the same commit. The pin protects
against accidental `_regen.py` runs but does not defeat an adversary
with merge access; for external trust, pin to a GitHub-immutable tag
URL like `/blob/v0.5.0rc1/keys/dev.pub.pem` rather than `main`.

The private half is **never** committed. Anyone running
`topofuzz keygen` generates their own keypair locally; the committed
public key matches the keypair that signs the v0.5 fixture bundle
at `packages/topofuzz/tests/fixtures/signed/`. To verify a run signed
by a different operator, point `topofuzz verify --pubkey <path>` at
their published pubkey.

## Standalone reference verifier

`topofuzz verify` is the production verify path. It depends only on
`cryptography + rfc8785 + typer + rich + pydantic + fuzzeval` — no
LLM SDKs, no LangGraph, no plotting. If you want to verify a topofuzz
attestation without installing topofuzz at all, the following
~30-line script does the cryptographic core in
stdlib + `cryptography`:

```python
#!/usr/bin/env python3
# verify_dsse.py — v0.5 reference verifier for topofuzz attestations.
# Standard base64 only; the DSSE v1.0 URL-safe MUST is handled in
# `topofuzz verify`. No JCS round-trip; this script prints the
# *authenticated* payload bytes for operator inspection only.
# Deps: stdlib + cryptography. Exit 0 on first verifying signature.
import base64, json, sys
from pathlib import Path
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import load_pem_public_key

if len(sys.argv) != 3:
    sys.exit("usage: verify_dsse.py <envelope.json> <pubkey.pem>")
env = json.loads(Path(sys.argv[1]).read_bytes())
pub = load_pem_public_key(Path(sys.argv[2]).read_bytes())
ptype = env["payloadType"].encode("utf-8")
payload = base64.b64decode(env["payload"], validate=True)
pae = (b"DSSEv1 " + str(len(ptype)).encode("ascii") + b" " + ptype + b" "
       + str(len(payload)).encode("ascii") + b" " + payload)
for s in env["signatures"]:
    try:
        pub.verify(base64.b64decode(s["sig"], validate=True), pae)
    except InvalidSignature:
        continue
    sys.stdout.buffer.write(payload)
    sys.exit(0)
sys.exit(f"FAIL: no signature verified ({len(env['signatures'])} tried)")
```

What it does:

1. Decodes the DSSE envelope JSON.
2. Loads the Ed25519 public key from a PEM SubjectPublicKeyInfo file.
3. Reconstructs the DSSE v1.0 Pre-Authentication Encoding (PAE)
   byte-exactly: `DSSEv1 SP len(type) SP type SP len(payload) SP payload`.
4. Verifies each signature against the PAE; the first match wins.
5. Prints the authenticated payload bytes to stdout (so the operator
   can pipe to `jq` or `python -m json.tool`).

What it deliberately omits relative to `topofuzz verify`:

* **URL-safe base64**: the DSSE v1.0 spec requires verifiers to accept
  both standard and URL-safe encodings. This reference script accepts
  standard only to stay short; production `topofuzz verify` accepts
  both.
* **JCS round-trip check**: a fully-spec-compliant verifier
  re-canonicalizes the parsed payload via RFC 8785 JCS and asserts
  the bytes match the authenticated bytes. This catches the rare
  "signer canonicalized incorrectly" failure mode. The script prints
  the authenticated bytes instead; the operator can compare manually.
* **Run-dir binding**: `topofuzz verify` re-hashes `run.sqlite` and
  re-canonicalizes the on-disk `manifest.json` against the envelope's
  signed copies. The reference script verifies only the envelope.
* **`payloadType` semantic binding**: integrators using this reference
  script as a security boundary **MUST** add an explicit
  `assert env["payloadType"] == EXPECTED_TYPE` before trusting the
  payload. The PAE-binding only catches *mutation of an existing
  envelope* — it does NOT catch a fresh signature over an
  attacker-chosen `payloadType` value (any signer with any key can
  produce a valid envelope claiming any type). `topofuzz verify` does
  enforce the canonical `application/vnd.topofuzz.run+json;v=<v>`
  string match; the reference script omits this check for minimality
  and that is why it is labeled a reference, not a security boundary.

**Reproducing the test:**

```bash
# From the repo root, after `topofuzz keygen` once on this machine
# (or just use the committed fixture):
python verify_dsse.py \
    packages/topofuzz/tests/fixtures/signed/attestation.json \
    keys/dev.pub.pem | python -m json.tool
```

Expected: pretty-printed payload JSON on stdout, exit code 0.
Tampering with any byte of the envelope flips the exit to nonzero.

## Verify-only install footprint

`pip install topofuzz` (base, no extras) pulls only:

| Package | Used for |
|---|---|
| `fuzzeval` | DSSE primitives + key management |
| `cryptography` | Ed25519 + PEM parsing |
| `rfc8785` | JCS canonicalization |
| `typer` + `rich` | CLI surface |
| `pydantic` | Manifest schema validation |

No `langgraph`, no `litellm`, no `anthropic`/`openai`/`google-genai`,
no `matplotlib`, no `jinja2`. The verify codepath through
`topofuzz.cli.verify` is hard-checked by
`packages/topofuzz/tests/test_import_guard.py` to confirm the
forbidden modules are not transitively loaded.

The `topofuzz[verify]` extras name is reserved for forward-compat
and ships **empty** in v0.5 because the base install is already
verify-safe. The `topofuzz[full]` extras pull the run-time stack
(LangGraph + LLM SDKs + plotting) and add roughly 400 MB of wheels.

## Cache-directory trust

`topofuzz run` uses `diskcache` for `LLMCache` and `JudgeCache`
under `--cache <dir>` (default `.cache/topofuzz`). `diskcache` is a
pickle-backed store. **Treat `--cache <dir>` as you would any
pickle file**: only point it at directories you control. The cache
layer at `fuzzeval/cache.py` defends against the most obvious
deserialization footgun by refusing to round-trip any value that is
not a `ContinuousScore` / `BinaryScore` (`JudgeCache`) or `bytes` /
`bytearray` (`LLMCache`), but a determined attacker
who can write arbitrary bytes to your cache directory can still
load object types you trust at attribute-access time. Do not check
`.cache/topofuzz/` into a shared repository, do not point `--cache`
at a network share, and prefer `--no-cache` in CI runners where the
cache root is shared across tenants. v0.6 will switch the on-disk
cache format to JSON-only to remove this surface entirely.

## Operator checklist

* **POSIX**: `~/.topofuzz/dev.pem` is written `chmod 0o600` and its
  parent directory `chmod 0o700` by `topofuzz keygen`. Audit with
  `ls -la ~/.topofuzz/`.
* **Windows**: NTFS ACLs inherit from the user profile directory,
  which is private by default. In shared environments
  (e.g. a build agent under a service account), explicitly remove
  group/everyone read with:

  ```powershell
  icacls $env:USERPROFILE\.topofuzz\dev.pem /inheritance:r `
      /grant:r "$env:USERNAME:(R,W)"
  ```

* **CI**: do not check `dev.pem` into a CI secret store yet — the
  v0.5 dev key is a developer artifact, not a release-signing key.
  Use `--no-cache` and a fresh per-run keypair in ephemeral CI
  containers; treat the resulting attestation as `[DEV-KEY]`-tagged.

## Reporting vulnerabilities

For findings that can be safely disclosed publicly, open an issue at
<https://github.com/kirannayudu/topofuzz/issues> with the label `security`.

For findings that require coordinated disclosure, use GitHub's private
vulnerability reporting at
<https://github.com/kirannayudu/topofuzz/security/advisories/new>. The
maintainer is alerted automatically and the report stays private until
a fix is published. A PGP public key + fingerprint will be added here
at v0.6.
