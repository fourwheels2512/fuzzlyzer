# Security Policy

## Supported Versions

Security fixes are published only for the current release line. At any given time, exactly one line is supported.

| Topofuzz version | Status |
|---|---|
| 1.0.x | Supported — security fixes published as point releases on this line |
| < 1.0 | Unsupported (the 0.x line was a pre-release iteration; upgrade to 1.0.x for security fixes) |

(On v1.1 ship, this table is updated in the same commit to mark v1.1.x supported and v1.0.x EOL per the one-line-supported policy.)

## Reporting a Vulnerability

Email: **kiran@modelbrew.ai**

Response SLA: acknowledge within 72 hours; coordinated-disclosure window 90 days from acknowledgment.

PGP key: Available on request via **kiran@modelbrew.ai**

Please include: topofuzz version, affected component (`topofuzz`, `fuzzeval`, `topofuzz publish`, `topofuzz verify`, etc.), reproduction steps, and a suggested fix if you have one.

For coordinated disclosure that prefers a non-email channel, GitHub Security Advisories on the project repository (private vulnerability reporting) is also accepted.

## API Key Handling

Topofuzz operates under a strict bring-your-own-key (BYO) model.

- **CLI**: vendor API keys are read from environment variables only: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`. The CLI never writes a vendor key to disk.
- **`topofuzz ui`**: by default, a key pasted into the in-browser key card is held **in memory only**, for the lifetime of the running process — never written to disk. The "Remember this key on this computer" checkbox is **opt-in and unchecked by default**; only when a user explicitly checks it does the UI write the key to `~/.topofuzz/keys/{vendor}` (plaintext; POSIX `chmod 0o600`, and on Windows an `icacls` ACL reset that grants Full Control to the current user only and drops inherited permissions — POSIX mode bits do not restrict access on NTFS). Unchecking "remember" (or clicking "Remove") deletes that file.
- In all cases, a vendor API key is never written to any cache file and never transmitted to any endpoint other than the declared vendor API (see **Network Egress** below).
- The on-disk secrets Topofuzz can create: the local Ed25519 attestation-signing key at `~/.topofuzz/dev.pem` (POSIX `chmod 0o600`; not a vendor key, never transmitted off-host), and — opt-in only, UI path — `~/.topofuzz/keys/{vendor}` as described above.
- The `topofuzz repro` command emits a shell script with log-redaction guards (see **Log Redaction** below); the generated script will not echo or log API keys even if invoked under `set -x`.

## Network Egress

Topofuzz makes outbound HTTPS requests **only** to the following endpoints, and only when explicitly invoked by the user:

- `generativelanguage.googleapis.com` (Gemini) — vendor LLM API
- `api.openai.com` (OpenAI) — vendor LLM API
- `api.anthropic.com` (Anthropic) — vendor LLM API
- `api.deepseek.com` (DeepSeek) — vendor LLM API
- `rekor.sigstore.dev` (Sigstore Rekor transparency log) — **only when `--rekor` is explicitly passed**; never on the default path
- `unpkg.com` (HTMX library CDN, `htmx.org@1.9.12`) — **only when `topofuzz ui` is invoked**; the request is issued by the user's browser when loading the served HTML page, not by the Python process. Vendoring HTMX locally ships in v0.6.
- `aistudio.google.com` (Google AI Studio key-issuance page) — **only when `topofuzz ui` is invoked and the user clicks the "Get a Gemini key" link** rendered by the no-key page; this is a user-initiated browser navigation, not an automatic request.

No telemetry, no analytics, no crash reporting, no auto-update checks. The Python process does **not** phone home on the default path.

To verify:
- `topofuzz scan` produces zero outbound network traffic.
- `topofuzz run --dry-run` produces zero outbound network traffic.
- Observe network traffic during any topofuzz invocation; only the endpoints listed above appear, and only the ones gated by their flag/subcommand.

## Cryptographic Attestation

Topofuzz signs every run bundle with DSSE (Dead Simple Signing Envelope, an in-toto / sigstore standard). The signed bundle can be re-verified offline with `topofuzz verify <run-dir>`.

The DSSE envelope is byte-exact against the [`secure-systems-lab/dsse`](https://github.com/secure-systems-lab/dsse) v1.0.0 reference vectors. Verification depends only on `cryptography + rfc8785` and does not require the LLM runtime stack.

The current v1.0 dev-key posture is summarised inline below: a single Ed25519 key at `~/.topofuzz/dev.pem`, POSIX `chmod 0o600`, unencrypted PEM, with a `[DEV-KEY]` watermark printed by `topofuzz verify` so an operator never mistakes a local-signed attestation for a production-grade one. Production-grade key escrow (KMS / Sigstore) is a v1.1 milestone with its own rotation policy ship.

## Standalone Reference Verifier

`topofuzz verify` is the production verify path: it accepts both standard and URL-safe base64 (DSSE v1.0 MUST), re-canonicalises the parsed payload via RFC 8785 JCS, and re-binds the run directory by re-hashing `run.sqlite` and `manifest.json` against the envelope's signed copies. To verify a DSSE-signed attestation **without installing topofuzz at all**, the following script does the cryptographic core in stdlib + `cryptography`:

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

What it does, in order: decodes the DSSE envelope JSON; loads a PEM public key via `cryptography.hazmat.primitives.serialization.load_pem_public_key`; reconstructs the DSSE v1.0 Pre-Authentication Encoding (`DSSEv1 SP len(type) SP type SP len(payload) SP payload`) byte-exactly; verifies each signature against the PAE and writes the authenticated payload bytes to stdout on the first match.

What it deliberately omits relative to `topofuzz verify`: URL-safe base64 acceptance, the RFC 8785 JCS round-trip check, and the run-directory binding. These all live in `topofuzz verify`; the reference script keeps the cryptographic core inspectable in one screen.

**Operator caveats** (the script is intentionally minimal; production verification belongs in `topofuzz verify`):

- The script verifies the **first** signature in `signatures[]` that matches the supplied pubkey, without checking the signature's `keyid` field. On a multi-signer envelope this means "some signature by the supplied key passed" — not "the signer named *X* signed this." Topofuzz v0.5 emits single-signer envelopes, so this degenerates to "the published `keys/dev.pub.pem` signed this"; verify the keyid yourself before trusting an envelope you did not produce locally.
- The script supplies the pubkey to `load_pem_public_key`, which accepts any PEM SPKI key type. Topofuzz signs with Ed25519; passing a non-Ed25519 PEM raises `TypeError` (not `InvalidSignature`) from `pub.verify(...)` and surfaces as a Python traceback. Likewise, envelopes missing `payloadType` / `payload` / `signatures` (or signature entries missing `sig`) raise `KeyError`, and a malformed base64 payload raises `binascii.Error`. **Treat any nonzero exit as verification failure** — distinguishing tamper from malformed input is out of scope for a reference verifier.

To reproduce against the committed v0.5 fixture:

```bash
python verify_dsse.py \
    packages/topofuzz/tests/fixtures/signed/attestation.json \
    keys/dev.pub.pem | python -m json.tool
```

Exit 0 on the unmodified fixture; nonzero with no payload on stdout after flipping any byte of the envelope's base64 payload (`packages/topofuzz/tests/test_security_md_script.py` runs both paths in CI to keep this doc from drifting).

## Log Redaction

Topofuzz auto-installs a redaction filter (`fuzzeval.log_redaction.SecretRedactionFilter`) on the root logger at `import fuzzeval` time. It scrubs the following patterns from log records before any handler sees them: `sk-…` (OpenAI / Anthropic / xAI dev-key formats), `xai-…`, `AKIA…` (AWS access key id), `AIza…` (Google AI Studio / Gemini), and `Bearer …` (RFC 6750 bearer tokens).

The `sk-…`, `xai-…`, `AKIA…` patterns cover vendor key formats beyond the four declared LLM endpoints as a defense-in-depth measure, so that secrets leaked from adjacent tooling do not surface in topofuzz logs either.

Log-redaction test coverage: 24 tests in `packages/fuzzeval/tests/test_log_redaction.py`.

**Report a redaction gap as a security vulnerability.** A leaked key in a log record is a P0.

## License-Acceptance Telemetry

None. Topofuzz transmits no acceptance event to any server.

A first-run CLI click-through (shipped in v1.0) creates a local-only flag file at `~/.topofuzz/license-accepted`; that file is never transmitted, and no telemetry surrounds it.

---

## Boundary Conditions

The claims above are load-bearing. Future contributors **must** keep this file in sync with the code on any change that touches:

- The set of outbound HTTP destinations (`packages/topofuzz/src/topofuzz/rekor.py`, the LLM SDK call sites, the UI templates under `packages/topofuzz/src/topofuzz/ui/`).
- The redaction filter (`packages/fuzzeval/src/fuzzeval/log_redaction.py`) — pattern additions or removals require updating **Log Redaction** above.
- The BYO-only posture (any code in `packages/topofuzz/src/topofuzz/cli.py` that reads a credential from anywhere other than `os.environ` requires updating **API Key Handling**).
- The on-disk secret list (any code that writes a secret to disk requires updating **API Key Handling**).
- The DSSE PAE construction in `fuzzeval/attestation.py` — any change to envelope shape, PAE byte layout, accepted base64 alphabet, or signature algorithm (currently Ed25519 via `Ed25519Signer`; switching to Ed448 or hybrid silently breaks the reference verifier) must mirror into the **Standalone Reference Verifier** code block above (the block is content-anchored on `DSSEv1` and re-extracted by `packages/topofuzz/tests/test_security_md_script.py`).

If you add any outbound HTTP path, you must also weaken the no-telemetry statement above to reflect what is now transmitted.
