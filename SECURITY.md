# Security Policy

## Supported Versions

Security fixes are published only for the current release line. At any given time, exactly one line is supported.

| Topofuzz version | Status |
|---|---|
| 0.5.x (current pre-release: `0.5.0rc1`) | Supported — security fixes published as point releases on this line |
| < 0.5 | Unsupported |

(Once v0.6 ships, this table is updated in the same commit to mark v0.6.x supported and v0.5.x end-of-life.)

## Reporting a Vulnerability

Email: **security-TBD@modelbrew.ai**

Response SLA: acknowledge within 72 hours; coordinated-disclosure window 90 days from acknowledgment.

PGP key: **[TBD — fingerprint will be published here]**

Please include: topofuzz version, affected component (`topofuzz`, `fuzzeval`, `topofuzz publish`, `topofuzz verify`, etc.), reproduction steps, and a suggested fix if you have one.

For coordinated disclosure that prefers a non-email channel, GitHub Security Advisories on the project repository (private vulnerability reporting) is also accepted.

## API Key Handling

Topofuzz operates under a strict bring-your-own-key (BYO) model.

- **Vendor** API keys are read from environment variables only: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`.
- **Vendor** API keys are never persisted to disk by Topofuzz, never written to any cache file, and never transmitted to any endpoint other than the declared vendor API (see **Network Egress** below).
- The only on-disk secret Topofuzz creates is the local Ed25519 attestation-signing key at `~/.topofuzz/dev.pem` (POSIX `chmod 0o600`); this is not a vendor key and is never transmitted off-host.
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

The signing-key rotation policy is published with the v0.6 key-store hardening release. Until then, the v0.5 dev-key posture is summarised inline below: a single Ed25519 key at `~/.topofuzz/dev.pem`, POSIX `chmod 0o600`, unencrypted PEM, with a `[DEV-KEY]` watermark printed by `topofuzz verify` so an operator never mistakes a local-signed attestation for a production-grade one. Production-grade key escrow (KMS / Sigstore) is a v1.0 milestone.

## Log Redaction

Topofuzz auto-installs a redaction filter (`fuzzeval.log_redaction.SecretRedactionFilter`) on the root logger at `import fuzzeval` time. It scrubs the following patterns from log records before any handler sees them: `sk-…` (OpenAI / Anthropic / xAI dev-key formats), `xai-…`, `AKIA…` (AWS access key id), `AIza…` (Google AI Studio / Gemini), and `Bearer …` (RFC 6750 bearer tokens).

The `sk-…`, `xai-…`, `AKIA…` patterns cover vendor key formats beyond the four declared LLM endpoints as a defense-in-depth measure, so that secrets leaked from adjacent tooling do not surface in topofuzz logs either.

Log-redaction test coverage: 24 tests in `packages/fuzzeval/tests/test_log_redaction.py`.

**Report a redaction gap as a security vulnerability.** A leaked key in a log record is a P0.

## License-Acceptance Telemetry

None. Topofuzz transmits no acceptance event to any server.

A first-run CLI click-through ships in v0.6 and will create a local-only flag file at `~/.topofuzz/license-accepted`; that file is never transmitted, and no telemetry surrounds it. Until v0.6 ships, no acceptance flag is written or read.

---

## Boundary Conditions

The claims above are load-bearing. Future contributors **must** keep this file in sync with the code on any change that touches:

- The set of outbound HTTP destinations (`packages/topofuzz/src/topofuzz/rekor.py`, the LLM SDK call sites, the UI templates under `packages/topofuzz/src/topofuzz/ui/`).
- The redaction filter (`packages/fuzzeval/src/fuzzeval/log_redaction.py`) — pattern additions or removals require updating **Log Redaction** above.
- The BYO-only posture (any code in `packages/topofuzz/src/topofuzz/cli.py` that reads a credential from anywhere other than `os.environ` requires updating **API Key Handling**).
- The on-disk secret list (any code that writes a secret to disk requires updating **API Key Handling**).

If you add any outbound HTTP path, you must also weaken the no-telemetry statement above to reflect what is now transmitted.
