# Privacy Policy — Topofuzz

## Data We Collect

**None.** Topofuzz is a local-first tool. The Licensor does not collect, log, transmit, or store any data about your use of Topofuzz.

## Data You Process Through Topofuzz

You supply evaluation data, agent topologies, and API keys to your local Topofuzz installation. This data is:

- Sent to the LLM vendor APIs you configure (`generativelanguage.googleapis.com`, `api.openai.com`, `api.anthropic.com`, `api.deepseek.com`) according to those vendors' privacy policies. **The vendor APIs you configure act as sub-processors for the data you send.** Review each vendor's Data Processing Addendum (Google, OpenAI, Anthropic, DeepSeek) before sending regulated data.
- Persisted locally in your run directory as `run.sqlite`, `manifest.json`, `report.html`, `attestation.json`, and related files. Retention is fully under your control — Topofuzz never copies or transmits these files.
- Optionally posted to the Sigstore Public Good Instance (`rekor.sigstore.dev`, operated by the Linux Foundation / OpenSSF) **only when you explicitly pass `--rekor`** — see `SECURITY.md` for the threat model of that flag. Rekor submissions are public and permanent.
- Never transmitted to the Licensor, Topofuzz maintainers, or any third party other than the vendors and (when explicitly opted in) the transparency log named above.

You are responsible for compliance with applicable data-protection law (GDPR, CCPA, HIPAA, etc.) for the evaluation data you process.

## Verification

You can verify this posture yourself:

- Observe network traffic during topofuzz invocation. The default path (no `--rekor`, no `topofuzz ui`) emits zero non-vendor egress; `topofuzz scan` and `topofuzz run --dry-run` emit zero egress.
- Inspect `~/.topofuzz/`. The principal local-only files Topofuzz writes there are `~/.topofuzz/dev.pem` (your developer signing key, never transmitted) and, once the v0.6 license click-through ships, `~/.topofuzz/license-accepted` (a local acceptance flag, never transmitted). The `--cache` directory (default `.cache/topofuzz`, configurable) is also local-only.
- Read `SECURITY.md` at the repo root for the full network-egress list and key-handling posture; every claim there is greppable in `packages/topofuzz/src/` and `packages/fuzzeval/src/`.

## Contact

**privacy-TBD@modelbrew.ai**
