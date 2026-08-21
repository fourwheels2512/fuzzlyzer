# Changelog — fuzzeval

All notable changes to the `fuzzeval` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases prior to `0.1.0` were development drops on the `0.1.0.dev0`
placeholder version and are not separately enumerated.

## [Unreleased]

No unreleased changes yet.

## [0.2.0] — 2026-05-16

Ships alongside `topofuzz` v1.0.0.

### Changed

- **License change from Apache 2.0 to Business Source License 1.1
  (BUSL-1.1).** Same terms as topofuzz: non-production / academic /
  small-scale commercial use (under USD $100,000 annual gross revenue)
  remain permitted without charge; production-scale use requires a
  commercial license — contact `kiran@modelbrew.ai`. Each release
  auto-converts to Apache 2.0 four years after publication. Full
  terms in `LICENSE`. Version 0.1.0 was released under the Apache
  License, Version 2.0, and remains under that license as published —
  the license change is not retroactive.
- **Consumer pin bumped to `fuzzeval>=0.2,<0.3`.** Supersedes the
  0.1.x line's `>=0.1,<0.2`; `topofuzz` v1.0.0's base install depends
  on this range.

## [0.1.0] — 2026-05-13

First public release. Ships alongside `topofuzz` v0.5.0rc1 as the
verify-path substrate and the LLM-judge/statistics/attestation
primitive layer.

Consumers should depend on `fuzzeval>=0.1,<0.2`; the 0.2 line may shift
signatures.

### Added

- **`fuzzeval.attestation`** — DSSE v1.0.0 envelope build + verify with
  byte-exact PAE encoding (`DSSEv1 SP len(type) SP type SP len(payload)
  SP payload`). Pinned against the
  [`secure-systems-lab/dsse`](https://github.com/secure-systems-lab/dsse)
  v1.0.0 reference vectors via a CI byte-compare gate.
- **`fuzzeval.canonical`** — RFC 8785 JCS canonicalization wrapper +
  stable `canonical_hash` round-trip used as the signed payload digest.
- **`fuzzeval.judge`** —
  - `SingleJudge`: one LLM judge call per cell, `JudgeCache`-backed,
    UNPARSEABLE fallback, rubric-template based.
  - `JuryJudge` (v0.5.0rc1 opt-in via `topofuzz run --jury-of-3`):
    three-vendor fan-out (Anthropic + OpenAI + Gemini) with
    `threading.Lock`-guarded accounting, vendor-name normalization,
    duplicate-model-VALUE rejection, mean-of-parseable aggregation,
    cumulative `judge_failure_rate`, and `judge_independence.vendor_overlap`.
  - `ContinuousScore` + `BinaryScore` pydantic types as the cache-safe
    judge-return contract.
- **`fuzzeval.stats`** —
  - Wilson CI.
  - Percentile bootstrap.
  - `percentile_cluster_bootstrap` — two-level paired resampler
    (outer = eval_id, inner = variant_id) with property-tested
    idempotence under permutation, length invariance, and CI-width
    monotonicity in n. Custom code; scipy does not do this natively.
- **`fuzzeval.cache`** —
  - `LLMCache` and `JudgeCache` on top of `diskcache` (extras
    `[cache]`).
  - Strict pickle-safety: refuses to round-trip any value that is not a
    `ContinuousScore` / `BinaryScore` (`JudgeCache`) or `bytes` /
    `bytearray` (`LLMCache`).
- **`fuzzeval.costest`** — cold + warm-cache dry-run estimator with
  90% confidence bands per the BLUEPRINT §12 cost-estimator contract.
- **`fuzzeval.keys`** — Ed25519 dev-keypair generation + load via
  `cryptography`. POSIX writes are chmod 0o600 on the private file with
  parent chmod 0o700; Windows inherits NTFS ACLs from the user profile.
  Atomic `os.replace` write path; `KeyAlreadyExistsError` /
  `KeyNotFoundError` typed exceptions.

### Packaging

- `fuzzeval` (base) — verify-path safe. Pulls `cryptography`, `rfc8785`,
  `pydantic`. ~15 MB.
- `fuzzeval[stats]` — adds `numpy + scipy` for the cluster bootstrap path.
- `fuzzeval[cache]` — adds `diskcache` for the on-disk cache path.
- Per-package `LICENSE` bundled into the wheel via `license-files =
  ["LICENSE"]` (PEP 639).
- Python 3.11 / 3.12 / 3.13 / 3.14 declared as supported; macOS, Linux,
  and Windows declared in classifiers.

[0.2.0]: https://github.com/fourwheels2512/fuzzlyzer/releases/tag/fuzzeval-v0.2.0
[0.1.0]: https://github.com/fourwheels2512/fuzzlyzer/releases/tag/fuzzeval-v0.1.0
