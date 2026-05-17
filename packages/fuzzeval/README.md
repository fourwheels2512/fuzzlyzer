# fuzzeval™

**Verify-path substrate for `topofuzz`™ and sibling multi-agent eval tools.**

`fuzzeval` ships the modules a verify-only install needs: DSSE
attestation, single + jury LLM judges, statistics, on-disk caches,
cost estimation, and key management. It has **no LangGraph dependency**
and **no LLM SDK dependency** at import time, so `pip install topofuzz`
(verify-only path) stays under ~15 MB.

## License

`fuzzeval` is licensed under the Business Source License 1.1 (BSL)
alongside `topofuzz`. Non-production use, internal development,
academic research, and small-scale commercial use (annual revenue
under USD $100,000) are **permitted without charge**. Production use
beyond that threshold, or offering `fuzzeval` as a managed service,
requires a commercial license — contact
**kiran@modelbrew.ai**. Each release converts to Apache 2.0
four years after publication. Full terms in [LICENSE](LICENSE).

## API Keys (BYO)

`fuzzeval`'s judge modules call LLM vendors only when **you** invoke
them with credentials. **Your keys stay on your machine.** Keys are
read from environment variables only (`GEMINI_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`), never
persisted by `fuzzeval`, and never transmitted to any endpoint other
than your declared vendor API. See [SECURITY.md](../../SECURITY.md)
for the full posture and [PRIVACY.md](../../PRIVACY.md) for the
no-telemetry statement.

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#install)

## Status

**v0.1.0** — released alongside `topofuzz` v0.5.0rc1 (2026-05-13).
Public API surface frozen for the 0.1.x line per
[BLUEPRINT_v5plus.md §17a](https://github.com/kirannayudu/topofuzz/blob/main/BLUEPRINT_v5plus.md).
Pinned upper bound: any consumer should depend on `fuzzeval>=0.1,<0.2`.

## Install

```bash
# Base install — verify-path safe (cryptography, rfc8785, pydantic).
pip install "fuzzeval==0.2.0"

# With statistics extras (numpy + scipy for cluster bootstrap).
pip install "fuzzeval[stats]==0.2.0"

# With on-disk cache support (diskcache for LLMCache + JudgeCache).
pip install "fuzzeval[cache]==0.2.0"

# All extras.
pip install "fuzzeval[stats,cache]==0.2.0"
```

## Public API

| Module | Purpose |
|---|---|
| `fuzzeval.attestation` | DSSE v1.0.0 envelope build + verify. Byte-exact against `secure-systems-lab/dsse` reference vectors. |
| `fuzzeval.canonical`   | RFC 8785 JCS canonicalization + stable `canonical_hash` round-trip. |
| `fuzzeval.judge`       | `SingleJudge` (single LLM call) and `JuryJudge` (3-vendor opt-in). UNPARSEABLE fallback, `judge_failure_rate`, `judge_independence`. |
| `fuzzeval.stats`       | Wilson CI, percentile bootstrap, two-level paired cluster bootstrap (the load-bearing piece for `variance_across_orderings` CIs). |
| `fuzzeval.cache`       | Pickle-safe `LLMCache` + `JudgeCache` with strict type guards (`ContinuousScore`/`BinaryScore` or `bytes`/`bytearray` only). |
| `fuzzeval.costest`     | Cold + warm-cache cost dry-run estimator with confidence bands. |
| `fuzzeval.keys`        | Ed25519 dev-key generation + load via `cryptography`; chmod 0o600 on POSIX. |

## Use from outside topofuzz

`fuzzeval` is intentionally separable. If you build your own
LLM-pipeline evaluator and want DSSE-signed bundles + a 3-vendor judge
without rewriting the crypto, depend on `fuzzeval` directly:

```python
from fuzzeval.judge import SingleJudge, JuryJudge
from fuzzeval.attestation import dsse_sign, dsse_verify
from fuzzeval.stats import wilson_ci, percentile_cluster_bootstrap
```

See
[BLUEPRINT_v5plus.md §17a](https://github.com/kirannayudu/topofuzz/blob/main/BLUEPRINT_v5plus.md)
for the public-API contract.

## License (full terms)

See the **License** section at the top of this README for the summary,
and [LICENSE](LICENSE) for the full text. The license file is also
bundled into the wheel under `dist-info/licenses/LICENSE` per PEP 639.

## See also

- [`topofuzz`](https://github.com/kirannayudu/topofuzz/blob/main/packages/topofuzz/README.md)
  — the LangGraph topology fuzzer that consumes `fuzzeval`.
- [`SECURITY.md`](https://github.com/kirannayudu/topofuzz/blob/main/SECURITY.md)
  — threat model + 30-line standalone DSSE verifier reference (stdlib +
  `cryptography` only).
- [`CHANGELOG.md`](https://github.com/kirannayudu/topofuzz/blob/main/packages/fuzzeval/CHANGELOG.md)
  — release history.
