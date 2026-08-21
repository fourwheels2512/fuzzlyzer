# topofuzz™ + fuzzeval™ monorepo

**Topology Safety Fuzzer for multi-agent LLM systems, plus the verify-path substrate that supports it.**

Source: [github.com/fourwheels2512/fuzzlyzer](https://github.com/fourwheels2512/fuzzlyzer) (BSL 1.1, source-available).

If you swap two agents in your LangGraph pipeline and the final decision flips, your topology is brittle. `topofuzz` finds those flips and prints the receipt. `fuzzeval` ships the shared infrastructure — DSSE attestation, judges (single + jury), statistics, caches, cost estimation — that `topofuzz` and other multi-agent eval tools build on.

## Packages in this monorepo

| Package | What it is | Install |
|---|---|---|
| [`packages/topofuzz`](packages/topofuzz/) | The full fuzzer: orchestrator, variants, jury-of-3, ablation, report.html, CEO PDF, repro, inspect, BYOP, langgraph-import. | `pip install topofuzz` |
| [`packages/fuzzeval`](packages/fuzzeval/) | Verify-path substrate: judges, statistics, attestation, cache, cost estimation. No LangGraph or LLM-SDK dependency at import time. | `pip install fuzzeval` (or comes bundled with `topofuzz`) |

See each package's own `README.md` for full feature lists, install instructions, and status.

## License

Both packages are licensed under the **Business Source License 1.1 (BSL)**. Non-production use, internal development, academic research, and small-scale commercial use (annual revenue under USD $100,000 from any use of the Licensed Work) are **permitted without charge**. Production use beyond that threshold, or offering either package as a managed service, requires a commercial license — contact **kiran@modelbrew.ai**. Each release converts to Apache 2.0 four years after publication. Full terms in each package's `LICENSE` file.

## API Keys (BYO)

Both packages call LLM vendors only when **you** invoke them with credentials. **Your keys stay on your machine.** Keys are read from environment variables only (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`), never persisted, and never transmitted to any endpoint other than your declared vendor API. See [`SECURITY.md`](SECURITY.md) for the full posture and [`PRIVACY.md`](PRIVACY.md) for the no-telemetry statement.

## Status

**v1.0.0** (2026-05-17). Production-stable engine, BSL-1.1 licensed. 1595-test suite passes. DSSE-signed run bundles. Single-judge by default; opt-in jury-of-3 (Anthropic + OpenAI + Gemini) wired end-to-end with jury-aware variance-of-orderings and Wilson 95% CIs. F8 Cross-Vendor Consistency card auto-leads with F4 per-agent attribution when `ablation.json` is present in the run dir.

**Post-v1.0.0 on `main` (pending v1.1.0 tag):** F7 counterfactual audit shipped — `topofuzz audit` runs per-node disparate-impact attribution on demographic input swaps (name / zip / age / gender), and `topofuzz publish --counterfactual-card` renders the buyer-facing PDF. The artifact is **adjacent** to ECOA / FHA / NYC LL144 / OCC SR 11-7 adverse-impact work — it is a candidate disparate-impact signal worth a follow-up formal audit, **not** a compliance determination (see `packages/topofuzz/README.md` Limitations). 1753-test suite. See `packages/topofuzz/CHANGELOG.md` for the [Unreleased] entry, including a `pip install topofuzz[full]==1.0.0` resolver bug (v1.0.1 patch pending).

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](packages/topofuzz/LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#)

## Repository layout

```
packages/
├── topofuzz/         the fuzzer
└── fuzzeval/         the verify-path substrate
examples/
└── gallery/          worked fixtures (lending, insurance, hiring, …)
keys/                 DSSE public keys (development)
SECURITY.md           security & threat-model posture
PRIVACY.md            no-telemetry statement
```

## Quick start

```bash
pip install topofuzz
export GEMINI_API_KEY=...
python -m topofuzz ui              # browser UI on http://localhost:8765
# or: topofuzz scan <fixture>      # CLI
```

See [`packages/topofuzz/README.md`](packages/topofuzz/README.md) for the full CLI surface and fixture authoring guide.
