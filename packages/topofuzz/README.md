# topofuzz

**Topology Safety Fuzzer for multi-agent LLM systems.**

If you swap two agents in your LangGraph pipeline and the final decision
flips, your topology is brittle. `topofuzz` finds those flips and prints
the receipt.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/kirannayudu/topofuzz/blob/main/packages/topofuzz/LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#install)

## Status

**v0.5.0rc1** — soft launch. Single-judge by default (Gemini); opt-in
jury-of-3 (Anthropic + OpenAI + Gemini) lives behind `--jury-of-3` and
ships stub-only in rc1 — real-keys integration test lands in v0.5.1.

DSSE-signed run bundles, byte-exact against the
[`secure-systems-lab/dsse`](https://github.com/secure-systems-lab/dsse)
v1.0.0 reference vectors. Verify-only install path (`pip install
topofuzz`, no LangGraph, no LLM SDKs, ~15 MB) is the design contract,
not an afterthought.

## What it does

1. Reads your compiled LangGraph and lifts it to a typed
   `Topology` (`fuzzeval.ir`).
2. Generates permuted variants per
   [BLUEPRINT §6](https://github.com/kirannayudu/topofuzz/blob/main/BLUEPRINT_v5plus.md):
   `order_swap`, `role_swap`, `decoding_param_swap`.
3. Runs every `(variant, eval_row)` cell through the LLM stack, judges
   each result against a rubric, persists traces.
4. Computes `variance_across_orderings` + cluster-bootstrap CI + ECR +
   Cohen's κ.
5. Signs the bundle (DSSE / Ed25519 / RFC 8785 JCS), emits HTML + JSON,
   prints a `file://` URL.
6. A fresh machine running `topofuzz verify` (or ~30 lines of stdlib +
   `cryptography` — see
   [SECURITY.md](https://github.com/kirannayudu/topofuzz/blob/main/SECURITY.md))
   reproduces the trust decision without re-running the LLMs.

## Install

```bash
# Full install: LangGraph adapter + LLM SDKs + plotting + signing (~400 MB).
pip install "topofuzz[full]==0.5.0rc1"

# Verify-only install: cryptography + rfc8785 only — no LLM stack (~15 MB).
pip install "topofuzz==0.5.0rc1"
```

Python 3.11+ on macOS, Linux, and Windows. Use a virtualenv.

## Quickstart

The demo lending fixture and 50-row eval CSV ship with the source tree.
Clone the repo to run the Quickstart:

```bash
git clone https://github.com/kirannayudu/topofuzz.git
cd topofuzz
pip install -e "packages/topofuzz[full]"
```

Generate a dev signing key (one time per machine):

```bash
topofuzz keygen
# Wrote private key: ~/.topofuzz/dev.pem
# Wrote public key:  ~/.topofuzz/dev.pub.pem
# [DEV-KEY] this is a developer-grade key (unencrypted PEM, no KMS).
```

Dry-run (no LLM judge calls — every cell gets `ContinuousScore(0.5)`,
useful for proving the install works without spending money):

```bash
topofuzz run \
    packages/topofuzz/src/topofuzz/fixtures/lending_4role.py:build_graph \
    --eval packages/topofuzz/src/topofuzz/fixtures/lending_eval.csv \
    --n 10 --dry-run --out runs --yes
```

The run prints a unique run directory at the end. The run id is a
UTC date + a 26-character ULID — yours will look different from this
example:

```text
Run id: 2026-05-13-01KRH4DF2T314C1X6D9HVE6QTK
Run dir: /abs/path/to/runs/2026-05-13-01KRH4DF2T314C1X6D9HVE6QTK
Status: ok
```

Substitute YOUR run id into the follow-up commands. Render the HTML
report and open the printed `file://` URL:

```bash
topofuzz report runs/2026-05-13-01KRH4DF2T314C1X6D9HVE6QTK
```

Verify the DSSE-signed bundle:

```bash
topofuzz verify runs/2026-05-13-01KRH4DF2T314C1X6D9HVE6QTK
# PASS [DEV-KEY]
```

For a real run, set `GEMINI_API_KEY` (or your judge of choice via
`--judge MODEL_ID`) and drop `--dry-run`. The cost estimator will print
`cold=$X ± $Y / warm-cache=$X' ± $Y' (90% CI)` and prompt before
submitting jobs; `-y` skips the prompt.

## Subcommand reference

| Subcommand | What it does |
|---|---|
| `topofuzz run`    | Permute topology, run all `(variant, eval_row)` cells, sign the bundle. |
| `topofuzz verify` | DSSE-verify an output bundle. Works without LLM SDKs installed. |
| `topofuzz report` | Render HTML report with voo bar + heatmap + per-agent influence panel (§6a) and [§A](https://github.com/kirannayudu/topofuzz/blob/main/BLUEPRINT_v5plus.md) banners. |
| `topofuzz keygen` | Generate a dev Ed25519 signing keypair in `~/.topofuzz/`. |
| `topofuzz scan`   | Static risk badge — zero-LLM-cost HIGH/MEDIUM/LOW rule-based scanner over the Topology IR (BLUEPRINT §5a). |

Full option lists: `topofuzz <cmd> --help`.

## What "variance across orderings" means

The headline number `variance_across_orderings` answers a single
question: if I shuffle the order in which agents in my pipeline get the
same input, **does the final answer change**? On the packaged lending
fixture, swapping `risk` ↔ `policy` flips a non-trivial number of
approve/deny calls — that is the kind of brittleness the tool surfaces.

The report also includes:

- **Per-agent influence (§6a)** — for each agent, `P(answer flips | agent moved between two variants) − P(answer flips | agent in same slot)`, with a 95% bootstrap CI per agent. Negative = stabilising. The signed `attestation.json` carries the same list under `findings.per_agent_influence`.
- **Per-eval variance** — which rows in the eval set are most order-sensitive.
- **ECR (Effective Cell Rank)** — degenerate-cell-aware effective sample size.
- **Cohen's κ + balanced accuracy** — judge-vs-ground-truth agreement.
- **`collapse_flag`** — fires when variants degenerate to constant outputs.

## Advanced

### Verify-only path (no LLM stack)

`pip install topofuzz` (base, no extras) installs only `cryptography`,
`rfc8785`, `pydantic`, `typer`, `rich`, and `fuzzeval`. No LangGraph,
no LiteLLM, no Anthropic / OpenAI / Google SDKs. `topofuzz verify` and
`topofuzz keygen` are the only subcommands guaranteed to work in this
mode; `run` / `report` / `scan` require `topofuzz[full]`. The boundary
is regression-guarded by `tests/test_import_guard.py`.

The `topofuzz[verify]` extras name is **reserved and empty** in v0.5 —
the base install is already verify-safe, so `pip install topofuzz[verify]`
and `pip install topofuzz` install the same deps. The extras name exists
for forward-compat documentation; do not interpret an empty `[verify]`
extra as a packaging bug.

For a fully standalone reproduction without installing topofuzz at all,
[`SECURITY.md`](https://github.com/kirannayudu/topofuzz/blob/main/SECURITY.md)
ships a ~30-line `verify_dsse.py` reference verifier that depends only on
the Python stdlib + the `cryptography` package (no `rfc8785`, no
`pydantic`).

### Jury-of-3 (opt-in, stub-only in v0.5.0rc1)[^jury]

Pass `--jury-of-3` to fan each judge call out to Anthropic + OpenAI +
Gemini, take the mean of parseable scores, and report
`judge_failure_rate` + `judge_independence.vendor_overlap` separately.
The schema ships in v0.5.0rc1 as INFORMATIONAL; the v1.0 promotion to a
HARD gate is a gate-logic change, not a schema bump.

```bash
export ANTHROPIC_API_KEY=...  # or via your secret manager
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...

topofuzz run <fixture>:build_graph --eval <rows.csv> --n 50 --jury-of-3
```

Missing keys surface as `judge_failure` cells, not as a kickoff error
(per the
[BLUEPRINT §A](https://github.com/kirannayudu/topofuzz/blob/main/BLUEPRINT_v5plus.md)
degraded-badge contract). The end-to-end real-keys integration test for
`topofuzz run --jury-of-3` is deferred to v0.5.1 — v0.5.0rc1 ships with
stub-only jury unit + soft-DoD coverage.

[^jury]: v0.5.0rc1 ships the jury opt-in wired end-to-end through the
    CLI, manifest, runner, and report. The deferred piece is an
    integration test that exercises three real vendors on `--n 10`.

### Reserved flag namespace

The following flags are reserved for v1.0 and raise a helpful error
today: `--conformance`, `--signoff`, `--judge-pool`, `--sandbox`,
`--key-uri`, `--kms`, `--allow-side-effects`, `--tenant-id`,
`--per-cell-entropy`. Don't build CI on them — they will repurpose.

## Citation

```bibtex
@software{nayudu2026topofuzz,
  title  = {topofuzz: Topology Safety Fuzzer for multi-agent LLM systems},
  author = {Nayudu, Kiran},
  year   = {2026},
  url    = {https://github.com/kirannayudu/topofuzz},
  version = {0.5.0rc1}
}
```

## License

Apache-2.0. See
[LICENSE](https://github.com/kirannayudu/topofuzz/blob/main/packages/topofuzz/LICENSE).
The license file is also bundled into the wheel under
`dist-info/licenses/LICENSE` per PEP 639.

## Acknowledgments

The topology-instability motivation is informed by ordering-effect
results in multi-agent LLM literature (Bajaj et al. 2026 in particular).
The DSSE attestation path is byte-exact against
[`secure-systems-lab/dsse`](https://github.com/secure-systems-lab/dsse)
v1.0.0 reference vectors.

## See also

- [`fuzzeval`](https://github.com/kirannayudu/topofuzz/blob/main/packages/fuzzeval/README.md)
  — verify-path substrate (DSSE + judge + stats + cache + costest + key
  management).
- [`BLUEPRINT_v5plus.md`](https://github.com/kirannayudu/topofuzz/blob/main/BLUEPRINT_v5plus.md)
  — full design contract.
- [`SECURITY.md`](https://github.com/kirannayudu/topofuzz/blob/main/SECURITY.md)
  — threat model + standalone verifier.
- [`CHANGELOG.md`](https://github.com/kirannayudu/topofuzz/blob/main/packages/topofuzz/CHANGELOG.md)
  — release history.
