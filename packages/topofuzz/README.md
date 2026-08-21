# topofuzz™

**Topology Safety Fuzzer for multi-agent LLM systems.**

If you swap two agents in your LangGraph pipeline and the final decision
flips, your topology is brittle. `topofuzz` finds those flips and prints
the receipt.

## License

Topofuzz is licensed under the Business Source License 1.1 (BSL).
Non-production use, internal development, academic research, and
small-scale commercial use (annual revenue under USD $100,000 from any
use of Topofuzz) are **permitted without charge**. Production use
beyond that threshold, or offering Topofuzz as a managed service,
requires a commercial license — contact **kiran@modelbrew.ai**.
Each release converts to Apache 2.0 four years after publication. Full
terms in [LICENSE](LICENSE).

## API Keys (BYO)

Topofuzz runs against the LLM vendor of your choice. **Your keys stay
on your machine.** Topofuzz reads keys from environment variables only
(`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`DEEPSEEK_API_KEY`), never persists them, and never transmits anything
other than your declared vendor API calls. As a convenience, the CLI
fills *missing* vendor API keys — only the four variables above, never
arbitrary environment variables — from a `.env` file in the current
directory (real environment variables always win; set
`TOPOFUZZ_NO_DOTENV=1` to disable). Run `topofuzz keys` to pre-flight
which keys are visible. See
[SECURITY.md](../../SECURITY.md) for the full posture and
[PRIVACY.md](../../PRIVACY.md) for the no-telemetry statement.

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#install)

## Status

**v1.0.0.** Single-judge by default (Gemini); opt-in jury-of-3
(Anthropic + OpenAI + Gemini) lives behind `--jury-of-3` and is wired
end-to-end through the CLI, manifest, runner, and report with
jury-aware VOO and Wilson 95% CIs.

DSSE-signed run bundles, byte-exact against the
[`secure-systems-lab/dsse`](https://github.com/secure-systems-lab/dsse)
v1.0.0 reference vectors. Verify-only install path (`pip install
topofuzz`, no LangGraph, no LLM SDKs, ~15 MB) is the design contract,
not an afterthought.

## Disclaimers

topofuzz is **research and engineering software**, distributed **with no
warranty** under the terms of the [LICENSE](LICENSE) file at the root of
this package.

- **Not a certification.** topofuzz measures one specific property —
  how much a multi-agent system's answers change when you shuffle the
  agent order. It does not certify safety, security, alignment,
  correctness, or fitness for any purpose, and makes no claim of
  compliance with HIPAA, GDPR, SOC 2, FedRAMP, FINRA, the EU AI Act,
  or any other regulatory regime.
- **No PII guidance.** Reports embed your eval inputs verbatim and
  LLM responses verbatim. If your eval set contains personal,
  confidential, or regulated data, treat `report.html`, `report.json`,
  `manifest.json`, and `run.sqlite` as carrying that data classification.
- **Topology fixtures execute Python.** `topofuzz run path/to/fx.py:g`
  calls `importlib.util.spec_from_file_location` and `exec_module` on
  the file you point at. Only run fixtures you trust; never run a
  fixture from an untrusted source (a Discord paste, an unverified PR,
  etc.).
- **Sigstore Rekor is permanent and public.** `--rekor` posts the
  DSSE envelope (containing manifest digest, judge model id, eval
  count, and timing) to a globally-replicated, immutable transparency
  log. There is no delete. Do not enable `--rekor` on runs whose
  metadata you would not publish on GitHub.
- **`[DEV-KEY]` is not a third-party-verifiable signature.** The
  developer key lives unencrypted at `~/.topofuzz/dev.pem`; anyone
  with read access to that file can forge attestations
  indistinguishably. See
  [`SECURITY.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/SECURITY.md)
  for the v0.5 threat model.

## What it does

1. Reads your compiled LangGraph and lifts it to a typed
   `Topology` (`fuzzeval.ir`).
2. Generates permuted variants per
   [BLUEPRINT §6](https://github.com/fourwheels2512/fuzzlyzer/blob/main/BLUEPRINT_v5plus.md):
   `order_swap`, `role_swap`, `decoding_param_swap`.
3. Runs every `(variant, eval_row)` cell through the LLM stack, judges
   each result against a rubric, persists traces.
4. Computes `variance_across_orderings` + cluster-bootstrap CI + ECR +
   Cohen's κ.
5. Signs the bundle (DSSE / Ed25519 / RFC 8785 JCS), emits HTML + JSON,
   prints a `file://` URL.
6. A fresh machine running `topofuzz verify` (or ~30 lines of stdlib +
   `cryptography` — see
   [SECURITY.md](https://github.com/fourwheels2512/fuzzlyzer/blob/main/SECURITY.md))
   reproduces the trust decision without re-running the LLMs.

## Install

```bash
# Full install: LangGraph adapter + LLM SDKs + plotting + signing (~400 MB).
pip install "topofuzz[full]"

# Verify-only install: cryptography + rfc8785 only — no LLM stack (~15 MB).
pip install topofuzz
```

Neither package is on PyPI yet (pre-launch); once published, install the
latest release unpinned as shown above. Do **not** pin
`topofuzz[full]==1.0.0` — the published `1.0.0` tag shipped a
mutually-unsatisfiable `fuzzeval` dependency pin in the `[full]` extra
(base required `fuzzeval>=0.2,<0.3`, `[full]` required
`fuzzeval[stats,cache]>=0.1,<0.2`) that makes that exact pinned command
fail to resolve. The fix is committed on `main` pending a `1.0.1`
release — see
[`CHANGELOG.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/packages/topofuzz/CHANGELOG.md)
`[Unreleased]` for details. If you must pin a version, wait for
`1.0.1` or check the CHANGELOG for the current resolvable pin.

Python 3.11+ on macOS, Linux, and Windows. Use a virtualenv.

## Quickstart

The demo lending fixture and 50-row eval CSV ship with the source tree.
Clone the repo to run the Quickstart:

```bash
git clone https://github.com/fourwheels2512/fuzzlyzer.git
cd topofuzz
pip install -e "packages/topofuzz[full]"
```

### First run: license acceptance

Every subcommand except `topofuzz verify` asks you to accept the BSL
1.1 terms once per machine — an interactive `y`/`N` prompt. CI jobs,
scripts, piped commands, and double-clicked launchers have no
terminal to answer that prompt, so pass `--accept-license` on the
**root** command instead; it persists the acceptance to
`~/.topofuzz/license-accepted` so it's a one-time cost per machine:

```bash
topofuzz --accept-license keygen
```

Placement matters: `--accept-license` must come **before** the
subcommand (`topofuzz --accept-license run ...`) — `topofuzz run ...
--accept-license` (after) is rejected as an unrecognized option.

Generate a dev signing key (one time per machine):

```bash
topofuzz keygen
# Wrote private key: ~/.topofuzz/dev.pem
# Wrote public key:  ~/.topofuzz/dev.pub.pem
# [DEV-KEY] this is a developer-grade key (unencrypted PEM, no KMS).
```

Zero-cost smoke test (`--simulate` stubs BOTH the agent rollouts and
the judge with deterministic responses — zero network calls, cannot
bill — proving the fixture, eval, and topology wire up correctly):

```bash
topofuzz run \
    packages/topofuzz/src/topofuzz/fixtures/lending_4role.py:build_graph \
    --eval packages/topofuzz/src/topofuzz/fixtures/lending_eval.csv \
    --n 10 --simulate --out runs --yes
```

Note: `--dry-run` alone only stubs the judge — agent rollouts still
bill real money (see `topofuzz run --help`). `--simulate` implies
`--dry-run`'s judge stub *and* stubs the agents, so it's the flag that
actually costs $0.

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
`--judge MODEL_ID`) and drop `--simulate`. The cost estimator will print
`cold=$X ± $Y / warm-cache=$X' ± $Y' (90% CI)` and prompt before
submitting jobs; `-y` skips the prompt.

## Subcommand reference

All 14 subcommands. "Cost" is whether the command makes billed LLM
calls; free commands make zero vendor API calls (or, for `ui`, zero
calls until you start a run from inside it).

**Core — fuzz + reproduce**

| Subcommand | What it does | Cost |
|---|---|---|
| `topofuzz scan`    | Static risk badge — zero-LLM-cost HIGH/MEDIUM/LOW rule-based scanner over the Topology IR (BLUEPRINT §5a). | Free |
| `topofuzz run`     | Permute topology, run all `(variant, eval_row)` cells, sign the bundle. | Bills (use `--simulate` for $0) |
| `topofuzz confirm` | Re-run screened verdict flips K times to separate the topology effect from LLM noise. | Bills |

**Attribution — which agent is responsible**

| Subcommand | What it does | Cost |
|---|---|---|
| `topofuzz ablate` | Counterfactual leave-one-out attribution on a run's flips — which node's swap closes the flip. Writes `ablation.json`. | Bills unless `--dry-run` |
| `topofuzz audit`  | Eval-axis counterfactual audit (name/zip/age/gender-swap) — which agent is biased on a disparate-impact axis. Writes `audit.json`. | Bills unless `--dry-run` |

**Artifacts — render what happened**

| Subcommand | What it does | Cost |
|---|---|---|
| `topofuzz report`  | Render HTML report with voo bar + heatmap + per-agent influence panel (§6a) and [§A](https://github.com/fourwheels2512/fuzzlyzer/blob/main/BLUEPRINT_v5plus.md) banners. | Free |
| `topofuzz inspect` | Side-by-side HTML inspector for one verdict flip, agent-by-agent. | Free |
| `topofuzz repro`   | Emit a standalone reproducer script for a verdict flip. | Free |
| `topofuzz publish` | Bundle report.html + top flipping inspect pages into a PDF (or a CEO/consistency/counterfactual card). | Free |

**Trust — sign and verify**

| Subcommand | What it does | Cost |
|---|---|---|
| `topofuzz verify` | DSSE-verify an output bundle. Works without LLM SDKs installed. | Free |
| `topofuzz keygen` | Generate a dev Ed25519 signing keypair in `~/.topofuzz/`. | Free |

**Entry points**

| Subcommand | What it does | Cost |
|---|---|---|
| `topofuzz keys`             | Pre-flight: list which vendor API keys are visible in the env. | Free |
| `topofuzz langgraph-import` | Import an existing LangGraph as a topofuzz custom workflow tile, for use in `topofuzz ui`. | Free |
| `topofuzz ui`               | Launch the topofuzz web UI at `http://localhost:<port>`. | Free to launch (runs started inside it bill) |

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
[`SECURITY.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/SECURITY.md)
ships a ~30-line `verify_dsse.py` reference verifier that depends only on
the Python stdlib + the `cryptography` package (no `rfc8785`, no
`pydantic`).

### Jury-of-3 (opt-in)

Pass `--jury-of-3` to fan each judge call out to Anthropic + OpenAI +
Gemini, take the mean of parseable scores, and report
`judge_failure_rate` + `judge_independence.vendor_overlap` separately.
The schema is INFORMATIONAL in v1.0.0; the v1.1 promotion to a HARD
gate is a gate-logic change, not a schema bump.

```bash
export ANTHROPIC_API_KEY=...  # or via your secret manager
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...

topofuzz run <fixture>:build_graph --eval <rows.csv> --n 50 --jury-of-3
```

Missing keys surface as `judge_failure` cells, not as a kickoff error
(per the
[BLUEPRINT §A](https://github.com/fourwheels2512/fuzzlyzer/blob/main/BLUEPRINT_v5plus.md)
degraded-badge contract).

### Cross-Vendor Consistency

The jury-of-N protocol assumes the N jurors are vendor-disjoint —
their failure modes correlate as little as the underlying models do.
That assumption fails when two jurors share a vendor (e.g.
`gemini-2.5-flash` + `gemini-2.5-pro` both count as Google), and the
fraction of jury-vote failures is a separate procurement signal.

`topofuzz publish --consistency-card OUT.pdf` produces a single-page
PDF derivative of the F5 publish PDF that surfaces these two findings.
Source must be a jury-of-N run (`--jury-of-3` set, or `judges` list of
≥ 2 in `report.json`); the command refuses single-judge sources rather
than emitting a misleading "cross-vendor" PDF from one juror.

The card classifies the run into one of four severities:

| Severity | Trigger |
|---|---|
| `concentrated` | `vendor_overlap` non-empty AND ≥ 2 jurors with resolved vendor data |
| `degraded` | jury-vote failure rate ≥ 10% (`_DEGRADED_THRESHOLD = 0.0995`, set just below 10% so a 0.0999 rate that displays as 10.0% under `.1f` formatting cannot slip through into the stable branch) |
| `stable` | disjoint, low failure rate, AND `distinct_vendor_count == total_jurors` (≥ 2) |
| `unknown` | partial telemetry — refuses to overstate |

Vendor names are normalised through `topofuzz.publish._normalize_vendor_name`
(strips trailing `_\d+` IR disambiguators) so `google` and `google_2`
do not double-count toward `distinct_vendor_count`. The
`self_preference_warning` field — surfaced separately when agents and
judges share vendors — is rendered as a Notes block on the card so a
procurement reader sees the same warning the source `report.json`
carries.

Severity precedence: `concentrated` outranks `degraded` because vendor
concentration is the more actionable procurement finding (the jury's
independence claim itself is conditional on disjointness, regardless
of the failure rate). The degradation signal is preserved on the
headline-stat sub-line and in the body paragraph.

```bash
topofuzz publish runs/<run-id> \
  --consistency-card consistency.pdf \
  --target "Acme Bank"
```

The `--target` flag substitutes into a "what this means for $TARGET"
paragraph. NFC normalisation + bidi-override / zero-width / homoglyph
rejection match the F6 CEO-card surface (shared via
`_validate_target_company`).

### Counterfactual Audit (F7, post-v1.0.0 on `main`)

`topofuzz audit RUN_DIR --max-cost-usd N --yes` runs per-node
disparate-impact attribution on a source run. For each
`(variant, canonical_eval, dim)` triple — where `dim` is one of
`name-swap`, `zip-swap`, `age-swap`, `gender-swap` — it generates a
paired counterfactual eval row (Maria → Mario, 11211 → 11212, 29 → 59,
etc.), runs the variant against it, and — when the verdict flips —
walks every upstream node and ablates it back to its canonical-run
response to attribute which agent caused the flip. The buyer-actionable
answer to "which agent is biased on this input axis?"

Source-run gate: `--jury-of-3` required (SingleJudge runs are rejected
at the gate so the leave-one-out re-judgments do not inherit the
source's ~26% per-judgment verdict noise — see BLUEPRINT §9a — noise-corrected metrics — and features F4 (ablation) / F7 (counterfactual audit)).

Writes `<run_dir>/audit.json`. The PDF builder
(`topofuzz publish RUN_DIR --counterfactual-card OUT.pdf
--target "Acme Bank"`) consumes the sidecar to render the per-dim
disparate-impact + per-node attribution tables. The artifact is
**adjacent** to ECOA / FHA / NYC LL144 / OCC SR 11-7 adverse-impact
work — it produces a candidate disparate-impact signal worth a
follow-up formal audit. It is **not** a compliance determination and
makes no certification claim (see the Limitations section above —
"makes no claim of compliance with HIPAA, GDPR, SOC 2, FedRAMP, FINRA,
the EU AI Act, or any other regulatory regime"). Treat the per-agent
attribution as a scoping signal for where a compliance team should
look first, not as a finding to file.

The publish-side guard refuses to render contradictory sidecars with
three distinct error messages (cap-hit / unreadable / mixed validity),
and the headline picker gates the "agent X contributed most" branch on
positive non-terminal attribution — the all-zero subcase falls through
to an honest 3-subcase clause distinguishing "math failed on some
cells" / "math ran cleanly and produced a zero answer" / "attribution
machinery never ran."

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
  url    = {https://github.com/fourwheels2512/fuzzlyzer},
  version = {1.0.0}
}
```

## License (full terms)

See the **License** section at the top of this README for the summary,
and [LICENSE](LICENSE) for the full text. The license file is also
bundled into the wheel under `dist-info/licenses/LICENSE` per PEP 639.

## Acknowledgments

The topology-instability motivation is informed by ordering-effect
results in multi-agent LLM literature (Bajaj et al. 2026 in particular).
The DSSE attestation path is byte-exact against
[`secure-systems-lab/dsse`](https://github.com/secure-systems-lab/dsse)
v1.0.0 reference vectors.

## See also

- [`fuzzeval`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/packages/fuzzeval/README.md)
  — verify-path substrate (DSSE + judge + stats + cache + costest + key
  management).
- [`BLUEPRINT_v5plus.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/BLUEPRINT_v5plus.md)
  — full design contract.
- [`SECURITY.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/SECURITY.md)
  — threat model + standalone verifier.
- [`CHANGELOG.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/packages/topofuzz/CHANGELOG.md)
  — release history.
- [`BACKLOG.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/packages/topofuzz/BACKLOG.md)
  — polish + hardening items deferred from prior steps; the
  v0.5.1 punch list. Contributors closing items reference this
  file in commit messages (`closes BACKLOG: <description>`).
