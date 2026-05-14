# Changelog — topofuzz

All notable changes to the `topofuzz` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases prior to `0.5.0rc1` were development drops on the `0.5.0.dev0`
placeholder version and are not separately enumerated.

## [Unreleased]

### Added

- **Flip-composition donut + per-eval variance sparkline** (Weekend 7
  step 18.5b, commit `9c722ec`). Two new hand-rolled SVG panels in
  the HTML report per the Variant D editorial-condensed sketch
  (`.planning/sketches/001-report-redesign/`). The §14 donut renders
  the signed `verdict_flip_rate.point` from `report.json` (donut
  centre) and decomposes it into approve→deny vs deny→approve
  splits computed live from `run.sqlite` (legend), with
  drift-detection that surfaces a stale-mirror warning when the
  signed and live numbers disagree by more than 0.005. The §9
  sparkline plots `np.nanvar` per eval sorted descending, accent-
  coloured at-or-above the §10a worst-case-candidate gate (sourced
  from `topofuzz.metrics.worstcase.DEFAULT_VARIANCE_GATE`, inclusive
  `>=` to match the promoter exactly). Both sections wrapped in
  test-extractable HTML sentinel comments; H2 suppressed entirely
  on disabled state to avoid a question-shaped void on pre-attest
  runs. Two-pass dual audit across six dimensions (code, RT, legal,
  blueprint+flow, UX/UI, cross-step) with all MAJOR + MINOR
  findings fixed; deferred polish items tracked in `BACKLOG.md`.
- **Variant D editorial-condensed report re-skin + pre-launch
  hardening** (Weekend 7 step 18.5a, commits `4ca51bd` + `53af646`
  + `ac28ac7` + `a44c56a`). Warm cream + serif editorial voice with
  2-col metadata grid, 4-stat verdict box, accessibility (aria-label
  + tabindex + focus ring) on tooltip terms. Pre-launch hardening:
  CSP meta + Referrer-Policy `no-referrer` on every report; LLM
  call timeout 60s default + 3600s ceiling via
  `TOPOFUZZ_LLM_TIMEOUT_SECS`; log-discipline tightening (no PII
  in `%s` args); CLI banner on `topofuzz run`. Legal: copyright
  credit to Kiran Nayudu (personal); NEW `NOTICE` files for DSSE +
  Sigstore Rekor schema + Acklam attribution.
- **`--rekor` opt-in for Sigstore Public Good Instance** (Weekend 6
  step 18, commit `4d9139d`). Stdlib-urllib-only Rekor client (no
  `sigstore` PyPI dep). Bounded synchronous retry (~30s ceiling, 3
  attempts). `[REKOR-DEFERRED]` banner on race-window failures.
  Live-validated against `rekor.sigstore.dev` (uuid `108e91…56f`,
  logIndex `1527092975`).
- **§17 natural-language summary panel** (Weekend 6 step 17, commit
  `25f3d79`). Prose verdict ("Your topology is **stable** /
  **unstable**: X% of answers flipped...") sourced from the signed
  `report.json` headline. Verdict-tiered styling
  (`nls-stable` / `nls-unstable` / `nls-unavailable`).
- **§10a worst-case viewer** (Weekend 6 step 16). Side-by-side
  variant cards showing the largest-score-gap flipping pair per
  eval, with per-node trace excerpts. Top-3 ranked by binary-verdict
  flip count, gated by per-eval variance ≥ 0.1.
- **Per-agent influence score (§6a)** (Weekend 5 step 15). New
  `topofuzz.metrics.compute_influence` computes
  `P(answer flips | A moved) − P(answer flips | A in same slot)`
  for each agent across the variant×eval grid, with a 95% percentile
  bootstrap CI over evals (B=1000, seed-pinned to 0 in the
  attestation path for byte-stable signed reports). Signed values
  surfaced — negative = stabilising — no clamping per the step-15
  kickoff decision. Ranked descending; degenerate rows
  (agent <2 variants, no moved pairs, no same-slot pairs) carry a
  `degenerate_reason` and sink to the bottom in node_id lex order.
- **`attestation.json` findings field**. The DSSE-signed payload
  gains a `findings` dict (alongside `headline`); the first occupant
  is `findings.per_agent_influence` — a ranked list of
  `{node_id, influence, ci_lower, ci_upper, degenerate}` entries
  (plus `degenerate_reason` when applicable). Old runs without
  `variant_orderings` in their manifest emit `findings={}`.
- **HTML report — per-agent influence panel**. New `<h2>` section
  in `report.html` renders a horizontal bar chart (matplotlib, error
  bars = 95% CI, blue = driver, green = stabilising) plus a textual
  rows table; placeholder SVG when influence is not computable.
  Capped at 32 rows to bound HTML footprint.
- **Manifest carries `variant_orderings`**. New parallel field
  alongside `variant_canonical_hashes` — a list of per-variant
  node_id sequences in slot order — so `topofuzz report` and
  `emit_run_attestation` can recompute influence post-hoc without
  the live Topology objects. Older `manifest.json` files without
  the field degrade gracefully (no crash; empty `findings`).
- **`topofuzz scan FIXTURE_PATH` static risk badge** (Weekend 5 step 14,
  BLUEPRINT §5a). Zero-LLM-cost rule-based scanner walks the Topology
  IR and emits a `HIGH` / `MEDIUM` / `LOW` badge plus matched-patterns
  + rationale. Four base rules from the §5a table (linear-3+-no-agg
  HIGH, linear-2-no-agg MEDIUM, terminal-aggregator LOW, mid-flow
  router MEDIUM) and two +1 RISK STEP promotions (all-share-model,
  temperature>0). Human banner → stderr, JSON → stdout (parse with
  `topofuzz scan ... | jq`). Exit 0 regardless of badge per DoD #11;
  exit 2 on bad fixture, exit 64 on out-of-scope LangGraph construct.

### Fixed

- N/A — scan was previously a stub raising a "not yet implemented" message.

## [0.5.0rc1] — 2026-05-13

First public release candidate of the v5+ build. Soft-launch to PyPI;
graduation to `0.5.0` follows smoke-test feedback per
`BLUEPRINT_v5plus.md` §14 Weekend 7.

### Added

- **Topology IR + LangGraph adapter** (Weekend 1 step 3). Pydantic
  `Topology` / `Node` / `Edge` models with JCS canonical-hash round-trip;
  walks a compiled `langgraph.graph.StateGraph` into the IR, with
  explicit "unsupported graph construct" errors for in-scope/out-of-scope
  features per BLUEPRINT §5.
- **Variant generators** (Weekend 1 step 4). `order_swap`, `role_swap`,
  `decoding_param_swap` per BLUEPRINT §6; deterministic under `--rng-seed`.
- **Async runner** with bounded semaphore + `fuzzeval.cache.LLMCache`
  integration (Weekend 2 step 5). JSONL trace + `run.sqlite` schema.
- **CLI orchestrator** (`topofuzz run`, Weekend 2 step 6). Reserved-flag
  rejection (`--conformance`, `--signoff`, `--judge-pool`, `--sandbox`,
  `--key-uri`, `--kms`, `--allow-side-effects`, `--tenant-id`,
  `--per-cell-entropy`) raises a v1.0-reserved error today so user habits
  do not form around squatting.
- **Single judge** (`SingleJudge`, Weekend 2 step 7). One Claude/Gemini
  call per `(variant, eval_row)` cell with `fuzzeval.cache.JudgeCache`
  + rubric template (`prompts/lending_judge.txt`).
- **Statistics** — Wilson CI, percentile bootstrap, two-level paired
  **cluster bootstrap** (`fuzzeval.stats`, Weekend 2 step 8).
- **Metrics** — `variance_across_orderings` with cluster-bootstrap CI,
  ECR (Effective Cell Rank) with degenerate-cell handling, Cohen's κ,
  balanced accuracy, `collapse_flag`, status-badge dispatch (Weekend 2 step 9).
- **DSSE signing** (Weekend 3 step 10a). DSSE v1.0.0 PAE byte-exact
  implementation in `fuzzeval.attestation`; Ed25519 keypair via
  `topofuzz keygen` writing `~/.topofuzz/dev.pem` (chmod 0o600 POSIX,
  NTFS ACLs on Windows); `topofuzz verify` in-process round-trip;
  tampering test.
- **Standalone verifier proof** (Weekend 3 step 10b). `pip install
  topofuzz` (no extras) installs only `cryptography + rfc8785 + pydantic
  + typer + rich + fuzzeval`; the `topofuzz verify` codepath is
  regression-guarded by an import-guard test that forbids `langgraph`,
  `litellm`, `anthropic`, `openai`, `google-genai`, `matplotlib`, and
  `jinja2`. `SECURITY.md` ships a ~30-line reference verifier that
  depends only on stdlib + `cryptography`, tested against a packaged
  signed fixture.
- **HTML report** (`topofuzz report`, Weekend 3 step 11). Jinja template
  + two inlined matplotlib SVGs (per-eval variance + score matrix);
  `[DEV-KEY]`, `[JURY]`, and degraded-status banners wired per
  BLUEPRINT §A; opens via `file://` URL on stdout.
- **DSSE test-vector CI gate** (Weekend 4 step 12a). Forked
  `secure-systems-lab/dsse` v1.0.0 reference vectors into
  `packages/fuzzeval/tests/test-vectors/dsse/`; CI byte-compare gate
  pins the PAE implementation against upstream.
- **Cross-platform path tests** (Weekend 4 step 12a). `~/.topofuzz/dev.pem`
  resolution + Windows-drive-letter handling in `loaders._split_spec`.
- **Jury-of-3 opt-in** (`--jury-of-3`, Weekend 4 step 12b). `JuryJudge`
  in `fuzzeval.judge` with `threading.Lock`-guarded accounting,
  vendor-name normalization, duplicate-model rejection, mean-of-parseable
  aggregation + UNPARSEABLE fallback + cumulative `judge_failure_rate`
  + `judge_independence.vendor_overlap`. CLI wiring constructs the jury
  past the prior bad-usage gate; stderr hints surface missing env keys,
  cache interaction, and the ~3x cost. Soft-DoD integration test
  (BLUEPRINT line 445) covered against `run_variant`.
- **BLUEPRINT cross-check polish** (step 12b polish): §8 5% sentinel
  gate on judge failure rate; `[JURY]` label propagated through
  `judge_protocol`; CLI label clarity; dead-link cleanup; preflight cost
  estimator annotates ~3x for jury.

### Packaging

- `topofuzz[full]` pulls LangGraph + LiteLLM + Anthropic + OpenAI +
  google-genai + matplotlib + jinja2 + keyring + tenacity + structlog +
  `fuzzeval[stats,cache]`. Roughly 400 MB.
- `topofuzz[verify]` ships **empty** in v0.5 — the base install
  (`pip install topofuzz`) is already verify-safe. The extras name is
  reserved for forward-compat and documentation.
- Per-package `LICENSE` bundled into the wheel via `license-files =
  ["LICENSE"]` (PEP 639). Build backend pinned `hatchling>=1.27` to
  guarantee finalized-spec license-files semantics.
- Python 3.11 / 3.12 / 3.13 / 3.14 declared as supported; macOS, Linux,
  and Windows declared in classifiers.

### CI gates (v0.5.0rc1 release-engineering)

- Cross-platform install smoke (`verify-install-smoke`) — macOS + Linux
  + Windows × Python 3.11 + Python 3.13 — installs the verify-only path
  from local source, asserts none of the 9 forbidden modules
  (`langgraph`, `litellm`, `anthropic`, `openai`, `google.genai`,
  `matplotlib`, `jinja2`, `numpy`, `scipy`) made it onto `sys.path`,
  runs `topofuzz verify` against the packaged signed fixture, and
  separately confirms a tampered-payload envelope is rejected with a
  nonzero exit code.
- Full-extras install smoke (`full-install-smoke`) — macOS + Linux +
  Windows × Python 3.13 — installs `topofuzz[full]` from local source
  and confirms the full-stack imports (`topofuzz.runner`,
  `topofuzz.report`, `langgraph`, `litellm`, `anthropic`, `openai`,
  `matplotlib`, `jinja2`) all succeed.
- Release-fixture pubkey-pin (`test_release_fixture_pins.py`) — pins
  the sha256 of `keys/dev.pub.pem` so any accidental re-run of
  `_regen.py` between v0.5.0rc1 and v0.5.0 GA trips the test.
- Release dry-run workflow (`release.yml`) — tag-triggered (`v[0-9]*`,
  `fuzzeval-v[0-9]*`), builds wheels + sdists for both packages, runs
  `twine check`. PyPI upload steps are committed-but-commented; they
  enable trusted-publishers OIDC after the first manual rc1 push.

### Known limitations (v0.5.0rc1)

- **Jury real-keys integration test deferred to v0.5.1.** v0.5.0rc1
  ships stub-only jury unit tests + the BLUEPRINT-line-445 soft-DoD
  integration test. The end-to-end `topofuzz run --jury-of-3 --n 10`
  against three real LLM vendors is a v0.5.1 deliverable. Per
  `BLUEPRINT_v5plus.md` line 445 this is explicitly soft — known issues
  in jury infra do not block ship.
- **`topofuzz scan` is a stub at the v0.5.0rc1 tag.** The static risk
  badge (BLUEPRINT §5a, step 14) lands post-rc1 on `main` and graduates
  to v0.5.0; see the `[Unreleased]` section above.
- **Worst-case viewer is a stub.** The §10a side-by-side trace excerpts
  (step 16) land in Weekend 5; the rendered report includes the "no
  instability detected" fallback path.
- **Per-agent influence score is a stub.** §6a (step 15) lands in
  Weekend 5; the score matrix in the report is variance-only.
- **`--rekor` (Sigstore transparency log) is reserved.** Step 18, lands
  Weekend 6 — the flag exists but currently raises.
- **DEV-KEY only.** Attestations are signed with an unencrypted PEM dev
  key. Age-encrypted PEM via `keyring` lands in v0.6; KMS in v1.0. See
  [`SECURITY.md`](https://github.com/kirannayudu/topofuzz/blob/main/SECURITY.md)
  for the threat model.

[0.5.0rc1]: https://github.com/kirannayudu/topofuzz/releases/tag/v0.5.0rc1
