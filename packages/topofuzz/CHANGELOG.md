# Changelog — topofuzz

All notable changes to the `topofuzz` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases prior to `0.5.0rc1` were development drops on the `0.5.0.dev0`
placeholder version and are not separately enumerated.

## [Unreleased] — pending v1.1.0 tag

Three feature/fix commits + several release-engineering fixes post-date
the `1.0.0` tag on `main` and will roll into the next release. The
1.0.0 lede's claim of "F1-F8 features all shipped" was an off-by-one —
F7 (counterfactual audit) shipped in `7948c94`, **after** the v1.0.0
cut. The 1.0.0 entry has been amended to read "F1-F6 and F8 features
shipped" to match what was actually under the tag. The git tag
`v1.0.0` (commit `f8beb04`) and any PyPI artifact published from that
tag still carry the original "F1-F8" claim and the v1.0.0 known
limitations enumerated below — readers of the on-disk CHANGELOG and
readers of the published wheel will see divergent text until v1.0.1
is cut.

### Known limitations of the v1.0.0 tag (not yet patched on PyPI)

These were present in `f8beb04` and require a v1.0.1 patch release to
clear from any installed wheel. The fixes themselves are committed on
`main` in `[Unreleased]` below.

- **`pip install topofuzz[full]==1.0.0` fails to resolve.** The base
  `dependencies` pinned `fuzzeval>=0.2,<0.3` while the `[full]` extra
  pinned `fuzzeval[stats,cache]>=0.1,<0.2` — mutually unsatisfiable.
  Anyone running `pip install topofuzz[full]==1.0.0` against the
  published wheel sees pip resolver fail with a `ResolutionImpossible`
  error. Fixed in `[Unreleased]` by aligning the `[full]` pin to
  `>=0.2,<0.3`.
- **F8 Stage A all-zero attribution contradiction.** When `ablation.json`
  was partial / cap-hit, the F8 consistency-card headline could read
  "agent X contributed most" alongside a "0.0%" stat panel — the same
  self-contradiction shape `45683cb` closed for F7. v1.0.0's
  `_extract_attribution_rows` did not gate on `attribution_pct > 0`.
  Fixed in `[Unreleased]` (`45683cb`).

### Dependencies (lockfile regen)

- **`uv.lock` regenerated.** Resolves `fuzzeval` from `0.1.0` (stale
  in the v1.0.0 lockfile carry-forward) to `0.2.0` matching the bumped
  package, AND fills in 11 missing optional-extras entries that the
  `[ui]` and `[pdf]` extras have declared since their introduction:
  `fastapi`, `greenlet`, `httptools`, `playwright`, `pyee`, `pypdf`,
  `python-multipart`, `starlette`, `uvicorn`, `uvloop`, `watchfiles`.
  All are direct or transitive deps of declared optional extras; no
  new direct deps were added to `pyproject.toml`. Lockfile diff
  +338/-2 lines. Procurement reviewers running SBOM checks should
  re-pull the lockfile after a `topofuzz` reinstall.

### Fixed (release engineering, post-tag)

- **Mutually-unsatisfiable `fuzzeval` constraint in `[full]` extra.**
  `packages/topofuzz/pyproject.toml` line 72 was `>=0.1,<0.2` while
  base line 29 was `>=0.2,<0.3`. Aligned to `>=0.2,<0.3` so
  `pip install topofuzz[full]` resolves. Surfaced by RT-specialist
  audit 2026-05-17. v1.0.1 patch release must rebuild + republish the
  wheel.
- **Test suite no longer aborts at collection on the CI install.** Five
  `tests/ui/` modules imported `topofuzz.ui` (which loads FastAPI
  eagerly) with no guard, so a `topofuzz[full]`-only environment — the
  exact one `ci.yml` syncs — hit 4 collection errors and ran zero
  tests. The root dev dependency group now mirrors the `ui` extra plus
  `pypdf`, so a bare `uv sync && uv run pytest` runs the UI suite, and
  the modules `importorskip("fastapi")` / `importorskip("pypdf")` so
  an install without those extras skips instead of erroring.
- **Two host-dependent test assertions.**
  `test_resolve_eval_path_absolute_no_duplicate_tries` counted the path
  across the whole error message, which also echoes it via `eval=…!r`
  — a double match on POSIX (Windows' doubled backslashes hid it); it
  now counts only the `tried` list. The counterfactual-card legal-caveat
  test collapses whitespace before matching, because where Chromium
  wraps the paragraph depends on the fonts installed on the host.
- **Install docs matched to the extras.** The root README Quick start
  told users to `pip install topofuzz` and run `topofuzz ui`, which
  exits with a missing-dependency error: the UI needs the `ui` extra
  and runs need `full`. Both READMEs now install from a clone with
  `-e packages/fuzzeval -e "packages/topofuzz[full,ui]"` (neither
  package is on PyPI yet, so plain pip cannot resolve `fuzzeval`), and
  the Quickstart's `cd topofuzz` is corrected to `cd fuzzlyzer`.
- **Web UI: "Insurance claim triage" failed on the first click of Run.**
  The preset shipped two starter cases but the run form requires three,
  so the untouched tile returned a 400. A third starter is added and a
  test pins every built-in workflow to at least three.
- **Web UI: custom workflow form rejected a normally filled-in chain.**
  The verdict-agent select latched onto the first agent typed (the only
  option at that moment) instead of defaulting to the last, which the
  linear-chain validator requires. It now defaults to the last agent
  unless the user picks one by hand.
- **Web UI: cost estimate overstated and out of sync.** The estimate
  priced the two blank padding cards (5 cases instead of 3), and the
  note beside "Run the test" kept the page-load figure after the
  estimate box updated. Both now count filled cases only and refresh
  together, including when cases are edited, added or removed.
- **Web UI: htmx is vendored.** `base.html` loaded htmx 1.9.12 from
  unpkg.com, so the UI's interactive parts failed offline or behind a
  restrictive firewall and every page load made a third-party request.
  The identical file (same SRI hash) now ships in `ui/static/`;
  `unpkg.com` is removed from the SECURITY.md egress list.
- **CLI output polish.** `scan` printed the badge twice
  ("HIGH RISK — HIGH risk — …"); `--max-cost-usd` help showed a literal
  `80%%`; `publish --help` and `report --help` lost the `[pdf]` /
  `[full]` extras names to Rich markup; and `run --simulate` logged a
  false "provider served a different model" warning per model because
  the offline stub echoed the request as a served-model id.
- **Web UI: results and history showed the raw cost meter.** The setup
  estimate and live cap are converted to real dollars (the meter runs
  2.75x hot), but the post-run total from `report.json` was shown
  unconverted, so a "$0.24" estimate was followed by a "$0.63" bill.
  Verified on a real Gemini run: 6 cells metered at $0.6257 price out
  at $0.2274 from their token counts. The total is now converted too.
- **Web UI: History was lost when the app restarted.** Runs were held
  in memory only. Each run now writes a small `ui_run.json` record and
  the registry reloads finished runs at startup; a record left at
  "running" by a closed app is shown as failed, not as an active run.
- **Flip inspector called every variant "Reordered".** On a linear
  chain there is one valid ordering, so the variants are role swaps and
  temperature lifts; both rendered two identical agent chains under a
  "two different orderings" headline. The inspector now says what the
  variant changed (re-ordered / roles swapped, with the moved slots
  marked / same order with one model setting changed). The web UI's
  "See an example that changed" link prefers an agents-swapped flip and
  accepts `?eval=<id>`; the CLI default selection is unchanged. The UI
  results headline no longer attributes every flip to a re-ordering.
- **CEO card misdescribed single-judge runs.** It stated "measured
  under a jury-of-3, not a single judge" regardless of the run, printed
  `Evals: —` (manifest key is `n_eval_rows`) and "(eval prompt
  unavailable)" (run.sqlite has no `evals` table), and badged every
  flip "Re-ordered". All four now reflect the source run.
- **`topofuzz confirm` left the run unverifiable and its results
  unused.** The pass appends repeats to `run.sqlite` and writes
  `confirm.json`, but nothing re-emitted the attestation: `topofuzz
  verify` then FAILed on `run_sqlite_sha256_match`, and the
  noise-corrected headline metrics stayed null in `report.json`.
  `confirm` now re-signs the run from the existing headline inputs when
  it finishes (and says so, or says why it could not).

### Added

- **F7 — `topofuzz audit` per-node counterfactual attribution.**
  Commit `7948c94`. For each `(variant, canonical_eval, dim)` triple,
  generates a paired counterfactual eval row (Maria → Mario for
  name-swap; age-swap, zip-swap, gender-swap likewise), runs the
  variant against it, and — when the verdict flips — walks every
  upstream node and ablates it back to its canonical-run response to
  attribute which agent caused the disparate-impact flip. The
  buyer-actionable answer to "which agent is biased on this input
  axis?". Source-run gate: `--jury-of-3` required (SingleJudge runs
  rejected at the gate so the leave-one-out re-judgments do not
  inherit the source's ~26% per-judgment verdict noise). Writes
  `<run_dir>/audit.json`; consumed by `topofuzz publish
  --counterfactual-card`.
- **Empty-token regression fixture** at
  `examples/gallery/langgraph-lending-4role/eval-empty-tokens.csv`
  (commit `45683cb`) — every row yields zero swaps across all four
  dims, pinning the `n_canonical_pairs == 0` branch of the F7 publish
  headline picker. The pre-existing `eval.csv` did not exercise this
  branch because rows contained "NN years old" phrases that the
  age-swap generator matched.

### Fixed

- **F7 counter-mismatch contradiction.** Commit `a348f6a`. Phase-1
  flip counts now authoritatively drive `by_dim` /
  `by_swap_pair.flipping_pairs` instead of being re-derived from
  Phase-2 leave-one-out `results[]`. Before this fix, when Phase 2
  was cap-hit before any triple completed (e.g. `$3` cap on `4 dims
  × 8 rows × jury-of-3`), `results[]` was empty so per-dim and
  per-swap flip counts collapsed to 0 while the top-level scalar
  `n_flipping_pairs` remained at 8 — and the buyer-PDF rendered the
  contradiction silently. New maps
  `n_flipping_pairs_by_swap_pair` + `n_flipping_pairs_by_dim` are
  incremented in Phase 1 alongside the scalar (same if-branch,
  mathematically lockstep) and threaded through `_aggregate`.
  `by_node` attribution remains Phase-2-derived (the F4 leave-one-out
  math is untouched). Publish-side guard at
  `_build_counterfactual_card_context` adds three distinct
  `PublishError` diagnostics: **cap-hit** (`n > 0`, well-formed
  entries, sum == 0 → "rerun with higher cap"), **fully malformed
  sidecar** (every entry missing/non-int/negative/non-dict →
  "sidecar unreadable — DO NOT raise cap"), **mixed validity**
  (some valid positives + stray malformed → "refusing to publish
  partial"). +7 regression tests; two passes of dual audit folded
  (pass-1: 1 MAJOR + 1 MINOR; pass-2: 2 MEDIUM).
- **All-zero attribution contradiction (F7 + F8 Stage A).** Commit
  `45683cb`. A 2026-05-17 partial-Phase-2 lending bias audit
  rendered a buyer-PDF whose title read "The intake agent
  contributed most to demographic-counterfactual verdict flips"
  alongside a "0.0%" stat panel — a self-contradiction the existing
  counter-mismatch guard did NOT catch because the flip counts
  themselves WERE internally consistent. The contradiction lived in
  the **attribution counters**: every Phase-2 leave-one-out re-judge
  had collapsed to `Score: NONE` on a HOLD response, so
  `by_node[*].flips_closed = 0` everywhere, but the headline picker
  still named the alphabetically-first non-terminal node.
  `_format_counterfactual_headline` now gates the "agent X
  contributed most" branch on `top_agent.attribution_pct > 0 AND
  any_nonterminal_attribution`; the all-zero subcase falls through
  to a 3-subcase honest attribution clause (math failed on some
  cells / math ran cleanly and produced a zero answer / attribution
  machinery never ran). F8 consistency card falls back to
  vendor-concentration when F4 attribution is all-zero, and surfaces
  a one-sentence `vendor_independence_note` breadcrumb so the
  buyer-PDF does not pretend `ablation.json` doesn't exist. The
  inconclusive-clause wording is honest about the union of
  non-meaningful kinds. `_extract_swap_pair_rows` and
  `_extract_attribution_rows` drop 0%-attribution contributors from
  drilldown cells. `PublishError` on malformed
  `n_ablation_points / n_ablation_points_meaningful` types (string,
  negative, etc.) and on `napm > nap`. +15 regression tests
  (`test_counterfactual.py` +1, `test_publish.py` +3,
  `test_publish_counterfactual_card.py` +11; the `45683cb` commit
  body says "+14" — off by one, reconciled here); two
  passes of dual audit folded (pass-1: 1 BLOCKER + 1 MAJOR + 1
  MAJOR + 1 MEDIUM + 2 MINOR; pass-2: 1 MAJOR + 1 MAJOR + 1 MAJOR +
  1 MEDIUM).

## [1.0.0] — 2026-05-17

First production-stable release. Engineering surface (DSSE attestation
byte-exact against the secure-systems-lab reference, dual-audit-every-
step discipline, F1-F6 and F8 features shipped) earns the 1.0 tag.
(The v1.0.0 commit body cites "1595 tests"; the canonical pytest
suite at HEAD `f8beb04` actually collected closer to 1700 — the 1595
figure was a stale carry-forward from an earlier session. Reported
here so the published number is not load-bearing. Current `main`
collects 1755 tests; 1753 pass + 2 platform-skip on Windows.) The 0.x line was a pre-release iteration; on this release
the Trove classifier moves to `Development Status :: 5 -
Production/Stable` and SECURITY.md `Supported Versions` table is
rebuilt around 1.0.x.

### Added

- **F8 — `topofuzz publish --consistency-card OUT.pdf`.** 1-page
  procurement-flavoured derivative of the F5 PDF / F6 CEO card. Pure
  replay over `report.json` — no LLM spend, no `run.sqlite` required.
  Auto-resolves `run_dir/ablation.json`: when present the headline
  leads with **F4 per-agent attribution** (e.g., "The risk role
  closed 7 of 11 ablation-eligible verdict flips on lending-4role
  (63.6%)"), and the vendor-concentration narrative demotes to a
  "Vendor independence check" notes block. When absent, the
  vendor-concentration headline is preserved byte-for-byte for
  back-compat. New severity `attribution` (neutral blue chip) joins
  concentrated/degraded/stable/unknown. Dual-audit 2 passes (45
  tests for F8 + Stage A; 1595 pass total). Leave-one-out overlap
  semantics disclosed in the body paragraph so the demo's
  63.6 + 60 + 54.5 = 178% does not pattern-match to "numbers don't
  add up." `ablation.partial: true` surfaces in lede + chip sub-line
  so a CRO who opens the source file does not catch a silent drop.

### Changed

- **Version bump 0.6.0 → 1.0.0.** Engineering rigor (dual-audit-every-
  step, DSSE byte-exact, F4 attribution validated against real
  Gemini/DeepSeek spend, F3 jury-aware VOO, F5+F6+F8 publish surface)
  earns the 1.0 tag. KMS-backed signing, Sigstore production keys,
  and the GH Pages activation are explicit **v1.1** milestones — the
  v1.0 `[DEV-KEY]` watermark on every `topofuzz verify` output is
  honest disclosure that the local-signed attestation is not
  production-grade. `Development Status` classifier moves Alpha → 5 -
  Production/Stable. `SECURITY.md` `Supported Versions` table moves
  0.5.x → 1.0.x; pre-1.0 lines marked unsupported.
- **Contact email cleanup.** All `-TBD@modelbrew.ai` placeholders
  (licensing / security / privacy) collapse to a single working alias
  `kiran@modelbrew.ai` across `SECURITY.md`, `PRIVACY.md`, both
  `LICENSE` files, `license_check.py`, and the test that pins its
  text. PGP fingerprint `[TBD]` now reads "Available on request via
  `kiran@modelbrew.ai`".
- **Root LICENSE removed.** Root `/LICENSE` was an Apache 2.0 orphan
  inherited from the deleted pramana-eval package; deleting it
  resolves the license-declaration contradiction with the BUSL-1.1
  `packages/*/LICENSE` files. Both packages remain BUSL-1.1 per
  their own LICENSE files; the change is a removal of a stale file,
  not a license change.
- **License change from Apache 2.0 to Business Source License 1.1
  (BUSL-1.1), effective v1.0.0.** Non-production use, internal
  development, academic research, and small-scale commercial use
  (annual gross revenue under USD $100,000 from any use of the
  Licensed Work) remain permitted without charge. Production use
  beyond that threshold, or offering Topofuzz as a managed service,
  requires a commercial license — contact
  `kiran@modelbrew.ai`. Each release auto-converts to Apache
  2.0 four years after publication. Full terms in `LICENSE` (root and
  per-package).
  Historical 0.5.x versions (including `0.5.0rc1`) were released under
  the Apache License, Version 2.0, and those published artifacts
  remain under their original license — the license change is not
  retroactive.

### Added

- **F3 — `--jury-of-3` wired with per-vendor vote breakdown.**
  The CLI flag was already present but defaulted to the BLUEPRINT trio
  (Anthropic + OpenAI + Gemini) without a way to override the models.
  This release wires `--jury-models <m1,m2,m3>` to override the trio
  with any three LiteLLM model ids (e.g.
  `google/gemini-2.5-flash,google/gemini-2.5-pro,deepseek/deepseek-chat`).
  Duplicate model ids are accepted via the new
  `JuryJudge(allow_duplicate_models=True)` opt-in — F3 single-vendor
  plumbing-validation mode where three identical Gemini-Flash calls
  exercise the majority-vote infrastructure without cross-vendor
  independence guarantees. Per-vendor vote breakdowns now persist in
  schema v6's new `judge_results.votes_json` column (forward-compat
  with v5 readers via PRAGMA-probe fallback in `inspect._load_cell`).
  The inspect view (`topofuzz inspect`) renders a "Jury vote
  breakdown" sub-panel showing each vendor's score + verdict + own
  rationale, with an explicit "TIE → DENY" pill for deadlocked juries
  (compliance-conservative default per F3 Pass-2 dual-audit
  M-Fuzzer-2). The headline `score_value` is the **majority-vote
  scalar** (`1.0` if strict majority of binarized votes approve, else
  `0.0`) — NOT the arithmetic mean — so the verdict pill and the
  per-vendor approve/deny tally always agree by construction. Raw
  per-vendor scores remain accessible via `ContinuousScore.votes` for
  any consumer that needs the mean. Manifest provenance: both the
  verbatim `--jury-models` string AND the post-parse
  `{vendor: model}` dict land in `cli_args` so repro tools can rebuild
  the trio without re-parsing. Schema bumped 5 → 6; idempotent
  migration via `ALTER TABLE judge_results ADD COLUMN votes_json`.
  DeepSeek added to `costest._RATE_USD_PER_1K` at `$0.001/1K` blended
  (~43% overestimate vs real DeepSeek, intentionally safe-side).

### Changed

- **`order_swap` is now edge-aware (BLUEPRINT §6 / BACKLOG Option B,
  v0.5.1).** The generator previously permuted the `nodes` tuple
  without consulting `topology.edges`, which on a near-linear-chain
  topology yielded variants whose execution order violated the graph's
  own data dependencies (the runner bypasses LangGraph's edge-driven
  scheduler and runs `nodes` in tuple order). `order_swap` now
  enumerates *valid linear extensions* of the DAG via Kahn-style
  backtracking (`_enumerate_topological_orderings` in
  `variants/generators.py`). Pinned nodes still keep their slot
  indices; the enumerator prunes any branch that cannot satisfy the
  pin. **Effect on the headline:** linear-chain fixtures (e.g.
  `lending_4role`) now report `order_swap`-contribution variants =
  just the original (1) instead of an inflated 8, because a strict
  chain has only one valid topological ordering — the verdict-flip
  signal from edge-invalid permutations was a
  scrambled-information-flow artifact, not topology fragility. The
  v0.5 CLI edge-blindness warning was replaced by a defence-in-depth
  regression-guard message that fires only if a future bug re-emits
  an edge-invalid permutation. Gallery hashes built before v0.5.1
  are NOT compatible — re-run any pre-v0.5.1 lending gallery.

### Added

- **`topofuzz inspect` — side-by-side flip inspector.** New CLI
  subcommand and UI route that render one verdict-flipping eval as a
  two-column HTML page: original ordering on the left, the flipping
  ordering on the right, each agent's raw output stacked agent-by-
  agent so a reader can see the exact step whose response changed
  when the ordering changed. Editorial palette matches `report.html`
  (warm cream + Source Serif). Risk phrases — deny / approve /
  concern stems — are highlighted server-side. Verdict pills derive
  from the §6 binarisation (`score >= threshold` when `passed` is
  None) so continuous-judge cells still read as APPROVE/DENY at a
  glance. `topofuzz inspect <run_dir> --eval <eid> [--out <path>]`;
  default destination `<run_dir>/inspect_<eid>.html` (atomic
  `os.replace` write to survive concurrent UI fetches). The UI's
  results page surfaces a "Inspect a flip →" button next to the
  full-report link whenever `headline.flip_count > 0`; the route
  returns HTML 200 with a friendly empty-state page when there's
  nothing to inspect (a clean run is not an error). Dual-audited
  twice (code + red-team, two passes); 14 new tests covering shape
  validation, UTF-8-safe truncation, byte-stability, CLI reserved/
  unknown flag rejection, and `Markup("")` empty-rationale handling.
- **`topofuzz langgraph-import` CLI subcommand + regulatory crosswalk
  doc.** Imports a user's existing compiled LangGraph (a `path:attr`
  spec, same shape as `topofuzz run`) and saves it as a BYOP custom
  workflow tile in `~/.topofuzz/custom-workflows/` — so a graph the
  user already wrote in code surfaces as a clickable tile in
  `topofuzz ui` alongside the gallery and form-built workflows. Auto-
  detects the module's `LLM_CONFIGS` if exported; `--llm-config
  NODE=MODEL` (repeatable) overrides per-node models from the command
  line. `--rubric` / `--rubric-file` are mutually exclusive sources
  for the rubric paragraph (PowerShell-friendly alternative for long
  text). Branching graphs, cycles, and multi-START topologies are
  refused with an actionable message — BYOP supports linear chains
  only. The same arbitrary-Python load warning the runner emits fires
  on first load of any imported fixture path (shared one-shot set
  across both code paths). New `docs/regulatory-crosswalk.md` maps
  topofuzz artifacts to common compliance map points (CFPB Reg B,
  OCC MRM bulletin, Fannie LL, MA AG settlements) as a starting
  hook for compliance conversations — explicitly NOT a compliance
  claim; the README's warranty disclaimer applies.
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
- **`model_served_version_resolver` canonical/served mismatch.** The
  CLI previously left the runner's served-version slot keyed under the
  bare-identity default, which mismatched LiteLLM's canonical
  `<vendor>/<model>` request id against the bare/snapshot id every
  provider returns in `response.model`. Net effect: per-call mismatch
  warning **and** 5-10× §7 cache miss on warm runs (empirically
  observed across the gallery in step 19a). New
  `topofuzz.runner.default_resolver` dispatches by canonical prefix:
  `gemini_resolver` strips `google/`, `openai_resolver` strips
  `openai/`, `anthropic_resolver` strips `anthropic/` and maps
  `*-latest` aliases via a vendored `_ANTHROPIC_SERVED_MAP` snapshot
  table (3.5 sonnet/haiku, 3.7 sonnet, 3 opus). The CLI now wires
  `default_resolver` into `execute_run` so the warning is gone and
  the cache works as designed. **One-time cache invalidation:**
  existing warm `~/.topofuzz/llm-cache/` entries written under
  canonical prefixes will not be reused; one cold run then back to
  normal. OpenAI snapshot mapping is prefix-strip-only in v0.5
  (deferred to v0.6 — tracked in `BACKLOG.md`).
- **Cross-step drift propagation into §17 NLS.** The §14 donut
  detected when `report.json`'s signed `verdict_flip_rate.point`
  disagreed with the live `n_flips/n_pairs` recount, but the §17
  natural-language summary — derived from the same signed value —
  silently rendered stale prose. `_render_summary_panel` now
  accepts a keyword-only `drift_flag` parameter and renders a
  complementary stale-prose warning inside the
  `.natural-language-summary` div with the same warn palette as the
  donut's drift-warning so a reader recognises the two as paired.

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
  [`SECURITY.md`](https://github.com/fourwheels2512/fuzzlyzer/blob/main/SECURITY.md)
  for the threat model.

[0.5.0rc1]: https://github.com/fourwheels2512/fuzzlyzer/releases/tag/v0.5.0rc1
