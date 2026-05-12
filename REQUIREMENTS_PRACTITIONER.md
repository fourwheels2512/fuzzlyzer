# pramana-eval: Practitioner / Eval-Engineering Requirements (Adversarial)

**Lens:** ML practitioner / eval engineer. Hostile.
**Date:** 2026-05-09.
**Scope:** What pramana-eval (the OSS benchmark + 15-phase reasoning schema) MUST do to survive ML-engineering and methodology scrutiny. The scholarly / Sanskrit-philology lens is being produced separately; this document deliberately stays in engineering and statistics.
**Companion docs read:** `pramana_test.py`, `README.md`, `SCORING_VALIDATION.md`, `SAMPLE_OUTPUT.md`, `SCHEMA_15_PHASE_DRAFT.md` (only sections needed; the audit summary substitutes for full re-read), `SCHEMA_15_AUDIT.md`.

The audit found CRITICAL/MAJOR issues in the 15-phase draft (citation hygiene, phase collapse, unscorable phases, cross-phase rules the worked example violates, expected format-adherence drop to 50–70%). The existing harness has documented false-pass / false-fail risks (`SCORING_VALIDATION.md`) and a vacuous n=10 (Wilson 95% CI ~[69%, 100%]). This document treats those as the floor, not the bar.

---

## 1. The Users / Reviewers We Must Survive

### 1.1 ML Engineer at a Frontier Lab (Anthropic, OpenAI, Google DeepMind, Meta, xAI)
Wants to plug pramana-eval into their internal eval rotation against a candidate model and report a number to leadership.
- *Q1:* "Can I run this against my model in <1 hour with one command, no Sanskrit homework, and get a JSON I can paste into our internal leaderboard schema?"
- *Q2:* "Does it integrate with Inspect AI? Our infra is Inspect-native; standalone Python harnesses are friction." (Inspect AI is the de facto framework for Anthropic, DeepMind, Grok per UK AISI's [Inspect Evals](https://inspect.aisi.org.uk/) collection of 200+ pre-built evals.)
- *Q3:* "What's the per-model cost ceiling and total wall time for a 'standard run'?"
- *Failure looks like:* multiple manual API key dances, Sanskrit-named CLI flags, no Inspect adapter, $40+ for one run, no clear "headline number" field in the JSON.

### 1.2 NeurIPS / ICML / ACL Methodology Reviewer
Reviewing a workshop paper or main-track submission that cites pramana-eval as evidence.
- *Q1:* "What is n? What is the 95% CI? On how many distinct frontier models is the result confirmed?"
- *Q2:* "Where are the per-problem traces, the format-vs-semantic split, and the subgroup breakdown?"
- *Q3:* "Is the test set held out from pre-training? What is the contamination-detection procedure?"
- *Failure looks like:* n<50, single-model claim, no contamination check, no inter-rater agreement on the LLM-judge component, headline figure that overlaps with chance under Wilson 95%.

### 1.3 Frontier-Lab Eval Team Deciding Whether to Cite in a Model Card
Anthropic, OpenAI, and Google model cards cite a small fixed set: GPQA-Diamond, MATH, MMLU, AIME, HumanEval, SWE-bench, LiveBench, FrontierMath, ARC-AGI. Horizontal "reasoning schema" benchmarks rarely make the cut.
- *Q1:* "Does this benchmark differentiate frontier models in a way no existing benchmark does? Where is the discrimination signal?"
- *Q2:* "Is it contamination-resistant the way LiveBench (monthly refresh) or FrontierMath (private hold-out) is?"
- *Q3:* "Who maintains it? Is there a versioning + refresh cadence I can rely on across our model release cycle?"
- *Failure looks like:* single-author repo, no refresh policy, no spread between top-tier models, score saturation predicted within one model generation.

### 1.4 OSS Maintainer (Inspect AI / lm-eval-harness / HELM)
Deciding whether to accept a contribution.
- *Q1:* "Does the eval definition fit our task interface? Is the scoring deterministic given a fixed completion?"
- *Q2:* "Does the test set ship under a permissive license with clear provenance?"
- *Q3:* "Does the contribution include a YAML registration file (Inspect Evals' [/register/ flow as of 8 May 2026](https://github.com/UKGovernmentBEIS/inspect_evals)) and pass their CI?"
- *Failure looks like:* judge-only scoring with no deterministic fallback, ambiguous license on problem text, no Inspect-compatible task definition, no smoke test in CI.

### 1.5 Contamination Hawk (Hacker News / r/MachineLearning red-team commenter)
Will run the published problems through MinHash against The Pile, RefinedWeb, FineWeb, and against ArXiv full-text. Will dump dataset SHA into Google to check for leakage on Common Crawl.
- *Q1:* "Show me the n-gram overlap report against open pre-training corpora. Show me a Google search of each problem stem."
- *Q2:* "What's stopping the next frontier-model training run from absorbing this?"
- *Q3:* "If you publish 1000 problems on GitHub, your benchmark is dead in 18 months. What's the held-out plan?"
- *Failure looks like:* no contamination check in repo, no private hold-out, problem stems found in pre-training data via simple search.

### 1.6 Reproducibility Reviewer (NeurIPS Datasets & Benchmarks track)
- *Q1:* "Can I re-run your published headline result and get within 1 problem of the same score?"
- *Q2:* "Are the model versions, SDK versions, Python version, OS, and seed pinned in the released JSON?"
- *Q3:* "Is the reproduce path covered in CI?"
- *Failure looks like:* "claude-opus-4-7" with no dated revision, no SDK version pin, manual-only reproduce path, CI that doesn't actually call a provider.

### 1.7 Sanskrit / Philosophy Reviewer (covered by the scholarly lens, but engineering must not block them)
This document does *not* address citation hygiene — that is the scholarly lens. Engineering must, however, expose machine-readable provenance metadata so the scholar can audit (REQ-DAT-006).

---

## 2. Requirements

Severity legend: **MUST** (ship-blocker) / **SHOULD** (credibility-blocker post-launch) / **NICE-TO-HAVE** (v0.3+).

### A. Schema Implementability As System Prompt

**REQ-SCH-001 [MUST]** — System prompt token budget upper bound.
The 15-phase system prompt MUST fit in ≤ 4,000 input tokens (cl100k tiktoken count). Acceptance: tokenize the canonical English system prompt with `tiktoken` for `cl100k_base` and `o200k_base`; both ≤ 4,000.
*Rationale:* Provider context windows differ; some open-weights deployments (vLLM with Llama-3-70B at 8K context) leave ≤ 4K for input after reasoning room. A schema that eats half the context is unusable.

**REQ-SCH-002 [MUST]** — Output token budget.
Default `max_tokens` MUST be ≥ 6,000 and ≤ 8,000, configurable. The audit predicts 3,000–6,000 token responses on non-trivial 15-phase outputs (Finding 5.2). Acceptance: 95th-percentile response length on the v0.2 100-problem set fits within the configured ceiling without truncation on at least 4 of {Claude Opus 4.7, GPT-5, Gemini 2.5 Pro, Llama-3-70B-Instruct, DeepSeek-V3}.

**REQ-SCH-003 [MUST]** — Per-phase token caps in the prompt itself.
The system prompt MUST specify a hard per-sub-field cap ("≤ 3 sentences, ≤ 60 tokens"). Acceptance: empirical mean tokens per sub-field on the 100-problem v0.2 run ≤ 80; max ≤ 200. Pre-empts Finding 5.2.

**REQ-SCH-004 [MUST]** — Format-pass baseline targets pre-registered before any 15-phase run.
Pre-register the headline target *before* running. Targets:
- ≥ 80% strict format-pass on Claude Opus 4.7 + Gemini 2.5 Pro + GPT-5 (n=100, Wilson 95% lower bound ≥ 71%).
- ≥ 50% on at least one open-weights frontier model (Llama-3-70B-Instruct or DeepSeek-V3).
- If 15-phase strict format-pass is < 60% on three frontier models, ship a *graded* (per-phase partial credit) format scorer alongside strict — strict-only on a 50–70% schema (Finding 5.1) is unreadable.

**REQ-SCH-005 [MUST]** — Behavior on schema overflow / refusal must be observable.
The harness MUST distinguish: (a) clean completion, (b) max-tokens truncation (`stop_reason="max_tokens"` or equivalent across providers), (c) safety filter refusal, (d) model-self-collapsed format (no YAML / fewer than 15 headers). Each is a separate `error_type` in the result JSON. Acceptance: unit test that injects all four conditions per provider and asserts correct `error_type` classification.

**REQ-SCH-006 [SHOULD]** — Provider-specific system-prompt placement parity.
Anthropic uses `system=...`, Google uses `system_instruction`, OpenAI uses a `{"role":"system"}` message, OpenAI Responses API uses `instructions=`. The harness MUST normalize to the canonical native channel for each provider AND expose a `--system-as-user` fallback for providers/models that ignore system prompts in some configurations (e.g., some o-series reasoning models). Acceptance: a regression test verifies the system prompt was actually accepted (probe with a sentinel instruction "If you received this prompt, your first line must be exactly `SCHEMA-LOADED`" run at the start of each provider's smoke test).

**REQ-SCH-007 [MUST]** — Determinism contract.
`temperature=0.0`, `top_p=1.0` (or provider equivalent), single seed. Document explicitly that providers are *not* bit-for-bit deterministic at temp 0; quantify expected non-determinism by running each problem 5x at temp 0 on the v0.2 set and reporting per-problem disagreement rate. Acceptance: `meta` block contains `temp_zero_disagreement_rate` for the calibration run.

---

### B. Scoring Methodology

**REQ-SCO-001 [MUST]** — Format scorer must specify, per phase, exhaustively:
(a) required header text (regex-anchored), (b) required ordered sub-fields, (c) minimum-content-tokens per sub-field, (d) allowed sentinels (e.g. `absent`, `(none in this problem)`).
Acceptance: a JSON-Schema-like spec file `format_spec.yaml` ships in the repo; the scorer is a generated function from that spec; changing the spec without regenerating fails CI.

**REQ-SCO-002 [MUST]** — False-pass / false-fail bounds for the format scorer measured, not assumed.
On a hand-labeled gold set of 50 model responses (mix of pass/fail/edge), format-scorer agreement with the human label MUST be ≥ 95%. Acceptance: published Cohen's kappa ≥ 0.85 against two independent human raters; disagreements published in `scoring_audit.md`.

**REQ-SCO-003 [MUST]** — Semantic scorer must publish its known false-pass / false-fail risks.
`SCORING_VALIDATION.md` already does this for the 6-phase keyword scorer (negation blindness, multi-answer ambiguity, keyword leakage in scaffolding, synonym enumeration, unicode hyphen, scoping-to-Nirnaya). The 15-phase semantic scorer MUST extend this list and MUST gate on a Z3-verified ground-truth answer (REQ-DAT-005) wherever the problem is mechanically verifiable. Keyword-only scoring is forbidden where Z3 / SymPy can verify (CSP, SAT, arithmetic).

**REQ-SCO-004 [MUST]** — LLM-judge component (if used) must publish:
1. Judge model + dated revision
2. Judge rubric, verbatim, in the repo
3. Inter-judge agreement (≥ 2 frontier models as judges, Cohen's kappa ≥ 0.7)
4. Self-consistency: re-run the same judge twice, agreement ≥ 0.85
5. Drift monitor: re-run quarterly on a fixed 25-trace canon set; alert if score moves > 5pp.

Acceptance: `judge_calibration.json` published with each release; CI runs the canon set monthly.

**REQ-SCO-005 [MUST]** — Cross-phase consistency rules must be mechanically scorable AND must be satisfiable by the schema's own worked example.
The audit's Finding 6.1 / 6.2 / 6.3 documents that the v0.1 schema's worked example violates its own cross-phase rules. Before launch, every cross-phase rule must:
- Have a deterministic implementation (regex / lemma overlap / Jaccard threshold, named).
- Pass on the canonical worked example.
- Pass on at least 80% of frontier-model traces in the calibration run, OR be downgraded to a soft signal with documented expected pass rate.

**REQ-SCO-006 [MUST]** — Per-phase pass/fail granularity AND a single headline metric.
Output JSON contains both `phase_scores: dict[phase_id, bool|float]` and a top-level `format_pass: bool`, `semantic_pass: bool`. The headline number for the leaderboard is the *strict* format+semantic joint pass rate; the graded score is reported as a secondary axis. No "average of subjective fields" headline.

**REQ-SCO-007 [SHOULD]** — Negation-aware semantic scoring.
Detect "not X" / "never X" / "no X" patterns in the answer span and toggle keyword-match polarity. Catches the Negation Blindness false-pass already documented for the 6-phase scorer.

**REQ-SCO-008 [MUST]** — Empty-keyword-list short-circuit must raise.
Already implemented for v0.1 (`pramana_test.py:740-746`). Regression test ensures no future problem ships with `ground_truth_keywords=[]`.

**REQ-SCO-009 [SHOULD]** — Audit log per scored response.
Each format/semantic decision logs *which check fired* with line numbers, so a reviewer can audit without re-reading the regex.

---

### C. Statistical Methodology

**REQ-STAT-001 [MUST]** — Minimum n for any headline claim.
Headline numbers MUST be reported on n ≥ 100. Justification: GPQA-Diamond at n=198 already has a Wilson 95% CI wider than the gap between the top four models ([Artificial Analysis: GPQA Diamond](https://artificialanalysis.ai/evaluations/gpqa-diamond)). For a binomial at p=0.85, the Wilson 95% half-width is ~7pp at n=100, ~5pp at n=200, ~3pp at n=500. n=10 (Wilson 95% [69%, 100%]) is forbidden as a headline.

**REQ-STAT-002 [MUST]** — Multi-model coverage for any "schema is portable" claim.
The "schema is portable" or "frontier models do this zero-shot" claim MUST be supported on ≥ 3 frontier models from ≥ 2 vendors AND ≥ 1 open-weights model. Acceptance: published table includes Claude Opus 4.7, GPT-5, Gemini 2.5 Pro, plus Llama-3-70B-Instruct or DeepSeek-V3, all at n ≥ 100.

**REQ-STAT-003 [MUST]** — Wilson 95% CI on every published rate.
No bare percentages without CI in README, paper, or release notes. Acceptance: a `lint_readme.py` CI step rejects any `\d+%` not followed by a CI.

**REQ-STAT-004 [MUST]** — Subgroup reporting.
At minimum: by problem type (CSP / SAT / multi-step deduction / syllogism / abductive / fallacy-detection / ambiguity), by difficulty (simple / moderate / complex), by source (paper-verbatim / hand-crafted / Z3-synthesized / sourced-from-existing-benchmark). Each subgroup must have n ≥ 15 to report; below that, mark "underpowered" not "no result."

**REQ-STAT-005 [MUST]** — Multi-run variance.
Each model MUST be run k=3 times on the headline set. Report mean + per-problem disagreement rate. Acceptance: `meta.runs_per_problem >= 3`. Pre-empts the temp-0 non-determinism caveat that the v0.1 README correctly flags but does not measure.

**REQ-STAT-006 [MUST]** — Pre-registered keyword groups / judge rubrics.
Keyword groups, judge rubrics, and cross-phase rules MUST be committed to a tagged git revision *before* the model is run. Acceptance: each result JSON contains `pre_registration_commit` SHA; CI rejects publishing a result whose `pre_registration_commit` is older than its `model_run_started_at`.

**REQ-STAT-007 [MUST]** — All failures published.
Per-problem traces (response, format report, semantic decision, error_type) MUST ship with the headline numbers. No cherry-picking. Acceptance: file size of published `*_results.json` is ≥ 5 MB for n=100 (sanity check that traces are present).

**REQ-STAT-008 [SHOULD]** — Power analysis for any "schema A beats schema B" claim.
If pramana-eval is ever used to argue 15-phase > 6-phase, or zero-shot > fine-tune, the comparison must include a McNemar's test (paired binary) at α=0.05, with documented power, and Wilson CIs on both arms.

---

### D. Dataset Requirements

**REQ-DAT-001 [MUST]** — Minimum problem count for credibility floor.
n ≥ 100 problems for v1.0. n ≥ 500 for v2.0. n=10 is acceptable only as a smoke test, never as a headline.

**REQ-DAT-002 [MUST]** — Problem-type coverage matrix.
At minimum 7 categories, each with ≥ 10 problems: (1) constraint satisfaction, (2) Boolean SAT (mix sat/unsat), (3) multi-step transitive deduction, (4) syllogism, (5) ambiguity / underdetermination, (6) fallacy detection (problems where the input contains a fallacy the model must catch), (7) abductive / counterfactual. The 15-phase schema's distinguishing phases (vyapti, upadhi, anupalabdhi) only earn their slots if the dataset has problems where those phases discriminate. Acceptance: each category has ≥ 10 problems; pre-registered which phases each category should exercise.

**REQ-DAT-003 [MUST]** — Difficulty distribution disclosed and bucketed.
Each problem tagged `simple` / `moderate` / `complex` with a documented criterion (e.g., for SAT: number of variables and clauses; for CSP: solution-space size). Distribution disclosed in README. Acceptance: `dataset_card.md` contains a table of (category × difficulty × n).

**REQ-DAT-004 [MUST]** — Provenance per problem.
Each problem record carries: `source` ∈ {paper-verbatim, hand-crafted, z3-synthesized, sourced-from-{logiQA, ZebraLogic, BIG-Bench-Hard, ...}}, `created_date`, `created_by`, `license`. Sourcing from another benchmark requires the upstream license to permit redistribution.

**REQ-DAT-005 [MUST]** — Z3 / SAT-solver / SymPy verification of synthetic problems.
Every Boolean SAT, CSP, and arithmetic problem MUST have a machine-checkable ground truth derived from a solver, not from a human guess. Acceptance: `verify_dataset.py` re-derives ground truths from solver and asserts equality with the stored `ground_truth`. CI runs this on every PR that touches `problems.json`.

**REQ-DAT-006 [MUST]** — Per-problem ground-truth specification with equivalence classes.
For each problem: (a) canonical answer string, (b) equivalence class (e.g., for "Alice gets fish": the set of all string permutations that mean the same), (c) acceptable synonym sets, (d) negation-sensitive flag. Acceptance: `ground_truth_spec` field per problem, documented schema.

**REQ-DAT-007 [MUST]** — Held-out private set.
A FrontierMath-style private hold-out of ≥ 50 problems, never published, used only for headline numbers. The public training-and-development set may be larger and is what users iterate against. Acceptance: `private/` dir excluded from public distribution; headline JSON references only the private set; verification that public/private problems do not overlap by stem-hash.
*Note from FrontierMath:* publish the contamination policy clearly to avoid the scandal Epoch hit when [it emerged OpenAI had access to most problems and contributors hadn't been told](https://siliconreckoner.substack.com/p/the-frontier-math-scandal). State explicitly: who has access, when, and under what NDA.

**REQ-DAT-008 [SHOULD]** — Dataset versioning and refresh cadence.
Either LiveBench-style monthly partial refresh ([LiveBench replaces ~1/6 of the dataset monthly, full refresh every 6 months](https://github.com/LiveBench/LiveBench)), or quarterly versioned releases (`v1.0`, `v1.1`, ...). Each release tagged in git, fingerprinted (SHA-256 of canonical JSON), and old versions remain runnable for back-comparison.

**REQ-DAT-009 [MUST]** — Dataset card.
A HuggingFace-compatible `dataset_card.md` covering: provenance, intended use, known biases, limitations, license, contact. Required for HF publication and for NeurIPS Datasets & Benchmarks track submission.

---

### E. Contamination Resistance

**REQ-CON-001 [MUST]** — Contamination check at release.
Run published problem stems through:
- Google Search (top-1 hit must not be the exact problem)
- The Pile (open) MinHash @ 8-gram, threshold 0.5
- Common Crawl recent snapshot via [CC search APIs]
- ArXiv full-text search

Publish results in `contamination_report.md` per release.

**REQ-CON-002 [MUST]** — Private hold-out as the headline source.
See REQ-DAT-007. Headline numbers come from the private set; public set exists for development and is treated as already-contaminated by definition once published.

**REQ-CON-003 [MUST]** — Contamination signal monitor.
Define a signal: if a model's score on the *public* set exceeds its score on the *private* set by more than 2σ (where σ is computed from cross-model variance), publish a contamination alert. Acceptance: `contamination_check.py` runs on every published result and emits a flag in the result JSON.

**REQ-CON-004 [SHOULD]** — Refresh policy committed to writing.
Either: (a) monthly partial refresh à la LiveBench, or (b) quarterly hold-out rotation, or (c) explicit acknowledgement that "this is a one-shot benchmark, not a tracker." Picking nothing is the default failure.

**REQ-CON-005 [MUST]** — Forbidden problem provenance.
Problems MUST NOT be sourced from existing benchmarks that have been in the wild > 24 months without explicit re-verification that they are not in pre-training data. (LogiQA, ProofWriter, RuleTaker etc. have all been absorbed by frontier model pretrains.)

---

### F. Reproducibility

**REQ-REP-001 [MUST]** — Pinned dependency versions.
`requirements.txt` and `pyproject.toml` MUST pin to exact versions for: `anthropic`, `google-genai`, `openai`, `inspect-ai`, `pytest`. CI runs against the pinned set.

**REQ-REP-002 [MUST]** — Pinned model versions in result JSON.
Each result JSON `meta` block must record the dated model revision (e.g., `claude-opus-4-7-20260301`, `gemini-2.5-pro-preview-2026-04-15`, `gpt-5-2026-04-01`). The convention varies by provider; document each. The current `pramana_test.py` records `model` but not the dated revision returned in API responses — fix.

**REQ-REP-003 [MUST]** — Full reproducibility metadata block.
Already partially implemented (`_build_meta_block`). Must extend to include: OS + version, CPU arch, Python version, every SDK + version, model name + dated revision, temperature, top_p, max_tokens, seed (if set), `pre_registration_commit` SHA, `dataset_version`, `schema_version`, run started/finished UTC timestamps, total_cost_estimate_usd, total_wall_time_seconds.

**REQ-REP-004 [MUST]** — One-command reproduce.
`pramana-eval reproduce <result-json-path>` MUST re-run the same problems against the same model+revision (if accessible) and report agreement rate vs the original. Documented as the primary reproducibility check.

**REQ-REP-005 [MUST]** — CI smoke test that actually calls a provider.
Use a free or low-cost model (e.g., `gemini-2.5-flash-lite` or `gpt-5-nano` with $5/month CI budget) on a 3-problem subset. Asserts JSON shape, format-scorer fires, semantic-scorer fires, no exceptions. Skips on PR from forks (no secrets); runs nightly on main.

**REQ-REP-006 [MUST]** — Frozen test-set fingerprint.
SHA-256 of the canonical sorted JSON of `(id, question, ground_truth_keywords, source)` tuples, recorded in `meta.dataset_fingerprint`. Changing any problem changes the fingerprint; any result whose fingerprint doesn't match the dataset version is rejected by the leaderboard ingester.

---

### G. Provider Abstraction

**REQ-PRO-001 [MUST]** — Provider contract documented.
A new provider MUST implement: `_build_client(env, **kwargs) -> Client`, `call_model(client, model, problem) -> str`, plus a `provider_meta()` returning supported features (system-prompt channel, max_output_tokens, supports_temperature_zero, dated-revision-extraction). One-page `CONTRIBUTING_PROVIDER.md`.

**REQ-PRO-002 [MUST]** — OpenRouter / Together / Fireworks support.
Reach to long-tail models (Llama-3-70B, DeepSeek-V3, Qwen3-72B, Mistral-Large) without per-provider integrations. Acceptance: `pramana-eval --provider openrouter --model meta-llama/llama-3.3-70b-instruct` works against the canonical 100-problem set.

**REQ-PRO-003 [MUST]** — Local-model support via vLLM or Ollama.
For credibility: scholars and small labs cannot afford frontier-API runs. Acceptance: `--provider vllm --base-url http://localhost:8000/v1` runs end-to-end against a locally-hosted Llama-3-70B-Instruct.

**REQ-PRO-004 [MUST]** — Inspect AI integration as first-class.
Ship as both: (a) standalone CLI (creator-controlled UX), (b) an Inspect AI task in the [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) repo via a `register/pramana_eval.yaml` PR. Per the [/register/ flow that opened 8 May 2026](https://github.com/UKGovernmentBEIS/inspect_evals), inspect_evals now accepts community evals via YAML pointing at an external code repo — pramana-eval can ride this with low maintenance burden.

**REQ-PRO-005 [SHOULD]** — lm-eval-harness adapter.
Eleuther's harness is the legacy standard for academic comparisons and many open-weights papers still cite it. A thin adapter is high-leverage, low-cost.

---

### H. Framework Integration

**REQ-FRA-001 [MUST]** — Standalone harness AND Inspect AI task.
Standalone gives creator-controlled UX (Sanskrit-named flags, custom JSON shape, the README's narrative framing). Inspect adapter gives distribution into Anthropic / DeepMind / Grok eval rotations. Ship both.

**REQ-FRA-002 [MUST]** — HuggingFace dataset publication.
Public test set (not private hold-out) ships as `modelbrew/pramana-eval` on HF Hub with a proper dataset card. Required for academic citation graph; HF is the de facto dataset registry.

**REQ-FRA-003 [SHOULD]** — Leaderboard hosted on HuggingFace Spaces.
Submission flow: vendor uploads a result JSON → automated validator checks `pre_registration_commit`, `dataset_fingerprint`, `meta` completeness → rejects or accepts → ranks by strict joint pass rate with Wilson 95% CI as the secondary sort key.

**REQ-FRA-004 [NICE-TO-HAVE]** — HELM / EleutherAI lm-eval-harness adapters as community contributions, not maintained in-house.

---

### I. Output and Reporting

**REQ-OUT-001 [MUST]** — Result JSON schema versioned and documented.
A `result_schema_v1.json` (JSON Schema spec) ships in the repo. Tools that ingest results validate against it. Required fields:
- `meta` (REQ-REP-003)
- `provider`, `model`, `dataset_version`, `schema_version`
- `n`, `format_passes`, `semantic_passes`, `format_pass_rate`, `semantic_pass_rate`, `joint_pass_rate`
- `wilson_ci_format`, `wilson_ci_semantic`, `wilson_ci_joint`
- `subgroup_results` (by category, difficulty, source)
- `phase_scores` (per problem, per phase)
- `results` (per-problem traces with response, format_report, semantic_decision, error_type, elapsed_seconds, provider_request_id, model_revision_returned)

**REQ-OUT-002 [MUST]** — Leaderboard data shape.
Stable: `{model, model_revision, vendor, run_date, dataset_version, schema_version, format_pass_rate, format_ci_low, format_ci_high, semantic_pass_rate, semantic_ci_low, semantic_ci_high, joint_pass_rate, joint_ci_low, joint_ci_high, total_cost_usd, total_wall_seconds, run_config_hash}`.

**REQ-OUT-003 [MUST]** — Per-problem trace publication is mandatory for any headline.
A claim of "X% format pass" without traces is unciteable. The 5-MB-floor sanity check (REQ-STAT-007) catches stripped publishers.

**REQ-OUT-004 [SHOULD]** — Markdown report auto-generated.
`pramana-eval report <result.json>` produces a Markdown report with summary table, subgroup tables, per-problem failures, and the prose decision rule. The current `_print_summary_table` is console-only.

---

### J. Versioning and Lifecycle

**REQ-VER-001 [MUST]** — SemVer for schema and dataset.
Schema version (`SCHEMA_VERSION = "1.0"` already exists) follows SemVer. Major bump = breaking phase changes; minor = new optional fields; patch = clarifications. Same for dataset version.

**REQ-VER-002 [MUST]** — Result-comparability across versions.
Cross-version comparisons (e.g., "model X on schema 1.0 vs schema 2.0") MUST surface the version axis in any report; the leaderboard MUST refuse to mix versions in a single ranking row.

**REQ-VER-003 [MUST]** — Deprecation policy.
A schema version is supported for ≥ 12 months after the next major release. `pramana-eval --schema-version 1.0` keeps working even after 2.0 ships.

**REQ-VER-004 [SHOULD]** — Migration tool between versions.
`pramana-eval migrate-results --from 1.0 --to 1.1 result.json` for non-breaking bumps.

---

### K. Cost and Runtime

**REQ-COST-001 [MUST]** — Cost ceiling for a "standard run."
For n=100 problems on a single frontier model, total API cost MUST be ≤ $15 at standard pricing. Measurement: run on Gemini 2.5 Pro and report. Frontier-lab evaluators won't pay $100 per run; the cost has to be smaller than a typical engineer's lunch.

**REQ-COST-002 [MUST]** — Wall-time ceiling.
n=100 single-model run wall time ≤ 60 minutes with parallelism enabled. Acceptance: `--concurrency 8` flag; benchmarked at ≤ 60 min on Gemini 2.5 Pro and Claude Opus 4.7.

**REQ-COST-003 [MUST]** — Parallelism / batching.
The current `pramana_test.py` is serial. v0.2 MUST add concurrent execution via `asyncio.gather` or a worker pool, with provider-aware rate-limit honoring (use the `Retry-After` header where present; fall back to exponential backoff already implemented).

**REQ-COST-004 [SHOULD]** — Cost estimate published per-run.
Each result JSON's `meta.total_cost_estimate_usd` is computed from per-provider published per-token rates × actual input/output tokens. Acceptance: cost-estimate within ±20% of the actual provider invoice.

---

### L. CI / Quality Gates

**REQ-CI-001 [MUST]** — AST + import + pytest baseline.
Already in place per repo convention. Pytest covers: format scorer on canonical pass / fail / edge fixtures; semantic scorer on AND-of-OR / digit-word / negation cases; provider client construction without API key set (mocked); empty `ground_truth_keywords` raises.

**REQ-CI-002 [MUST]** — Schema validation in CI.
Every problem in `problems.json` validates against the `Problem` JSON Schema. Every result JSON in `examples/` validates against `result_schema_v1.json`.

**REQ-CI-003 [MUST]** — Smoke run that calls a provider.
See REQ-REP-005. Nightly on main, not on PRs from forks.

**REQ-CI-004 [MUST]** — Result-format regression test.
A frozen example result JSON in `tests/fixtures/`; CI rejects if the produced shape diverges.

**REQ-CI-005 [MUST]** — Format-scorer regression test on a hand-labeled gold set.
50+ hand-labeled responses with expected scorer output. Any change to the format scorer must explain (in PR description) which fixture pass/fail flips and why.

**REQ-CI-006 [SHOULD]** — Linter for README CI bounds.
A `lint_readme.py` step rejects any bare `\d+%` not adjacent to a CI like `[x%, y%]`. Pre-empts regressions of the v0.1 README's "100% on n=10" framing.

**REQ-CI-007 [MUST]** — Z3 solver verification step in CI.
For every problem with `category` ∈ {sat, csp, arithmetic}, `verify_dataset.py` re-derives the ground truth and asserts equality. Required by REQ-DAT-005.

---

## 3. Anti-Patterns To Forbid

1. **Single-model headline.** A "100% on Gemini" claim is a single data point; not a benchmark result. (Already partly acknowledged in v0.1 README; must be enforced in v0.2.)
2. **n < 50 headlines.** Wilson 95% CI on n=10 is [69%, 100%] for p=1.0 and [17%, 69%] for p=0.4 — these intervals overlap; the comparison is not statistically separated. n=10 stays a smoke test, never a result.
3. **Sanskrit-only error messages and CLI flags.** `--samshaya-only` is hostile to engineers. English flags + Sanskrit metadata.
4. **Keyword-only scoring on solver-verifiable problems.** SAT, CSP, arithmetic must be Z3-verified. Keyword scoring there is malpractice.
5. **LLM-judge as the sole scoring channel.** Judge-only without a deterministic floor is a black box; cite [Anthropic's eval-engineering posture](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — judges drift, judges cost, judges disagree. Pair every judge call with a deterministic check or a published agreement rate.
6. **Cherry-picked traces.** All failures published with the headline. The 5MB floor is a sanity check; the real rule is "every claim has a per-problem trace next to it."
7. **Re-using LogiQA / ProofWriter / RuleTaker problem stems verbatim** without contamination check (REQ-CON-005).
8. **A worked example that violates the schema's own cross-phase rules** (per Audit Finding 6.1 / 6.2 — v0.1 schema does this; v0.2 must not).
9. **Headline numbers without Wilson 95% CI.** Already a v0.1 anti-pattern in informal communication; lint at CI.
10. **Schema changes without a version bump.** "Patch fix to a phase rule" silently invalidates back-comparisons.
11. **Floating model names like `claude-opus-4-7`.** Capture the dated revision returned in the API response, every time.
12. **Mixing public-set scores into the headline.** Public-set numbers go in the development table; private hold-out goes in the headline (REQ-DAT-007).
13. **No contamination report at release.** A "we didn't check" stance is worse than a "we found 2 stems on Google" stance.
14. **Goodhart-friendly format scoring.** A scorer that rewards literal "Wherever ... there is" template matches without checking that the *referent* of the wherever clause matches the inference is gameable; add semantic checks where feasible.
15. **A headline metric that conflates format and semantic.** They measure different things; report separately AND jointly.

---

## 4. Open Engineering Questions (Founder Decisions Needed Before Drafting)

1. **Schema scope: 6-phase OR 15-phase OR both?** The 6-phase result is shippable today (Gemini 100/100, n=10 — promote to 100). The 15-phase is REVISE-WITH-MAJOR-FIXES per the audit and will likely land at 50-70% format-pass per Audit Finding 5.1. Recommend: ship 6-phase as v1.0 and 15-phase as v2.0-experimental with explicit caveat. Founder must decide whether to delay launch.

2. **Public vs private hold-out ratio.** FrontierMath uses 290 private / Tier 4 + 48 private of 290 total. LiveBench rolls 1/6 monthly. Pick one. Recommend: 60% public (developer-facing, contamination-presumed), 40% private (headline-only). Decide before recruiting problem authors.

3. **Judge model selection.** Cross-vendor (Claude judges OpenAI outputs and vice versa) gives independence but costs more and doubles operational complexity. Single-judge cheaper and simpler but biased toward judge's own model family. Recommend: ensemble of Claude Opus + Gemini 2.5 Pro as judges, fall back to Z3 / regex where possible.

4. **Inspect AI integration: register-only YAML or full plugin?** Register YAML is low-touch (per [the 8 May 2026 inspect_evals flow](https://github.com/UKGovernmentBEIS/inspect_evals)) but limits UX customization; full plugin is more work. Recommend: ship register YAML at v1.0, plugin at v1.1.

5. **Dataset license.** CC-BY-4.0 (academic-friendly, allows commercial) vs CC-BY-NC-4.0 (kills commercial reach but protects against extraction by frontier vendors) vs MIT. Pick before publishing on HF Hub. Recommend: Apache-2.0 for code, CC-BY-4.0 for problem data.

6. **Refresh cadence.** Monthly LiveBench-style is high commitment (one full-time eng-month/year). Quarterly versioned releases lighter. "Static, never refresh" kills the benchmark within 12 months. Recommend: quarterly versioned + private hold-out rotation every 6 months.

7. **Cost budget for the public benchmark calibration run.** A 4-model × 100-problem × 3-runs calibration is ~ 1200 API calls × ~ $0.05/call = ~$60. Not the bottleneck. But adding Llama-3-70B and DeepSeek-V3 via OpenRouter at $0.10/call is ~$60 more. Total budget for v1.0 calibration: ~$200. Founder must approve.

8. **Naming and provenance attribution.** The schema is grounded in classical Sanskrit literature; the OSS eval is a derivative engineering artifact. Decide the name (`pramana-eval` is fine for the engineering tool; "Navya-Nyaya benchmark" overclaims). Decide the citation footer relationship to Sathish 2026 (already handled in v0.1 README appropriately; carry forward).

9. **Single-author vs multi-maintainer.** Single-author OSS benchmarks are read as "one person's hobby project" by frontier labs. Recruit ≥ 2 co-maintainers from different orgs before v1.0 announcement. Founder decides whom.

10. **Whether to publish failure analyses on contamination.** Some labs prefer to know contamination signals privately. Recommend: public failure log + private NDA channel for vendors who request prerelease access.

---

## Sources

- [Inspect AI homepage (UK AISI)](https://inspect.aisi.org.uk/)
- [UKGovernmentBEIS/inspect_evals (200+ pre-built evals; new /register/ flow May 2026)](https://github.com/UKGovernmentBEIS/inspect_evals)
- [LiveBench / contamination-limited LLM benchmark](https://github.com/LiveBench/LiveBench)
- [LiveBench paper (arXiv 2406.19314)](https://arxiv.org/pdf/2406.19314)
- [FrontierMath (Epoch AI, 290 private + 48 Tier 4 private)](https://epoch.ai/frontiermath)
- [The FrontierMath scandal (Silicon Reckoner)](https://siliconreckoner.substack.com/p/the-frontier-math-scandal)
- [GPQA Diamond benchmark (Artificial Analysis): n=198, CI wider than top-4 gap](https://artificialanalysis.ai/evaluations/gpqa-diamond)
- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
