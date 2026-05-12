# TSF v4 vs v5_minimal vs v5+ — Plain English Comparison

*One-page side-by-side. Read this in 5 minutes, decide which version to build.*

---

## What is TSF in one sentence?

A tool that takes a multi-agent LLM system (like a 4-agent loan-approval chain), rearranges the agents in every order, runs each ordering against the same set of test cases, and tells you how often the final answer changes. The headline result is a number: *"your agents disagree 31% of the time when reordered."*

All three versions do this. They differ in **how rigorously, how cryptographically, and how expensively.**

---

## Side-by-side

| Question | **v4** (regulator-grade) | **v5_minimal** (weekend toy) | **v5+** (OSS-honest) |
|---|---|---|---|
| **How long to build?** | 6–10 weeks full-time, or 12–14 weeks side-project | 2 weekends (~15–20 hours) | **7–9 weekends side-project (~100–130 hr honest)** |
| **How much money out of pocket?** | $3,000–$5,000 (calibration runs, KMS, trademarks, dev API calls) | $20–$80 (a few dev API calls) | **~$800** (dev iteration + 1 cold + 2 warm + 1 jury-of-3 partial) |
| **How many lines of code?** | ~5,000–8,000 | ~485 | ~1,450 |
| **How many "Definition of Done" gates?** | 25 | 4 | **14 (13 hard + 1 soft jury gate)** |
| **Who is the intended user?** | CISO at a bank or pharma company needing compliance evidence | A LangGraph developer who saw the Bajaj 2026 paper on Hacker News | A LangGraph developer who wants a real eval tool, not a toy |
| **Tells you WHICH agent to fix?** | No — gives variance number with CI; no per-agent attribution | No | **YES — per-agent influence score with 95% CI ranks the agents by how much each one drives instability** |
| **Static risk badge before any LLM call?** | No | No | **YES — `tsf scan` returns HIGH/MEDIUM/LOW in <1s with zero LLM cost** |
| **Visual "look at this failure" panel?** | No — JSON metrics only | No | **YES — top-3 worst-disagreement panel: same input, two orderings, opposite answers, side-by-side traces with divergence highlighted** |
| **What does the signature on the report mean?** | Cloud KMS signs every report; anchored in Sigstore Rekor (public transparency log) + a second log; auditable years later | A SHA256 hash printed in the report. Not really a signature. | DSSE-signed by a dev key on your laptop (with a `[DEV-KEY]` watermark). Anyone can verify with a 30-line script (only needs `cryptography` + `rfc8785`, ~15MB install), or with `pip install tsf[verify]` — no need to install the full 400MB `tsf`. Rekor opt-in. KMS reserved for v1.0. |
| **How many judges score each answer?** | 3 mandatory (Claude + GPT + Gemini, 2-of-3 majority) with vendor-disjointness requirements | 1 (Claude) | 1 by default, 3 if you pass `--jury-of-3` |
| **Bootstrap confidence intervals?** | BCa with two-level cluster resampling (eval × variant), n=10,000 | Percentile, n=2,000, no clustering | Percentile with two-level cluster resampling, n=5,000 (the v4 fix that actually matters) |
| **Multiple-comparisons correction?** | Benjamini-Hochberg-Yekutieli FDR | None | None (single headline, no multi-test) |
| **Per-cell entropy with Miller-Madow correction?** | Mandatory at k≥5 replicates | None | Deferred to v0.6 (marginal payoff, 5× LLM cost) |
| **Sandboxing for untrusted graphs?** | Firecracker microVM with read-only rootfs, dropped capabilities, seccomp, egress allowlist | None ("trust your own code") | None in v0.5; `--sandbox` reserved namespace for v1.0; README documents trust boundary |
| **Reference fixture quality?** | 5 fixtures, lending-200 with full Gebru datasheet + κ=0.84 label-quality study (2 annotators on 30 rows) | 1 fixture, 50 rows, no claims | 1 fixture, **50 rows**, single-annotator documented, no κ claim |
| **Variant generators?** | 10 (order, role, sequential-to-parallel, dissenter, judge-position, remove-aggregator, tool-set, streaming, decoding-param, structured-output) | 2 (order, role) | 3 (order, role, decoding-param) |
| **Reports include cryptographic transparency-log proofs?** | Mandatory at TDS-Audit (Sigstore Rekor); ≥2 logs at TDS-Regulator | No | Opt-in via `--rekor` flag |
| **What does the report look like?** | HTML + PDF (WeasyPrint) + JSON + DSSE envelope + Rekor proof + second-log proof | HTML + JSON | HTML + JSON + DSSE envelope (+ optional Rekor proof) |
| **CLI commands?** | `tsf init / run / report / verify / resume / diff / model-status` + `tds-validate` standalone tool + Python SDK + REST API + OpenAPI | `tsf run`, `tsf report` | `tsf run`, `tsf report`, `tsf verify` |
| **Frameworks supported?** | LangGraph + CrewAI + AutoGen + custom DSL | LangGraph only | LangGraph only |
| **Sales / business artifacts?** | 10-slide pitch deck, 3 CISO outreach emails, pricing model ($5K/run, $200K ACV), trademark filings US+EU+UK, SOC 2 Type II clock | None | None |
| **Standards-body work?** | NIST AISI Month 3, EU AI Office Month 5, ISO PAS Month 18, CEN/CENELEC, ANSI INCITS, 8 conferences over 24 months | None | None |
| **Multi-tenant SaaS roadmap?** | Postgres row-level security, per-tenant KMS DEK, Stripe billing, FastAPI control plane, Firecracker per job, Q2–Q4 rollout | None | None |
| **Probability of shipping on time?** | 25–35% in 2 weeks, 70–80% in 6 weeks (per v4's own admission) | 80%+ in 2 weekends | **45–55% on time at 6 weekends; 75% within 8 weekends** (recalibrated by RT specialist) |
| **Forward-compatible with v4 if you want to upgrade later?** | n/a (this IS v4) | No — schema, flags, and signing all change | **Yes** — `tds_version` field, reserved CLI flag namespace, DSSE byte-exact, JCS canonical hash, `tenant_id` placeholder all preserved |
| **Brand signal on HN / GitHub?** | High in theory (real cryptographic attestations), but most viewers don't care about the cryptography | Low (looks like a weekend project) | Medium-high (real stats + real signing + jury-capable, looks serious) |

---

## What does each version actually do when you run it?

### v4 — what `tsf run` does

1. Imports your LangGraph in a Firecracker microVM with dropped capabilities.
2. Walks the graph into a fully-typed Pydantic IR including every decoding parameter, tool schema hash, cache layout hash, and provider extras.
3. Generates up to 32+ variants across 10 generators, deduplicates via NetworkX graph isomorphism.
4. Cross-products with a 200-row eval set (you must supply with Gebru datasheet + κ-study).
5. Runs a cold + warm-cache cost estimator, asks you to confirm a $890 cold estimate.
6. Enqueues 6,400 LLM-call jobs to Redis via arq with idempotency keys.
7. Each agent rollout is cached on a 9-tuple key including `tenant_id`, model-served-version, prompt hash, tool schema hash, cache-control layout hash.
8. Three judges (Claude Opus 4.7 + GPT-5 + Gemini 2.5 Pro) score every (variant, eval) cell with 2-of-3 majority and UNPARSEABLE fallback.
9. A 120-cell calibration shard is judged against a 100-row human gold set; Fleiss κ + per-judge κ-vs-gold reported with bootstrap CIs.
10. Metrics scorer computes variance, ECR, balanced accuracy, Cohen's κ, latency_instability, per-cell Miller-Madow entropy, all with two-level cluster bootstrap (n=10,000) and BH-Yekutieli FDR correction.
11. Renders HTML + WeasyPrint PDF. JCS-canonicalizes the JSON. Wraps in DSSE envelope. Signs DSSE PAE preimage with a cloud KMS key. Submits to Sigstore Rekor + a second transparency log.
12. Emits `report.html`, `report.pdf`, `report.json`, `report.json.dsse`, `report.json.dsse.rekor.json`, `report.json.dsse.secondary.json`.
13. `tds-validate` standalone tool verifies signature, JCS round-trip, Rekor inclusion, key URI resolution, CRL freshness (fail-closed), and conformance level (Basic / Audit / Regulator).

### v5_minimal — what `tsf run` does

1. Imports your LangGraph in the current Python process (trust your own code).
2. Walks the graph into a 4-field Pydantic IR (name, nodes, edges, entrypoint).
3. Generates 8–12 variants across 2 generators.
4. Runs against a 50-row eval CSV you supply.
5. Prints a one-line cost estimate, asks for confirmation.
6. Runs 8 × 50 = 400 LLM rollouts with `asyncio.Semaphore(8)`.
7. Each rollout cached on a 3-tuple key.
8. One judge (Claude Opus 4.7) scores every (variant, eval) cell.
9. Metrics scorer computes variance (with one percentile-bootstrap CI), ECR, and Cohen's κ (point estimates only).
10. Renders HTML report with a banner disclaimer that this is a single-judge demo.
11. SHA256 hash of the report printed at the bottom.
12. Emits `report.html`, `report.json`. That's it.

### v5+ — what `tsf run` does

1. Imports your LangGraph in the current Python process (trust your own code; `--sandbox=docker` stub for Linux/WSL2 if you opt in).
2. Walks the graph into a 5-class Pydantic IR (Topology + LLMConfig with model/temp/max_tokens/seed + Node + Edge). JCS canonical hash on every IR.
3. Generates 8–16 variants across 3 generators (order, role, decoding-param).
4. Cross-products with a 100-row eval CSV (lending fixture or yours).
5. Runs a **cold + warm-cache** cost estimator with per-line breakdown, prints `cold=$320 ± $60 / warm-cache=$72 ± $14 (90% CI). Continue?`
6. Runs 8 × 100 = 800 LLM rollouts with `asyncio.Semaphore(8)`, cached on a 6-tuple key including `tenant_id="local"` placeholder (forward-compat to multi-tenant).
7. **Single judge by default** (Claude Opus 4.7). If you pass `--jury-of-3`, three judges with UNPARSEABLE fallback and `judge_failure_rate > 5% → degraded` gate.
8. Records `judge_independence` field (vendor overlap, self-preference warning) as INFORMATIONAL — no hard gate.
9. Metrics scorer computes variance with **two-level cluster bootstrap CI** (eval × variant, n=5,000), ECR with degenerate-cell handling, Cohen's κ + balanced accuracy with bootstrap CIs, `collapse_flag` per normative thresholds.
10. Per-cell entropy with Miller-Madow correction is opt-in via `--per-cell-entropy`.
11. Renders HTML report with banner conditionals (DEV-KEY watermark, vendor-overlap warning, degraded badge, unstable-judge badge).
12. JCS-canonicalizes the JSON. Wraps in DSSE envelope. **DSSE PAE preimage is byte-exact per the DSSE v1.0 spec** (forward-compat to v4 KMS signing). Signed with Ed25519 age-encrypted PEM dev key.
13. `--kms=aws|gcp` opt-in for cloud KMS signing. `--rekor` opt-in for Sigstore Rekor inclusion.
14. Emits `report.html`, `report.json`, `report.json.dsse` (+ optional `.rekor.json`), `traces/*.jsonl`, `run.sqlite`.
15. `tsf verify` does in-process signature + JCS round-trip + optional Rekor lookup. No standalone validator tool (deferred to v1.0).

---

## What v5+ keeps from v4 (the items worth their weight)

1. **DSSE byte-exact PAE construction** — so signatures are real cryptographic artifacts that v1.0 KMS signing slots into without breaking. Test vectors forked from `secure-systems-lab/dsse` v1.0.0 and gated in CI.
2. **JCS canonical hash on the IR** — so reports are byte-comparable and tamper-evident.
3. **Two-level cluster bootstrap** — the v4 stats fix that actually matters; variant-only stratification produces CIs that are too narrow. v5_minimal had this wrong.
4. **Cohen's κ + balanced accuracy** (not Spearman ρ) on binary outcomes — v4's correct call.
5. **Jury-of-3 as opt-in architecture** with `judge_independence` informational field — reusable by Pramana for v0.2 semantic scoring.
6. **Cold + warm-cache cost estimator** — the v4 detail that prevents the founder from burning $400 on a single misconfigured run.
7. **Degenerate-cell handling** — bootstrap CIs on all-zero/all-one outcome rows return `ci_lower=null, ci_upper=null, degenerate_flag=true`. Required to avoid crashes on edge cases.
8. **Reserved CLI flag namespace** + `tds_version` + `tenant_id` placeholder — forward-compat to v4 without breaking changes.

## What v5+ drops from v4 (the items that exist for a customer who doesn't exist in OSS Year 1)

1. **Cloud KMS as default signing key** → use age-PEM dev key with `[DEV-KEY]` watermark; KMS is `--kms` opt-in.
2. **Sigstore Rekor mandatory at TDS-Audit** → `--rekor` opt-in.
3. **≥2 transparency logs at TDS-Regulator** → no Regulator conformance; opt-in single Rekor only.
4. **`tds-validate` standalone verifier tool** → `tsf verify` in-process. Standalone is a v1.0 problem.
5. **CRL fail-closed in the verifier** → no signatures issued in OSS Year 1 that anyone would want to revoke.
6. **Calibration shard ≥120 cells with κ-vs-100-row-human-gold per judge** → no second human annotator on Day 1; single-annotator-documented fixture suffices.
7. **`tsf model-status` deprecation watcher** → only matters for customers running quarterly TDS-Audit. No customers.
8. **`judge_independence.vendor_disjoint == true` HARD gate at TDS-Regulator** → field is informational; no hard gate.
9. **`latency_instability` as fourth headline metric** → defer; three metrics are enough for v0.5.
10. **BH-Yekutieli FDR correction across 32 variants** → single headline comparison, no multi-test correction needed.
11. **Pitch deck, CISO outreach emails, pricing model** → no sales motion in OSS Year 1.
12. **24-month standards Gantt** (NIST / ISO / CEN / ANSI / conferences / trademark filings US+EU+UK) → 3–7 year horizon, not Year 1.
13. **Multi-tenant SaaS infrastructure** (Postgres RLS, per-tenant DEK, FastAPI control plane, Stripe billing, REST API + OpenAPI, BYO KMS) → no SaaS in Year 1.
14. **Firecracker microVM sandboxing** → `--sandbox=docker` stub only; founder is on Windows where Firecracker doesn't run anyway.
15. **Gebru datasheet + label-κ study for the reference fixture** → single-annotator-documented note suffices for OSS demo.
16. **STRIDE per-component threat model + Vulnerability Disclosure Policy + Key Rotation Ceremony doc** → 1-page `SECURITY.md` covers the OSS minimum.
17. **SOC 2 Type II clock** → no paying customers, no SOC 2 needed.
18. **Multi-framework adapters** (CrewAI, AutoGen, custom DSL) → LangGraph only in v0.5; revisit after first user request.

---

## The bottom-line recommendation

**Build v5+.**

- It's the same brand signal as v4 at one-third the cost and one-third the calendar time.
- It's the same forward-compat as v4 (KMS + Rekor + jury are flags, not architecture changes).
- It's enough rigor that a sophisticated reviewer on HN can't dismiss it as a toy (real cluster bootstrap, real DSSE signatures, real `judge_independence` disclosure).
- It explicitly accommodates Pramana as the second product in the same repo (shared `mbcore` package, Pramana-aware interfaces).

The risk with v4 is paying enterprise-grade cost for OSS-grade payoff and missing the brand-signal window.
The risk with v5_minimal is shipping something that looks like a weekend project and gets dismissed as such.
**v5+ threads the needle.**

---

**End of comparison.**
