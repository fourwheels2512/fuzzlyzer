# FIX_PLAN_AUDIT.md — Adversarial Red-Team Review of FIX_PLAN.md

**Auditor disposition:** Hostile. Default to flagging.
**Audit date:** 2026-05-09
**Document audited:** `FIX_PLAN.md` (dated 2026-05-09)
**Auditor's stake:** ~190 hours of solo founder time over 8–12 weeks. The cost of letting a bad plan ship is large; the cost of finding holes now is small.

---

## 1. Verdict

**REVISE-WITH-MAJOR-FIXES.**

The plan is structurally serious and traceable — fix IDs, severity tags, source-requirement back-links, calendar layout. It is also (a) optimistic on time, (b) mis-sequenced in two specific ways that compound, (c) silently dependent on artifacts it never schedules creating (notably `INDEPENDENCE_RECEIPTS.md` named in SCH-C-001 as a MUST), and (d) systematically under-severitied on three items that are actually launch-blockers under their own source-requirement language. The plan is not REJECT — the architecture is right, the locked decisions are coherent, and the founder will reach launch with most of it intact. But executing as written produces a Phase 7 review-gate failure (no founder will get through reviewer signoffs in 20 hours), a likely Phase 5/6 timeline blowout, and at least one credibility-grade hole (no INDEPENDENCE_RECEIPTS.md) that re-creates the v0.1 audit failure pattern in a new disguise. Five specific phases need redesign; eight items need to be added; three items need severity upgrades; two need resequencing. Budget recalculates to **~250–280 hours over 14–16 weeks**, not 190 hours over 8–12.

---

## 2. ADD list

These are concrete fixes the plan is missing and a hostile reviewer will note.

| FIX | Statement | Phase | Why missing | Source req | Severity | Time |
|---|---|---|---|---|---|---|
| **NEW-1.9** | Create `INDEPENDENCE_RECEIPTS.md` — per-phase pre-Sathish PRIMARY/COMMENTARY citation, named edition, English-language scholar pre-dating Sathish 2026, year of that scholar's publication. This is **named verbatim** in SCH-C-001 acceptance criterion. The plan references SCH-C-004 (FIX-3.11) which addresses the *honest contribution range* but never addresses SCH-C-001's *receipts file*. These are different requirements. | 1 (or early 3) | Plan covers SCH-C-004, SCH-C-005 but silently drops SCH-C-001. SCH-C-001 is a MUST. | SCH-C-001 | BLOCKER | 4 hr |
| **NEW-3.16** | Verify Sathish 2026's actual *hetvābhāsa* list by direct quotation from his paper. If Annambhaṭṭa's Navya, demote Phase 13 from "divergence" to "shared with sharper rules." | 3 | Plan has FIX-3.11 (range 30–45%) but the *load-bearing* SCH-C-002 verification step is not its own fix. Audit Finding 1.4 explicitly called this out: it's "one paper-read away." | SCH-C-002, audit Finding 1.4 | BLOCKER | 1 hr |
| **NEW-3.17** | Phase-count justification table (15 phases × {classical anchor, non-redundancy with neighbors, evaluability tag}). SCH-D-004 is a MUST. The plan addresses operational distinctness (FIX-3.10) and per-phase tags (FIX-3.9, FIX-3.14) but never the unified justification artifact SCH-D-004 demands. | 3 | A reviewer asking "why 15?" finds no single document answer. | SCH-D-004 | BLOCKER | 2 hr |
| **NEW-3.18** | Schema "What this measures" / "What this does not measure" sections. SCH-F-001 and SCH-F-002 are both MUSTs. SCH-F-003 (causality vs decoration) is in the plan as FIX-3.13 but F-001/F-002 are not. | 3 | F-001 names three different claims with different burdens of proof; the plan never picks. | SCH-F-001, SCH-F-002 | BLOCKER | 1 hr |
| **NEW-3.19** | "Whose Navya-Nyāya?" disclosure paragraph in schema doc — TS pedagogical synthesis, NOT Bengal line, with explicit decline-rationale (founder's lack of Sanskrit). Named in CANONICAL_ANCHOR_RECOMMENDATION §6 as the pre-emption to the Bengal-line objection (Risk 1). | 3 | Plan locks Tarka-Saṃgraha in §0 but never schedules the *disclosure paragraph* — the artifact that closes the OQ-001 / SCH-OQ-001 question. | SAN-OQ-001, SCH-A-005 (alternatives considered), Anchor-doc Risk 1 | BLOCKER | 30 min |
| **NEW-3.20** | "Translator's Note / Linguistic Disclosure" section (founder cannot read Sanskrit; all primary-text claims mediated through Phillips, Mohanty, Jha, Phillips-Tatacharya, Athalye-Bodas). SCH-A-006 is a MUST. The plan has FIX-1.6 (positionality) but SCH-A-006 demands a *separate* explicit linguistic disclosure positioned in schema doc + README. | 1 or 3 | Conflated with FIX-1.6; SCH-A-006 acceptance asks for the explicit "Translator's Note" section by name. | SCH-A-006 | BLOCKER | 30 min |
| **NEW-3.21** | Test-of-the-test: run the schema's own worked example through the harness and assert it passes every format check, every cross-phase consistency rule, every semantic constraint. SCH-D-003 is a MUST. The plan has FIX-3.7 (worked example *internally* satisfies its own rules at design time) but no end-to-end run-through-the-harness test. | 6 (after scorer rewrite) | This is the v0.1 audit's Finding 6.1/6.2/6.3 in one phrase. The plan addresses the *design* of the worked example but not its *passing the actual harness*. | SCH-D-003, REQ-CI-005 | BLOCKER | 2 hr |
| **NEW-5.9** | Per-problem expected-phase-discrimination tag: which phases (8/9/10/13/14) the problem is *designed* to exercise. REQ-DAT-002 acceptance demands "pre-registered which phases each category should exercise." | 5 | Plan has FIX-5.1 (categories with min-counts) but not the per-problem expected-discrimination annotation, which REQ-DAT-002 acceptance criterion requires. | REQ-DAT-002 | BLOCKER | 3 hr |
| **NEW-6.16** | Provider-specific system-prompt placement parity test (sentinel "SCHEMA-LOADED" probe). REQ-SCH-006 is SHOULD with a regression-test acceptance. Critical for the OpenAI-Responses-API and o-series-reasoning-model channels where the system prompt is silently ignored under some configurations. | 6 | A frontier-lab reviewer running on GPT-5 reasoning will hit this within an hour. | REQ-SCH-006 | SHOULD (effective BLOCKER for cost-credibility) | 1 hr |
| **NEW-7.0** | Reviewer-recruitment outreach starts in Week 3 (parallel to Phase 4 Mithila), NOT Week 8. Reviewer time-to-respond is 4–8 weeks for a Sanskrit-fluent academic; recruiting in Week 8 puts reviewer signoff at Week 12–16. | Move outreach piece of FIX-7.1/7.3 to Week 3 | The plan treats Phase 7 as a Week 8–9 phase but the *outreach component* must start when Phase 4 starts. | SCH-G-001, SCH-G-004, SAN-J-001 | BLOCKER | 0 hr (resequence; see RESEQUENCE) |
| **NEW-8.9** | Pre-mortem document naming the 5 most likely failure scenarios + first-response plan for each. Saves the founder from improvising under stress at week 6 / week 10. | 1 | Risk register §5 is single-line per phase; pre-mortem is the agent-side analog of incident-response runbook. | hygiene / risk-management | SHOULD | 2 hr |
| **NEW-CC-9** | A `THIRD_PARTY_LICENSES.md` listing the dependency stack (anthropic, google-genai, openai, inspect-ai, z3-solver, pytest) and their licenses. Standard for OSS releases citing Apache 2.0. | 8 | Apache 2.0 license discipline + REQ-REP-001 pinned-deps demand a separate license inventory. | OSS hygiene, REQ-REP-001 | SHOULD | 1 hr |
| **NEW-CC-10** | A `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1 standard). For a project actively soliciting reviewer engagement, the absence of one is a yellow flag. | 1 | Academic OSS norm. The plan has CONTRIBUTING_PROVIDER.md (REQ-PRO-001) but no CoC. | OSS norm | SHOULD | 30 min |
| **NEW-6.17** | A version migration tool (`pramana-eval migrate-results --from 1.0 --to 1.1 result.json`). REQ-VER-004 is SHOULD; flagged as deferred-to-v1.1 implicitly but not listed in the plan's §10. | 6 or §10 | Quietly missing. | REQ-VER-004 | SHOULD (defer to v1.1 explicitly OR implement) | 2 hr |
| **NEW-8.10** | Sathish-author-courtesy outreach: email Sathish a pre-publication courtesy copy of the schema + paper, inviting comment, 2 weeks before public launch. Closes the SCH-C / R4 attack surface from "did you even tell him?" to "I told him; here's the response or non-response." | 8 | Plan addresses Sathish citation in artifacts (CC-6) and contribution accounting (FIX-3.11) but not the *direct courtesy contact*. Cheap; high-leverage. | SCH-C-001, professional courtesy | SHOULD | 30 min |

**Rough additional time from ADDs:** ~20 hr.

---

## 3. REMOVE list

The plan is mostly tight, but two items are duplicative or low-leverage:

| FIX | Why remove or merge |
|---|---|
| **FIX-7.5 vs FIX-7.7** | Both produce signoff artifacts. FIX-7.5 (`EXTERNAL_AUDIT.md` + NOTICE acknowledgement) and FIX-7.7 (`LAUNCH_CHECKLIST.md` requiring three-of-three clearance) overlap. Merge: one file, `EXTERNAL_AUDIT.md`, with a checklist section at top. Saves ~1 hr. |
| **FIX-6.4 + FIX-6.5 acceptance overlap** | FIX-6.4 ("format scorer rewrite from `format_spec.yaml`") and FIX-6.5 ("cross-phase consistency rules implementation") share the gold-set audit acceptance criterion. The 50-response gold-set audit (REQ-SCO-002) is one artifact; build it once, consume it twice. The plan currently double-counts the audit time. Save ~2 hr. |
| **FIX-2.4 + CC-5** | FIX-2.4 (grep-and-eradicate forbidden phrases at Week 1–2) and CC-5 (always-on standard) restate the same rule. Keep CC-5 as the always-on lint gate; reduce FIX-2.4 to "establish the lint gate; one-time pass" — already the implied scope, but make it explicit so it's not re-done in Phase 8. |

Net REMOVE savings: ~3 hr.

---

## 4. RESEQUENCE list

Three sequencing problems, two of which are load-bearing.

### 4.1 The Mithila-Phase-3 chicken-and-egg (Lens 9, the Catch-22)

The plan sends Phase 4's Mithila letter in Week 3, AFTER Phase 3's schema redesign (Weeks 2–3, 25 hr). This is correct as written — the letter cannot reference a schema that doesn't exist. **But:** Phase 7 reviewer signoffs depend on Mithila response, AND the schema redesign in Phase 3 makes assumptions (e.g., Phase 9/10 reorder, Annambhaṭṭa anchor scope) that Mithila could overrule. If Mithila responds in Week 7 with substantive changes, those propagate back to Phase 3 — but Phase 5 and 6 have been built on the pre-Mithila schema for 4 weeks.

**Resequence:** Add **FIX-3.0** (Week 2, 1 hr): "Pre-Mithila schema sketch" — a 2-page outline (anchor decisions, phase list, key cites) sent to Mithila as an *advance copy* in Week 2, with the full schema-doc following in Week 3 once the founder has Mithila's preliminary indication of interest. This shortens the response-window-to-substantive-feedback loop by ~2 weeks and reduces the rework risk on Phases 5/6.

**Better:** **Move all reviewer outreach (FIX-7.1, FIX-7.3) to Week 3, parallel with FIX-4.1/4.2/4.3.** The single most expensive thing in academic outreach is calendar latency; running Mithila + Sanskrit-fluent reviewer + Indian-philosophy academic outreach in parallel (3 letters Week 3, 6-week window closing Week 9) means Phase 7 has signoffs to incorporate, not signoffs to wait for. As currently written, Phase 7 (Week 8–9, 20 hr) cannot complete in 20 hr if reviewer recruitment + reading + response takes the typical 4–6 weeks of academic calendar latency.

### 4.2 Phase 5 contamination check is too late

FIX-5.8 (contamination check) is the *last* item in Phase 5. **Move it earlier:** any problems that fail contamination check force replacement; doing it last means the work is potentially wasted. Run contamination check incrementally as problems are authored, not as a final pass.

### 4.3 FIX-3.13 (causality vs decoration ablation) is in the *schema design* phase but the ablation runs in *calibration* (Phase 6)

Currently grouped under Phase 3 (schema doc), but the actual ablation experiment (free-form CoT vs schema-prompted on held-out subset) runs as part of FIX-6.15 calibration. **Resequence:** keep the *design* in Phase 3 (the section in the schema doc) but explicitly schedule the *run* as FIX-6.15.1 in Phase 6. This is a 4-hr addition that's currently invisible in the cost.

---

## 5. RE-PRIORITIZE list

The hostile-reviewer reading of source requirements demands these severity changes.

| FIX | Current | Should be | Rationale |
|---|---|---|---|
| **FIX-3.12** (drafting timeline disclosure) | SHOULD | **BLOCKER** | SCH-C-005 *as written* says SHOULD, but its acceptance criterion is "if claims of pre-Sathish independent thinking are made, they are documentable in repo history." The plan in §0 *makes that claim* ("Pre-Sathish evidence documented in `memory\project_vedic_ai_idea.md`") so the disclosure is now claim-supporting infrastructure, not optional. Treat as MUST. |
| **FIX-2.3** (ASCII prompt comment) | SHOULD | **BLOCKER** | SAN-B-004 marks SHOULD but its "if and only if" structure (a 3-line comment naming each ASCII term + IAST canonical form must exist for the ASCII-in-prompt to be defensible) makes the comment itself the precondition for the ASCII keep. Without it, ASCII in the prompt is not defensible per SAN-B-004. Therefore the comment is a BLOCKER for the ASCII-keep decision. |
| **FIX-6.14** (Markdown auto-report) | SHOULD | **BLOCKER** | Frontier-lab reviewers (REQUIREMENTS_PRACTITIONER §1.1, §1.6) explicitly want a "JSON I can paste into our internal leaderboard schema" and a one-command reproduce path with markdown summary. A benchmark whose results can only be read by re-running the harness is unciteable. The Markdown auto-report is what makes the headline number embeddable in tweets, papers, model cards. SHOULD-level severity hides the credibility risk. |
| **FIX-8.1** (companion paper) | SHOULD (strongly) | **SHOULD-but-renamed** | SCH-G-003 also marks SHOULD; consistency intact. **However,** the plan's §10 deferral list does NOT include the paper, and FIX-8.6 (arXiv submission) sits in Phase 8. Either commit to the paper as SHOULD-launch-bundled (current posture) or move the paper to v1.1 explicitly. The current "SHOULD (strongly)" is a hedge that the founder will resolve under launch pressure by dropping the paper. **Pre-decide now:** paper at launch (SHOULD-bundled, 16 hr is real; see §6) or paper in v1.1 (then explicitly defer FIX-8.1 and FIX-8.6 to §10). |
| **FIX-1.2** (NOTICE rewrite) | BLOCKER | BLOCKER (sufficient) — but split | The single FIX-1.2 covers 8 source requirements (SCH-E-001/002/003/005, SAN-A-002, SAN-E-001/002/003). One 2-hr fix that lands all 8 acceptance criteria perfectly is unrealistic. **Split into FIX-1.2a (independent posture + Sathish citation), FIX-1.2b (Apache-vs-vidyā-dāna asymmetry), FIX-1.2c (Mithila/living-tradition acknowledgement), FIX-1.2d (founder positionality)** — same 2 hr total but auditable per-acceptance. |

**Plan §10 deferred-list MUST audit:** Per Lens 4 in the audit brief, check whether anything deferred is actually a MUST. The plan defers the following:

- *LangSmith / Inspect Evals full plugin* — REQ-PRO-004 is MUST but plan addresses the **register-only YAML** path (FIX-6.2, FIX-8.4). Per Practitioner OQ-4, register-only is acceptable at v1.0. **Correctly deferred.**
- *HELM / lm-eval-harness adapters* — REQ-PRO-005 is SHOULD, REQ-FRA-004 is NICE. **Correctly deferred.**
- *Sanskrit-medium publication* — SAN-OQ-007 is OQ. **Correctly deferred.**
- *Multi-darshana expansion* — SAN-OQ-002 is OQ. **Correctly deferred.**
- *Quarterly refresh cadence* — REQ-DAT-008 is SHOULD; REQ-CON-004 is SHOULD with the explicit-acknowledgement option. **Acceptable to defer IF the "this is a one-shot benchmark, not a tracker" acknowledgement is in the launch artifacts.** Plan does not currently include this acknowledgement. **ADD: a one-line "refresh cadence: TBD post v1.0" statement in the dataset card.**
- *Co-author recruitment* — Practitioner OQ-9 (single-author OSS read as hobby). **Quietly deferred without acknowledgement; the founder accepts the "single-maintainer permanently" locked decision but a hostile reviewer at a frontier lab will still flag this.**

---

## 6. Time-estimate corrections

| Phase | Plan estimate | Realistic estimate | Variance | Why |
|---|---|---|---|---|
| **Phase 0** | 2 hr | 2 hr | 0% | Realistic. |
| **Phase 1** | 10 hr | **14 hr** | +40% | Naming candidate research (FIX-1.1) at 1 hr is optimistic — sandhi check + PyPI/GH/arXiv collision search + IAST validation across 4–5 candidates is 2–3 hr alone. NOTICE/README/GLOSSARY/CITATIONS are dense, multi-source-requirement documents; first-pass-then-review cycle on each is the realistic model. |
| **Phase 2** | 8 hr | 10 hr | +25% | The grep-and-eradicate (FIX-2.4) + IAST corrections (FIX-2.2) interact: every IAST fix introduces new visual hits in CI lint that need re-reviewing. Multi-school + Buddhist + vidyā-dāna paragraphs (FIX-2.5/2.6/2.7/2.8) require source-checking against named scholars, which is per-paragraph 30–45 min, not 15. |
| **Phase 3** | 25 hr | **40 hr** | **+60%** — most underestimated phase | 15 schema fixes averaging 1.5 hr each is the optimistic floor; the founder is non-Sanskrit-trained and *every* primary-source citation requires verification against named edition. FIX-3.10 (operational distinctness — pair-by-pair contrasting examples for ~14 phase-pairs) at 4 hr is unrealistic; this is a 8–10 hr piece. FIX-3.7 (worked example satisfies own rules) at 3 hr is unrealistic if the worked example is non-trivial; the v0.1 audit found this is precisely where errors hide. Buffer to 35 hr in the plan's §5 risk register acknowledges this; revise to **40 hr realistic, 50 hr if descope to 12-phase fails**. |
| **Phase 4** | 5 hr active | 8 hr active | +60% | Letter writing at 2 hr assumes one draft + send. Realistic: founder draft → 1 review cycle (FIX-8 §8 lists this) → send. Plus the additional "advance sketch in Week 2" recommended in §4.1 above (+1 hr). Plus FIX-4.5 parallel outreach if no response in 2 weeks (+1 hr). Plus founder calendar-emotional cost of waiting and following up. |
| **Phase 5** | 40 hr | **60 hr** | **+50%** — second most underestimated | 100 problems at 25 hr (FIX-5.2) = 15 min/problem. For a problem with Z3-verifiable ground truth + per-problem ground-truth spec + provenance + difficulty tag + expected-phase-discrimination tag (NEW-5.9), 30–40 min/problem is realistic. That's 50 hr just for FIX-5.2 + NEW-5.9. Add Z3 verification debugging (some Z3 specs will fail, requiring problem rework), contamination iteration (NEW-5.8 moved earlier per §4.2), the categorization rebalancing pass — realistic is 60 hr. |
| **Phase 6** | 50 hr | **70 hr** | **+40%** — third most underestimated | The 50-hr estimate assumes provider abstractions work cleanly across Anthropic / Google / OpenAI native + OpenRouter + vLLM/Ollama. In practice, each provider has 1–3 surprise quirks (rate-limit headers, system-prompt-channel weirdness in REQ-SCH-006, dated-revision extraction varying by API). REQ-SCH-006 sentinel test (NEW-6.16) catches some but not all. CI flakes across 4 providers at 4 hr (FIX-6.13) is unrealistic; nightly smoke runs need debugging cycles. The k=3 calibration runs at 4 hr (FIX-6.15) ignores the failure mode where the first run's results force schema/scorer revisions and trigger a full re-run. |
| **Phase 7** | 20 hr | **20 hr active + 25 hr calendar latency absorbed** | — | Active hours acceptable IF reviewer outreach is moved to Week 3 per §4.1. As written (outreach in Week 8), the 20-hr estimate is structurally impossible — reviewer recruitment alone is 4–6 weeks calendar. |
| **Phase 8** | 30 hr | **40 hr** if paper bundled, **18 hr** if paper deferred | +33% (paper bundled) | FIX-8.1 paper at 16 hr is *very* tight for a primary-source-grounded ~6–10-page paper from a non-Sanskrit-trained founder. Realistic: 24–30 hr first-time, with at least one round of self-review against citation hygiene. Decide now whether the paper is at-launch (40 hr Phase 8) or v1.1 (18 hr Phase 8). |
| **TOTAL** | **190 hr / 12 weeks** | **~260 hr / 14–16 weeks (paper bundled)** or **~240 hr / 14 weeks (paper deferred)** | **+37% to +47%** | |

The 190-hour figure is **~37% optimistic** (paper-deferred path) and **~47% optimistic** (paper-bundled path). Either is over the audit brief's ≥30% flag threshold; both are flagged.

---

## 7. New risks to add to register

The current §5 Risk Register lists 9 risks. Six are missing:

| Risk | Phase | Why it matters | Mitigation |
|---|---|---|---|
| **Founder loses motivation at week 6** | 5/6 | The founder is solo, self-funded, with no daily team accountability. The 4-week mid-stretch (parallel Phase 5+6 = 90 hr together) is the canonical motivation-collapse window. modelbrew is the day job; pramana-eval is the side project; week 6 is when the side project gets quietly de-prioritized. | Mid-project pre-commitment device: a public week-3 announcement (HN comment, X thread, blog post stub) that creates social cost to abandonment. Or: a 1-week pause built into Week 6 with explicit "go/no-go" decision. |
| **Mithila response is critical and demands changes that delay launch by 4 weeks** | 4 → 7 → 8 | Plan §5 has "Mithila non-response" risk but not "Mithila *substantive* response demanding rework." This is the *good* outcome that breaks the timeline. If a Mithila *paṇḍita* says "your Phase 9 is wrong, here's how Gaṅgeśa actually treats *upādhi*," the founder must either (a) accept rework and slip 4 weeks, (b) decline the feedback and document the disagreement (per SCH-G-001 acceptance), or (c) compromise. | Pre-commit to (b) as the default if rework is >1 week. Build the "disagreement log" template now. |
| **n=100 calibration produces ugly results that demand schema revision** | 6 → 3 | What if 15-phase strict format-pass on 4 frontier models is 35% (not 50–70%)? Audit Finding 5.1 already flagged 50–70% as the realistic range; below 50% is plausible. The plan's §5 risk for Phase 6 ("Inspect AI integration breaks UX") misses this much larger risk. | Pre-register the failure threshold: "if format-pass < 50% on three of four frontier models, run a graded scorer and frame the headline as graded, not strict." Don't hide the failure; reframe the metric. |
| **Founder opportunity-cost regret at week 8** | 7 | At week 8, the founder will have spent ~120 hours and seen no public output yet. modelbrew's competing demands will be intense. The risk is that Phase 7's reviewer-wait-time looks like idle time but isn't. | Make week 7–9 productive on a *different* axis: the companion paper draft (FIX-8.1) overlaps Phase 7 calendar-wise, so use the wait-time on the paper. Plan currently puts the paper in Week 10–12, leaving Week 7–9 idle-feeling. **Move FIX-8.1 paper drafting to start Week 7.** |
| **CI cost > $5/month** | 6 | REQ-REP-005 nightly smoke run with provider call has no cost ceiling stated. If the smoke run accidentally invokes a frontier model on the full 100-problem set instead of the 3-problem subset, the founder hits a $20/night surprise charge. | CI smoke run hard-coded to ≤3 problems on `gpt-5-nano` or `gemini-2.5-flash-lite`. Cost-cap alarm at $10/month. Documented in CONTRIBUTING.md. |
| **Repository-identity decision (OD-4) blocks repo bootstrap** | 0 → 1 | OD-4 (personal handle vs new neutral name vs new GitHub org) is in the plan's open-decisions list but it blocks Phase 0's "Initialize new project structure" (FIX-0.4). The repo URL appears in NOTICE, CITATIONS, paper, dataset card; changing it post-Phase-1 is expensive. | Decide OD-4 in Day 1, before FIX-0.4. The decision is 30 minutes; the cost of getting it wrong is ~3 hr of repo/cite churn. |

Plus: **the §5 risk register itself uses single-line mitigations that do not survive contact with reality.** A pre-mortem (NEW-8.9 above, scheduled now in Phase 1) is a better artifact.

---

## 8. The single biggest flaw

**The plan treats Phase 7 (reviewer signoffs) as a 20-hour Week 8–9 phase when reviewer recruitment is the longest-calendar-latency item in the entire project, AND it puts reviewer outreach in Phase 7 itself rather than Phase 4.** A Sanskrit-fluent reviewer at Mithila / Sampūrṇānanda / Madras / Harvard / SOAS / Vienna is not someone who reads a schema document and signs off in two days; the realistic recruitment-to-reading-to-response cycle is 4–6 weeks calendar. The plan, as written, sends the Mithila letter Week 3 (correct) but sends the Sanskrit-fluent reviewer letter and the Indian-philosophy academic letter Week 8 (wrong). Phase 7 cannot complete in Week 9 because the responses won't arrive until Week 12–14. The founder will then either (a) ship without signoffs (violating SCH-G-005's "minimum bar of external scrutiny" MUST, the same SCH-G-005 the plan explicitly cites in FIX-7.7), or (b) slip launch to Week 14–16. (a) is the credibility-grade catastrophic failure the locked decision "careful academic posture" was specifically designed to avoid; (b) is what the plan should have called for from day one. The fix is structurally simple — move all three reviewer outreach letters to Week 3 parallel with Mithila — but the plan as written quietly sets up the founder to choose between violating their own MUST and slipping by a month, under launch-pressure, in a state that's been working for 12 weeks. This is the failure that kills the project.

---

## 9. Recommended action

**Execute the plan with the modifications below, in this order.**

1. **Pre-execute (Day 0 of Phase 0):** Read this audit. Decide OD-4 (repo identity). Decide companion-paper-at-launch vs paper-in-v1.1. These two decisions cascade into Phase 0/1 and Phase 8/§10 respectively.

2. **Adopt the §4 RESEQUENCE moves before any Phase 3 work begins:**
   - Move FIX-7.1 (Sanskrit-fluent reviewer outreach) and FIX-7.3 (Indian-philosophy academic outreach) to Week 3 parallel with Phase 4. Allocate 2 hr each for letter writing now, not Week 8.
   - Add NEW-3.0: pre-Mithila schema sketch in Week 2 as advance copy.
   - Move FIX-5.8 contamination check earlier in Phase 5 (incremental, not final).
   - Move FIX-8.1 paper drafting start to Week 7 (overlapping Phase 7 calendar-wait).

3. **Adopt the §2 ADD list (~20 hr additional) — specifically NEW-1.9 (INDEPENDENCE_RECEIPTS.md) is non-negotiable.** SCH-C-001 names this file by name as a MUST.

4. **Adopt the §5 severity upgrades** — particularly FIX-3.12 (drafting timeline) to BLOCKER. The §0 locked decision "Pre-Sathish evidence documented in memory file" is not actually documented yet; the disclosure is the artifact that backs that claim.

5. **Revise total budget to 240–260 hours, 14–16 weeks calendar.** Communicate the revised budget to whoever the founder is accountable to (cofounder if any, life partner, modelbrew leadership). The 190/12 figure will not survive contact with execution; setting the expectation at 250/15 and beating it is better than the reverse.

6. **Build NEW-8.9 (pre-mortem) as the very first Phase 1 deliverable.** Name the 5 most likely failure modes. The pre-mortem itself is a 2-hour artifact that prevents 30+ hours of mid-execution course-correction.

7. **At Week 6 (mid-stretch motivation-collapse window), pause for 1 day and run a forensic check** against this audit. Compare actuals to estimates. If >50% over on Phase 5 or Phase 6, descope to 80 problems / drop FIX-6.5 cross-phase rules to soft-signals / lower the bar for Phase 13's worked-example self-consistency check. Pre-commit to the descope triggers now, in writing, in the plan, so they are not negotiated under stress.

**Do not begin execution without items 1–6.** The fix is two days of pre-work plus an updated FIX_PLAN.md v1.1. The cost of skipping is the §8 single-biggest-flaw scenario plus the time-estimate-blowout scenario, compounding.

The plan is good. It is not good enough to execute as-is. After these modifications, the project ships in 14–16 weeks with all three SCH-G-005 review-bar boxes ticked and the audit-grade citation hygiene the v0.1 failure was specifically about.

---

**End of FIX_PLAN_AUDIT.md**

Word count: ~2,800. Audit posture: hostile per brief. Cited fix-IDs (FIX-1.1 through FIX-8.10), source-requirement IDs (SCH-A through SAN-J, REQ-SCH through REQ-CI), audit findings (1.1 through 6.4), and locked decisions. The verdict is REVISE-WITH-MAJOR-FIXES; the project is not killed, but execution as written produces a Phase 7 timing failure compounded by a Phase 5/6 budget blowout, with at least one MUST (SCH-C-001 INDEPENDENCE_RECEIPTS.md) silently dropped from the plan. Address these and ship.
