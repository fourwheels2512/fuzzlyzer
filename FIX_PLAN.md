# FIX_PLAN.md — Project Blueprint

**Date:** 2026-05-09
**Project status:** Reset from "modelbrew Path B certification revenue" (audit-killed) to "independent academic OSS benchmark, Tarka-Saṃgraha-anchored, careful 8–12 week launch."
**Founder:** Kiran Nayudu (Indian diaspora; ML/SaaS background; not formally Sanskrit-trained)
**Current repo:** `REPO_ROOT\`
**This document:** complete sequenced fix list across 8 phases, with severity tags, time estimates, file references, and source-requirement traceability.

**Companion audit:** `FIX_PLAN_AUDIT.md` (REVISE-WITH-MAJOR-FIXES verdict). Audit's ADD list (10 items: NEW-1.9 INDEPENDENCE_RECEIPTS, NEW-3.16–3.21 schema-doc additions, NEW-5.9 phase-discrimination tag, NEW-6.16 sentinel test, NEW-8.9 pre-mortem + CoC + THIRD_PARTY_LICENSES), severity upgrades (FIX-3.12, FIX-2.3, FIX-6.14 → BLOCKER), resequencing (FIX-7.1 and FIX-7.3 reviewer outreach moves to Week 3 parallel with Mithila), and time correction (~250 hr / 14–16 weeks, not 190 / 8–12) MUST be incorporated during execution. Treat FIX_PLAN_AUDIT.md as authoritative for execution-time fixes to this blueprint.

---

## 0. Decisions locked in (do not re-litigate)

| Decision | Choice |
|---|---|
| Project ownership | Independent side project — NOT modelbrew-branded |
| Schema scope | 15-phase only; archive 6-phase as v0 legacy |
| Launch posture | Careful academic (~8–12 weeks) |
| Sanskrit-lens adoption | Full (institutional outreach, vidyā-dāna disclaimer, IAST throughout) |
| Existing 6-phase | Fix the *hetvābhāsa*-list bug, then freeze |
| Project naming | Declared hybrid (IAST-correct Sanskrit + English suffix); name TBD |
| Founder positionality | Indian diaspora, not Sanskrit-trained — disclosed |
| Primary outreach | Mithila Institute (Darbhanga) |
| Pre-Sathish evidence | Documented in `memory\project_vedic_ai_idea.md` |
| Funding model | Self-funded out of pocket, ~$500–800 budget |
| Buddhist scope | Explicitly Naiyāyika; invite Buddhist parallel as future work |
| Maintainers | Solo permanently |
| Canonical anchor | Annambhaṭṭa's *Tarka-Saṃgraha* (Athalye-Bodas 1930), with Mithila Institute consultation; Phillips for TC-tier escalation |

---

## 1. What we're building (one paragraph)

An open-source, primary-source-grounded LLM reasoning benchmark anchored in the Naiyāyika tradition of *pramāṇa-śāstra* via Annambhaṭṭa's *Tarka-Saṃgraha*. A 15-phase reasoning schema delivered as a system prompt; a 100-problem dataset (60 public + 40 private); a Python harness with Inspect-AI integration; calibration runs on 3 frontier models + 1 open-weights; a companion arXiv preprint. The project is independent of modelbrew, solo-maintained, and self-funded. Direct revenue: $0. Indirect value: academic credibility, brand association for the founder's modelbrew work, paper #8 in the founder's publishing pipeline.

---

## 2. Severity legend

- **BLOCKER** — gates public launch
- **SHOULD** — credibility-blocker post-launch; justify in writing if waived
- **NICE** — improves quality, not load-bearing

---

## 3. Phase-by-phase fix list

### PHASE 0 — Critical bug fix + archive (Day 1, ~2 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **0.1** | Replace Gautama's Old-Nyāya five-fold *hetvābhāsa* list with Annambhaṭṭa's Navya five (*savyabhicāra, viruddha, satpratipakṣa, asiddha, bādhita*) | `pramana_test.py:236-240` | BLOCKER | SAN-C-004 | 30 min |
| **0.2** | Update NOTICE to reflect Navya list (lines 17, 45–47) | `NOTICE` | BLOCKER | SAN-C-004 | 15 min |
| **0.3** | Tag current state as `v0.1-archive` in git; commit message documenting the schema migration | git | BLOCKER | hygiene | 10 min |
| **0.4** | Initialize new project structure for v2 (separate dir or branch) | filesystem | BLOCKER | hygiene | 30 min |

**Phase 0 acceptance:** the existing pramana_test.py is internally consistent (Navya list throughout), tagged, frozen.

---

### PHASE 1 — Project re-foundation (Week 1, ~10 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **1.1** | Choose new project name: declared hybrid, IAST-correct, sandhi-correct, no PyPI/GH/arXiv collision. Propose 4–5 candidates; founder picks one | new repo | BLOCKER | SAN-G-001/002/003 | 1 hr propose + founder decide |
| **1.2** | Rewrite NOTICE: independent project, Tarka-Saṃgraha anchor, Apache-2.0-vs-vidyā-dāna disclaimer, Mithila living-tradition acknowledgement, Sathish citation as related work, founder positionality, Indian-diaspora disclosure | `NOTICE` | BLOCKER | SCH-E-001/002/003/005, SAN-A-002, SAN-E-001/002/003 | 2 hr |
| **1.3** | Rewrite README: project scope, measurement target, statistical caveats, link to GLOSSARY/CITATIONS, no orientalist or Hindutva framing | `README.md` | BLOCKER | SCH-E-004, SAN-H-001/002/003, SCH-F-001/002 | 2 hr |
| **1.4** | Create `CITATIONS.md` — named-edition register (Athalye-Bodas 1930 for TS; Phillips-Tatacharya 2004 for TC; Ganganatha Jha for NS; Mohanty 1989, Matilal 1998, Ganeri 2001, Potter EIP Vols. 2/6) | `CITATIONS.md` | BLOCKER | SCH-A-002 | 1 hr |
| **1.5** | Create `GLOSSARY.md` — every Sanskrit term in IAST + Devanāgarī + named-Indologist gloss with page reference | `GLOSSARY.md` | BLOCKER | SAN-B-001, SCH-B-002 | 2 hr |
| **1.6** | Add POSITIONALITY section to README: founder background, Indian-diaspora disclosure, no Sanskrit training, intent of consultative engagement | `README.md` | BLOCKER | SAN-H-003 | 30 min |
| **1.7** | Create `LIVING_TRADITION.md` (or NOTICE section): name ≥3 living institutions, ≥1 named scholar, intent for ongoing engagement | `LIVING_TRADITION.md` | BLOCKER | SAN-E-002, SCH-E-002 | 30 min |
| **1.8** | Update `LICENSE` (Apache 2.0 stays for code) and add dataset-license decision (CC-BY-4.0 recommended) | `LICENSE`, `DATASET_LICENSE` | BLOCKER | Practitioner OQ-5 | 15 min |

**Phase 1 acceptance:** new identity established, naming locked, NOTICE/README/GLOSSARY/CITATIONS/LIVING_TRADITION all coherent, positionality disclosed.

---

### PHASE 2 — Sanskrit hygiene (Week 1–2, ~8 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **2.1** | Generate `SANSKRIT_FIX_REGISTER.md` — every Sanskrit term in repo with current rendering, IAST correction, sandhi rule, file:line | new file | BLOCKER | SAN-B-003 | 2 hr |
| **2.2** | Apply ~25 IAST corrections across all .md files (*pramāṇa, anumāna, pratyakṣa, śabda, pañcāvayava, hetvābhāsa, savyabhicāra, prakaraṇasama, sādhyasama, kālātīta, nirṇaya, saṃśaya, vyāpti, upādhi, parāmarśa, ...*) | all .md | BLOCKER | SAN-B-001/002 | 2 hr |
| **2.3** | Update `pramana_test.py:189-247` system prompt: keep ASCII for tokenizer-portability reasons but add 3-line comment naming canonical IAST forms | `pramana_test.py` | SHOULD | SAN-B-004 | 30 min |
| **2.4** | Grep-and-eradicate forbidden phrases across all .md files: "Vedic", "ancient wisdom", "Hindu logic", "Sanskrit revival", "Indian logic" without scare-quotes, "5000-year", "rishis", "dharma" outside scoped use | all .md | BLOCKER | SCH-E-004, SAN-A-001, SAN-D-003, SAN-H-001/002 | 1 hr |
| **2.5** | Add Apache-2.0-vs-vidyā-dāna asymmetry paragraph to NOTICE | `NOTICE` | BLOCKER | SAN-E-003 | 30 min |
| **2.6** | Add multi-school *pramāṇa* enumeration paragraph (Cārvāka 1, Buddhism 2, Sāṃkhya/Yoga 3, Nyāya 4, Prābhākara 5, Bhāṭṭa/Advaita 6) | schema doc, README | BLOCKER | SAN-I-001 | 45 min |
| **2.7** | Add Buddhist-logic acknowledgement paragraph (Diṅnāga, Dharmakīrti, *trairūpya*, *anyāpoha*) to schema doc source survey | schema doc | BLOCKER | SAN-D-001/002 | 1 hr |
| **2.8** | Add "Schema is Naiyāyika; Buddhist parallel invited as future work" scope statement | README, schema doc | BLOCKER | SAN-I-003 | 15 min |

**Phase 2 acceptance:** zero ASCII-Sanskrit hits in human-facing docs; zero forbidden phrases; multi-school + Buddhist-logic + vidyā-dāna acknowledgements all in place.

---

### PHASE 3 — Schema v2 redesign (Week 2–3, ~25 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **3.1** | Re-anchor schema in *Tarka-Saṃgraha* (Athalye paragraph numbers, named edition); replace all TC-decimal-subsection citations | `SCHEMA_15_v2.md` | BLOCKER | SCH-A-001/002, audit Findings 1.1–1.3 | 4 hr |
| **3.2** | Fix Phase 8 *vyāpti-pañcaka* representation (use Phillips/Tatacharya 2004 page citations; remove "twenty-four definitions" error) | schema doc | BLOCKER | SCH-C-002, audit Finding 1.2 | 1 hr |
| **3.3** | Resolve Phase 12 vs Phase 13 operational distinctness or merge | schema doc | BLOCKER | SCH-D-001, audit Finding 2.1 | 2 hr |
| **3.4** | Fix Phase 4 (*siddhānta-vibhāga*) evaluability — either define gold annotations or downgrade to design-synthesis | schema doc | BLOCKER | SCH-D-005, audit Finding 3.4 | 1 hr |
| **3.5** | Fix Phase 7 (*dṛṣṭānta*) vs Phase 11 (*udāharaṇa*) distinctness | schema doc | BLOCKER | SCH-D-001, audit Finding 2.4 | 1 hr |
| **3.6** | Phase 13 uses Annambhaṭṭa's Navya five (consistent with Phase 0 code fix) | schema doc | BLOCKER | SAN-C-004 | 30 min |
| **3.7** | Worked example must satisfy own rules — Phase 8 *vyāpti* matches Phase 11 *udāharaṇa*; Phase 13 fallacy presence aligns with Phase 15 epistemic status | schema doc | BLOCKER | SCH-D-003, audit Findings 6.1/6.2/6.3 | 3 hr |
| **3.8** | Dependency graph + cross-phase consistency rules with mechanical implementation specs | schema doc | BLOCKER | SCH-D-002 | 2 hr |
| **3.9** | Per-phase tag: mechanical / lookup / judge / human-only | schema doc | BLOCKER | SCH-D-005 | 30 min |
| **3.10** | Operational-distinctness test pair-by-pair: contrasting examples + dissociating problem instance | schema doc | BLOCKER | SCH-D-001 | 4 hr |
| **3.11** | Honest Sathish-divergence accounting: range 30–45%, derivation method, per-phase tally | schema doc | BLOCKER | SCH-C-004, audit Finding 4.1 | 1 hr |
| **3.12** | Drafting timeline disclosure: cite memory file as evidence of pre-Sathish "Vedic/Upaniṣadic for AI" → "Honest Agent" → narrowed to *pramāṇa* trajectory | schema doc, README | SHOULD | SCH-C-005 | 1 hr |
| **3.13** | Schema-causality vs schema-decoration ablation design (free-form CoT vs schema-prompted on held-out subset) | schema doc | BLOCKER | SCH-F-003 | 1 hr |
| **3.14** | Per-phase classical-source citation with role tag [PRIMARY] / [COMMENTARY] / [MODERN] | schema doc | BLOCKER | SCH-A-004 | 2 hr |
| **3.15** | Format spec as `format_spec.yaml` (per-phase header regex, sub-fields, min-content tokens) | `format_spec.yaml` | BLOCKER | REQ-SCO-001 | 1 hr |

**Phase 3 acceptance:** schema v2 written; passes its own worked example; every phase has classical citation; every phase is operationally distinct; every phase is evaluation-tagged.

---

### PHASE 4 — Mithila outreach (Week 3, then 4-week response window)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **4.1** | Identify named contact at Mithila Institute (current head; Sanskrit-medium plus English-speaking faculty) | research | BLOCKER | SAN-E-001 | 1 hr |
| **4.2** | Draft letter (English + brief Sanskrit greeting) with: project intent, schema document attachment, ask for review/feedback, 4-week window, no commitment expected | `outreach/mithila_letter.md` | BLOCKER | SAN-E-001 | 2 hr |
| **4.3** | Send letter via post + email | logistics | BLOCKER | SAN-E-001 | 30 min |
| **4.4** | `LIVING_TRADITION_OUTREACH.md` — log attempts, dates, contacts, responses | new file | BLOCKER | SAN-E-001 | 30 min |
| **4.5** | If no Mithila response in 2 weeks, parallel outreach to Sampūrṇānanda + Madras Sanskrit College | logistics | SHOULD | SAN-OQ-004 | 1 hr |

**Phase 4 acceptance:** letter sent; outreach log documents attempts; response window is open (Phase 7 closes it).

---

### PHASE 5 — Dataset construction (Week 4–7, parallel to Phase 6, ~40 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **5.1** | Define 7 problem categories with min-counts (≥10 each): CSP, Boolean SAT, multi-step deduction, syllogism, ambiguity, fallacy detection, abductive | `problems.json` schema | BLOCKER | REQ-DAT-002 | 2 hr |
| **5.2** | Build 100-problem corpus (60 public + 40 private) | `problems.json`, `problems_private.json` | BLOCKER | REQ-DAT-001/007 | 25 hr |
| **5.3** | Z3-verify all SAT/CSP/arithmetic problems (`verify_dataset.py`) | new script | BLOCKER | REQ-DAT-005 | 4 hr |
| **5.4** | Per-problem ground-truth spec: canonical answer + equivalence classes + synonym sets + negation flag | `problems.json` | BLOCKER | REQ-DAT-006 | 3 hr |
| **5.5** | Per-problem provenance: source / created_date / created_by / license / category / difficulty | `problems.json` | BLOCKER | REQ-DAT-004 | 2 hr |
| **5.6** | Dataset card (HuggingFace-compatible) | `dataset_card.md` | BLOCKER | REQ-DAT-009 | 1 hr |
| **5.7** | Replace 3 paper-verbatim problems from old 6-phase set with newly constructed problems | `problems.json` | BLOCKER | independence | 2 hr |
| **5.8** | Contamination check (Google search top-1; The Pile MinHash @ 8-gram thr 0.5; Common Crawl recent; arXiv full-text) | `contamination_report.md` | BLOCKER | REQ-CON-001 | 4 hr |

**Phase 5 acceptance:** 100 problems across 7 categories; Z3-verified; ground-truth specs complete; contamination report shows no leakage.

---

### PHASE 6 — Harness rewrite (Week 4–7, parallel to Phase 5, ~50 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **6.1** | Provider abstraction: Anthropic, Google, OpenAI native + OpenRouter + vLLM/Ollama local; unified `provider_meta()` | `providers/` | BLOCKER | REQ-PRO-001/002/003 | 8 hr |
| **6.2** | Inspect AI integration: register-only YAML for inspect_evals | `inspect_register.yaml` | BLOCKER | REQ-PRO-004 | 2 hr |
| **6.3** | Token budget enforcement: ≤4K input, ≤8K output; observable error_type for truncation/refusal/overflow | harness | BLOCKER | REQ-SCH-001/002/005 | 3 hr |
| **6.4** | Format scorer rewrite for 15-phase from `format_spec.yaml` (generated function); false-pass/false-fail audit on 50-response gold set | `scorer.py`, `scoring_audit.md` | BLOCKER | REQ-SCO-001/002 | 8 hr |
| **6.5** | Cross-phase consistency rules implementation (Phase 8↔11 *vyāpti/udāharaṇa* match; Phase 13→15 *bhrama* propagation; etc.) | `scorer.py` | BLOCKER | REQ-SCO-005 | 4 hr |
| **6.6** | Pre-registration commit SHA + dataset fingerprint in result JSON; CI rejects publishing if registration is older than run start | `meta.py`, CI | BLOCKER | REQ-STAT-006, REQ-REP-006 | 2 hr |
| **6.7** | Wilson 95% CI on every reported rate; `lint_readme.py` rejects bare `\d+%` | `lint_readme.py`, CI | BLOCKER | REQ-STAT-003, REQ-CI-006 | 2 hr |
| **6.8** | Subgroup reporting: by category, difficulty, source; min n=15 per subgroup or marked underpowered | reporting | BLOCKER | REQ-STAT-004 | 2 hr |
| **6.9** | k=3 runs per problem; per-problem disagreement rate measured | harness | BLOCKER | REQ-STAT-005 | 1 hr |
| **6.10** | Cost + wall-time tracking; cost ≤$15/run, wall-time ≤60 min | harness | BLOCKER | REQ-COST-001/002 | 2 hr |
| **6.11** | Concurrent execution (asyncio.gather + provider rate-limit honor) | harness | BLOCKER | REQ-COST-003 | 4 hr |
| **6.12** | Result JSON schema v1 (`result_schema_v1.json`) | `result_schema_v1.json`, validator | BLOCKER | REQ-OUT-001 | 2 hr |
| **6.13** | CI: AST + pytest, schema validation, smoke run with provider, regression on hand-labeled fixtures, Z3 step on dataset | `.github/workflows/` | BLOCKER | REQ-CI-001/002/003/005/007 | 4 hr |
| **6.14** | Markdown auto-report: `[name] report <result.json>` | reporting | SHOULD | REQ-OUT-004 | 2 hr |
| **6.15** | Run n=100 calibration on Claude Opus 4.7, GPT-5, Gemini 2.5 Pro, Llama-3-70B-Instruct (≥3 frontier from ≥2 vendors + ≥1 open-weights) | calibration | BLOCKER | REQ-STAT-002 | 4 hr (mostly wait) |

**Phase 6 acceptance:** harness runs end-to-end on 4 models; CI green; result JSON validates; calibration data published with full per-problem traces.

---

### PHASE 7 — Review gates (Week 8–9, ~20 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **7.1** | Recruit Sanskrit-fluent reviewer: TS + Tarka-Dīpikā in Sanskrit + ≥1 non-Nyāya darshana familiarity. Candidates: Mithila Institute faculty; Karuṇa Bhattacharya; Ramanuja Tatacharya; Parimal Patil (Harvard); Ganeri's doctoral students | outreach | BLOCKER | SAN-J-001/002, SCH-G-001 | 4 hr |
| **7.2** | Recruit external citation auditor (different person; checks every cite in <15 min per SCH-A-003) | outreach | BLOCKER | SCH-G-002 | 2 hr |
| **7.3** | Approach Indian-philosophy academic for substantive feedback (Harvard South Asian / Oxford OCHS / JNU / BHU / IIT Madras Humanities / Heidelberg STAR) | outreach | BLOCKER | SCH-G-004 | 2 hr |
| **7.4** | Mithila response window closes — incorporate feedback or document disagreements | tracking | BLOCKER | SAN-E-001 | varies |
| **7.5** | Reviewer signoffs in `EXTERNAL_AUDIT.md` and NOTICE | NOTICE, `EXTERNAL_AUDIT.md` | BLOCKER | SCH-G-002, SAN-J-002 | 2 hr |
| **7.6** | Apply reviewer-requested fixes (likely 5–15 specific corrections) | various | BLOCKER | dependent | 4–8 hr |
| **7.7** | Pre-launch checklist sign-off: all three of {Sanskrit reviewer, citation auditor, Indian-philosophy academic} cleared | `LAUNCH_CHECKLIST.md` | BLOCKER | SCH-G-005 | 1 hr |

**Phase 7 acceptance:** three external sign-offs documented; reviewer feedback incorporated or disagreements logged.

---

### PHASE 8 — Companion paper + launch (Week 10–12, ~30 hours)

| FIX | Statement | Files | Severity | Source req | Time |
|---|---|---|---|---|---|
| **8.1** | Draft arXiv companion paper (~6–10 pages): philological grounding, schema design, calibration results, methodological caveats | `paper/main.tex` | SHOULD (strongly) | SCH-G-003 | 16 hr |
| **8.2** | Final NOTICE / README / GLOSSARY / CITATIONS / SCHEMA review for consistency | all docs | BLOCKER | hygiene | 4 hr |
| **8.3** | HuggingFace dataset publication (public 60-problem set with dataset card) | HF Hub | BLOCKER | REQ-FRA-002 | 1 hr |
| **8.4** | Inspect Evals register-YAML PR | inspect_evals repo | BLOCKER | REQ-PRO-004 | 1 hr |
| **8.5** | Public GitHub push: tagged release with all artifacts | GitHub | BLOCKER | launch | 1 hr |
| **8.6** | arXiv submission (preprint, no venue commitment yet) | arXiv | SHOULD | SCH-G-003 | 1 hr |
| **8.7** | Announcement copy: HN / Twitter / LinkedIn — no orientalist or Hindutva framing; n-floor caveats prominent; Sathish + Mithila + Buddhist-logic acknowledgements visible | `launch_copy/` | BLOCKER | SCH-E-004, SAN-H-001/002 | 3 hr |
| **8.8** | Post-launch monitoring: respond to issues, track contamination signals, plan v1.1 refresh cadence | ongoing | NICE | maintenance | ongoing |

**Phase 8 acceptance:** repo public, arXiv preprint live, HF dataset live, announcement posted.

---

## 4. Cross-cutting requirements (always-on standards)

These apply across every phase, not as discrete fixes:

- **CC-1** No bare `\d+%` in any doc — Wilson 95% CI required adjacent
- **CC-2** Every Sanskrit term in IAST in human-facing docs
- **CC-3** Every citation [PRIMARY] / [COMMENTARY] / [MODERN] tagged
- **CC-4** Every claim verifiable in <15 min via archive.org / SEP / named scholar
- **CC-5** No "Vedic" / "ancient wisdom" / "Hindu logic" framings anywhere
- **CC-6** Sathish 2026 cited in NOTICE, README, schema doc, paper
- **CC-7** ModelBrew funding/affiliation disclosed in NOTICE per SCH-E-005
- **CC-8** No PR merges without CI green (AST + pytest + schema validation + Z3)

---

## 5. Risk register (per phase)

| Phase | Top risk | Mitigation |
|---|---|---|
| 0 | Git history conflicts when archiving | Tag explicitly, branch cleanly |
| 1 | Naming choice collision or sandhi error | Run sandhi check + PyPI/GH/arXiv collision search before commit |
| 2 | Missed forbidden phrases in copy | Automated grep in CI; manual final pass |
| 3 | Schema redesign exceeds 25 hr | Buffer to 35 hr; if exceeded, descope to 12-phase |
| 4 | Mithila Institute non-response | Document attempts; parallel Sampūrṇānanda; non-response satisfies the bar per SAN-E-001 |
| 5 | 100 problems is too ambitious in 4 weeks | Drop to 80 with 50/30 split; document the descope |
| 6 | Inspect AI integration breaks UX | Ship standalone first; Inspect register YAML as separate PR |
| 7 | No reviewer found | Extend outreach window; lower bar to "PhD student in Indian philosophy" |
| 8 | arXiv paper exceeds budget | Ship benchmark without paper; companion as v1.1 |

---

## 6. Open decisions still to make

These can be deferred to the relevant phase but listed here for visibility:

1. **OD-1** — Project name (Phase 1, FIX-1.1) — needs founder pick from 4–5 proposed candidates
2. **OD-2** — Dataset license (Phase 1, FIX-1.8) — recommend CC-BY-4.0
3. **OD-3** — Refresh cadence (Phase 8, post-launch) — recommend quarterly versioned releases
4. **OD-4** — Repo identity (personal handle vs new neutral name vs new GitHub org)
5. **OD-5** — Long-term Indian-academic relationship plan (SAN-OQ-008)
6. **OD-6** — Paid commissioned-translation passages (SAN-OQ-005) — defer until Mithila responds

---

## 7. Estimated total effort

| Phase | Solo founder hours | Calendar weeks |
|---|---|---|
| 0 | 2 | 1 day |
| 1 | 10 | Week 1 |
| 2 | 8 | Week 1–2 |
| 3 | 25 | Week 2–3 |
| 4 | 5 (active) + 4 weeks waiting | Week 3 send; Week 7 close |
| 5 | 40 | Week 4–7 |
| 6 | 50 | Week 4–7 (parallel) |
| 7 | 20 | Week 8–9 |
| 8 | 30 | Week 10–12 |
| **Total** | **~190 hours focused** | **~12 weeks calendar** |

Cash budget: ~$165 books + ~$200 calibration API + ~$200 honorarium + ~$50 postage = **~$615**, within the $500–800 envelope.

---

## 8. What requires founder input (blocking the agent)

| Need | When | What founder provides |
|---|---|---|
| Name pick from 4–5 candidates | Phase 1 start | Final name |
| Sign off Mithila letter draft before send | Phase 4 | Approval to send |
| Pick a Sanskrit-fluent reviewer from candidates | Phase 7 start | Named reviewer |
| Approve final NOTICE/README copy before launch | Phase 8 | Approval |

Everything else can be drafted, executed, and tested by the agent without blocking on founder.

---

## 9. Source-requirement traceability

Every fix references a source requirement from one of:
- `REQUIREMENTS_SCHOLAR.md` (SCH-*)
- `REQUIREMENTS_PRACTITIONER.md` (REQ-*)
- `REQUIREMENTS_SANSKRIT.md` (SAN-*)
- `SCHEMA_15_AUDIT.md` (audit Findings)

Phases 0–8 collectively address ~110 of the ~130 requirements across the three RT docs. Remaining ~20 are NICE-to-have items deferred to v1.1+ and listed in §10 below.

---

## 10. Deferred to v1.1+

- LangSmith / Inspect Evals full plugin (vs register-only YAML)
- HELM / lm-eval-harness adapters (NICE)
- Sanskrit-medium publication (SAN-OQ-007)
- Multi-darshana expansion (SAN-OQ-002 — Buddhist parallel)
- Vertical eval sets (audit Path A)
- Quarterly refresh cadence (Practitioner OQ-6)
- Co-author recruitment on paper (SAN/SCH co-authoring)
- IGNCA / Central Sanskrit University engagement (SAN-OQ-006)

---

**End of FIX_PLAN.md**

This document is the master blueprint. All phase work should reference fix IDs (FIX-N.M) for traceability. RT-audit recommended before execution begins.
