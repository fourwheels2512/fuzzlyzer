# Analysis of the three RT requirements documents

**Date:** 2026-05-09
**Source documents:**
- `REQUIREMENTS_SCHOLAR.md` (~6,800 words, 40 reqs)
- `REQUIREMENTS_PRACTITIONER.md` (~3,800 words, ~60 reqs)
- `REQUIREMENTS_SANSKRIT.md` (~5,400 words, ~30 reqs)
- `SCHEMA_15_PHASE_DRAFT.md` (failed draft, REVISE-WITH-MAJOR-FIXES)
- `SCHEMA_15_AUDIT.md` (audit of the failed draft)

---

## Quality scores

| Lens | Falsifiability | Concreteness | Trust |
|---|---|---|---|
| Scholar | High | High | High |
| Practitioner | Highest (numbers everywhere) | Highest | High |
| Sanskrit | High | Highest on domain | High-Medium |

All three did their jobs. They are non-overlapping by design. Total surface ~130 requirements, ~16,000 words.

---

## The single most valuable finding

**The current codebase has an internal contradiction nobody caught before:**

- `pramana_test.py:236-240` lists *Savyabhichara, Viruddha, Prakaranasama, Sadhyasama, Kalaatita* — **Gautama's Old Nyāya** five-fold list (~2nd c. BCE).
- `SCHEMA_15_PHASE_DRAFT.md` Phase 13 lists *savyabhicāra, viruddha, satpratipakṣa, asiddha, bādhita* — **Annambhaṭṭa's Navya-Nyāya** list (17th c.).

The project's own two artifacts disagree about which Nyāya it is. A specialist will read this as ignorance or concealment. **Independent of the schema redesign, this is a real bug in current code.** Both Scholar (SCH-C-002) and Sanskrit (SAN-C-004) flagged it.

---

## What each agent uniquely caught

### Scholar
- Named-edition register (Athalye-Bodas, Phillips-Tatacharya, Ganganatha Jha) — turns "TC 5.1" from unfalsifiable to verifiable
- 15-minute verifiability budget per citation
- Drafting-timeline disclosure (SCH-C-005) — temporal defense against Sathish-mirroring
- Schema-causal vs schema-decorative ablation (SCH-F-003) — control comparison nobody else demanded
- "Worst defensible review" acceptance test as launch gate

### Practitioner
- Cost ceiling ($15/run) + wall-time ceiling (60 min)
- 5MB result-JSON floor — catches cherry-picked traces
- Temp-0 non-determinism *measurement* (k=3 runs per problem)
- Pre-registration commit SHA in result JSON
- Inspect AI register-only-YAML flow (8 May 2026) — low-touch distribution
- Contamination signal monitor (public > private by 2σ → alert)

### Sanskrit
- Line-by-line corrections for ~20+ Sanskrit terms with file:line refs
- Vedic-vs-darshana category error (Navya-Nyāya is NOT Vedic; rejected "Vedic AI" branding residue)
- Buddhist-logic prehistory (Diṅnāga, Dharmakīrti, *trairūpya*, *anyāpoha*) — substantively unavoidable, missed by other lenses
- Living-tradition outreach to named institutions (Mithila Institute, Sampūrṇānanda, Madras Sanskrit College) — 3 letters + 4 weeks
- Multi-school *pramāṇa* enumeration (Cārvāka 1, Buddhist 2, Sāṃkhya 3, Nyāya 4, Mīmāṃsā 6) — prevents sectarian erasure
- Pole A (Hindutva instrumentalization) vs Pole B (extractive orientalism) — both real audiences

---

## Cross-document tensions

1. **Time/cost vs rigor.** Practitioner's "ship fast, $200 calibration, ≤$15/run" vs Sanskrit's "4-week consultation, $200–500 per pandit translation." Both can be true but launch timeline absorbs the slower one.
2. **Schema scope: 6-phase or 15-phase?** Practitioner OQ-1: ship 6-phase as v1.0, 15-phase as v2.0-experimental. Scholar and Sanskrit assume 15-phase. User wanted 15.
3. **Sample size floors.** Scholar n≥50, Practitioner n≥100. Higher wins.
4. **Decolonial framing tone.** Sanskrit's S5 reviewer more activist than Scholar's R3. Defensible but tendentious in places.

---

## Gaps (all three docs missed)

- Project name and repo identity (all flag, none decide; user already said: not modelbrew, not pramana-eval)
- Launch timeline / sequencing
- Funding model for translations + consultations (who pays the pandit?)
- Fate of existing 6-phase work (keep, archive, migrate?)
- Co-maintainer recruitment

---

## Quality concerns

- Scholar agent over-length (agent itself flagged this)
- Sanskrit agent's "Vedic AI" obsession re-litigates a point already settled by user
- Practitioner's k=3 runs may be premature for v1.0 (triples cost) — could be SHOULD not MUST

---

## Top critical findings (the launch-blocker MUSTs)

1. Fix the Old-Nyāya / Navya-Nyāya *hetvābhāsa* list contradiction in current code
2. All Sanskrit terms → IAST + sandhi correction (afternoon's work, blocking)
3. Citation discipline: named editions, no decimal section numbers
4. n≥100 with private holdout + multi-model coverage
5. Sanskrit-fluent + non-Nyāya darshana reviewer signoff before launch
6. Living-tradition consultation attempt (4-week minimum window — start early)
7. Sathish citation in all artifacts + honest 30–45% contribution range
8. Cross-phase consistency rules must pass on the schema's own worked example

---

## Overall judgment

The three documents are complementary:
- Scholar = "will this survive peer review at JIP/PEW?"
- Practitioner = "will this survive an ML engineer trying to use it?"
- Sanskrit = "will this survive a pandit reading it?"

Skip any one and you fail with that audience.

The single highest-leverage next move is fixing the *hetvābhāsa*-list contradiction in current code — that's a real bug independent of the schema redesign and it gates everything downstream.

---

## Files saved at project root

- `SCHEMA_15_PHASE_DRAFT.md` — failed 15-phase draft
- `SCHEMA_15_AUDIT.md` — audit of the failed draft
- `REQUIREMENTS_SCHOLAR.md` — scholar lens RT
- `REQUIREMENTS_PRACTITIONER.md` — practitioner lens RT
- `REQUIREMENTS_SANSKRIT.md` — Sanskrit/Vedas lens RT
- `ANALYSIS_RT_REQUIREMENTS.md` — this file

**End of analysis.**
