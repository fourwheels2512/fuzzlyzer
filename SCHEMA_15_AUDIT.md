# Red Team Audit: 15-Phase Navya-Nyaya Reasoning Schema

**Auditor disposition:** Hostile. Default to flagging.
**Audit date:** 2026-05-09
**Document audited:** `SCHEMA_15_PHASE_DRAFT.md` (v0.1)
**Sources consulted:** SEP (Gangesha), Wikipedia (Nyaya Sutras, Tarka-Sangraha, Vyapti, Udayana), Wisdomlib (multiple essays on Navya-Nyaya inference, hetvabhasa, anupalabdhi, samshaya), arXiv:2604.04937 (Sathish 2026 abstract), Wisdomlib Vatsyayana commentary articles.

---

## 1. Verdict

**REVISE-WITH-MAJOR-FIXES.**

The schema's *architecture* is more defensible than its *citation hygiene*. The 15-phase decomposition is operationally workable for at least 11 of the 15 slots, and the Navya-Nyaya core (Phases 8/9/10) is a real divergence from Sathish 2026. But the citation layer — specifically the granular section references like "Tattvacintamani Anumana-khanda 5.1" and "Tarka-Sangraha §43" — does not survive scrutiny. At least one substantive misattribution (the "vyapti-pancaka = survey of 24 prior definitions" claim in Phase 8) was found. If shipped without remediation, a single competent Sanskrit-philosophy reviewer in the OSS community can dismantle the foundational-document claim in an afternoon. The schema is not approve-as-is; it is also not REJECT, because the underlying philosophical structure is sound. Fix the citations, redesign Phase 7 vs Phase 11, and tighten Phases 12 vs 13 (overlap risk), and the schema is shippable.

The single most damaging finding is in **Lens 1 (citation verification)**: the section-numbering scheme used throughout the document (e.g., "TC Anumana-khanda 5.1", "Tarka-Sangraha §43", "§§54-67") does not correspond to any standardized edition that this auditor could verify. These citations have the *appearance* of granular primary-source grounding while in fact riding on (a) the Stanford Encyclopedia article's *thematic* subsection numbering (5.1, 5.2, 5.3), which is SEP's editorial structure, NOT TC's; and (b) the Athalye/Bahulikar editions' paragraph numbering, which is *not standardized* across editions. This is the kind of error that makes a benchmark unshippable as "primary-source grounded."

---

## 2. Citation Verification Table

| # | Phase | Citation as claimed | Verified? | Evidence consulted | Severity if wrong |
|---|---|---|---|---|---|
| 1 | Phase 8 (Vyapti) | "Tattvacintamani Anumana-khanda 5.1 — the famous *vyapti-pancaka* (Gangesha's survey of and answer to twenty-four prior definitions of pervasion)" | **NO — substantive error** | Wisdomlib "Anumana-Inference in Nyaya" 4.3.2 lists *5 provisional definitions + 6 vishesha-vyapti = 11 definitions*, not 24. SEP says "twenty-four attempts" but this is in TC's broader polemics, *not* the vyapti-pancaka. The drafter has conflated two different things: (a) vyapti-pancaka = Gangesha's own five-fold analytic treatment, (b) the "twenty-four" rejected definitions are a separate polemic *preceding* the pancaka. The "5.1" subsection number appears to be the SEP article's editorial structure, not TC's. | **CRITICAL** |
| 2 | Phase 9 (Upadhi) | "Tattvacintamani Anumana-khanda 5.3. SEP notes Gangesha devotes 'more than a tenth' of the inference chapter to *upadhi*. Tarka-Sangraha §52." | **PARTIALLY** — the "more than a tenth" SEP quote is verified; the "5.3" section number is SEP's thematic numbering, not TC's; "TS §52" is unverified — Tarka-Sangraha has no canonical §-numbering scheme. | SEP fetch confirmed the "more than a tenth" phrasing. Wikipedia Tarka-Sangraha article gives no §-numbering. Athalye and Bahulikar editions use *paragraph* numbering that varies between editions. | **MAJOR** |
| 3 | Phase 10 (Paramarsa) | "Tattvacintamani Anumana-khanda 5.2" | **NO** — same problem as #1 and #2: the "5.2" is SEP editorial, not TC. | SEP article's structure has subsections 5.1/5.2/5.3 by topic, which the drafter has apparently mistaken for TC section numbers. | **MAJOR** |
| 4 | Phase 13 (Hetvabhasa) | "Tarka-Sangraha §§54–67 (5-fold Navya list, not Gautama's older 5-fold list)" | **PARTIALLY** — the *content* (Annambhatta's Navya five-fold: savyabhicara, viruddha, satpratipaksa, asiddha, badhita) is confirmed by Wisdomlib article on Anumana in Indian Philosophy. The §-range is unverified and likely fabricated. | Wisdomlib confirms Annambhatta's 5-fold list with the same five names. Asiddha sub-types (asraya-, svarupa-, vyapyatva-) confirmed. The "§§54-67" range, however, has no source. | **MAJOR (citation), MINOR (content)** |
| 5 | Phase 6 (Pramana) | "Tarka-Sangraha §§30–47 (Annambhatta's treatment of each, including *alaukika pratyaksa*)" | **NO** — §-numbering unverified. *Content* about alaukika pratyaksa subtypes (samanyalaksana, jnanalaksana, yogaja) is canonical. | Wisdomlib article on alaukika-pratyaksa confirms the three subtypes. The §§30-47 range itself has no canonical source. | **MAJOR (citation)** |
| 6 | Phase 2 (Samshaya) | "NS 1.1.23 + Vatsyayana, listing five sources of doubt: samana-dharma, aneka-dharma, vipratipatti, upalabdhi-avyavastha, anupalabdhi-avyavastha" | **YES** — verified. | Wisdomlib essays confirm: Vatsyayana lists five sources; *Vacaspati and Uddyotakara reduce to three.* The schema correctly uses Vatsyayana's five-fold. | OK |
| 7 | Phase 3 (Prayojana) | "NS 1.1.24 ('yam artham adhikrtya pravartate tat prayojanam' — Vatsyayana's gloss)" | **YES** — the Sanskrit phrase is canonical Vatsyayana commentary on prayojana. The Springer JIP article in the Sources list is on this exact passage. | Wikipedia Nyaya Sutras article and IEP Nyaya entry both cite NS 1.1.24 for prayojana. | OK |
| 8 | Phase 4 (Siddhanta) | "NS 1.1.26–29 (the four-fold typology of *siddhanta*: sarvatantra, pratitantra, adhikarana, abhyupagama)" | **YES** — the four-fold typology is canonical. Sutra range plausible. | Standard reference. The four-fold is well-attested in IEP/Wikipedia. | OK |
| 9 | Phase 13 — Sathish's hetvabhasa list | Drafter claims Sathish uses "Gautama's older 5-fold list (savyabhicara, viruddha, prakaranasama, sadhyasama, kalatita)" | **UNVERIFIED — not directly checkable from abstract** | Sathish abstract enumerates only the 6 phase names. Whether Sathish's *hetvabhasa* sub-list inside Phase 5 actually uses Gautama's list (versus Annambhatta's) was not verifiable from the abstract. **The drafter's claim about Sathish's specific list is asserted without showing receipts.** | **MAJOR** — if Sathish actually used the Navya list, the schema's "fourth major divergence" collapses. |
| 10 | Phase 14 (Anupalabdhi/Abhava) | "Atmatattvaviveka; Tarka-Sangraha on abhava; Raghunatha *Padartha-Tattva-Nirupanam*" | **YES (general)** — Udayana's perceptual treatment of absence (visesanata-pratyasatti) is well-attested. | Wisdomlib entries on Nyayakusumanjali on Anupalabdhi; Wikipedia Udayana. Raghunatha's treatment of abhava as a real category is canonical. | OK at the general level; specific section citations not verified. |

**Net citation health: 4 of 10 sampled citations have CRITICAL or MAJOR issues. The pattern is: the *content* is usually defensible, but the *granular section numbering* ("5.1", "§43", etc.) is fabricated or pulled from secondary editorial structures and mislabeled as primary-source location.**

---

## 3. Findings by Lens

### Lens 1 — Primary Source Verification (most important)

**Finding 1.1 [CRITICAL]: The TC "Anumana-khanda 5.1 / 5.2 / 5.3" section numbers are SEP editorial, not TC structure.**
The Stanford Encyclopedia of Philosophy article on Gangesa uses subsection numbers (5.1, 5.2, 5.3, 5.4) as its own thematic organization. Gangesha's *Tattvacintamani* itself does not use this kind of decimal-section numbering — it is organized by *khanda* (Pratyaksa, Anumana, Upamana, Sabda) and within each by topic-sections without standardized decimal numerals. The drafter has lifted SEP's editorial subsection numbers and presented them as if they were primary-text locations. A reader trying to find "Tattvacintamani Anumana-khanda 5.1" in a Sanskrit edition will fail.
**Fix:** Replace with verifiable references — e.g., "the *vyapti-pancaka* section of TC's Anumana-khanda" (no decimal), or cite the specific Bombay Sanskrit Series page range, or cite Mohanty's *Gangesa's Theory of Truth* (1989) by page.

**Finding 1.2 [CRITICAL]: The vyapti-pancaka is NOT a survey of 24 prior definitions.**
Phase 8 says "*vyapti-pancaka* (Gangesha's survey of and answer to twenty-four prior definitions of pervasion)." Two distinct things have been collapsed:
- *Vyapti-pancaka* is Gangesha's *own* exposition of *five* analytic formulations of pervasion.
- The "twenty-four (close) attempts" is a separate polemical pre-survey, mentioned in SEP, that Gangesha rejects *before* arriving at his own formulation.
Wisdomlib's "Anumana — Inference in Nyaya" 4.3.2 explicitly lists 5 provisional + 6 vishesha-vyapti = 11 definitions — neither matches "24". This is exactly the kind of citation error a Sanskritist reviewer will spot in seconds.
**Fix:** Either (a) drop the "twenty-four" claim and call it "*vyapti-pancaka*: Gangesha's five-fold analytic treatment of pervasion in TC's Anumana-khanda," or (b) keep "twenty-four" but disambiguate: "Gangesha rejects ~24 prior attempts before presenting his own five-fold treatment (*vyapti-pancaka*)."

**Finding 1.3 [MAJOR]: Tarka-Sangraha §-numbering has no canonical source.**
The schema cites "TS §43", "§49", "§50", "§52", "§§54-67". Tarka-Sangraha is "about 15 pages" (per Wikipedia) and follows the *uddesa-laksana-pariksa* schema; standard editions (Athalye 1918, Bodas 1918, Bahulikar) use *paragraph numbering* that is **not stable across editions**. The Athalye edition itself notes that some paragraphs were renumbered between editions. The §-numbering used in the schema appears to be drafter-fabricated or drawn from a single edition without disclosure.
**Fix:** Either cite the specific edition (e.g., "Athalye 1918 §43") or replace with topic references ("Annambhatta's section on *vyapti*"). Better: cite Bahulikar's English translation by page number from the publicly available PDF.

**Finding 1.4 [MAJOR]: The claim about Sathish's hetvabhasa list is unverified.**
The schema's headline divergence #4 is that Sathish uses Gautama's older 5-fold list while this schema uses Annambhatta's Navya 5-fold. The Sathish abstract (verified) names only the 6 phases — it does not specify which hetvabhasa list he uses inside Phase 5. The drafter asserts that Sathish uses Gautama's list (savyabhicara, viruddha, prakaranasama, sadhyasama, kalatita) without quoting the paper. **If Sathish actually uses the Navya list, the "fourth major divergence" collapses entirely, and the net independent contribution drops from ~60% to ~50%.**
**Fix:** The drafter must read Sathish's full paper and quote the exact list Sathish uses. Without this verification, the divergence claim is unsupported.

**Finding 1.5 [MAJOR]: Phase 11's source citation is internally inconsistent.**
Phase 11 cites "NS 1.1.32–39" for the five-membered argument. But standard accounts cite NS 1.1.32 (the enumeration) and 1.1.33–37 (the individual members), with 1.1.38–39 covering related material. The Wikipedia Nyaya Sutras article confirms NS 1.1.32 and NS 1.1.39 as the relevant sutras for the five members; the inclusive range "1.1.32–39" is plausible but unverified. Mild compared to other findings.
**Fix:** Verify against the Ganganatha Jha translation (linked in archive.org) and tighten.

**Finding 1.6 [MINOR]: NS 1.1.23 vs the actual definitional sutra.**
Phase 2 cites "NS 1.1.23 (definition of *samshaya*)." Different editions number this differently — some give the samshaya definition at NS 1.1.23, others at 1.2.1 (or place it in adhyaya 2.1). Confirm against the same edition used elsewhere in the schema.
**Fix:** Pin the edition.

---

### Lens 2 — Operational Distinctness

**Finding 2.1 [MAJOR]: Phases 12 and 13 risk operational collapse on adversarial problems.**
Phase 12 is *tarka* (counterfactual reductio); Phase 13 is *hetvabhasa-pariksa* (fallacy audit). The drafter claims they are operationally distinct. In the worked example (§5), Phase 12's "Contradiction with: Phase 4's *pratitantra*" and Phase 13's "Satpratipaksa: present-but-managed" both flag the *same* underlying issue: the puzzle's underdetermination. A frontier LLM running this prompt would predictably produce *overlapping diagnostic content* in both phases. The conceptual distinction (negate-and-derive vs. audit-against-known-patterns) is real, but the *output* on a given problem is often the same observation phrased two ways.
**Fix:** Add an explicit anti-duplication rule in the prompt: "Content that appeared in Phase 12 cannot be the primary fallacy claim in Phase 13; if the same defect is the relevant one, Phase 13 must cite *which named Navya fallacy* in canonical form, while Phase 12 must show the derivation steps."

**Finding 2.2 [MAJOR]: Phase 9 (upadhi) and Phase 13's *satpratipaksa* + *asiddha* are conceptually adjacent in ways that will collapse in practice.**
The drafter argues that *upadhi* is "an external condition that secretly carries the inference" while a *hetvabhasa* is "intrinsic to the hetu's relation to sadhya." This distinction is real in classical texts but is *very subtle* and frontier LLMs will not reliably distinguish them under output pressure. *Vyapyatva-asiddha* (a sub-type of asiddha named in Phase 13) means "concomitance with major term cannot be proved" — which is precisely what an upadhi shows. The same defect can be reported as either, and a strict rubric will sometimes mark the model wrong for a correct-but-other-categorized observation.
**Fix:** Either (a) add an explicit *priority* rule (if the defect is a hidden enabling condition, it goes in Phase 9; if it is an unestablished attribution, it goes in Phase 13), or (b) accept overlap and design the scorer to credit the model when *either* phase catches the defect.

**Finding 2.3 [MINOR]: Phase 1 and Phase 5 have a cleaner separation than Phase 5 and Phase 11's *pratijna*.**
Phase 5 specifies *answer type*; Phase 11's *pratijna* states "*<paksa>* has *<sadhya>*". The pratijna already implicitly carries the answer's structure. On simple problems Phase 5 will be a strict subset of what Phase 11 commits to. Operationally distinct, but the redundancy hurts in token budget.
**Fix:** Acceptable, but note in the schema doc.

**Finding 2.4 [MAJOR]: Phase 7 (drstanta) and Phase 11.3 (udaharana) — distinction is too fine.**
The drafter pre-flagged this in §7. The auditor concurs: this distinction *exists* in the texts but is not crisp enough to support a separate phase. On practical problems, models will produce structurally identical content for both, with minor wording variation. The drstanta will tend to be a slightly-rephrased udaharana. This is ~one wasted phase.
**Fix:** Merge into Phase 11 with a sub-field "external paradigm reference" requirement, OR keep Phase 7 but redesign it to mandate *cross-domain transfer* (e.g., "the drstanta must be from a different problem class than the paksa") and *enforce that mechanically* via domain-tag detection.

**Finding 2.5 [MAJOR]: Phase 3 (prayojana) is structurally vulnerable to LLM confabulation.**
The "Downstream use of the answer" field will be filled with plausible-sounding but unverifiable text on most benchmark problems. A logic puzzle has no real-world prayojana; the model will manufacture one. This phase carries less weight than its slot suggests.
**Fix:** Either (a) make Phase 3 conditional on problem class, or (b) replace "downstream use" with the harder, more checkable "what would change if the answer flipped" — and *only* score that field.

---

### Lens 3 — Evaluability

**Finding 3.1 [CRITICAL]: Phase 4 (siddhanta-vibhaga) cannot be reliably scored.**
The four-fold typing (sarvatantra/pratitantra/adhikarana/abhyupagama) requires the scorer to know which background tenets *should* be classified into which bucket, *for every problem in the benchmark*. There is no mechanical rule. An LLM-judge will be inconsistent. Two judges will disagree. The example in §5 marks "Classical 2-valued logic" as sarvatantra and "exactly one suspect is guilty" as pratitantra — defensible but non-unique. A different reasonable annotation would mark "exactly one is guilty" as adhikarana (granted for the puzzle).
**Fix:** Either (a) provide the correct typing as part of the gold answer for each problem (heavy annotation cost), or (b) score only that *some* typing is provided and that the typings are *internally consistent* (lighter, but loses semantic signal).

**Finding 3.2 [MAJOR]: Phase 7's "Why both layperson and expert would accept this" is unscorable except by LLM-judge.**
This is a hand-wavy criterion. Two judges will give different verdicts on the same drstanta. The format check is trivial; the semantic check is undefined.
**Fix:** Replace with a *concrete* criterion: "the drstanta must be a textbook example named in at least one introductory logic / mathematics resource." Make it lookup-checkable.

**Finding 3.3 [MAJOR]: Phase 9's "Pervades sadhya: yes/no | Pervades hetu: yes/no" is fragile.**
The two pervasion claims are themselves universal generalizations that are hard to verify mechanically. The drafter's worked example marks the candidate upadhi "speakers are speaking sincerely" as "pervades sadhya: yes; pervades hetu: no" — but this is a contestable judgment. A scorer cannot reliably grade this without an LLM-judge that itself has access to a Navya-Nyaya specialist's training. Realistically, any answer that *fills the four cells* will pass; the semantic check is undefined.
**Fix:** The phase remains valuable for forcing the model to *consider* upadhi, but accept that scoring it gives a weak signal. Document this honestly in the scoring methodology.

**Finding 3.4 [MAJOR]: Phase 12's "Does the contradiction actually follow from the supposition?" is essentially asking the LLM-judge to re-do the reasoning.**
This is a hidden-cost evaluation. To grade Phase 12, the judge must run the entire reductio chain and verify it. On hard problems, the judge will be wrong sometimes, and a Phase-12 score is then mostly noise.
**Fix:** Restrict Phase 12 evaluation to *structural* checks (presence of "Suppose:", "Contradiction with: Phase X", at least one intermediate step) and *do not* attempt to grade the reductio's validity automatically. Save the reductio-validity check for human spot-audit on a small sub-sample.

**Finding 3.5 [MINOR]: Phase 14's "Counterfactual locus" criterion is hand-wavy.**
"Is the counterfactual locus the right place to have looked?" requires problem-by-problem judgment.
**Fix:** Acceptable as LLM-judge criterion but score as a soft signal, not a hard pass/fail.

**Finding 3.6 [MAJOR]: Cross-phase consistency rules need a formal scorer that doesn't yet exist.**
The schema repeatedly invokes "cross-phase consistency check" (Phase 10's hetu must match Phase 8 verbatim; Phase 11's udaharana must match Phase 8's vyapti; Phase 15 must reflect Phases 9, 13, 14 verdicts). Each of these requires specific implementation:
- "Verbatim match" — what counts? Lemmatization? Stem match? Synonym tolerance? Without specifying, the scorer will be brittle.
- The Phase 11 → Phase 8 udaharana match in the worked example actually fails if checked literally: Phase 8 vyapti is "if exactly-one-thief and exactly-one-statement-true are simultaneously satisfied"; Phase 11 udaharana is "wherever a constraint set admits multiple satisfying models." Different content.
**Fix:** Either spec the matching algorithm precisely (recommend: bag-of-lemmatized-content-words, ≥0.6 Jaccard) or downgrade these to "semantic correspondence (LLM-judge)" with documented unreliability.

---

### Lens 4 — Sathish-mirroring under disguise

**Finding 4.1 [MAJOR]: Net-independent-contribution claim of "60%" is high.**
Honest accounting:
- Sathish's 6 phases map to the schema's Phases 2, 6, 11, 12, 13, 15. That's 6/15 = **40% direct preservation** — schema's own claim. Defensible.
- Phases 1, 3, 4, 5 are framing/typing additions. These are *real additions* but each is a small slice; collectively ~25% of weight.
- Phases 7, 8, 9, 10 are the Gangesha-derived core. Phase 8 (vyapti as separate commitment) is **the** real divergence. Phase 10 (paramarsa) is a real second divergence. Phase 9 (upadhi) is a real third divergence. Phase 7 (drstanta separated from udaharana) is *not* substantively new — a Sathish defender will say "this is just a sub-bullet of pancha avayava."
- Phase 14 (anupalabdhi) is a real divergence.
- Phase 13's "Navya list vs Gautama list" claim is a divergence *only if* the unverified claim about Sathish's list is correct. If Sathish actually used Annambhatta's list, Phase 13 collapses to Sathish's Phase 5 with cosmetic differences. **This is a load-bearing claim that has not been verified.**

**Honest net independent contribution: 30–45%, depending on (a) whether the Sathish list claim survives, and (b) whether Phase 7 is separately defensible. Not 60%.**

**Finding 4.2 [MAJOR]: A Sathish defender's strongest counterattack would be on Phase 9 (upadhi) being implicit in Sathish's hetvabhasa Phase.**
*Vyapyatva-asiddha* (sub-type of asiddha, present in both Gautama's and Annambhatta's lists) covers cases where the concomitance cannot be proved. A Sathish defender would say: "Upadhi-pariksa is just a careful name for vyapyatva-asiddha analysis, which is already in my Phase 5." The drafter's distinction (upadhi as external condition vs. hetvabhasa as intrinsic defect) is correct but textually subtle. The defender's argument has surface plausibility — *this is the rebuttal to anticipate.*
**Fix:** In the schema document, add a paragraph explicitly comparing upadhi vs. vyapyatva-asiddha and showing they are *not* coextensive, with classical citation. Without this, the most predictable academic attack on the schema is unanswered.

---

### Lens 5 — LLM Implementability Stress Test

**Finding 5.1 [CRITICAL]: 15-phase format-adherence will be substantially below 6-phase. The 9/10 Gemini result will not transfer.**
The user's pramana-eval result is 9/10 format adherence on a 6-phase schema. With 15 phases, format-adherence is structurally harder by combinatorics (more fields to get right, more cross-references to align). Sathish reports 40% strict format adherence on 6-phase. The user's result of 90% on 6-phase is plausibly because of stricter prompt engineering / smaller / cleaner test set. **For 15 phases, realistic expected strict format adherence is 50–70% even on Gemini 2.5 Pro / GPT-5 class models, possibly lower on first run.**
**Fix:** Plan for format-relaxed scoring (e.g., partial credit per phase) before launch; do not assume the 9/10 number transfers.

**Finding 5.2 [MAJOR]: Token budget concern.**
A full 15-phase response with all sub-fields populated, on a non-trivial problem, will run **3,000–6,000 tokens**. On a benchmark of 1,000 problems × 5 models, that is substantial. Many open models have 4K–8K output limits; this schema fills them. Several frontier models have *thinking* budgets that will further bloat the trace.
**Fix:** Provide an explicit per-phase token budget in the system prompt. Specify maximum 1–3 sentences per sub-field, hard.

**Finding 5.3 [MAJOR]: Phases 8/9/10 are the most cognitively demanding *and* the most likely to be mistakenly merged by smaller models.**
On models below frontier (e.g., 7B-class open models the OSS community will benchmark with), Phases 8, 9, and 10 will collapse into "the inference step." Distinguishing pervasion-statement, hidden-condition-search, and paksa-application requires sustained meta-cognition that small models lack. The benchmark will show *near-zero* score on these three phases for sub-frontier models, which is informative *if* the framing is "these phases are the discriminating signal" — but it must be framed honestly.
**Fix:** Add a "model-class expectations" section to the launch documentation.

**Finding 5.4 [MAJOR]: Phase 13's five-fallacy enumeration will trigger over-flagging.**
Forcing the model to answer "present | absent" for *each* of five fallacies on every problem creates pressure to mark something "present" to look thorough. On clean problems this will produce false positives that then propagate to Phase 15 (`Bhrama`). Cross-phase consistency rule "if any fallacy present, status = Bhrama" amplifies the damage.
**Fix:** Require the *reason* for "present" to be substantive (LLM-judge or keyword check); allow Phase 15 to override `Bhrama` to `Sasamshaya` if Phase 13's "present" finding is weak.

**Finding 5.5 [MINOR]: Model-output ordering may not honor the schema's logical-dependency graph.**
Even with explicit ordering instructions, frontier models sometimes back-fill earlier phases after committing to a conclusion. The schema's "no forward references" rule is good but not enforceable from the output alone.
**Fix:** Acceptable risk; flag in launch documentation.

---

### Lens 6 — Cross-Phase Consistency Claims

**Finding 6.1 [MAJOR]: "Phase 11's syllogism tokens must match Phase 8 vyapti tokens" — unwinnable on phrasing variation.**
The worked example fails this check on a strict reading (different content in Phase 8 vyapti vs. Phase 11 udaharana). The author of the schema *himself* did not satisfy his own cross-phase rule in his worked example. This is a self-inflicted scorer-brittleness problem.
**Fix:** Loosen to "semantic equivalence (LLM-judge)" or relax to "shared content lemmas ≥ 0.5 overlap."

**Finding 6.2 [MAJOR]: "If Phase 13 marks any fallacy 'present', Phase 15 must be Bhrama" — creates contradictions with Phase 12's confirmation.**
In the worked example, Phase 13 marks *satpratipaksa* "present-but-managed" while Phase 15 reports *Sasamshaya* (not Bhrama). The drafter's own example violates the rule. The "present-but-managed" hedge is doing all the work, but the rule says "any present → Bhrama" without qualification.
**Fix:** Add a fourth value to Phase 13's tag-set: `present | absent | present-but-managed`. Rule then becomes: any unmanaged "present" → Bhrama; "present-but-managed" → Sasamshaya allowed.

**Finding 6.3 [MAJOR]: Phase 14 ↔ Phase 5 consistency rule is too tight.**
Rule: "If Phase 5 said the answer type is 'no such X exists' (negative), then Phase 14 must say `yes`." But many problems have a negative *answer* without making a "no X exists" *claim* — they have a Boolean false answer. Phase 5's category "no such X exists" is narrower than "negative answer." The rule will produce false-negatives on the consistency check.
**Fix:** Separate "answer is False" (Boolean negative) from "answer is the non-existence of X" (existential negative). Apply Phase 14 only to the latter.

**Finding 6.4 [MINOR]: Phase 9's verdict integration with Phase 15 is undefined.**
"If Phase 9 found an unrestricted upadhi, status must be `Sasamshaya` or `Bhrama`." But if the upadhi is restricted by Phase 4's *abhyupagama* (as in the worked example), the rule does not specify what Phase 15 should do. The example says "Phase 4's *abhyupagama*, so the restriction is satisfied" — but then Phase 15 is `Sasamshaya` for *other* reasons. The rule's interaction with restriction-satisfaction is underspecified.
**Fix:** Spell out the four cases explicitly: (no upadhi found / upadhi found and restriction satisfied / upadhi found, restriction not in scope / upadhi found, defeats inference) → Phase 15 verdict.

---

### Lens 7 — Verifying the Pre-Flagged Weaknesses

The drafter pre-flagged five. Auditor's ranking:

1. **Phase 1 (Jijnasa) prefatory grounding.** Drafter's self-assessment: "weakest classical anchor". Auditor verdict: **understated**. The phase is fine in operation but the citation is genuinely weak. *Jijnasa* is a Vatsyayana commentary term, not a sutra term. The rebuttal "every Nyaya treatise opens with this" is true but does not promote the citation. **Recommend: relabel as "design synthesis" rather than "primary-grounded".**

2. **Phase 5 (Prameya repurposing).** Drafter: "pedagogical extension". Auditor verdict: **accurate**. The repurposing is honest but a Sanskrit-pure reviewer will object. Fixable by adding a 1-sentence concession in the phase description. *Most fixable of the five.*

3. **Phase 9/10 reorder (upadhi before paramarsa, against Gangesha's expository order).** Drafter: "benchmark-driven reordering". Auditor verdict: **accurate; less damaging than feared**. The cognitive logic argument is sound. A Navya-Nyaya scholar will note the reversal but will not call it fatal. *Acceptable as-is.*

4. **Annambhatta vs Gautama hetvabhasa.** Drafter: claims Annambhatta's list is sharper. Auditor verdict: **the substance is correct, but the divergence-from-Sathish claim is unverified** (see Finding 1.4). *This is the most damaging of the five if the Sathish list claim falls.*

5. **Phase 14 placement.** Drafter: "rejected as a sixth pramana, treated as a phase". Auditor verdict: **understated**. Phase 14 will trigger trivially-no on most non-negative problems and add benchmark noise. The "balanced benchmark mix is required" caveat is real and load-bearing for Phase 14's value.

**Most damaging of the five if left unfixed: weakness #4 (the Sathish list claim).** Because the schema's "60% independent contribution" claim depends on it, and verification is one paper-read away.

---

## 4. The Single Biggest Flaw

**The schema's granular section-citation scheme ("Tattvacintamani Anumana-khanda 5.1", "Tarka-Sangraha §43", "§§54–67") is a stylistic appearance of primary-source rigor that does not survive even cursory verification.** The decimal-section numbers for TC are SEP editorial structure, not TC's own; the §-numbers for Tarka-Sangraha do not exist in any standardized edition. A reviewer with a Sanskrit library — exactly the type of reviewer the project needs to validate the benchmark as a "primary-source-grounded" foundation — will spot this within the first hour and conclude that the document is *gestures toward* primary-source grounding rather than *delivers* it. This is the kind of finding that, once raised in OSS-community discussion (HN, Reddit r/IndianPhilosophy, ML/Twitter/X), will durably damage the project's credibility regardless of whether the underlying philosophical structure is sound. The structure is mostly sound; the citations are the embarrassment risk.

---

## 5. Recommended Action

**REVISE-WITH-MAJOR-FIXES.** Concrete actions, ranked by criticality:

1. **CRITICAL — Replace fabricated section numbers with verifiable references.** Audit every "TC Anumana-khanda X.Y" and "TS §N" reference. Replace with: (a) topic-name references ("the *vyapti-pancaka* section"), (b) page references to a specific named edition ("Bahulikar 1985 p. 47"), or (c) standard sutra numbers where applicable. *Estimated effort: 2 hours.*

2. **CRITICAL — Fix the vyapti-pancaka / 24-definitions conflation in Phase 8.** Either drop the "twenty-four" or disambiguate it from the *vyapti-pancaka* (which is five). *Estimated effort: 15 minutes.*

3. **MAJOR — Verify Sathish's actual hetvabhasa list.** Read the full Sathish 2026 paper. Quote the exact list. If it is Gautama's, the divergence claim stands. If it is Annambhatta's, demote Phase 13 from "fourth major divergence" to "preserved with sharper consistency rules." *Estimated effort: 1 hour.*

4. **MAJOR — Resolve Phase 7 vs Phase 11 (drstanta vs udaharana) in one of two ways:** (a) Merge Phase 7 into Phase 11 as a sub-field, recovering one phase slot; or (b) Add a *mechanically enforceable* domain-tag rule to make drstanta out-of-domain. *Estimated effort: 30 minutes design + reflow downstream phase numbers.*

5. **MAJOR — Add anti-collapse rules between Phases 9 and 13, and between Phases 12 and 13.** Specify priority / non-duplication. *Estimated effort: 30 minutes.*

6. **MAJOR — Tighten cross-phase consistency rules to be mechanically scorable.** Specify the matching algorithm (lemma overlap, Jaccard threshold, etc.). Add the `present-but-managed` tag to Phase 13. Separate "Boolean false" from "no such X exists" in Phase 14. *Estimated effort: 1 hour.*

7. **MAJOR — Plan for 50–70% format adherence on 15-phase, not 90%.** Add per-phase token budgets to the prompt. Plan partial-credit scoring. Document model-class expectations. *Estimated effort: 2 hours.*

8. **MAJOR — Add the upadhi vs. vyapyatva-asiddha defense paragraph.** Pre-empt the Sathish defender's strongest counterattack. *Estimated effort: 30 minutes.*

9. **MINOR — Recalibrate the "60% independent contribution" claim to "30–45%, mostly via Phases 8/9/10/14".** Honesty here strengthens, not weakens, the document. *Estimated effort: 5 minutes.*

10. **MINOR — Document the §7 weaknesses more sharply.** Phase 1 grounding is weaker than the drafter says; Phase 14 placement risks are larger than the drafter says. *Estimated effort: 15 minutes.*

**Total estimated remediation effort: 8–10 hours.** After remediation, the schema is **APPROVE-WITH-MINOR-REVISIONS-grade** and ready for OSS launch.

If the user proceeds without remediation, the most likely failure mode is *credibility damage from the citation layer* — a blog post or HN comment from a Sanskritist tearing apart the section-numbering scheme — rather than philosophical-architectural failure. The architecture is mostly sound; the citation hygiene is not.

---

**End of audit.**
