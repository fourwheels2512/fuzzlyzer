# Scholar-Lens Requirements for pramana-eval

**Document type:** Adversarial requirements (scholarly/academic lens)
**Date:** 2026-05-09
**Audience:** Founder of pramana-eval (Kiran Nayudu, ModelBrew AI), pre-launch
**Purpose:** Specify falsifiable conditions the project must satisfy to survive scrutiny by Sanskrit philologists, Navya-Nyaya specialists, and peer reviewers at venues like NeurIPS, ACL, *Journal of Indian Philosophy* (JIP), and *Philosophy East and West* (PEW).
**Companion document:** `REQUIREMENTS_PRACTITIONER.md` (ML lens — out of scope here).
**Posture:** Hostile. The project's previous internal draft (`SCHEMA_15_PHASE_DRAFT.md`) failed audit (`SCHEMA_15_AUDIT.md`) on citation hygiene, terminological misuse, unverified comparison claims, and inflated independent-contribution accounting. These requirements pre-empt those failures and others.

---

## 1. The reviewers we must survive

Every scholarly requirement below is justified by reference to one or more of the following archetypes. Each archetype has a hostile probe and a known failure mode. The project must defeat every probe, not just the easy ones.

### R1. The Sanskrit philologist (citation accuracy hawk)

A reader fluent in classical Sanskrit with research-library access. Checks citations against the Sanskrit text.

- *Probes:* "You cite Tattvacintamani Anumana-khanda 5.1 — which edition uses that section number? SEP uses 5.1 as its own thematic subsection; is that what you mean? Tarka-Sangraha is fifteen pages — which edition's paragraph numbering is your '§43'? Show me the Sanskrit phrase grounding your gloss of *vyāpti-pañcaka*."
- *Failure mode:* Section numbers that look granular but resolve nowhere. **Already happened in v0.1 (audit Findings 1.1, 1.2, 1.3).**

### R2. The Navya-Nyaya specialist (terminological fidelity, Gangesha-onwards)

A scholar who has read Mohanty, Phillips, Ganeri, Bhattacharyya, Matilal, Wada — and will treat Old-Nyaya / Navya-Nyaya conflation as disqualifying.

- *Probes:* "You use *saṃśaya* the way Vatsyayana uses it, your Phase 13 list is Annambhatta's, your Phase 8 leans on Gangesha — whose system is the schema inside? *Vyāpti-pañcaka* is Gangesha's own five-fold analytic treatment, not 'a survey of twenty-four prior definitions' — where is that in TC? Phase 9 (*upādhi*) vs Phase 13 (*vyāpyatva-asiddha*) — coextensive or independent? Show me the passage that separates them."
- *Failure mode:* Treating the tradition as a unified system rather than a 1500-year argument with internal disagreement.

### R3. The post-colonial / orientalism-skeptic reviewer

Will reject on framing grounds before reading the schema.

- *Probes:* "Why is a Bay-Area founder in 2026 deciding what counts as the canonical formulation of a 1000-year-old tradition? Does the framing imply AI 'recovers' or 'unlocks' ancient wisdom? Whose translation are you using — did you credit them? What is your relationship to current *paṇḍita*s at BHU, Mithila Institute, Sampurnanand, Madras Sanskrit College?"
- *Failure mode:* Copy that valorizes the tradition as exotic / ancient / mystical; conflation of "Indian logic" with spirituality; absent living-tradition acknowledgement.

### R4. The IP-aware / prior-art reviewer (the Sathish-2026 question)

Knows about Sathish 2026 (arXiv:2604.04937) and will ask whether this work is a relabel.

- *Probes:* "What is in your schema that is not in Sathish? Show me phase-by-phase with a primary-text citation pre-dating Sathish 2026 and locatable independent of his bibliography. Your '60% independent contribution' number — derive it. When did you first read Sathish, and what was drafted before vs after?"
- *Failure mode:* A draft that mirrors Sathish's six phases at the core and dresses additions on top, while presenting itself as independent derivation.

### R5. The reproducibility hawk

Cares only that another researcher can re-trace every citation and re-run every claim.

- *Probes:* "Pick any Sanskrit term — within 15 minutes can I find an English gloss attributed to a recognized Indologist? Pick any divergence claim against Sathish — can I verify it from his paper in 30 minutes? Are translations credited by translator + edition + page?"
- *Failure mode:* Citations that demand library access readers don't have; quotes attributed to "the tradition" with no human translator named.

### R6. The peer reviewer at a Navya-Nyaya-aware venue

JIP, PEW, JOAS, *Asian Philosophy*; or NLP/ML workshops with humanities scope. Will desk-reject "AI does ancient logic" submissions on tone unless the scholarly contribution is explicit and bounded.

- *Probes:* "What is the philological contribution vs the empirical contribution vs the engineering contribution? Was this read by anyone who can read Devanagari? Why is this a benchmark and not a translation note — what is your normative theory of 'good Navya-Nyaya reasoning' and who dissents?"
- *Failure mode:* Document presenting itself as a contribution to Indian-philosophy scholarship without a Sanskrit-fluent reader having checked it.

### R7. The ML-conference area chair who quietly forwards Sanskrit-flavored submissions to a Sanskritist friend

The lethal one — the reviewer the founder will not see, asked off-record "is this real?", whose two-line answer determines whether the submission reads as "interesting framing, sound benchmark" or "amateur orientalism."

- *Whispered probes:* "Is the citation structure plausible to someone who has read Gangesha? Does this read like the author has seen the Sanskrit, or pieced it together from Wikipedia and SEP?"
- *Failure mode:* What the audit caught — SEP editorial subsection numbers mislabeled as primary-text locations.

---

## 2. Requirements

Severity legend: **MUST** = blocker for public launch. **SHOULD** = strongly required, justify in writing if waived. **NICE-TO-HAVE** = improves credibility, not load-bearing.

### A. Citation rigor and primary-source grounding

#### SCH-A-001 — Granularity floor for every Sanskrit citation
- **Statement:** Every citation to a Sanskrit primary text in any public-facing artifact (schema document, README, NOTICE, paper, website) MUST identify (a) the named text, (b) a specific named edition (e.g., "Athalye & Bodas, 2nd ed., Bhandarkar OR Institute, Poona, 1930"), and (c) one of: a canonical sutra/karika number where one exists in standard scholarly use, OR a page range in the named edition, OR an explicit topic-section name (e.g., "the *vyapti-pancaka* section of the Anumana-khanda"). Decimal subsection numbers (e.g., "5.1") MUST NOT appear unless they are the cited edition's own numbering AND that edition is named in the same sentence or footnote.
- **Severity:** MUST
- **Rationale:** Pre-empts R1 (Sanskrit philologist) and R7 (whispered area-chair check). The v0.1 draft failed precisely here (audit Findings 1.1, 1.2, 1.3): SEP's editorial subsection numbers (5.1/5.2/5.3) were lifted and presented as TC's own. Tarka-Sangraha §-numbering does not exist canonically; the v0.1 §43, §49, §50, §52, §§54-67 references are not anchored to any named edition and could not be verified.
- **Acceptance criterion:** A reviewer with internet access (no library card needed) can, within 15 minutes per citation, either (a) find the cited passage in a freely-available scan (archive.org Athalye-Bodas 1930; Ganganatha Jha's NS translation on archive.org; Wisdomlib for Sanskrit text + English digest) and confirm the cited text exists at the cited location, or (b) reach a named scholar's page reference (Phillips, Mohanty, Potter, Matilal, Ganeri, Bronkhorst, Bhattacharyya) that backs up the cited claim. We test 100% of citations in this document, not a sample. A single unverifiable citation in a public artifact fails this requirement.

#### SCH-A-002 — Canonical edition register
- **Statement:** The repository MUST contain a `CITATIONS.md` (or equivalent) that names, for each primary text used, the *one* edition the project treats as canonical. The minimum register includes:
  - **Nyaya-Sutras of Gautama:** Ganganatha Jha's translation (1939, reprinted Motilal Banarsidass) for English; Taranatha Nyaya-Tarkatirtha & Amarendramohan Tarkatirtha's edition (Calcutta Sanskrit Series, 1936-1944) for Sanskrit.
  - **Nyaya-Bhashya of Vatsyayana:** Ganganatha Jha translation (1984, MLBD reprint).
  - **Nyaya-Varttika of Uddyotakara:** Ganganatha Jha translation in *Indian Thought* / MLBD reprint where available.
  - **Tattvacintamani of Gangesha (Anumana-khanda):** Stephen Phillips & N.S. Ramanuja Tatacharya, *Epistemology of Perception* (American Institute of Buddhist Studies / Columbia, 2004) for Pratyaksa-khanda; Phillips, *Epistemology in Classical India* (Routledge, 2012) for inferential coverage; J.N. Mohanty, *Gangesa's Theory of Truth* (1989) where applicable. For Sanskrit: the Bibliotheca Indica edition (Kamakhyanatha Tarkavagisa, ASB Calcutta, 1884-1901) or the Chowkhamba edition.
  - **Tarka-Sangraha of Annambhatta:** Athalye (ed.) & Bodas (tr.), 2nd edition, Bhandarkar OR Institute, Poona, 1930 (available on archive.org). Where paragraph numbers are cited, they are Athalye's, and this MUST be stated.
  - **Atmatattvaviveka of Udayana:** N.S. Dravid translation (Indian Council of Philosophical Research, 1995) where used.
  - **Padartha-Tattva-Nirupanam of Raghunatha Siromani:** Karl Potter's translation/summary in *Encyclopedia of Indian Philosophies* Vol. 2 (Princeton/MLBD).
- **Severity:** MUST
- **Rationale:** R1, R5. Without a named-edition register, "TC 5.1" is an unfalsifiable claim. With one, every citation can be checked against a single artifact. This also forces the founder to actually consult these editions rather than triangulate from secondary digests (the v0.1 failure mode).
- **Acceptance criterion:** `CITATIONS.md` exists; every citation in every public artifact references an edition listed in `CITATIONS.md` or explicitly justifies an exception. CI-style check: regex over schema document for `(NS|TC|TS|TS§|Anumana-khanda|Tarka-Sangraha)` matches; each match has a corresponding edition reference within ±50 lines.

#### SCH-A-003 — Verifiability budget
- **Statement:** A reader with English-only access and no Sanskrit background MUST be able to verify each *philosophical* claim (not each Sanskrit token) in the schema document by reading at most one named scholarly secondary source from {Phillips 2012, Matilal 1998 *The Character of Logic in India*, Mohanty 1989 *Gangesa's Theory of Truth* / 1992 *Reason and Tradition in Indian Thought*, Ganeri 2001 *Philosophy in Classical India*, Potter EIP Vol. 2 / Vol. 6, Bhattacharyya & Potter EIP Vol. 13, Bronkhorst 2011 *Buddhism in the Shadow of Brahmanism*, *Stanford Encyclopedia of Philosophy* entries on Gangesa, Nyaya-Vaisesika, Epistemology in Classical India}.
- **Severity:** MUST
- **Rationale:** R5 (reproducibility) and R7 (whisper-test). If the project's claims require a Sanskritist's library, no peer reviewer outside the field can audit them, and the project becomes unfalsifiable.
- **Acceptance criterion:** Pick 10 substantive claims at random from the schema document; for each, cite one of the named sources by author + year + page or chapter. If 10/10 succeed, pass. If <10/10, the missed claims must be removed or rewritten.

#### SCH-A-004 — Distinguish primary text, classical commentary, modern restatement
- **Statement:** Every citation MUST be tagged with one of three roles: **PRIMARY** (the root sutra/karika or the named *kara*'s own text — NS, TC, TS, NV, NVTT, AT, PTN), **COMMENTARY** (classical Sanskrit commentary on a primary text — Vatsyayana on NS, Annambhatta's own *Dipika* on TS, Govardhana's *Nyaya-Bodhini*, Mathuranatha's commentaries on TC), or **MODERN** (any 19th-century-onwards scholarly restatement, including Phillips, Mohanty, Matilal, Wikipedia, SEP, Wisdomlib, IEP). The schema's *foundational* claims must rest on PRIMARY or COMMENTARY; MODERN sources are corroborative only.
- **Severity:** MUST
- **Rationale:** R1, R2. The audit found that several v0.1 "primary" claims actually derive from SEP (a MODERN tertiary source) — the dominant failure mode of the previous draft. Tagging makes this visible.
- **Acceptance criterion:** Every citation in the schema document carries one of `[PRIMARY]`, `[COMMENTARY]`, `[MODERN]`. No phase grounds its existence solely on `[MODERN]` sources. If a phase has only `[MODERN]` support, it is flagged as "design synthesis" not "primary-source-grounded."

#### SCH-A-005 — Conflict-resolution policy across Sanskrit sources
- **Statement:** Where primary sources disagree (e.g., Vatsyayana lists *five* sources of doubt at NS 1.1.23, while Vacaspati and Uddyotakara reduce them to three — the audit confirmed this divergence), the schema MUST state which authority it follows AND name at least one alternative position with citation. Silent selection of one tradition is forbidden.
- **Severity:** MUST
- **Rationale:** R2. A Navya-Nyaya specialist reads silent selection as either ignorance of the disagreement or concealment of it. Either is disqualifying.
- **Acceptance criterion:** For every Sanskrit term where a known intra-tradition disagreement exists, the schema document includes an explicit "alternatives considered" line (≤2 sentences) with citation. Minimum coverage: *samshaya*'s sub-types (5 vs 3), *vyapti*'s definition (Gangesha's five-fold vs. earlier and later refinements), *hetvabhasa* enumeration (Gautama's 5 vs. Annambhatta's Navya 5), *pramana* count (4 in Nyaya, 6 in Mimamsa, 1 in Carvaka), *abhava* status (separate pramana for Bhatta Mimamsa, perceptual for Nyaya).

#### SCH-A-006 — Sanskrit-language source acknowledgement
- **Statement:** The project MUST acknowledge in writing that the founder does not read Sanskrit (if true), and that all primary-text claims are mediated through named English-language scholars. This acknowledgement is positioned in the schema document and the README, not buried.
- **Severity:** MUST
- **Rationale:** R3 (orientalism), R6 (peer reviewer). Hiding the absence of Sanskrit literacy invites discovery; disclosing it pre-empts the attack and signals epistemic humility.
- **Acceptance criterion:** A "Translator's Note / Linguistic Disclosure" section appears in the schema document and README, listing the translators relied on and explicitly stating the founder's relationship to the source language.

---

### B. Terminological fidelity

#### SCH-B-001 — IAST transliteration with consistent diacritics
- **Statement:** All Sanskrit terms in formal documents (schema design doc, paper if any) MUST use IAST (International Alphabet of Sanskrit Transliteration) with full diacritics: *vyāpti*, *upādhi*, *pakṣa*, *sādhya*, *parāmarśa*, *anupalabdhi*, *abhāva*, *hetvābhāsa*. ASCII transliteration (used in `pramana_test.py` system prompt for portability) is acceptable in the prompt itself but is NOT acceptable in the schema-defining document.
- **Severity:** MUST
- **Rationale:** R2. Consistent IAST is the table stakes for a scholarly artifact. Inconsistent diacritics ("Pratyaksha" vs "pratyakṣa" vs "pratyaksa") are a dead giveaway that the author has not engaged with the scholarly literature.
- **Acceptance criterion:** A single Sanskrit-term glossary appears in the document; every term is given in IAST first, then once in ASCII alongside if the document uses ASCII elsewhere. Diacritic consistency is checked: any term appearing twice must appear with the same diacritics both times. Standard reference: Apte's Sanskrit-English dictionary (online: SanskritDocuments / Cologne Digital Sanskrit Dictionaries) for normalization.

#### SCH-B-002 — Glosses cited to a named Indologist
- **Statement:** Every Sanskrit technical term used in the schema MUST have, in the glossary, an English gloss attributed to a named Indologist with edition and page reference. Anonymous glosses ("commonly translated as...") are forbidden.
- **Severity:** MUST
- **Rationale:** R5, R6. A gloss without attribution is folk-philology. A gloss with attribution is checkable.
- **Acceptance criterion:** Glossary entries take the form: `vyāpti — "invariable concomitance" (Phillips 2012, p. 47); also "pervasion" (Mohanty 1992, p. 110)`. Minimum coverage: every term named in any phase of the schema — at minimum *jijñāsā, saṃśaya, prayojana, siddhānta, sarvatantra, pratitantra, adhikaraṇa, abhyupagama, prameya, pramāṇa, pratyakṣa, anumāna, upamāna, śabda, dṛṣṭānta, vyāpti, vyāpti-pañcaka, upādhi, paramārśa, pakṣadharmatā, pakṣa, sādhya, hetu, pañca-avayava, pratijñā, udāharaṇa, upanaya, nigamana, tarka, hetvābhāsa, savyabhicāra, viruddha, satpratipakṣa, asiddha, bādhita, prakaraṇasama, sādhyasama, kālātīta, anupalabdhi, abhāva, nirṇaya*.

#### SCH-B-003 — Author-disambiguated concept use
- **Statement:** When a single Sanskrit term has different operative meanings across authors (Gautama vs Vatsyayana vs Gangesha vs Annambhatta vs Raghunatha), the schema MUST state which author's usage governs each phase and cite the operative passage.
- **Severity:** MUST
- **Rationale:** R2. The audit's lens-2 findings (operational distinctness) are downstream of this — Phase 9's *upādhi* and Phase 13's *vyāpyatva-asiddha* collapse into each other when the schema is silent about whose system it is inside.
- **Acceptance criterion:** The schema design document has a "Whose system is this?" section that, for each phase, names the governing author and cites the passage. Mixed-system phases (e.g., Phase 13 using Annambhatta's list while Phase 8 leans on Gangesha) are explicitly flagged with rationale.

#### SCH-B-004 — Contested-concept warning
- **Statement:** If a phase relies on a concept where modern scholarly consensus is contested (the audit's *vyapti-pañcaka* counting issue is example #1: Wisdomlib gives 5+6=11, SEP mentions "twenty-four" in a different context, Phillips 2012 walks through the actual five-fold treatment), the schema MUST include a "contested" flag and reference at least two scholarly positions.
- **Severity:** MUST
- **Rationale:** R1, R2. The previous draft's "twenty-four prior definitions" claim conflated SEP's polemical-pre-survey count with the *pañcaka* itself (audit Finding 1.2). Contested-flag discipline prevents this whole class of error.
- **Acceptance criterion:** A reviewer can run a check: every phase's "Classical source(s)" line includes either a "settled" or "contested" tag. Contested phases reference ≥2 scholars who hold different positions, with citations.

#### SCH-B-005 — No silent semantic substitution
- **Statement:** Where the schema uses a classical term for a non-classical purpose (the v0.1 Phase 5 *prameya-vislesana* was honest about this — repurposing *prameya* from "the twelve metaphysical objects of cognition" to "answer-type analysis" — and flagged it; this is the *correct* template), the document MUST flag the repurposing explicitly with a "design note" paragraph.
- **Severity:** MUST
- **Rationale:** R2, R3. Silent semantic substitution is the failure mode that gets papers desk-rejected at JIP. The v0.1 Phase 5 design note is the template to generalize.
- **Acceptance criterion:** Every phase whose Sanskrit term is used in a benchmark-pragmatic rather than text-faithful sense bears a "Design note" paragraph naming the divergence. Audit reads each phase and flags any that does benchmark-pragmatic work without the note. (Phase 1 *jijñāsā* and Phase 5 *prameya* are the v0.1 candidates already flagged; Phases 3 *prayojana* and 4 *siddhānta-vibhāga* in benchmark settings deserve scrutiny.)

---

### C. Defensibility against the Sathish-mirroring claim

#### SCH-C-001 — Independent-derivation receipt
- **Statement:** For every phase whose existence is defended as not-derived-from-Sathish, the project MUST produce a pre-Sathish-2026 primary-text citation (PRIMARY or COMMENTARY in the SCH-A-004 sense) where the operative concept appears, AND that citation must be locatable independent of Sathish's bibliography.
- **Severity:** MUST
- **Rationale:** R4 (IP-aware reviewer). The previous draft's "60% independent contribution" claim (audit Finding 4.1) is unsupported because the receipts are not laid out. The honest number is 30-45%; the project must establish *that* number rigorously rather than inflate.
- **Acceptance criterion:** A `INDEPENDENCE_RECEIPTS.md` (or table in the schema doc) lists, per phase: (phase number, pre-1900 primary citation with named edition, English-language scholar who has discussed this concept independently of Sathish, year of that scholar's publication if it pre-dates Sathish 2026). For each phase claimed as a divergence from Sathish, the receipt must show that the divergence rests on classical material accessible independent of Sathish.

#### SCH-C-002 — Sathish-list verification
- **Statement:** The claim that Sathish 2026 uses Gautama's older five-fold *hetvābhāsa* list (versus Annambhatta's Navya five-fold) MUST be verified by direct quotation from Sathish's paper, or removed.
- **Severity:** MUST
- **Rationale:** R4. Audit Finding 1.4 flagged this as a load-bearing unverified claim. If the divergence collapses, the schema's Phase 13 "fourth major divergence" claim collapses with it, and the independent-contribution accounting drops further.
- **Acceptance criterion:** The schema document quotes Sathish's exact list of fallacy names with section reference in his paper. If his list is Annambhatta's, the schema honestly demotes Phase 13 from "divergence" to "shared with sharper scoring rules." If his list is Gautama's, the divergence stands and is documented as such.

#### SCH-C-003 — Falsification test for the independent-derivation claim
- **Statement:** The independent-derivation claim is operationalized as: an auditor selects any phase, demands a pre-Sathish citation, and the project produces it within the SCH-A-003 verifiability budget. Failure on any phase falsifies the global claim.
- **Severity:** MUST
- **Rationale:** R4. Without a falsification test, the claim is unfalsifiable and therefore not scientific.
- **Acceptance criterion:** An external auditor (selected per SCH-G-002) runs the falsification test on all 15 phases. The project's claim is "X of 15 phases are independently derivable" where X is the auditor's count, not the founder's.

#### SCH-C-004 — Honest contribution accounting
- **Statement:** Public claims about independent-contribution percentage MUST be accompanied by the calculation method (which phases count as direct preservation, which as framing additions, which as Gangesha-derived, which as design synthesis) and the specific phase tallies. The headline number must be a range that includes the audit's lower bound (30-45% per audit Finding 4.1), not a single inflated number.
- **Severity:** MUST
- **Rationale:** R4, R6. The "60% independent contribution" headline of the v0.1 draft was inflated. Honesty here strengthens, not weakens, the project's standing.
- **Acceptance criterion:** Headline contribution numbers in README, paper abstract, and any pitch-style documents are stated as a range (e.g., "30-45% independent of Sathish, depending on whether Phase 7's drstanta-versus-udaharana separation and Phase 13's Navya-vs-Gautama list count as substantive divergences"). Per-phase tally is auditable.

#### SCH-C-005 — Drafting-timeline disclosure
- **Statement:** The schema document MUST include a one-paragraph drafting-timeline disclosure: when did the founder first read Sathish 2026, when was each draft of the schema written, in what order, and what was consulted.
- **Severity:** SHOULD
- **Rationale:** R4. The strongest defense against "you mirrored Sathish" is "I read him on date X, and here are the dated commits / writeups showing my pre-Sathish thinking." If the order was opposite, it is better to disclose than concealed.
- **Acceptance criterion:** A 1-paragraph timeline appears in the schema doc; if claims of pre-Sathish independent thinking are made, they are documentable in repo history.

---

### D. Schema design rigor

#### SCH-D-001 — Operational-distinctness test for each phase
- **Statement:** Two phases A and B are "operationally distinct" iff there exists a pair of LLM outputs (one for phase A, one for phase B) on the same problem that no reasonable annotator would confuse, AND there exists at least one problem on which a model can pass A but fail B (or vice versa). A phase that fails this test must be merged with its neighbor or redesigned.
- **Severity:** MUST
- **Rationale:** R6 (peer reviewer). The audit's Lens 2 (Findings 2.1, 2.2, 2.4) found that v0.1 Phases 7-vs-11, 9-vs-13, and 12-vs-13 risk operational collapse. A schema that is structurally redundant inflates its phase count without epistemic gain; reviewers will dismiss it.
- **Acceptance criterion:** For each adjacent or thematically-overlapping phase pair, the schema document supplies (a) a pair of contrasting example outputs and (b) a problem instance demonstrating dissociation. If either is missing, the phase pair is flagged for redesign.

#### SCH-D-002 — Phase-dependency contract
- **Statement:** The schema MUST include a directed dependency graph: which phases consume the output of which earlier phases, and what the consumption contract is (which fields are read, what semantic constraint they impose). Cross-phase consistency rules MUST cite specific field references and matching algorithms.
- **Severity:** MUST
- **Rationale:** R6. The audit's Lens 6 (Findings 6.1-6.4) found that v0.1's cross-phase consistency rules are unscorable as stated and self-violating in the worked example. Without a formal dependency contract, the schema is vibes.
- **Acceptance criterion:** A dependency graph (as a figure or table) names each edge `(producer_phase, consumer_phase, field_name, matching_rule)`. Matching rules are one of: `exact_string`, `lemma_match(threshold=N)`, `semantic_equivalence(judge=...)`, `none`. The worked example satisfies every edge constraint by construction (the v0.1 worked example failed its own Phase 8 → Phase 11 udaharana-vyapti match per audit Finding 6.1).

#### SCH-D-003 — The worked example must satisfy the schema's own rules
- **Statement:** The worked example bundled with the schema MUST pass every format check, every cross-phase consistency rule, and every semantic constraint stated in the schema design document. If the founder cannot produce a passing worked example, the schema's rules are too strict and must be loosened OR the example must be redone.
- **Severity:** MUST
- **Rationale:** R6, R7. The v0.1 worked example violated its own cross-phase rules (audit Finding 6.1, 6.2) — Phase 13 "present-but-managed" while Phase 15 said *Sasamshaya*, contradicting "any present → Bhrama"; Phase 8 vyapti and Phase 11 udaharana had different content despite the verbatim-match requirement. This is self-discrediting.
- **Acceptance criterion:** A test script (or a documented manual review) runs the worked example through every rule in the schema and reports zero violations. If violations exist, they are either fixed in the example or fixed in the rules.

#### SCH-D-004 — Phase count justification
- **Statement:** The choice of 15 phases (as opposed to Sathish's 6, or 9, or 12, or 21) MUST be defended on a non-arbitrary criterion. Acceptable criteria include: each phase corresponds to a named *padartha* in NS 1.1.1 OR a named technical innovation of Gangesha OR a named methodological commitment (with citation). Unacceptable: "we picked 15 because it's a nice round number" or "to maximize scoring resolution."
- **Severity:** MUST
- **Rationale:** R6. A phase count without a principled basis invites suspicion that the count was reverse-engineered from a target.
- **Acceptance criterion:** Each of the 15 phases maps to one row of a justification table: `phase | classical anchor | non-redundancy with neighbors | evaluability`.

#### SCH-D-005 — Mechanical-scorability vs. judge-scorability disclosure
- **Statement:** Each phase's evaluation criterion MUST be tagged as one of: `mechanical` (regex / parser), `lookup` (against a fixed gold answer), `judge` (LLM-as-judge with documented prompt), or `human-only` (cannot be automated). The fraction of judge/human-only phases is a published quality metric.
- **Severity:** MUST
- **Rationale:** R5 (reproducibility) and R6 (peer reviewer). The audit's Lens 3 (Findings 3.1-3.6) showed that several v0.1 phases are unscorable except by LLM judge, which the document did not disclose. Hidden judge-dependency is a reproducibility failure.
- **Acceptance criterion:** Every phase carries a tag; aggregate is reported (e.g., "5/15 mechanical, 4/15 lookup, 5/15 judge, 1/15 human-only"). Judge prompts are versioned and shipped in the repo.

---

### E. Attribution and ethics

#### SCH-E-001 — Sathish 2026 cited as foundational related work
- **Statement:** Sathish 2026 (arXiv:2604.04937) MUST be cited in the schema document, the README, the NOTICE, and any companion paper, as the proximate prior work that motivated the project AND the project must distinguish whether the relationship is "extension," "alternative formulation," or "co-discovery from primary sources."
- **Severity:** MUST
- **Rationale:** R4. The current NOTICE handles this for the 6-phase pramana-eval; the new 15-phase schema MUST do the same or risk being read as concealment.
- **Acceptance criterion:** Sathish citation appears in all four artifacts; the relationship is named in one of three terms above; no public document presents the schema as derived solely from primary sources without acknowledging Sathish's catalytic role.

#### SCH-E-002 — Living tradition acknowledgement
- **Statement:** The project MUST acknowledge that Navya-Nyaya is not a dead tradition. Living institutions and scholars include Mithila Institute of Postgraduate Studies and Research in Sanskrit Learning (Darbhanga), Sampurnanand Sanskrit University (Varanasi), Madras Sanskrit College, Rashtriya Sanskrit Sansthan, and individual *paṇḍita*s and academics in India and abroad. The acknowledgement is not boilerplate; it names at least three living institutions or scholars by name.
- **Severity:** MUST
- **Rationale:** R3 (orientalism). Treating the tradition as ancient-and-mystical rather than ancient-and-still-practiced is the canonical orientalism tell.
- **Acceptance criterion:** A "Living tradition" paragraph appears in the schema document and is cross-linked from the README.

#### SCH-E-003 — Translator and edition credit
- **Statement:** Every English-language quotation from a primary or commentarial Sanskrit text MUST be attributed to the translator, edition, year, and page. Public-domain status is not a substitute for credit.
- **Severity:** MUST
- **Rationale:** R3, R5. Treating English-translated Sanskrit as if it had no human translator is both ethically wrong (it erases the translator's labor) and methodologically wrong (different translators make different choices, which a reader needs to evaluate).
- **Acceptance criterion:** `grep` for any quoted phrase in the schema document; each has a footnote with translator + edition + page.

#### SCH-E-004 — No "ancient wisdom recovered by AI" framing
- **Statement:** Public-facing copy MUST NOT use any of: "unlocking ancient wisdom," "recovering lost knowledge," "AI rediscovers Indian logic," "5000-year-old reasoning," "the East meets the West," or close paraphrases. The framing is "open-source benchmark grounded in classical Indian logic literature, designed for falsifiable evaluation of LLM reasoning behavior."
- **Severity:** MUST
- **Rationale:** R3, R6. Orientalist framing is the single fastest desk-reject signal at JIP/PEW and the single fastest credibility-loss signal in HN/X discourse.
- **Acceptance criterion:** Founder-authored review of all public copy (README, NOTICE, website, blog post draft, tweet thread, paper abstract) for forbidden phrases. Add a CI grep for the worst offenders if paranoid.

#### SCH-E-005 — Funding and affiliation disclosure
- **Statement:** Every public artifact MUST clearly state the founder's affiliation (ModelBrew AI) AND any commercial relationship the project has with that affiliation. The project is OSS and "independent of ModelBrew branding," but the founder's professional context MUST be visible.
- **Severity:** MUST
- **Rationale:** R3, R6. Hidden commercial backing of an "independent OSS" project is a credibility risk.
- **Acceptance criterion:** README and NOTICE state ModelBrew AI funding/sponsorship; benchmark results are not used in ModelBrew product marketing without separate disclosure.

---

### F. The schema's epistemic claims (what it actually measures)

#### SCH-F-001 — Stated measurement target
- **Statement:** The schema document MUST state, in one paragraph, what the benchmark measures. The target is one of: (a) "LLM ability to produce structured outputs that pattern-match a Navya-Nyaya format," (b) "LLM ability to perform the cognitive operations Navya-Nyaya names," (c) "the portability of a Navya-Nyaya-flavored reasoning prompt across models." These are *different* claims with different burdens of proof.
- **Severity:** MUST
- **Rationale:** R6. The README hedges between (a) and (c); the schema document should pick. Conflating the three is the single biggest interpretive risk for a hostile reviewer.
- **Acceptance criterion:** A "What this measures" section names the target and lists the alternatives explicitly rejected.

#### SCH-F-002 — Stated boundary against epistemic over-claim
- **Statement:** The schema document MUST explicitly state what it does NOT measure: the model's metaphysical commitments, its capacity for *liberation* in the soteriological sense (Nyaya's *niḥśreyasa*), its ability to do Sanskrit philology, its closeness to a particular *paṇḍita*'s view. These are not benchmark-suitable targets.
- **Severity:** MUST
- **Rationale:** R3, R6. Pre-empts the reading "this benchmark measures whether the model has achieved Nyaya-style epistemic success," which the project cannot defend.
- **Acceptance criterion:** A "What this does not measure" section appears, with the four exclusions named or close paraphrases.

#### SCH-F-003 — Distinguish "the schema works" from "the model happens to do the right thing"
- **Statement:** The benchmark report MUST distinguish two outcomes: (i) the model produced the right phases AND its reasoning depended on those phases (schema-causal); (ii) the model produced the right phases AND the right answer, but the answer was reachable without the schema (schema-decorative). At minimum, the document acknowledges that without an ablation (compare schema-prompted vs. unschema-prompted on identical problems), this distinction cannot be drawn.
- **Severity:** MUST
- **Rationale:** R6, R5. A benchmark that cannot distinguish these is measuring decoration. The current pramana-eval (6-phase) does not do this ablation; the 15-phase version, given its higher cost, must.
- **Acceptance criterion:** The schema document includes a "Schema-causality vs schema-decoration" section that proposes an ablation design (free-form-CoT vs schema-prompted on a held-out subset, with answer-correctness compared). The benchmark report flags any result that has not run this ablation.

#### SCH-F-004 — Sample-size and statistical-power floor
- **Statement:** Public claims of model capability differences MUST be backed by sample sizes adequate to distinguish those claims. n=10 with overlapping Wilson CIs (the existing README already states this honestly) MUST NOT be re-cited as a benchmark result; the 15-phase version requires n≥50, ideally n≥100 with held-out problems.
- **Severity:** MUST
- **Rationale:** R5, R6. A peer reviewer will not allow "100% vs 40% on n=10" to be cited as a result without statistical caveats. The current README handles this; the 15-phase version must inherit the discipline.
- **Acceptance criterion:** Every claim of relative model performance carries a sample size, a confidence interval, and a statistical-significance statement (or explicit disclaimer of insufficient power).

---

### G. Publication and review path

#### SCH-G-001 — Pre-launch Sanskrit-fluent review
- **Statement:** Before public launch (defined as: tagged release, blog post, social post, conference submission), the schema document MUST be read and approved (or read and revised based on feedback) by at least one Sanskrit-fluent reader who has formal training in Indian philosophy. Acceptable readers: a Sanskritist at a research university (any country), a current PhD student in Indian philosophy, a *paṇḍita* affiliated with a recognized Sanskrit institution.
- **Severity:** MUST
- **Rationale:** R6, R7. A document that has not been read by a Sanskrit-fluent reviewer is a liability. The audit found citation errors that any *paṇḍita* would catch in an afternoon.
- **Acceptance criterion:** A named reviewer (with consent for naming) is acknowledged in the schema document, with a one-line statement of what they reviewed and at what level. If the reviewer requested changes that the founder did not adopt, those disagreements are noted (an "unresolved" log).

#### SCH-G-002 — External audit on citation hygiene
- **Statement:** Before public launch, an external auditor (different person from SCH-G-001's reviewer, ideally an information professional or another scholar) MUST audit the citation register for verifiability per SCH-A-001 / SCH-A-003. This audit is documented in the repo.
- **Severity:** MUST
- **Rationale:** R5, R7. Citation hygiene was the single biggest v0.1 failure (audit's "Single Biggest Flaw" section). It deserves a separate auditor.
- **Acceptance criterion:** An `EXTERNAL_AUDIT.md` records who audited, when, what they checked, and the results. Pass requires zero CRITICAL findings and ≤2 MAJOR findings (with documented response).

#### SCH-G-003 — Companion paper or technical report
- **Statement:** The project SHOULD produce a companion paper (arXiv preprint at minimum; preferred venues: a workshop at a major NLP/ML conference like ACL's WiNLP, or *Journal of Indian Philosophy*, or *Religions* (open-access, computational-religion-friendly), or *Asian Philosophy*). The paper makes the philological grounding rigorous in a way that a README does not allow.
- **Severity:** SHOULD
- **Rationale:** R6. An OSS benchmark without a paper is harder to cite and harder for the academic community to engage with on its own terms.
- **Acceptance criterion:** Either an arXiv preprint exists at launch, or the schema document is structured such that it could *be* the preprint with minimal reformatting.

#### SCH-G-004 — Engagement with at least one Indian-philosophy academic before launch
- **Statement:** Before public launch, the project MUST circulate the schema to at least one Indian-philosophy academic at a recognized department (e.g., a Nyaya specialist at Harvard South Asian Studies, Oxford OCHS, JNU, BHU, Hyderabad, IIT Madras Humanities, Heidelberg STAR), seeking critical feedback. Either the feedback is incorporated, or the disagreements are documented.
- **Severity:** MUST
- **Rationale:** R3, R6, R7. This is the single highest-leverage scholarly de-risking step.
- **Acceptance criterion:** A named academic (with consent for naming, or anonymized with department + role) is acknowledged. If no academic engages despite reasonable outreach, the outreach attempts are documented (3+ contacts, ≥4 weeks elapsed).

#### SCH-G-005 — Minimum bar of external scrutiny
- **Statement:** Define "minimum bar" as the conjunction of: SCH-G-001 (Sanskrit-fluent reviewer), SCH-G-002 (citation auditor), SCH-G-004 (Indian-philosophy academic). All three MUST be cleared before public launch. SCH-G-003 (companion paper) is strongly recommended but not blocking.
- **Severity:** MUST
- **Rationale:** R6, R7. Setting the bar explicitly prevents the founder from negotiating it down under launch pressure.
- **Acceptance criterion:** A pre-launch checklist exists in the repo that requires sign-off on all three.

---

## 3. Anti-patterns to forbid

The following are explicit prohibitions, drawn from the failed v0.1 draft, common "AI meets Indian philosophy" failure modes, and observed peer-review desk-rejection grounds.

1. **AP-001 — Decimal subsection numbers presented as primary-text locations.** "Tattvacintamani Anumana-khanda 5.1" without an edition reference. Decimal numbers belong to SEP, IEP, or specific named editions; not to canonical Sanskrit texts as such.

2. **AP-002 — Section-numbering in Tarka-Sangraha that is not Athalye/Bodas paragraph numbering with explicit attribution.** The v0.1 "§43, §49, §50, §52, §§54-67" references have no canonical source. If using Athalye paragraphs, name him; if not, use topic references.

3. **AP-003 — Conflating SEP / Wikipedia / Wisdomlib / IEP claims with primary-text claims.** Tertiary scholarly sources are corroborative. They are not, themselves, the Tattvacintamani.

4. **AP-004 — "Ancient wisdom recovered by AI" framing.** Forbidden in all public-facing copy (per SCH-E-004).

5. **AP-005 — Treating Indian logic as monolithic.** Old Nyaya is not Navya-Nyaya. Vatsyayana's *samshaya* is not Annambhatta's. Gangesha's *vyapti* is a specific innovation, not "the Nyaya position." Schema documents that paper over internal disagreement get desk-rejected.

6. **AP-006 — "60% independent contribution" or any precise contribution percentage that has not been derived in writing.** Either show the math or use a range.

7. **AP-007 — Citing "the tradition" or "Sanskrit sources" without naming texts and translators.** The phrase "as the tradition holds" is forbidden in scholarly artifacts.

8. **AP-008 — Worked examples that violate the schema's own rules.** A worked example is the project's promise to itself; if it fails, the rules are wrong, the example is wrong, or both.

9. **AP-009 — Claims about Sathish's content that have not been verified by direct quotation.** The audit caught one (the hetvabhasa-list claim, audit Finding 1.4); future drafts must avoid the pattern.

10. **AP-010 — Mixing IAST and ASCII transliteration inconsistently within a scholarly document.** Forbidden per SCH-B-001.

11. **AP-011 — Sanskrit terms with no English-language gloss attribution.** Forbidden per SCH-B-002.

12. **AP-012 — Single-author contribution narrative concealing relevant prior art.** The project's relationship to Sathish 2026 must be disclosed in every artifact (per SCH-E-001).

13. **AP-013 — Repackaging Wikipedia or SEP prose without acknowledgement.** This is plagiarism even when paraphrased; the project's foundational document must read like the founder thought through the philosophy, not curated tertiary sources.

14. **AP-014 — Marketing the OSS launch via spiritual or religious framing.** Nyaya is a logic and epistemology tradition, not a meditation tradition. Conflating the two is a category error and an orientalism tell.

15. **AP-015 — Scoring rubrics that are unscorable in practice.** Cross-phase consistency rules must be either mechanically scorable (with the algorithm specified) or honestly tagged as judge-scorable with documented unreliability. Hidden judge-dependency is a reproducibility crime.

---

## 4. Open scholarly questions

These are unresolved at the time of writing. The founder must make a call (or escalate to a Sanskrit-fluent advisor) before launch.

1. **OQ-001 — Whose Navya-Nyaya is canonical for the schema?** Gangesha himself, Mathuranatha's *Mathuri*, Annambhatta's pedagogical synthesis, or Raghunatha's revision? Each has different commitments. The v0.1 draft mixed Gangesha (Phases 8/9/10) with Annambhatta (Phase 13) without justifying the mix. *Defer to:* a named Indian-philosophy academic (per SCH-G-004) who can advise on whose system to anchor in.

2. **OQ-002 — How to handle the Bhatta-Mimamsa challenge to *anupalabdhi* as separate pramana?** Phase 14 sides with Nyaya/Udayana against Bhatta. Should the schema acknowledge that Nyaya's position here is contested? In peer review at JIP, a Mimamsa-aware reviewer will ask. *Defer to:* the schema's "alternatives considered" discipline (SCH-A-005); name the Bhatta position and decline it with reason.

3. **OQ-003 — Is *paramarsa* doing real work in benchmark settings?** Gangesha treats it as a real mental event; an LLM cannot have mental events. If the phase is purely formalist (require the template "Since X and Y, therefore Z"), is calling it *paramarsa* honest, or is it a relabeling of *upanaya*? *Defer to:* the SCH-D-001 operational-distinctness test.

4. **OQ-004 — Can the schema's contribution be honestly framed as "philological design + benchmark engineering" rather than "primary-source-grounded foundational document"?** The latter framing fails under audit. The former is more defensible and arguably more accurate. *Defer to:* the founder's strategic call, but the requirements above are agnostic — both framings work as long as the citations and acknowledgements are honest.

5. **OQ-005 — Is the project legitimate without a Sanskrit-reading founder?** The honest answer is: yes if the founder's role is harness-builder + benchmark designer + rigorous citer, with Sanskrit-fluent reviewers in the loop. The honest answer is: no if the founder claims primary-source authority. SCH-A-006 mandates the disclosure that resolves this.

6. **OQ-006 — Where does the schema position itself relative to Buddhist logic (Dignaga, Dharmakirti)?** Nyaya developed in dialogue with — and partly against — Buddhist logic. The schema does not reference this dialogue. A Buddhist-logic-aware reviewer will ask why. *Defer to:* a one-paragraph "scope" statement that explicitly limits the schema to Nyaya-Vaisesika materials and acknowledges the Buddhist-Indian-logic relationship without trying to cover it.

7. **OQ-007 — Should the project pursue a philological venue or a computational venue?** A philological venue (JIP, *Religions*, JOAS, *Asian Philosophy*) treats the contribution as scholarship; a computational venue (NeurIPS workshop, ACL workshop) treats it as benchmarking. The bars are different; the framing must match. *Defer to:* the founder's strategic call after seeing reviewer feedback.

8. **OQ-008 — How is the project's relationship to the Indian academic establishment going to be managed long-term?** Indian Sanskrit institutions have historically been wronged by extractive Western "wisdom" projects. The right relationship is probably collaborative, not extractive — but operationalizing that takes work the founder may not be ready to do at launch. *Defer to:* an explicit "future engagement" paragraph that names this as an ongoing commitment, not a launch-blocker.

---

## End-of-document acceptance test

The hostile question at the end is: **"If a competent Sanskrit-fluent reviewer reads this whole project for an afternoon, what is the worst thing they will write in their review?"**

For the v0.1 draft, the answer was: *"The granular section citations are fabricated or mislabeled, the vyapti-pancaka claim is wrong, the contribution percentage is inflated, and the worked example violates its own rules. This is not primary-source-grounded; it is gestures-toward-primary-source-grounded."*

After meeting these requirements, the worst defensible review should be: *"The schema is design synthesis informed by Phillips, Mohanty, and the Nyaya-Sutra translations, with explicit acknowledgement of which moves are pedagogical extension and which rest on primary text. The benchmark contribution is the harness, the test set, and the operational-distinctness discipline. The author does not read Sanskrit and discloses this. The citation register is auditable in 30 minutes. I disagree with their framing of paramarsa as a benchmark phase, but the disagreement is principled."*

That is the bar. Anything worse than that paragraph is a launch-blocker.

---

**End of requirements document.**
