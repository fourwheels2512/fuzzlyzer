# Sanskrit / Darshana-Specialist Requirements for pramana-eval

**Document type:** Adversarial requirements (Sanskrit-philology and darshana-systems lens)
**Date:** 2026-05-09
**Audience:** Founder of pramana-eval (Kiran Nayudu, ModelBrew AI), pre-launch
**Companion documents already produced:** `REQUIREMENTS_SCHOLAR.md` (generic-academic / peer-reviewer lens), `REQUIREMENTS_PRACTITIONER.md` (ML / methodology lens). Read those first; this document deliberately does not duplicate them.

**Posture:** Hostile, from inside the tradition. The previous two RT documents handle citation hygiene, edition discipline, statistical power, and benchmark engineering. They do not — because they cannot — diagnose Sanskrit-internal errors: sandhi failures, school-conflations, mistranslated technical concepts, sectarian framings that erase Buddhist/Mīmāṃsā/Cārvāka positions, and the Vedic-vs-darshana category error that the founder's earlier rejected branding "Vedic AI" suggests is still latent in the project's connotative environment. This document specifies what only a Sanskritist / darshana-systems specialist can specify.

---

## 1. The reviewers we must survive (Sanskrit-domain specifically)

### S1. The Mithila / Sampurnanand / Madras / Tirupati paṇḍita in the Navya-Nyaya parampara

A traditionally-trained Sanskrit scholar from one of the active Navya-Nyaya transmission lineages — for example, the Mithila Institute of Post-Graduate Studies and Research in Sanskrit Learning (Darbhanga, Bihar; established 1951, the institutional successor to the Mithila school where Gaṅgeśa Upādhyāya himself taught), Sampūrṇānanda Saṃskṛta Viśvavidyālaya (Vārāṇasī), Madras Sanskrit College, the Rāṣṭriya Saṃskṛta Saṃsthāna (Tirupati / Delhi / Jaipur branches), or Kavikulaguru Kālidāsa Saṃskṛta University (Ramtek). Reads Sanskrit fluently; trained in *Tarka-Saṃgraha* + *Dīpikā* the way Western philosophy students are trained in Aristotle's *Categories*; has memorized at least a *vyāpti-pañcaka* recital.

- **Probes:** "Whose paramparā are you in? Have you sat with anyone who reads Mathurānātha's *Mathurī*? Do you take *vyāpti* in Gaṅgeśa's sense or Raghunātha's revision? You write *anumāna* — but in *which* school's anumāna? You realize your 'Pancha Avayava' is not sandhi-correctly written — it is *pañcāvayava*; the second-letter *a* is long because of *savarṇa-dīrgha* sandhi between *pañca* + *avayava*. This is something a first-year Sanskrit student gets right."
- **Failure mode:** Treating Navya-Nyāya as a corpus to mine rather than a parampara to consult; ASCII Sanskrit; sandhi-incorrect compounds; silent selection between Gaṅgeśa, Mathurānātha, Raghunātha, Annambhaṭṭa, Viśvanātha, Jagadīśa.

### S2. The Western Indologist (Phillips / Ganeri / Bronkhorst archetype)

Reads Sanskrit, has produced critical editions or careful translations, has watched the discipline absorb three decades of "Indian logic in dialogue with analytic philosophy" debates, and is allergic to over-translations.

- **Probes:** "Why are you using Matilal's term *Indian logic* without scare-quotes? You realize Matilal's framing was contested by Ganeri, Mohanty, and Bhattacharyya, each pulling the concept toward different Western philosophical traditions. Which translation tradition's *vyāpti* is your *vyāpti*: Phillips's 'pervasion-as-extensional-relation', Mohanty's 'invariable concomitance with phenomenological residue', Ganeri's 'inferential warrant'? Did you cite Matilal's *Logic, Language, and Reality* (1985) or his *The Character of Logic in India* (1998)? Different commitments."
- **Failure mode:** Citing one translator selectively without acknowledging that the translation is theoretically loaded; treating "Indian logic" as a stable, agreed object; assuming Phillips's philosophical framing is neutral rather than analytic-philosophy-inflected.

### S3. The Buddhist Studies specialist

Reads the Naiyāyika position as one side of a 1500-year debate with Buddhist *pramāṇavāda* (Diṅnāga, c. 480–540 CE; Dharmakīrti, c. 600–660 CE). Knows that Navya-Nyāya's mature form would be unintelligible without the Buddhist-logic prehistory.

- **Probes:** "Where is Diṅnāga? Where is the *trairūpya* (the three-fold inferential mark: *pakṣa-dharmatā*, *sapakṣe sattvam*, *vipakṣe asattvam*) that Gaṅgeśa is *responding to* in his *vyāpti-pañcaka*? Where is *anyāpoha* (the Buddhist exclusion theory of meaning) that the Navya-Nyāya theory of *śābda-pramāṇa* is *defining itself against*? You present Navya-Nyāya as a sealed Hindu system. It is not — it is a polemical achievement *only* legible against the Buddhist interlocutor it spent six centuries defeating."
- **Failure mode:** Treating Navya-Nyāya as a self-enclosed Hindu epistemology and erasing the Buddhist contribution to its formation.

### S4. The Sanskrit-medium contemporary commentator

A scholar who teaches and publishes *in Sanskrit* (not just *about* Sanskrit). Members of this community publish in journals like *Saṃskṛta-Pratibhā*, *Saṃskṛta-Vimarśaḥ*, and the *Pandit* journal of Sampūrṇānanda Saṃskṛta Viśvavidyālaya. They will not read a Western OSS benchmark, but the field will read what they say about it.

- **Probes (translated from Sanskrit):** "Has anyone in this project read a Sanskrit *vṛtti* on this material? Have they consulted a *bhāṣya-paramparā*? Or is this another Anglophone re-extraction from the four standard Western translations?"
- **Failure mode:** Assuming the entire scholarly conversation lives in English. It does not.

### S5. The decolonial-academia critic

Suspicious of any project where Indian philosophical material moves through Western tech infrastructure. Reads "AI does Indian logic" as a continuation of the orientalist project. Their critique survives even if the project is technically perfect, *unless* the project has done specific work to neutralize their priors.

- **Probes:** "What is the consent relationship to the living tradition? Whose labor is being extracted? Is this Bay-Area-tech-as-good-faith-engagement or Bay-Area-tech-as-extractive-mining-with-better-PR? When have the Indian-academy *paṇḍita*s been compensated for the conceptual scaffolding this project rides on? Does the Apache-2.0 license re-encode the very property frame that classical Nyāya — as a gift-economy *vidyā-dāna* tradition — does not recognize?"
- **Failure mode:** Treating tradition as public-domain raw material; absent compensation/collaboration with living institutions; orientalist framing in the marketing copy; assuming that legal public domain status (Sanskrit texts > 1000 years old) settles the ethical question.

### S6. The traditional Sanskrit grammarian (Pāṇinian / sandhi-correctness lens)

A grammarian trained in the Pāṇinian system (*Aṣṭādhyāyī* + *Mahābhāṣya* + *Kāśikā* + *Siddhānta-Kaumudī* + *Laghu-Kaumudī*). Cares about sandhi correctness, vowel length, retroflex consonants, anusvāra, visarga, and the difference between *ṛ* and *r*. Will judge the project's seriousness in five seconds by reading any compound term.

- **Probes:** "Your prompt says 'Pancha Avayava' as two words. The compound is *pañcāvayava* (पञ्चावयव), one word with internal *savarṇa-dīrgha* sandhi. Your prompt says 'Sadhyasama' — should be *sādhyasama* (long *ā* in the first syllable). 'Kalaatita' is double-vowelled ASCII; the form is *kālātīta* (long *ā*, long *ā*, long *ī*). 'Prakaranasama' should be *prakaraṇasama* (retroflex *ṇ*; the dental *n* is wrong). 'Anumana' should be *anumāna*. 'Pratyaksha' should be *pratyakṣa*. 'Shabda' should be *śabda*. 'Pramana' should be *pramāṇa* (retroflex *ṇ*, long *ā*). The project's name is therefore not 'pramana' — it is *pramāṇa*. The ASCII is not just an aesthetic choice; it is a competence signal that fails immediately."
- **Failure mode:** ASCII-default Sanskrit; sandhi failures; vowel-length errors; dental-vs-retroflex confusion; consonant-aspiration confusion (*c* vs *ch*, *t* vs *th*).

---

## 2. Requirements

Severity legend: **MUST** = blocker for public launch. **SHOULD** = strongly required, justify in writing if waived. **NICE-TO-HAVE** = improves credibility, not blocking.

---

### A. Vedic vs Darshana category error

Navya-Nyāya is **not** Vedic. The Veda is *śruti* — Ṛk, Yajur, Sāma, Atharva, plus the Brāhmaṇas, Āraṇyakas, and Upaniṣads — composed (in its earliest layers) c. 1500–500 BCE. Nyāya is one of the six āstika *darśana*s (orthodox philosophical schools) — Nyāya, Vaiśeṣika, Sāṃkhya, Yoga, Mīmāṃsā, Vedānta. The darśanas accept the Veda as authoritative (which is what makes them *āstika*) but they are not themselves Veda. Navya-Nyāya specifically begins with Gaṅgeśa Upādhyāya's *Tattvacintāmaṇi* in 12th–13th century CE — over two thousand years after the latest layer of Veda was redacted. The category error matters because: (a) it is the single most common tell of a non-specialist, (b) the founder's earlier rejected branding "Vedic AI" indicates the connotation is still latent in the project's psychological environment, (c) it conflates *śruti*-authoritative testimony (which is what Mīmāṃsā centrally argues about) with the *darśanic* logical-epistemological apparatus (which is what Nyāya develops). These are categorically different things.

#### SAN-A-001 — No "Vedic" framing of Navya-Nyāya in any artifact
- **Statement:** No public-facing artifact (README, NOTICE, paper, abstract, blog post, social post, conference submission, repo description, package metadata, dataset card, leaderboard copy) MAY describe Navya-Nyāya, Nyāya, *pramāṇa*, *anumāna*, the *pañcāvayava*, or any of the Sanskrit technical terms used by the schema as "Vedic." The correct terms are *darśanic*, *philosophical*, or specifically *Nyāya / Navya-Nyāya / Naiyāyika*.
- **Severity:** MUST
- **Rationale:** Prevents the category error S1, S2, and S5 will all flag instantly. Closes off the residue from "Vedic AI."
- **Acceptance criterion:** Grep over every public artifact for `[Vv]edic`, `[Vv]eda`, `[Ss]hruti`, `[Śś]ruti`. Each hit either (a) is removed, (b) is a deliberate, scoped reference (e.g., "Nyāya is one of the six *āstika darśana*s, all of which accept the Veda as authoritative without themselves being Vedic literature"), or (c) is in a quoted secondary source where the original author used the term and the quotation is correctly attributed.

#### SAN-A-002 — Positive identification of where Navya-Nyāya sits in the Indian philosophical taxonomy
- **Statement:** The schema document, README, and any companion paper MUST contain a one-paragraph orientation that places Navya-Nyāya correctly: as a *darśana*, not Veda; as a Hindu *āstika* school in dialogue with Buddhist *pramāṇavāda*; as a 12th–13th-century-CE refinement of the older Nyāya (4th–8th-century-CE *bhāṣya-paramparā* of Vātsyāyana, Uddyotakara, Vācaspati Miśra, Udayana) which itself derives from Akṣapāda Gautama's *Nyāya-Sūtras*.
- **Severity:** MUST
- **Rationale:** A Sanskritist who reads three lines of orientation can detect immediately whether the author understands the taxonomy. Closes off S1 and S2's first-pass dismissal.
- **Acceptance criterion:** The orientation paragraph names: (a) the six darśanas, (b) Nyāya's status as one of them, (c) the distinction between *śruti*, *smṛti*, *sūtra*, *bhāṣya*, *vārtika*, *ṭīkā*, and the *cintāmaṇi*-type independent treatise; (d) the approximate centuries for old Nyāya vs Navya-Nyāya. A Sanskritist reading it can mark each clause "correct" / "incorrect" / "unclear."

#### SAN-A-003 — No conflation of *pramāṇa* with *śruti*-authority
- **Statement:** The schema's *śabda* phase (currently labeled "Shabda" in `pramana_test.py:215`) MUST NOT default-define *śabda-pramāṇa* as "Vedic testimony." In the Naiyāyika treatment that the schema explicitly inherits (Annambhaṭṭa, Tarka-Saṃgraha), *śabda* is **trustworthy linguistic testimony in general** — including from a reliable expert (*āpta*) on any domain — not specifically Vedic revelation. Mīmāṃsā restricts the cognitive privilege to Veda; Nyāya does not. The schema is Naiyāyika, so it must say what Nyāya means.
- **Severity:** MUST
- **Rationale:** Prevents the doubled error of (a) Vedic-darshana conflation and (b) Mīmāṃsā-Nyāya conflation in one stroke. The *śruti*-restricted reading of *śabda* is a *Mīmāṃsā* commitment, not a Naiyāyika one.
- **Acceptance criterion:** The `Shabda` phase description in `pramana_test.py` and in the schema document defines *śabda* as "trustworthy verbal testimony from a reliable source (*āpta-vākya*), including but not limited to authoritative texts" — explicitly *not* "Vedic testimony only." A footnote acknowledges that Mīmāṃsā restricts the privilege to Veda and that the schema follows Nyāya, not Mīmāṃsā.

---

### B. Sanskrit transliteration and grammatical correctness

The current `pramana_test.py` system prompt (lines 189–247) uses ASCII Sanskrit ("Pancha Avayava," "Sadhyasama," "Kalaatita," "Pramana," "Anumana," "Shabda," "Pratyaksha"). The schema draft (`SCHEMA_15_PHASE_DRAFT.md`) uses a mix of ASCII and partial-IAST. The NOTICE file uses pure ASCII (e.g., "Pancha Avayava" line 17, "Pratyaksha" line 45, "Savyabhichara" line 46). Each form has specific Sanskrit-grammatical defects that any first-year Sanskrit student would catch.

#### SAN-B-001 — IAST as the canonical rendering in all documentation
- **Statement:** All documentation files (`README.md`, `NOTICE`, `SCHEMA_*_DRAFT.md`, any companion paper, any web copy, dataset cards, blog posts) MUST use IAST (International Alphabet of Sanskrit Transliteration, ISO 15919-compatible for Sanskrit) with full diacritics for every Sanskrit term. Specific renderings to use:
  - *pramāṇa* (NOT "Pramana", NOT "pramana", NOT "pramaaNa")
  - *anumāna* (NOT "Anumana")
  - *pratyakṣa* (NOT "Pratyaksha", NOT "Pratyaksa")
  - *upamāna* (NOT "Upamana")
  - *śabda* (NOT "Shabda", NOT "Sabda")
  - *pañcāvayava* (NOT "Pancha Avayava" — single compound, sandhi-correct, see SAN-B-002)
  - *pratijñā* (NOT "Pratijna" — final long *ā*; the *jñā* is a conjunct, not "jna")
  - *hetu*, *udāharaṇa*, *upanaya*, *nigamana* — note retroflex *ṇ* in *udāharaṇa* and *nigamana*
  - *tarka* (this one is fine in ASCII)
  - *hetvābhāsa* (NOT "Hetvabhasa" — long *ā* twice; the *vā* and *bhā* are both long)
  - *savyabhicāra* (NOT "Savyabhichara" — the *cā* is *c* + long *ā*, not "cha")
  - *viruddha* (this one is fine, but the *ddh* is *d* + aspirated *dh*, a single conjunct)
  - *prakaraṇasama* (NOT "Prakaranasama" — retroflex *ṇ*)
  - *sādhyasama* (NOT "Sadhyasama" — long *ā* in the first syllable; the *dhy* is *dh* + *y*)
  - *kālātīta* (NOT "Kalaatita" — long *ā*, long *ā*, long *ī*; the ASCII double-vowel is a Harvard-Kyoto-ish hack, not IAST)
  - *nirṇaya* (NOT "Nirnaya" — retroflex *ṇ*)
  - *saṃśaya* (NOT "Samshaya" — anusvāra *ṃ* and *ś*)
  - *vyāpti* (long *ā*)
  - *upādhi* (long *ā*)
  - *parāmarśa* (long *ā*; *rś* not *rsh*; the schema draft's "Paramarsa" misses both diacritics)
  - *pakṣa* (retroflex *ṣ*)
  - *sādhya* (long *ā*)
  - *sapakṣa*, *vipakṣa* (retroflex *ṣ*)
  - *jijñāsā* (the *jñā* is a conjunct, long *ā* twice)
  - *prayojana* (this one has no diacritics; it is fine)
  - *siddhānta* (long *ā*)
  - *prameya* (no diacritics)
  - *padārtha* (long *ā*)
  - *anupalabdhi* (no diacritics)
  - *abhāva* (long *ā*)
  - *Naiyāyika* (long *ā*; the school-adjective)
  - *darśana* (with *ś*)
  - *Mīmāṃsā* (long *ī*, long *ā*, anusvāra)
  - *Vaiśeṣika* (with *ś* and retroflex *ṣ*)
  - *Sāṃkhya* (long *ā*, anusvāra)
  - *trairūpya* (long *ū*; the Buddhist-logic three-mark term)
  - *anyāpoha* (long *ā*; Diṅnāga's exclusion theory)
- **Severity:** MUST
- **Rationale:** S1 and S6 read ASCII-Sanskrit as a competence signal. The cost of fixing this is approximately one afternoon. The cost of *not* fixing it is the project being read as an outsider's gestural appropriation regardless of its actual scholarly content.
- **Acceptance criterion:** A `GLOSSARY.md` ships with the repo with one entry per Sanskrit term in the schema, in the form: `*term-IAST* (Devanāgarī) — gloss-attributed-to-named-Indologist (Author Year, p. N)`. Every documentation file references the term in IAST first; a hyperlinked or footnote ASCII alternative may follow only when needed for code-portability reasons. A reviewer can `grep` for ASCII variants ("Pancha Avayava", "Pramana", "Anumana", "Shabda", "Pratyaksha", "Hetvabhasa", "Savyabhichara", "Sadhyasama", "Kalaatita", "Nirnaya", "Samshaya", "Paramarsa", "Pratijna") in any non-`.py` file, and zero hits remain in human-facing documentation.

#### SAN-B-002 — Sandhi correctness in Sanskrit compounds
- **Statement:** Every Sanskrit compound term used in the schema MUST be sandhi-correct per Pāṇinian rules and rendered as a single word where the compound is a *samāsa*. Specifically, the following compounds require correction:
  - *pañcāvayava* (पञ्चावयव) — NOT "Pancha Avayava" (two words). This is *pañca* (five) + *avayava* (member). The terminal *a* of *pañca* and initial *a* of *avayava* combine via *savarṇa-dīrgha* sandhi (Pāṇini 6.1.101: *akaḥ savarṇe dīrghaḥ*) to form long *ā*. The *pañcāvayava* form is confirmed by Wisdomlib's lexical entry (`wisdomlib.org/definition/pancavayava`) and Apte's *Practical Sanskrit-English Dictionary*. The schema's current "Pancha Avayava" form is sandhi-incorrect AND non-IAST.
  - *vyāpti-pañcaka* (व्याप्तिपञ्चक) — when written as one compound, the *i* of *vyāpti* and *p* of *pañcaka* do not undergo sandhi (no vowel meeting). Acceptable as one word with hyphen for readability.
  - *hetvābhāsa* (हेत्वाभास) — *hetu* + *ābhāsa*. The *u* of *hetu* and *ā* of *ābhāsa* combine via *guṇa* sandhi (Pāṇini 6.1.87: *ād guṇaḥ*) to *o*, then *avādeśa* yields *etv* + *ābhāsa* → *hetvābhāsa*. The current "Hetvabhasa" misses both the long *ā*s.
  - *paksadharmatā* should be *pakṣadharmatā* (retroflex *ṣ*).
  - *vyāpyatva-asiddha* should be *vyāpyatvāsiddha* if compounded (sandhi: *a* + *a* → *ā*) or kept hyphenated if treated as a phrase. The schema draft is inconsistent.
  - *Tattvacintāmaṇi* (तत्त्वचिन्तामणि) — long *ā*, retroflex *ṇ*. Consistent with the form used by Phillips, Mohanty, and the standard Bibliotheca Indica edition.
  - *Anumāna-khaṇḍa* (अनुमानखण्ड) — long *ā* in *anumāna*; retroflex *ṇḍ* in *khaṇḍa*. The schema draft writes "Anumana-khanda" — both diacritics missing.
- **Severity:** MUST
- **Rationale:** Sandhi correctness is *the* basic competence test for Sanskrit literacy. S6 (the Pāṇinian grammarian) will not read past the first compound if it is sandhi-incorrect.
- **Acceptance criterion:** Every Sanskrit compound in any documentation file is verified by either (a) Apte's *Practical Sanskrit-English Dictionary* (online at `dsal.uchicago.edu/dictionaries/apte/`), (b) Monier-Williams *Sanskrit-English Dictionary* (online at `sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/web/webtc/indexcaller.php`), or (c) a Sanskrit-fluent reviewer's signoff. Each verification is logged in `GLOSSARY.md` next to the term.

#### SAN-B-003 — Quote each currently-defective Sanskrit term and its corrected form
- **Statement:** Before launch, the project MUST produce a `SANSKRIT_FIX_REGISTER.md` listing every Sanskrit term currently appearing in the codebase (`pramana_test.py`, `NOTICE`, `README.md`, `SCHEMA_15_PHASE_DRAFT.md`, `SAMPLE_OUTPUT.md`, `SCORING_VALIDATION.md`) with: the current ASCII rendering, the corrected IAST rendering, the source file and line number, and the Sanskrit-grammatical justification (which sandhi rule, which vowel-length, which retroflex). This register is reviewed and signed off by a Sanskrit-fluent reviewer before any launch artifact updates.
- **Severity:** MUST
- **Rationale:** The current corpus has at least the following defects, quoted explicitly:
  - `pramana_test.py:189` — "Navya-Nyaya" should be "Navya-Nyāya"
  - `pramana_test.py:204` — "Samshaya" should be *saṃśaya*
  - `pramana_test.py:205-207` — "Samana Dharma Upapatti, Aneka Dharma Upapatti, Vipratipatti, Upalabdhi Avyavastha, Anupalabdhi Avyavastha" should be *samāna-dharma-upapatti, aneka-dharma-upapatti, vipratipatti, upalabdhi-avyavasthā, anupalabdhi-avyavasthā*
  - `pramana_test.py:209` — "Pramana" should be *pramāṇa*
  - `pramana_test.py:213` — "Pratyaksha" should be *pratyakṣa*
  - `pramana_test.py:214` — "Anumana" should be *anumāna*
  - `pramana_test.py:215` — "Upamana" should be *upamāna*
  - `pramana_test.py:215` — "Shabda" should be *śabda*
  - `pramana_test.py:217` — "Pancha Avayava" should be *pañcāvayava* (single compound, sandhi-correct)
  - `pramana_test.py:220` — "Pratijna" should be *pratijñā*
  - `pramana_test.py:222` — "Udaharana" should be *udāharaṇa*
  - `pramana_test.py:223` — "Upanaya" should be *upanaya* (no diacritic) — fine
  - `pramana_test.py:223` — "Nigamana" should be *nigamana* — fine in ASCII
  - `pramana_test.py:232` — "Hetvabhasa" should be *hetvābhāsa*
  - `pramana_test.py:236` — "Savyabhichara" should be *savyabhicāra* (NOT "savyabhicara" with a half-Sanskritized *cha*)
  - `pramana_test.py:237` — "Viruddha" — fine
  - `pramana_test.py:238` — "Prakaranasama" should be *prakaraṇasama* (retroflex *ṇ*)
  - `pramana_test.py:239` — "Sadhyasama" should be *sādhyasama* (long *ā*)
  - `pramana_test.py:240` — "Kalaatita" should be *kālātīta* (long *ā*, long *ā*, long *ī*)
  - `pramana_test.py:241` — "Nirnaya" should be *nirṇaya* (retroflex *ṇ*)
  - `NOTICE:17` — "Samshaya, Pramana, Pancha Avayava, Tarka, Hetvabhasa, Nirnaya" — all defective per above
  - `NOTICE:45-47` — same defects re-occur
  - `SCHEMA_15_PHASE_DRAFT.md` passim — "Pancha-avayava" (two words hyphenated; should be one), "Tattvacintamani" (no diacritics), "Anumana-khanda" (no diacritics), etc.
- **Severity:** MUST
- **Acceptance criterion:** `SANSKRIT_FIX_REGISTER.md` exists, lists every term and correction, and is signed off by a named Sanskrit-fluent reviewer (per SAN-G-001 below).

#### SAN-B-004 — Distinguish the prompt-portability ASCII channel from the documentation channel
- **Statement:** ASCII Sanskrit MAY remain in the LLM-facing system prompt (`pramana_test.py:189-247`) for the *narrow, defensible reason* that some tokenizer / model behaviors are sensitive to non-ASCII Unicode. This is acceptable IF AND ONLY IF: (a) a comment in the file explicitly notes the choice and the canonical IAST forms, (b) the documentation channel everywhere else uses IAST, and (c) the project tests whether IAST in the prompt actually changes scoring (REQ-PRACTITIONER-equivalent: a controlled comparison of ASCII vs IAST prompt on the same problems).
- **Severity:** SHOULD
- **Rationale:** A blanket ban on ASCII-in-prompts overconstrains; a blanket allowance of ASCII everywhere undersignals competence. The Sanskritist understands the tokenizer concern when it is named; what they cannot forgive is silent ASCII default with no explanation.
- **Acceptance criterion:** A 3-line comment in `pramana_test.py` near the system prompt names each ASCII term, gives its IAST canonical form, and either cites a tokenizer-behavior result (with link to the experiment) or flags this as an unverified hypothesis to test.

---

### C. Concept-name accuracy within the correct darshana

The schema uses concepts whose technical meaning differs across darshanas. Selecting a meaning silently is a school-conflation error.

#### SAN-C-001 — Anumāna in the Naiyāyika sense, not the Buddhist sense
- **Statement:** The schema MUST explicitly state that *anumāna* is treated in the Naiyāyika sense — five-membered (*pañcāvayava*: *pratijñā, hetu, udāharaṇa, upanaya, nigamana*) — and acknowledge that the Buddhist (Diṅnāga, Dharmakīrti) treatment is three-membered (*pakṣa-dharmatā, sapakṣe sattvam, vipakṣe asattvam* — the *trairūpya*). The five-member vs three-member distinction is *the* most-debated structural difference between Hindu and Buddhist logic in the 5th–13th centuries.
- **Severity:** MUST
- **Rationale:** S3 (Buddhist Studies specialist) will dock the project immediately if it presents the Naiyāyika five-member as "the Indian inferential structure." It is the *Hindu Naiyāyika* version. The Buddhist version is shorter and arguably more elegant — Diṅnāga collapsed *upanaya* and *nigamana* as redundant rhetoric.
- **Acceptance criterion:** The schema document's Phase 11 (*pañcāvayava*) section contains a "Buddhist alternative" sub-paragraph naming the *trairūpya* and citing at least one Buddhist-logic source (e.g., Tom Tillemans, *Persons of Authority*, 1999; Richard Hayes, *Diṅnāga on the Interpretation of Signs*, 1988; or Mark Siderits, *Indian Philosophy of Language*, 1991).

#### SAN-C-002 — Upamāna: Naiyāyika vs Mīmāṃsaka definition
- **Statement:** *Upamāna* (analogy / similarity-based cognition) is treated differently across schools:
  - In **Nyāya** (following Annambhaṭṭa, *Tarka-Saṃgraha*; Gautama, *Nyāya-Sūtra* 1.1.6): *upamāna* yields knowledge of the relation between a name and the entity named, by similarity to a known case. The standard example: a forest-dweller who has never seen a *gavaya* (wild ox) is told it is "like a cow"; on later seeing one, they correctly identify it.
  - In **Mīmāṃsā** (Kumārila Bhaṭṭa, *Ślokavārttika*): *upamāna* gives knowledge of the *similarity itself* between two objects. The example differs: knowing a remembered cow is similar to a present *gavaya* yields the *upamiti* "my cow is similar to this *gavaya*."
  - In **Advaita Vedānta** (Dharmarāja, *Vedāntaparibhāṣā*): closer to Mīmāṃsā but with theological framing.
  
  The schema MUST state which school's *upamāna* it is using. The current `pramana_test.py:215` line ("Upamana: <analogies to known cases, if any>") is ambiguous but defaults to a non-technical "analogy" sense that matches *neither* school precisely.
- **Severity:** MUST
- **Rationale:** Conflating Naiyāyika *upamāna* with Mīmāṃsaka *upamāna* is the kind of school-error that any Indian-philosophy specialist (S1, S2, S3, S4) catches in seconds. It is also the kind of error the founder cannot detect without a school-aware reviewer because the surface English ("analogy") is the same.
- **Acceptance criterion:** The schema document's *upamāna* discussion (currently absent — the schema collapses the four pramāṇas into Phase 6) is rewritten to state explicitly: "We follow Annambhaṭṭa (Naiyāyika) treatment of *upamāna* as cognition of the *name-entity relation* via similarity." A footnote acknowledges the Mīmāṃsā divergence and cites Kumārila *Ślokavārttika* on *upamāna*.

#### SAN-C-003 — Śabda: Naiyāyika trustworthy testimony, not Mīmāṃsaka Vedic-restricted testimony
- **Statement:** Already specified in SAN-A-003. Repeated here for school-conflation completeness: the schema MUST treat *śabda* as Naiyāyika (general *āpta-vākya*), explicitly not Mīmāṃsaka (*Veda*-restricted), and the documentation must say so.
- **Severity:** MUST
- **Rationale:** Internal consistency with SAN-A-003.
- **Acceptance criterion:** Same as SAN-A-003.

#### SAN-C-004 — Hetvābhāsa: Old Nyāya vs Navya-Nyāya five-fold lists
- **Statement:** The Old Nyāya (Gautama, NS 1.2.4–9) five-fold *hetvābhāsa* list is: *savyabhicāra* (irregular / inconclusive), *viruddha* (contradictory), *prakaraṇasama* (the reason-equal-to-the-question — i.e., the issue itself), *sādhyasama* (the reason-equal-to-the-thing-to-be-proved — i.e., begging the question), *kālātīta* (mistimed / belated). The Navya-Nyāya / Annambhaṭṭa (*Tarka-Saṃgraha*) five-fold list is: *savyabhicāra*, *viruddha*, *satpratipakṣa* (counter-balanced by an opposing inference), *asiddha* (unestablished — sub-types *āśraya-asiddha*, *svarūpa-asiddha*, *vyāpyatva-asiddha*), *bādhita* (sublated by another *pramāṇa*). The two lists overlap in only two members (*savyabhicāra*, *viruddha*) and the divergence is *substantive* — *satpratipakṣa* and *bādhita* are technically richer concepts than *prakaraṇasama* and *kālātīta*. A schema that mixes the two lists silently is committing school-conflation.

  The current `pramana_test.py:236-240` lists: *Savyabhichara, Viruddha, Prakaranasama, Sadhyasama, Kalaatita*. **This is Gautama's old list, not Annambhaṭṭa's Navya list.** The 6-phase prompt is therefore using Old Nyāya, not Navya-Nyāya. The 15-phase schema draft (`SCHEMA_15_PHASE_DRAFT.md` Phase 13) correctly switches to Annambhaṭṭa's Navya list. **The two artifacts are using different lists** without acknowledging it. A Navya-Nyāya specialist reviewing both will flag this as either ignorance of the difference or concealment of the schema migration.
- **Severity:** MUST
- **Rationale:** This is the exact failure mode the audit (Finding 1.4) flagged for Sathish-comparison purposes. From the Sanskrit-domain side, the failure is more direct: the project's own two artifacts disagree about which Nyāya it is.
- **Acceptance criterion:** Either (a) the 6-phase prompt is migrated to Annambhaṭṭa's Navya list (i.e., replace *prakaraṇasama, sādhyasama, kālātīta* with *satpratipakṣa, asiddha, bādhita*) AND the NOTICE/README is updated to reflect the migration, OR (b) the project explicitly maintains two version lines — "pramāṇa-eval-old (6-phase, Gautama hetvābhāsa)" vs "pramāṇa-eval-navya (15-phase, Annambhaṭṭa hetvābhāsa)" — with the schema-version axis surfaced everywhere.

---

### D. Buddhist logic interactions

Navya-Nyāya's mature form is unintelligible without the Buddhist-logic prehistory. Diṅnāga (c. 480–540 CE) developed the *trairūpya* doctrine of inferential marks; Dharmakīrti (c. 600–660 CE) refined it and added the *anyāpoha* theory of meaning. Naiyāyikas from Uddyotakara (c. 6th–7th c.) onwards spent six centuries in dialectical engagement with this Buddhist apparatus. Gaṅgeśa's *Tattvacintāmaṇi* (12th–13th c.) — the founding Navya-Nyāya text — is structured as a sustained response to and refinement of this earlier debate. To present Navya-Nyāya as a "Hindu logic system" sealed from Buddhism is to falsify its history.

#### SAN-D-001 — Acknowledge the Buddhist-logic interlocutor in any historical framing
- **Statement:** The schema document's "Source survey" section (currently `SCHEMA_15_PHASE_DRAFT.md §2`) MUST acknowledge that Uddyotakara's *Nyāya-Vārttika* and subsequent Navya-Nyāya developments are responses to Buddhist *pramāṇavāda* (Diṅnāga, Dharmakīrti). The sentence "Defends NS against Buddhist (esp. Dignaga / Dharmakirti) critique" is in the current draft but is a one-liner; expand to a paragraph naming (a) Diṅnāga's *Pramāṇasamuccaya*, (b) Dharmakīrti's *Pramāṇavārttika*, (c) the *trairūpya* doctrine and its relation to *vyāpti*, (d) the *anyāpoha* doctrine and its relation to Naiyāyika theories of universals (*sāmānya*).
- **Severity:** MUST
- **Rationale:** S3's strongest probe. Closes off the "sealed-Hindu-system" framing.
- **Acceptance criterion:** The Source Survey paragraph names Diṅnāga's *Pramāṇasamuccaya* and Dharmakīrti's *Pramāṇavārttika* with approximate dates, and the schema document's discussion of *vyāpti* (Phase 8) acknowledges that Gaṅgeśa's five-fold treatment is a response to the prior Buddhist *trairūpya* analysis — not an isolated Hindu innovation.

#### SAN-D-002 — Cite at least one Buddhist-logic secondary source
- **Statement:** The schema's bibliography (or `CITATIONS.md` from REQUIREMENTS_SCHOLAR) MUST include at least one named Buddhist-logic scholar's work, e.g.:
  - Richard Hayes, *Diṅnāga on the Interpretation of Signs* (Kluwer, 1988)
  - Tom Tillemans, *Scripture, Logic, Language: Essays on Dharmakīrti and his Tibetan Successors* (Wisdom, 1999)
  - Mark Siderits, *Indian Philosophy of Language* (Kluwer, 1991)
  - Tillemans, *Persons of Authority: The sTon pa tshad ma'i skyes bur sgrub pa'i gtam of A lag sha Ngag dbang bstan dar* (Steiner, 1993)
  - Birgit Kellner et al., eds., *Religion and Logic in Buddhist Philosophical Analysis* (2010)
- **Severity:** SHOULD
- **Rationale:** S3's verifiability test. A bibliography that names only Hindu-side scholars (Phillips, Mohanty, Matilal, Ganeri, Bhattacharyya) is signaling a one-sided reading.
- **Acceptance criterion:** At least one Buddhist-logic monograph in the bibliography.

#### SAN-D-003 — No claim that Navya-Nyāya logic is "Indian logic" or "the Indian view of inference"
- **Statement:** Public-facing artifacts MUST NOT describe Navya-Nyāya as "Indian logic" without scare-quotes or qualifier. The term "Indian logic" is Matilal's coinage (most prominently *The Character of Logic in India*, 1998), and was contested in Mohanty's, Ganeri's, and Bhattacharyya's framings. The single-bucket framing erases (a) Buddhist *pramāṇavāda*, (b) Jain *naya-pramāṇa* theories, (c) Cārvāka skepticism, (d) Mīmāṃsā's distinct *pramāṇa* enumeration. The defensible framing is "the Naiyāyika tradition of *pramāṇa-śāstra*" or "Navya-Nyāya logic and epistemology."
- **Severity:** SHOULD
- **Rationale:** S2 and S3 will both flag the term. S5 (decolonial critic) will flag it as homogenizing.
- **Acceptance criterion:** Grep over public artifacts for `Indian logic` and `Indian inference`. Each occurrence is either removed, qualified ("Indian logic (Matilal's term, contested)"), or scoped ("Naiyāyika logic and epistemology").

---

### E. Living tradition and consent

Navya-Nyāya is a living tradition. Active institutions:
- **Mithila Institute of Post-Graduate Studies and Research in Sanskrit Learning**, Darbhanga, Bihar — established 1951 by Maharaja Kameshwar Singh of Raj Darbhanga, inaugurated by Rajendra Prasad. The institutional successor to the Mithila school where Gaṅgeśa Upādhyāya himself taught in the 12th–13th c. Publishes the journal *Mithilā Bhāratī*. Holds the largest collection of Navya-Nyāya manuscripts in the world.
- **Sampūrṇānanda Saṃskṛta Viśvavidyālaya**, Vārāṇasī — formerly the Government Sanskrit College (founded 1791 by Jonathan Duncan as one of the earliest British-era Sanskrit institutions). Strong department of *Nyāya-Vaiśeṣika*.
- **Madras Sanskrit College**, Chennai — founded 1906; strong Naiyāyika tradition descending from the South Indian *paṇḍita* lineages.
- **Rāṣṭriya Saṃskṛta Saṃsthāna** (Central Sanskrit University since 2020), with branches in Tirupati, Jaipur, Sringeri, Bhopal — public-funded, *paṇḍita*-tradition oriented.
- **Kavikulaguru Kālidāsa Saṃskṛta Viśvavidyālaya**, Ramtek (Maharashtra) — younger institution but with active *Nyāya-śāstra* department.
- **The Asiatic Society**, Kolkata — preserves Bengali-school Navya-Nyāya tradition (the line through Raghunātha Śiromaṇi, Mathurānātha, Jagadīśa Tarkālaṅkāra, and Gadādhara Bhaṭṭācārya).
- **Householder *paṇḍita* lineages** in Bengal, Odisha, Bihar, Tamil Nadu, and Andhra still teaching *Tarka-Saṃgraha* + *Dīpikā* outside formal academic settings.

#### SAN-E-001 — Pre-launch consultation with at least one Sanskrit institution
- **Statement:** Before public launch, the project MUST attempt formal consultation with at least one of the named Sanskrit institutions above. "Attempt" means: (a) a written letter or email of introduction to a named department head or *paṇḍita*; (b) sharing the schema document, NOTICE, and proposed launch artifacts; (c) inviting critique or correction; (d) offering a 3-week minimum response window before launch. If the institution responds, feedback is incorporated or disagreements documented (per REQUIREMENTS_SCHOLAR SCH-G-004). If no institution responds despite three documented attempts and a 4-week window, that fact is itself disclosed in the NOTICE.
- **Severity:** MUST
- **Rationale:** S5's strongest probe — "what is your consent relationship to the living tradition?" — is unanswerable without this. The cost (3 letters + 4 weeks) is small; the cost of the absence is durable: the project will be read by a non-trivial Indian academic audience as extractive regardless of its technical merits. This is the single highest-leverage anti-decolonial-critique step.
- **Acceptance criterion:** A `LIVING_TRADITION_OUTREACH.md` (or equivalent section in NOTICE) records the institutions contacted, dates, contact persons, and outcomes (response received and incorporated; response received and disagreed; no response within 4 weeks). A Mithila Institute or Sampūrṇānanda contact is strongly recommended given their primary parampara position.

#### SAN-E-002 — Living-tradition acknowledgement in the NOTICE
- **Statement:** The NOTICE file MUST contain a "Living tradition" section that names at least three living institutions or scholars (per REQUIREMENTS_SCHOLAR SCH-E-002, deepened here). Naming should include at least one institution in India and one Sanskrit-medium scholar by name where possible. Acceptable named living scholars (in 2026): N. S. Ramanuja Tatacharya (Madras Sanskrit College tradition; co-translator of TC with Phillips); Karuna Bhattacharya (Kolkata, Bengali Navya-Nyāya line); Jonardon Ganeri (named in his Western-academic capacity but with continuing engagement with Indian institutions); the current head of the Mithila Institute (consult institutional website for current name).
- **Severity:** MUST
- **Rationale:** Goes beyond REQUIREMENTS_SCHOLAR's generic "name three" by requiring at least one Indian institution and one named scholar.
- **Acceptance criterion:** NOTICE contains a "Living tradition" section with ≥ 3 named entities; ≥ 1 is an India-based institution; ≥ 1 is an individual named scholar.

#### SAN-E-003 — No claim of equivalence between OSS license and tradition
- **Statement:** The NOTICE's existing disclaimer ("ModelBrew AI makes no claim of ownership, authorship, or authority over the Navya-Nyaya tradition...") MUST be expanded to acknowledge that **Apache 2.0 licensing of the harness does not, and cannot, license the underlying tradition**. The tradition is not the project's to license; the project releases its own engineering work under Apache 2.0 and disclaims any intent to extend that licensing frame to the *vidyā-paramparā*.
- **Severity:** MUST
- **Rationale:** S5's epistemic-property critique. Apache 2.0 is a property regime; classical *vidyā-dāna* (the gift of knowledge) is a different regime. The two are not commensurable, and pretending the OSS license covers the source material is the colonialism the founder is trying to avoid.
- **Acceptance criterion:** A paragraph in NOTICE explicitly states: "The Apache 2.0 license applies to the engineering artifacts in this repository (the test harness, the format/semantic scorer, the dataset packaging, the documentation). It does not, and cannot, extend to the underlying Navya-Nyāya tradition, which is sustained by living *paṇḍita-paramparā*s and is not the project's to license. Users of the harness do not, by virtue of the Apache 2.0 grant, acquire any right or authority over the tradition itself."

---

### F. Translation politics

#### SAN-F-001 — Disclose translator commitments
- **Statement:** When the schema's documentation cites a translation of a Sanskrit term, the citation MUST acknowledge the translator's theoretical orientation:
  - **Bimal Krishna Matilal** (1935–1991): analytic-philosophy framing; influenced by Quine, Strawson, Davidson; tends to formalize Indian logic in first-order-logic / set-theoretic terms; *The Character of Logic in India* (1998), *Logic, Language, and Reality* (1985).
  - **Jitendra Nath Mohanty** (1928–2024): phenomenological framing influenced by Husserl; treats *pramāṇa* as a cognitive-act phenomenon; *Gaṅgeśa's Theory of Truth* (1989), *Reason and Tradition in Indian Thought* (1992).
  - **Stephen Phillips**: textual-philological framing combined with analytic-epistemology engagement; co-edited *Epistemology of Perception* (with N. S. Ramanuja Tatacharya, 2004); *Epistemology in Classical India* (2012).
  - **Jonardon Ganeri**: contemporary-philosophy framing, drawing on philosophy-of-mind and self-models; *Philosophy in Classical India* (2001), *The Concealed Art of the Soul* (2007), *The Lost Age of Reason* (2014).
  - **Sibajiban Bhattacharyya** (1923–2019): Bengali-school heir to a paṇḍita lineage; tends to preserve Sanskrit technicality with minimum Western-philosophy dressing.
  - **Karl Potter** (1927–): encyclopedic-summary framing in the *Encyclopedia of Indian Philosophies* series; consensus-aggregation rather than original-voice.
  - **Wisdomlib / IEP / SEP**: digest-secondary sources; useful but not authorities.
  - These commitments matter because, e.g., Phillips's "pervasion" and Mohanty's "invariable concomitance" carry different implicit metaphysics for *vyāpti*, and citing Phillips alone hides a metaphysical commitment.
- **Severity:** SHOULD
- **Rationale:** S2's probe. Treating English-language Sanskrit translations as if they were the Sanskrit original is a methodological error.
- **Acceptance criterion:** When a translation is cited in load-bearing argumentative position, a one-clause acknowledgement of the translator's framing accompanies the citation, e.g., "Phillips's 'pervasion' (analytic-philosophy framing); Mohanty's 'invariable concomitance' (phenomenological framing)."

#### SAN-F-002 — When in doubt, prefer Sanskrit-original term to English translation
- **Statement:** In schema-defining documentation, the canonical reference MUST be the Sanskrit term (in IAST). English translations are aids, not substitutes. Phrasings like "the *vyāpti* (which Phillips translates as 'pervasion' and Mohanty as 'invariable concomitance')" are preferred over phrasings like "pervasion (*vyāpti*)."
- **Severity:** SHOULD
- **Rationale:** The Sanskrit term is the stable referent across translation traditions; the English gloss is the variable.
- **Acceptance criterion:** Style guide in the repo (or `STYLE.md`) names this convention.

---

### G. The naming problem

#### SAN-G-001 — The hybrid "pramana-eval" name
- **Statement:** "pramana-eval" is a hybrid: Sanskrit philosophical noun + English engineering suffix. The hybrid is itself a stance — the project sits between two epistemic worlds — and is defensible IF declared. The project MUST disclose, in the README, that the name is a deliberate hybrid and explain the choice. Defensibility hinges on:
  - Acknowledging that *pramāṇa* (the Sanskrit term, in IAST) means "means of valid cognition" in the philosophical context; calling a benchmark "pramāṇa" implies the benchmark *is* a means of valid cognition, which is conceptually odd. The benchmark *tests* whether an LLM exhibits *pramāṇa*-like reasoning — different ontological category.
  - Considering whether the hybrid form is an asset (signals the cross-tradition engagement) or a liability (looks like a marketing portmanteau).
- **Severity:** SHOULD
- **Rationale:** S1, S2, and S5 will all probe the naming. The defense is "we know what we are doing and here is why" — declared hybridity is defensible; undeclared hybridity reads as casual appropriation.
- **Acceptance criterion:** README contains a "Naming" paragraph: (a) the term *pramāṇa* in IAST with gloss, (b) acknowledgement that the benchmark is not itself a *pramāṇa* but a test of *pramāṇa*-like reasoning, (c) brief justification for the hybrid choice (e.g., engineering-recognizability + tradition-acknowledgement) or proposal of an alternative.

#### SAN-G-002 — Avoid name collisions with other Sanskrit-named projects
- **Statement:** Before public launch, the project MUST search PyPI, NPM, GitHub topics, and arXiv abstracts for prior uses of "pramana-eval", "pramaana-eval", "pramāṇa-eval", "navya-nyaya-bench", "nyaya-bench", "tarkasangraha", and similar candidate names, and disclose any collisions. The 2026 Sathish paper itself uses "Pramana" as part of its title; the project's name MUST be distinguishable from Sathish's.
- **Severity:** MUST
- **Rationale:** Naming hygiene + Sathish-mirroring concern (already handled in REQUIREMENTS_SCHOLAR SCH-C, repeated here for naming-completeness).
- **Acceptance criterion:** A `NAMING_DUE_DILIGENCE.md` (or section in NOTICE) records the search and outcomes.

#### SAN-G-003 — If the project chooses an alternative name, it must be Sanskrit-grammatically defensible
- **Statement:** If, after SAN-G-001 reflection, the founder chooses a different name, the alternative MUST be either (a) fully English (e.g., "Naiyāyika Reasoning Bench"), (b) fully Sanskrit (e.g., *pramāṇa-mīmāṃsā* — though this collides with Hemacandra's 12th-c. work and would be a school-conflation), or (c) a declared hybrid with the Sanskrit component sandhi-correct and IAST-rendered. Names that are non-IAST ASCII Sanskrit (like the current "pramana-eval") are the worst of all worlds.
- **Severity:** NICE-TO-HAVE
- **Rationale:** Naming integrity if the founder chooses to rename.
- **Acceptance criterion:** If renamed, the alternative passes IAST + sandhi check.

---

### H. Sectarian / political / decolonial considerations

The current era (2026) has heightened sensitivity around dharma-related projects originating from Western tech contexts. Two failure poles:
- **Pole A (Hindutva-instrumentalization):** the project being read as Hindu-nationalist instrumentalization of philosophy ("ancient Vedic logic proves Hinduism is the original logic; AI confirms it"). The "Vedic AI" branding the founder rejected was on this trajectory.
- **Pole B (extractive-orientalism):** the project being read as Bay-Area-tech extracting from Indian tradition without consent or compensation. This is the S5 critique.

Both poles are real audiences. Threading the needle requires deliberate framing.

#### SAN-H-001 — No "Hindu", "Vedic", "ancient wisdom" framing
- **Statement:** Public-facing artifacts MUST NOT use any of: "Hindu logic," "Vedic logic," "Sanatan logic," "ancient Indian wisdom," "the wisdom of the rishis," "the gift of dharma," "the lost reasoning of the Vedas." The defensible framing is: "an open-source benchmark grounded in the *Naiyāyika* (Navya-Nyāya) tradition of *pramāṇa-śāstra*, designed for falsifiable evaluation of LLM reasoning behavior." This is the disciplined version of REQUIREMENTS_SCHOLAR SCH-E-004, deepened with specific Hindutva-coded phrases the Sanskrit specialist would also flag.
- **Severity:** MUST
- **Rationale:** Defends against Pole A (Hindutva-instrumentalization).
- **Acceptance criterion:** Grep over public artifacts for the listed phrases; zero hits.

#### SAN-H-002 — No "extracting wisdom" or "AI rediscovers" framing either
- **Statement:** Already covered by REQUIREMENTS_SCHOLAR SCH-E-004. Repeated here for symmetry: "AI rediscovers ancient wisdom," "machine intelligence catches up to the rishis," "Sanskrit AI revival" — all forbidden. Defends against Pole B (extractive-orientalism).
- **Severity:** MUST
- **Rationale:** Symmetry with SAN-H-001.
- **Acceptance criterion:** Same as REQUIREMENTS_SCHOLAR SCH-E-004; this requirement deepens but does not duplicate.

#### SAN-H-003 — Acknowledge the project's specific positionality
- **Statement:** The README MUST contain a "Positionality" or "About this project" paragraph stating: (a) the founder's background (ModelBrew AI, ML/data infrastructure, not a Sanskrit specialist or Indian-philosophy academic), (b) the founder's relationship to the tradition (likely diaspora; if so, named), (c) the explicit choice not to brand the project as "Vedic" or use spiritual/religious framing, (d) the intent of consultative engagement with Indian academic institutions (per SAN-E-001).
- **Severity:** SHOULD
- **Rationale:** Pre-empts both Pole A and Pole B by being legible about who is doing what and why.
- **Acceptance criterion:** A 1-paragraph "Positionality" section exists in README.

---

### I. Buddhist / Jain / Mīmāṃsā / Cārvāka boundary cases

The schema's *pramāṇa* list (in `pramana_test.py:213-215`: *pratyakṣa, anumāna, upamāna, śabda*) is the **Naiyāyika 4-fold list**. Other schools differ:

- **Cārvāka** (materialist skeptics, c. 6th c. BCE onwards, surviving primarily through opponents' citations — Mādhava's *Sarvadarśanasaṃgraha*, 14th c.): accepts only *pratyakṣa*. One-fold.
- **Buddhism** (Diṅnāga, Dharmakīrti): accepts only *pratyakṣa* + *anumāna*. Two-fold. *Śabda* is reduced to an indirect form of *anumāna*; *upamāna* is rejected as not yielding new cognition.
- **Vaiśeṣika** (Kaṇāda's *Vaiśeṣikasūtra*; close relative of Nyāya): accepts only *pratyakṣa* + *anumāna*. Two-fold (some Vaiśeṣika authors include *śabda* under *anumāna*).
- **Sāṃkhya** (*Sāṃkhya-Kārikā*, Īśvarakṛṣṇa, c. 4th–5th c.): accepts *pratyakṣa, anumāna, āptavacana* (= *śabda*). Three-fold.
- **Yoga** (Patañjali's *Yogasūtra* 1.7): same three as Sāṃkhya.
- **Nyāya / Naiyāyika**: four-fold (the schema's list).
- **Bhāṭṭa Mīmāṃsā** (Kumārila Bhaṭṭa): adds *arthāpatti* (presumption / postulation) and *anupalabdhi* (non-apprehension). Six-fold.
- **Prābhākara Mīmāṃsā** (Prabhākara): adds *arthāpatti* but not *anupalabdhi*. Five-fold.
- **Advaita Vedānta** (Dharmarāja, *Vedāntaparibhāṣā*): six-fold, following Bhāṭṭa.
- **Jaina**: classifies under *pratyakṣa* (further subdivided into ordinary and extraordinary) and *parokṣa* (mediated cognition; subdivided into *smṛti, pratyabhijñā, tarka, anumāna, āgama*). Functionally similar to four-fold but different conceptual structure (Hemacandra's *Pramāṇamīmāṃsā*).

Presenting the Naiyāyika 4-fold as "the Indian view" without acknowledging these is sectarian Hindu erasure of Buddhist, Jain, and Cārvāka positions.

#### SAN-I-001 — Acknowledge non-Naiyāyika *pramāṇa* enumerations
- **Statement:** The schema document MUST contain a paragraph (in the source survey or in the *pramāṇa* phase discussion) acknowledging that the four-fold *pramāṇa* enumeration is **Naiyāyika-specific** and that other schools differ:
  - Cārvāka: 1 (*pratyakṣa* only)
  - Buddhism, Vaiśeṣika: 2 (*pratyakṣa, anumāna*)
  - Sāṃkhya, Yoga: 3 (+ *śabda*)
  - Nyāya: 4 (+ *upamāna*) — the schema's choice
  - Prābhākara Mīmāṃsā: 5 (+ *arthāpatti*)
  - Bhāṭṭa Mīmāṃsā, Advaita Vedānta: 6 (+ *anupalabdhi*)
- **Severity:** MUST
- **Rationale:** S3 (Buddhist Studies), S4 (Sanskrit-medium), and S5 (decolonial — sensitive to sectarian erasure) will all flag the Hindu-Naiyāyika single-bucket framing.
- **Acceptance criterion:** A "Pramāṇa enumeration across schools" subsection exists in the schema document, with the six positions named and citations to standard scholarly references for each (e.g., Eli Franco for Cārvāka; Hayes for Buddhism; Larson for Sāṃkhya; Taber for Mīmāṃsā).

#### SAN-I-002 — The schema's *anupalabdhi* phase (Phase 14) MUST acknowledge the school dispute
- **Statement:** The current 15-phase schema treats *anupalabdhi* as a *phase* (Phase 14) under Udayana's reading, *not* as a separate sixth *pramāṇa*. This is the Naiyāyika position *against* Bhāṭṭa Mīmāṃsā. The schema document already partially acknowledges this (in `SCHEMA_15_PHASE_DRAFT.md` Phase 14's "Design note") but the acknowledgement must be expanded: name Kumārila explicitly, name the *Ślokavārttika* passage, and explain that the schema's choice is school-aligned (Naiyāyika), not neutral.
- **Severity:** MUST
- **Rationale:** Continues the SAN-A-005-style "alternatives considered" discipline from REQUIREMENTS_SCHOLAR.
- **Acceptance criterion:** Phase 14's "Design note" cites Kumārila *Ślokavārttika* on *anupalabdhi* and acknowledges the schema's Naiyāyika alignment.

#### SAN-I-003 — If the schema is open-sourced as "the" framework, document its sectarian commitments
- **Statement:** Any framing that positions the schema as *the* canonical framework for evaluating LLM reasoning in an "Indian-philosophy" mode is forbidden. The schema is *one* canonical framework — the Naiyāyika one, in its Navya-Nyāya phase, in dialogue with Buddhist *pramāṇavāda*. A Buddhist-logic-grounded benchmark (using *trairūpya*, *anyāpoha*, *svalakṣaṇa* / *sāmānyalakṣaṇa* perception distinction) would be a different schema. A Mīmāṃsaka benchmark would be a different schema. The project's own framing must make this explicit.
- **Severity:** SHOULD
- **Rationale:** Pre-empts S3 and S4's "why are you erasing my tradition?" critique.
- **Acceptance criterion:** A "Schema scope and sectarian commitments" paragraph in README and schema document explicitly says: this is the Naiyāyika schema; a Buddhist-logic schema, a Mīmāṃsaka schema, a Jaina schema would each be different.

---

### J. Pre-launch Sanskrit-fluent review (deepens REQUIREMENTS_SCHOLAR SCH-G-001)

#### SAN-J-001 — The reviewer's competence specifically includes Sanskrit reading and darshana-school awareness
- **Statement:** REQUIREMENTS_SCHOLAR SCH-G-001 specifies "at least one Sanskrit-fluent reader who has formal training in Indian philosophy." This document deepens that to require: the reviewer must be able to read at least *Tarka-Saṃgraha* + *Tarka-Dīpikā* in Sanskrit (not in English translation), AND must be familiar with at least one school other than Nyāya (Buddhist *pramāṇavāda*, Mīmāṃsā, Jaina, or Vedānta). A reviewer who can only read Phillips and Mohanty in English does not satisfy this requirement.
- **Severity:** MUST
- **Rationale:** A reviewer who cannot read *Tarka-Saṃgraha* in Sanskrit will not catch sandhi errors or school-conflations — i.e., the failure modes this document is designed to catch. Asking for Sanskrit-reading competence specifically is the only way to ensure the review is from inside the tradition rather than alongside it.
- **Acceptance criterion:** The named reviewer's qualifications (in `LIVING_TRADITION_OUTREACH.md` or NOTICE) include explicit Sanskrit-reading competence and at least one non-Nyāya darshana familiarity. Acceptable named reviewers include: any current faculty member at the Mithila Institute (Darbhanga); Sampūrṇānanda Saṃskṛta Viśvavidyālaya Nyāya department faculty; Madras Sanskrit College *paṇḍita*s; Karuna Bhattacharya (Calcutta); N. S. Ramanuja Tatacharya (Madras); Jonardon Ganeri's NYU/SOAS doctoral students with Sanskrit competence; Jay Garfield (Smith); Parimal Patil (Harvard); Daniel Stuart (South Carolina, Buddhist-side); Birgit Kellner (Vienna).

#### SAN-J-002 — The reviewer reviews the Sanskrit-specific items in this document
- **Statement:** The reviewer named under SAN-J-001 specifically signs off on: (a) `SANSKRIT_FIX_REGISTER.md` (per SAN-B-003), (b) `GLOSSARY.md` (per SAN-B-001), (c) the school-conflation requirements (SAN-C-001 to SAN-C-004), (d) the Buddhist-acknowledgement paragraph (SAN-D-001), (e) the *pramāṇa* enumeration paragraph (SAN-I-001).
- **Severity:** MUST
- **Rationale:** The reviewer's signoff is itself the verifiable artifact.
- **Acceptance criterion:** A "Sanskrit-domain reviewer" line in NOTICE names the reviewer and lists the specific items reviewed.

---

## 3. Anti-patterns to forbid (Sanskrit-domain specific)

The following are explicit prohibitions, drawn from common "AI-meets-Sanskrit-philosophy" failure modes. They do not duplicate REQUIREMENTS_SCHOLAR's anti-patterns; they are Sanskrit-specific.

1. **SAN-AP-001 — "Vedic AI" / "ancient Vedic logic" / "Vedic reasoning system" framings.** Category error per SAN-A-001. The most damaging single failure.

2. **SAN-AP-002 — Mixing Old Nyāya (Gautama's *prakaraṇasama, sādhyasama, kālātīta*) with Navya-Nyāya (Annambhaṭṭa's *satpratipakṣa, asiddha, bādhita*) in the *hetvābhāsa* list silently.** The current `pramana_test.py` (Old Nyāya list) and `SCHEMA_15_PHASE_DRAFT.md` (Navya list) already commit this, undisclosed (per SAN-C-004).

3. **SAN-AP-003 — ASCII Sanskrit ("Pancha Avayava," "Pramana," "Hetvabhasa") as the default rendering in human-facing documentation.** Per SAN-B-001. ASCII in code is defensible; ASCII in documentation is not.

4. **SAN-AP-004 — Sandhi-incorrect compound words** like "Pancha Avayava" (should be *pañcāvayava*), "Hetvabhasa" (should be *hetvābhāsa*), "Anumana-khanda" (should be *Anumāna-khaṇḍa*). Per SAN-B-002.

5. **SAN-AP-005 — Using Western-Indologist English translations as if they were the Sanskrit original**, without acknowledging the translator's theoretical framing (per SAN-F-001).

6. **SAN-AP-006 — Presenting the Naiyāyika 4-fold *pramāṇa* enumeration as "the Indian view of knowledge sources"** (per SAN-I-001). This erases Buddhist 2-fold, Cārvāka 1-fold, Sāṃkhya 3-fold, Mīmāṃsā 6-fold positions.

7. **SAN-AP-007 — Treating Navya-Nyāya as a sealed Hindu system without acknowledging the Buddhist *pramāṇavāda* prehistory** (Diṅnāga, Dharmakīrti, *trairūpya*, *anyāpoha*). Per SAN-D-001.

8. **SAN-AP-008 — Defaulting *śabda-pramāṇa* to "Vedic testimony"** (per SAN-A-003 and SAN-C-003). This is Mīmāṃsā, not Nyāya.

9. **SAN-AP-009 — Calling Navya-Nyāya "Indian logic" without scare-quotes or qualifier** (per SAN-D-003). Matilal's contested term.

10. **SAN-AP-010 — Citing only Hindu-side Indologists** (Phillips, Mohanty, Matilal, Ganeri, Bhattacharyya, Potter) **without any Buddhist-logic scholar** (Hayes, Tillemans, Siderits, Kellner) in the bibliography. Per SAN-D-002.

11. **SAN-AP-011 — Failing to consult a living Indian Sanskrit institution** (Mithila / Sampūrṇānanda / Madras / Tirupati / Ramtek) before launch, when the cost of consultation is three letters and four weeks. Per SAN-E-001.

12. **SAN-AP-012 — Apache-2.0-licensing framing that elides the licensing-vs-tradition asymmetry.** The license covers the harness, not the *vidyā-paramparā*. Per SAN-E-003.

13. **SAN-AP-013 — Hindutva-coded marketing phrases** ("Sanatan logic," "Vedic AI," "the wisdom of the rishis," "dharmic reasoning") in any public artifact. Per SAN-H-001. Different failure mode from orientalist phrases (SAN-AP-014); both are forbidden.

14. **SAN-AP-014 — Orientalist phrases** ("AI rediscovers ancient wisdom," "machine intelligence and Sanskrit revival," "the East-meets-West epistemology"). Per SAN-H-002.

15. **SAN-AP-015 — Naming the project as if the benchmark itself is a *pramāṇa*** (which is conceptually odd: the benchmark *tests* whether a model exhibits *pramāṇa*-like reasoning; the benchmark is not itself a *pramāṇa*). Per SAN-G-001.

---

## 4. Open Sanskrit-domain questions (founder decisions needed)

These require either a specific founder decision or a specific named-expert consultation. Each names a recommended consultant where possible.

1. **SAN-OQ-001 — Whose Navya-Nyāya is canonical: Mithila or Bengal?** The transmission lineage of Navya-Nyāya has two main branches after Gaṅgeśa: (a) the **Mithila line** (Vardhamāna Upādhyāya, 14th c.; the Mithila Institute's parampara) which preserves Gaṅgeśa's expository style; (b) the **Bengal line** (Raghunātha Śiromaṇi, 16th c.; Mathurānātha, 16th c.; Jagadīśa Tarkālaṅkāra, 17th c.; Gadādhara Bhaṭṭācārya, 17th c.) which produced the more technically refined and influential later Navya-Nyāya. The two have substantively different commitments on (e.g.) the ontology of *abhāva*, the analysis of *avacchedaka*, the treatment of *paryāpti*. The schema does not currently say which it follows.
   - **Defer to:** Either the Mithila Institute (consult about the Mithila line) or Karuna Bhattacharya / the Asiatic Society Calcutta (consult about the Bengal line). The pedagogical text the schema actually relies on (*Tarka-Saṃgraha*) is a 17th-c. Maharashtrian / Tamil pedagogical synthesis that is school-neutral on the Mithila/Bengal split — meaning the schema's de facto answer is "we are using the *Tarka-Saṃgraha* digest, which postdates and partly transcends the Mithila/Bengal split." This is defensible but should be declared.

2. **SAN-OQ-002 — Should the schema acknowledge a Buddhist counterpart?** A natural future direction is a Buddhist-logic benchmark using *trairūpya*, *svalakṣaṇa*-perception, and *anyāpoha*-style semantics. The 15-phase schema is sectarian Hindu by construction. Should the project: (a) declare itself permanently Naiyāyika and invite a parallel Buddhist project, or (b) plan a multi-darshana expansion where the framework abstracts over schools?
   - **Defer to:** A Buddhist-logic specialist consultation. Recommended: Birgit Kellner (Austrian Academy of Sciences, Vienna), Tom Tillemans (emeritus Lausanne), Daniel Stuart (USC), or any Tibetan Buddhist Studies scholar with *pramāṇavāda* background.

3. **SAN-OQ-003 — Translation-stack disclosure: which English translations does the schema read through?** The schema currently leans on Phillips's framing (analytic-philosophy-inflected). Should the documentation explicitly use a Mohanty (phenomenological) framing for *paramarśa* — since Mohanty's "reflective synthesis" is closer to what the phase is doing — even if Phillips is cited elsewhere?
   - **Defer to:** Founder's strategic call after a Sanskrit-fluent reviewer signoff.

4. **SAN-OQ-004 — Mithila Institute or Sampūrṇānanda for primary outreach?** Both are appropriate. Mithila is the Navya-Nyāya parampara historically; Sampūrṇānanda has a larger active Nyāya department and more English-comfortable faculty. Tactical consideration: Mithila response is more authoritative; Sampūrṇānanda response is faster and more likely to come.
   - **Defer to:** Founder's outreach choice; recommend trying Mithila first with a 4-week window, then Sampūrṇānanda + Madras Sanskrit College in parallel if Mithila does not respond.

5. **SAN-OQ-005 — Should the project commission a critical translation of any specific Sanskrit passage?** For load-bearing claims (e.g., what Gaṅgeśa actually says about *vyāpti-pañcaka* in *Tattvacintāmaṇi* Anumāna-khaṇḍa), commissioning a 1–2 page targeted translation from a *paṇḍita* would be high-value. Cost: probably $200–500 per passage. This produces a defensible primary-source citation that survives even hostile auditing.
   - **Defer to:** Mithila Institute or the Asiatic Society Calcutta for translator referral.

6. **SAN-OQ-006 — Position toward Indian ministry-of-culture / IGNCA-type institutions.** The Indira Gandhi National Centre for the Arts (IGNCA) and the Central Sanskrit University (Rāṣṭriya Saṃskṛta Saṃsthāna) are public-funded; engagement with them carries political coloration in 2026 (the post-2014 Hindutva-influenced cultural policy environment). An academic engagement with Mithila Institute or Sampūrṇānanda is school-tradition-aligned without the same political coloration. Engaging IGNCA is not forbidden but should be a deliberate choice with awareness of the framing.
   - **Defer to:** Founder's strategic call. Recommend: prefer Mithila / Sampūrṇānanda / Madras / Asiatic Society for primary outreach; engage IGNCA only if a specific scholar there has expressed interest, and disclose the engagement.

7. **SAN-OQ-007 — Sanskrit-medium publication possibility.** A long-form possibility: publishing a description of the project in *Saṃskṛta-Pratibhā* or a similar Sanskrit-medium journal. This is high-effort (requires either a Sanskrit-fluent co-author or commissioned translation) but extremely high-credibility within the Indian Sanskrit-medium scholarly world. Probably v2.0 or later.
   - **Defer to:** Strategic decision after v1.0 reception.

8. **SAN-OQ-008 — Long-term relationship with the living tradition.** REQUIREMENTS_SCHOLAR OQ-008 raises this generically; the Sanskrit-domain version is more concrete: should the project (a) compensate Indian institutions through Apache-2.0-license-disclaiming royalties from any commercial offshoot (e.g., a ModelBrew product that uses pramāṇa-eval results), (b) commit a percentage of any commercial revenue to a Sanskrit institution / a fund for *paṇḍita*-line preservation, (c) include Sanskrit-institution scholars on any future paper as co-authors? Current de facto answer (none of the above) is the extractive default.
   - **Defer to:** A specific founder decision before launch. Recommend at minimum a public statement of intent on this question, even if the operationalization is post-launch.

---

## 5. End-of-document acceptance test

The hostile question at the end is: **"If the founder of the Mithila Institute, a Buddhist-logic specialist from Vienna, and a decolonial-academia critic at JNU each read this whole project for an afternoon, what is the worst thing each will write in their reviews?"**

For the current state (`pramana_test.py` v0.1, `SCHEMA_15_PHASE_DRAFT.md` v0.1, current NOTICE), the reviews are:

- **Mithila paṇḍita:** *"This is an outsider-extraction with sandhi-incorrect Sanskrit, mixes Old Nyāya and Navya-Nyāya in the *hetvābhāsa* list without disclosure, presents the Naiyāyika 4-fold *pramāṇa* as if it were the Indian view, and has not consulted any institution in the parampara. The 'Vedic AI' connotation is not even denied in the NOTICE. The harness may be technically competent; the philosophy is amateur."*

- **Buddhist-logic specialist (Vienna):** *"Six centuries of Buddhist-Naiyāyika debate are absent. The schema treats Gaṅgeśa as if he developed *vyāpti-pañcaka* in a Hindu vacuum rather than in response to Diṅnāga's *trairūpya* and Dharmakīrti's refinements. The bibliography names no Buddhist-logic scholar. This is a Hindu-monolingual reading of a fundamentally polemical tradition."*

- **Decolonial critic (JNU):** *"Apache-2.0 licensing of a tradition the project did not consent-engage with; ASCII-Sanskrit competence failure; no living-institution outreach; Bay-Area-tech extraction with better-than-average disclaimers but the same underlying property frame. The NOTICE's 'no claim of ownership over the tradition' line is necessary but not sufficient — the project still extracts conceptual labor without compensation or acknowledgement of the *paramparā* it depends on."*

After meeting these requirements, the worst defensible reviews should be:

- **Mithila paṇḍita:** *"The IAST is correct, the sandhi is correct, the school-conflation issues are disclosed, the institution was consulted (and we declined further engagement / and we gave specific feedback that was incorporated). I disagree with treating *upādhi* as a benchmark phase — Gaṅgeśa's treatment is more subtle than the schema suggests — but the disagreement is a principled one and the schema acknowledges it. The hybrid name is declared."*

- **Buddhist-logic specialist:** *"Diṅnāga and Dharmakīrti are acknowledged; the *trairūpya* is named; at least one Buddhist-logic scholar is in the bibliography. The schema is explicitly Naiyāyika and explicitly invites a Buddhist parallel. I would have preferred a multi-school benchmark from the start, but a sectarian benchmark with disclosed sectarian commitments is defensible."*

- **Decolonial critic:** *"The living tradition is consulted (or attempted-consulted with disclosure); the licensing-vs-tradition asymmetry is named; the positionality is disclosed; the Hindutva and orientalist phrasings are absent; future engagement intent is articulated. I still think the underlying extraction-frame is intact, but the project has done the work to neutralize the easy critiques. I will engage on the harder ones."*

That is the bar. Anything worse than these three paragraphs is a Sanskrit-domain launch-blocker.

---

## Sources

- Wisdomlib lexical entry for *pañcāvayava* (sandhi-correct form): `wisdomlib.org/definition/pancavayava`
- Wisdomlib lexical entry for *hetvābhāsa* (five-fold Navya list): `wisdomlib.org/definition/hetvabhasa`
- Apte, V. S. *The Practical Sanskrit-English Dictionary*, online edition: `dsal.uchicago.edu/dictionaries/apte/`
- Monier-Williams, *Sanskrit-English Dictionary*, Cologne Digital Sanskrit Dictionaries: `sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/web/webtc/indexcaller.php`
- Mithila Institute of Post-Graduate Studies and Research in Sanskrit Learning, Darbhanga (founded 1951): see the Internet Archive collection for *History of Navya-Nyāya in Mithila* (Dineshchandra Bhattacharya, 1958, reprinted Mithila Institute 1987)
- Sanskrit and Vedic learning in Mithila — Wikipedia overview confirming the Mithila Institute's role
- Tarka-Saṃgraha — Wikipedia overview; Annambhaṭṭa, 17th c.; standard pedagogical Navya-Nyāya text
- Athalye, Y. V., and M. R. Bodas, ed. & trans. *Tarka-Saṃgraha of Annambhaṭṭa*, 2nd ed., Bhandarkar Oriental Research Institute, Poona, 1930. (Available archive.org.)
- Phillips, Stephen, and N. S. Ramanuja Tatacharya, trans. *Epistemology of Perception: Gaṅgeśa's Tattvacintāmaṇi, Jewel of Reflection on the Truth (about Epistemology) — The Perception Chapter*. American Institute of Buddhist Studies / Columbia, 2004.
- Phillips, Stephen. *Epistemology in Classical India: The Knowledge Sources of the Nyāya School*. Routledge, 2012.
- Mohanty, J. N. *Gaṅgeśa's Theory of Truth*. 1989, MLBD reprint.
- Matilal, B. K. *The Character of Logic in India*. SUNY Press, 1998.
- Ganeri, Jonardon. *Philosophy in Classical India: The Proper Work of Reason*. Routledge, 2001.
- Ganeri, Jonardon. *The Lost Age of Reason: Philosophy in Early Modern India 1450–1700*. Oxford, 2014.
- Hayes, Richard P. *Diṅnāga on the Interpretation of Signs*. Kluwer, 1988.
- Tillemans, Tom J. F. *Scripture, Logic, Language: Essays on Dharmakīrti and his Tibetan Successors*. Wisdom, 1999.
- Siderits, Mark. *Indian Philosophy of Language*. Kluwer, 1991.
- Stanford Encyclopedia of Philosophy entries: "Gaṅgeśa," "Epistemology in Classical Indian Philosophy," "Nyāya-Vaiśeṣika."

---

**End of Sanskrit-domain requirements document.**
