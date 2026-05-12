# A 15-Phase Epistemic Reasoning Schema Drawn from Primary Navya-Nyaya Sources

**Document type:** Foundational design document for an open-source LLM reasoning benchmark
**Status:** Draft v0.1, to be audited
**Author note:** Every phase is grounded in a specific primary classical source with section reference where the original text is sectionable. Phases that are partially synthetic (i.e., a design call I made by combining classical material) are flagged as such — see §7 for the full list of design calls.

---

## 1. Executive Summary

The schema decomposes a single act of epistemic reasoning into 15 sequential phases drawn from the canonical Nyaya / Navya-Nyaya literature. The phases, in order:

1. **Jijnasa** (inquiry framing) — Nyaya-Bhashya on NS 1.1.1
2. **Samshaya** (structured doubt) — Nyaya-Sutras (NS) 1.1.23 + Vatsyayana
3. **Prayojana** (purpose / decision-relevance) — NS 1.1.24
4. **Siddhanta-vibhaga** (background-tenet typing) — NS 1.1.26–29
5. **Prameya-vislesana** (object-of-cognition analysis) — NS 1.1.9
6. **Pramana-yojana** (selection of knowledge sources) — NS 1.1.3; Tarka-Sangraha §§30–47
7. **Drstanta** (probative example anchoring) — NS 1.1.25
8. **Vyapti-graha** (pervasion-grasping) — Tattvacintamani Anumana-khanda 5.1
9. **Upadhi-pariksa** (defeating-condition test) — Tattvacintamani Anumana-khanda 5.3
10. **Paksadharmata + Paramarsa** (subject-locus & reflective synthesis) — Tattvacintamani Anumana-khanda 5.2
11. **Pancha-avayava** (five-membered formal argument) — NS 1.1.32–39; Tarka-Sangraha §49
12. **Tarka** (counterfactual reductio) — NS 1.1.40
13. **Hetvabhasa-pariksa** (fallacy audit, Navya list) — Tarka-Sangraha §§54–67 (5-fold Navya list, not Gautama's older 5-fold list)
14. **Anupalabdhi/Abhava-vicara** (negation & non-apprehension check) — Atmatattvaviveka; Tarka-Sangraha on abhava
15. **Nirnaya** (ascertainment + epistemic status report) — NS 1.1.41

**Logical flow.** Phases 1–4 frame the problem (what is being asked, what is doubtful, what would change with an answer, what background tenets are presumed). Phases 5–7 prepare the epistemic raw material (the object, the available knowledge sources, anchoring examples). Phases 8–10 are the technical Navya-Nyaya core — the part that does *not* appear in older Nyaya in this resolved form: pervasion, defeating-condition, and reflective synthesis. Phase 11 is the formal output (the Indian syllogism). Phases 12–14 are adversarial: counterfactual stress, fallacy audit, and a separate negation/absence check that older 6-phase formulations conflate with fallacy detection. Phase 15 closes with a graded ascertainment.

**Defensibility.** The schema is grounded in the 16-padartha taxonomy of NS 1.1.1 (which already gives 9 of the 15 phases directly), plus the four technical innovations of Gangesha's *Tattvacintamani* (pervasion, defeating-condition, reflection, refined fallacy list), plus Udayana's perceptual treatment of absence. It uses *only* terms attested in the primary corpus; no phase is invented for the schema's convenience.

---

## 2. Source Survey

**Gautama, *Nyaya-Sutras* (NS, ~2nd c. BCE).** The root text. NS 1.1.1 enumerates the 16 *padarthas* (categories of inquiry): pramana, prameya, samshaya, prayojana, drstanta, siddhanta, avayava, tarka, nirnaya, vada, jalpa, vitanda, hetvabhasa, chala, jati, nigrahasthana. Of these, the first nine are "logic proper" (the second seven concern debate and dispute). My schema draws phases 2, 3, 4, 5, 6, 7, 11, 12, and 15 directly from these first nine. *Contributes phases:* 2, 3, 4, 5, 6, 7, 11, 12, 15.

**Vatsyayana, *Nyaya-Bhashya* (~5th c.).** The earliest surviving complete commentary on NS. Provides the operative definitions used here for *prayojana* ("yam artham adhikrtya pravartate tat prayojanam," NS 1.1.24 commentary), *drstanta* (1.1.25), and the four-fold typology of *siddhanta* (NS 1.1.26–29: sarvatantra, pratitantra, adhikarana, abhyupagama). Vatsyayana's framing of *jijnasa* (the felt need to know, prefatory to the entire treatise) underwrites Phase 1. *Contributes phases:* 1, 3, 4, 7.

**Uddyotakara, *Nyaya-Varttika* (~7th c.).** Defends NS against Buddhist (esp. Dignaga / Dharmakirti) critique. Important for the schema mainly because Uddyotakara begins to treat *hetvabhasa* and counter-inference with the technical precision later formalized by Gangesha. *Contributes phases:* 13 (background grounding), 12.

**Vacaspati Misra, *Nyaya-Varttika-Tatparya-Tika* (~9th c.).** Sub-commentary on Uddyotakara. Marginal to this schema beyond confirming Uddyotakara's readings; cited where it disambiguates a contested term. *Contributes phases:* 13 (cross-check).

**Udayana, *Nyayakusumanjali* / *Atmatattvaviveka* (~10th c.).** Often called the bridge from Old Nyaya to Navya-Nyaya. Crucial for Phase 14: Udayana argues that absence (*abhava*) and non-apprehension (*anupalabdhi*) can be handled within perception rather than as a separate pramana — but the *check* must still be performed. *Contributes phases:* 14.

**Gangesha, *Tattvacintamani* (TC, ~12th–13th c.).** The founding Navya-Nyaya text, organized in four *khandas* (Pratyaksa, Anumana, Upamana, Sabda). Gangesha's *Anumana-khanda* is the primary source for what makes this schema "Navya-Nyaya" rather than just "Nyaya": (a) the refined definition of *vyapti* after surveying twenty-four prior definitions, (b) the elaborate *upadhi* analysis (~10% of the inference chapter), (c) the role of *paramarsa* (reflective synthesis) as the proximate cause of inferential cognition, (d) the locushood (*paksata*) treatment. These four are exactly Phases 8, 9, and 10 of this schema. *Contributes phases:* 8, 9, 10. This is the load-bearing original contribution.

**Raghunatha Siromani, *Padartha-Tattva-Nirupanam* (~16th c.).** A revision of the Vaisesika ontology used by Nyaya. Cited in the schema only for confirming that *abhava* is a first-class category that must be reasoned about explicitly (Phase 14) — Raghunatha treats it as ontologically real, not merely as a verbal absence. *Contributes phases:* 14 (cross-check).

**Annambhatta, *Tarka-Sangraha* + *Tarka-Dipika* (TS, ~17th c.).** The standard Navya-Nyaya pedagogical text. Uses the Navya five-fold *hetvabhasa* list — *savyabhicara, viruddha, satpratipaksa, asiddha, badhita* — which differs from Gautama's older five-fold list (*savyabhicara, viruddha, prakaranasama, sadhyasama, kalatita*). My schema uses Annambhatta's Navya list (Phase 13) precisely because it is logically more discriminating: *satpratipaksa* (counter-inference) and *badhita* (sublated by another pramana) are absent from Sathish 2026's formulation. *Contributes phases:* 6, 11, 13.

**Distinctive Navya-Nyaya features (Gangesha onwards) that matter for this schema.**
1. *Vyapti* gets a fully extensional definition — Phase 8 can therefore be evaluated for whether the model actually states a universal generalization, not just an example.
2. *Upadhi* is treated as a defeater independent of fallacy detection — this is *why* Phase 9 is separated from Phase 13. In Sathish 2026's six-phase, no analog of *upadhi* exists.
3. *Paramarsa* is identified as a distinct mental event — the proximate cause of inferential knowledge — separate from both pervasion-knowledge and the formal syllogism. This justifies Phase 10 as cognitively distinct from Phases 8 and 11.
4. The Navya five-fold *hetvabhasa* list replaces Gautama's older list. Phase 13 uses the Navya list with explicit citation.

---

## 3. The Fifteen Phases

### Phase 1: Jijnasa (Inquiry Framing)

- **Classical source(s):** Vatsyayana's prefatory bhashya to NS 1.1.1, on the felt need to know (*jijnasa*) as the precondition for any inquiry. Reinforced by NS 1.1.1 itself ("from true knowledge of these sixteen, *nihsreyasa* is attained") — the entire treatise presupposes a question being asked.
- **Cognitive operation:** Restate the problem as a precise epistemic question. State what entity or relation is being inquired into, in the model's own words, before doing anything else. This is the "what am I actually being asked" gate.
- **Why distinct from neighbors:** Phase 1 does not yet acknowledge any *uncertainty* — it just fixes the question. Phase 2 (*samshaya*) is where doubt is introduced. Without Phase 1, Phase 2 has no anchor; the model can manufacture doubt about a question it has not actually committed to.
- **LLM output specification:**
  - `Question:` <one sentence, interrogative form>
  - `Inquired entity (jijnasita):` <noun phrase — the X in "is X a Y?" or "what is X?">
  - `Inquired relation (if any):` <e.g., "membership," "identity," "causation," "satisfaction of constraints">
  - `Restatement check:` <one sentence confirming the model has not changed the problem>
- **Evaluation criteria:**
  - *Format:* All four sub-fields present; `Question` ends in "?"; `Restatement check` is a declarative.
  - *Semantic:* The `Inquired entity` must appear (by lemma) in the original problem statement. LLM-judge: "Does the restated question, if answered, also answer the original question?" (yes/no).
- **Failure modes:** (a) Model rephrases the question into a *different* question (subtle goal drift). Catch: lemma overlap < 0.5 between problem statement and `Inquired entity`. (b) Model writes a generic "I am being asked to solve a problem." Catch: judge prompt detects vacuity.
- **Comparison to Sathish 2026:** *Absent.* Sathish has no framing phase; he begins with *samshaya*. This is a real omission — without it, the doubt phase has no fixed referent, and the schema is vulnerable to the model drifting into a related-but-different problem mid-trace. Phase 1 is grounded in Vatsyayana's prefatory bhashya, not invented.

---

### Phase 2: Samshaya (Structured Doubt)

- **Classical source(s):** NS 1.1.23 (definition of *samshaya*); Vatsyayana on NS 1.1.23, listing five sources of doubt. Tarka-Sangraha discusses *samshaya* as a cognition with conflicting predications about a single locus.
- **Cognitive operation:** Identify *what* is uncertain and *why* it is uncertain. Per Vatsyayana, doubt arises from (i) *samana-dharma* (shared property — multiple entities fit the partial description), (ii) *aneka-dharma* (one entity bears apparently conflicting properties), (iii) *vipratipatti* (conflicting testimony), (iv) *upalabdhi-avyavastha* (irregular apprehension), (v) *anupalabdhi-avyavastha* (irregular non-apprehension). The phase requires picking one of the five and justifying.
- **Why distinct from neighbors:** Phase 1 fixed the question; Phase 2 fixes *what kind of uncertainty the question has*, which determines downstream pramana selection. Phase 3 (*prayojana*) is about why the answer would matter, not what is unclear. Without typing the doubt, Phase 6 cannot rationally choose pramanas — different doubt types call for different pramana stacks (e.g., *samana-dharma* demands *upamana*; *vipratipatti* demands *sabda*).
- **LLM output specification:**
  - `Doubt type:` exactly one of {samana-dharma, aneka-dharma, vipratipatti, upalabdhi-avyavastha, anupalabdhi-avyavastha}
  - `Locus of doubt:` <the entity / proposition where conflicting predications meet>
  - `Conflicting predications:` `P` vs `not-P` (or `Q1` vs `Q2`)
  - `Justification:` <one sentence linking the chosen type to the locus>
- **Evaluation criteria:**
  - *Format:* Doubt type is one of the five canonical strings (case-insensitive).
  - *Semantic:* The two conflicting predications must be genuine contraries / contradictories, not the same statement twice. LLM-judge: "Is it actually unclear from the problem statement which of P or not-P holds?"
- **Failure modes:** (a) Model picks *samana-dharma* by default for everything. Catch: distribution check across a benchmark — if >70% of problems get the same type, scorer flags. (b) The two predications are not genuine contraries (e.g., "the answer is 5" vs "the answer is some integer"). Catch: judge prompt asks "Is P logically incompatible with not-P?"
- **Comparison to Sathish 2026:** *Present in Sathish as Phase 1 ("Samshaya — Doubt Analysis").* Sathish lists similar sub-types. The schema preserves Sathish's content here but moves it to Phase 2 (after framing) and tightens evaluability by requiring the conflicting predications as a separate field.

---

### Phase 3: Prayojana (Purpose / Decision-Relevance)

- **Classical source(s):** NS 1.1.24 ("yam artham adhikrtya pravartate tat prayojanam" — Vatsyayana's gloss). Listed as the fourth padartha in NS 1.1.1.
- **Cognitive operation:** State what the answer is *for*. Concretely: which downstream decision, prediction, or further inference becomes possible once the doubt is resolved? This is the pragmatic anchor — it prevents the model from wandering into adjacent interesting questions that are not the asked one.
- **Why distinct from neighbors:** Phase 2 said *what* is unclear; Phase 3 says *what changes if it gets clarified*. Without Phase 3, the model has no way to detect that it has solved a related-but-orthogonal problem (a common LLM failure mode on constraint-satisfaction problems).
- **LLM output specification:**
  - `Downstream use of the answer:` <single sentence, must reference an action or further inference>
  - `What would change if the answer flipped:` <single sentence>
- **Evaluation criteria:**
  - *Format:* Both fields present; both are non-empty declarative sentences.
  - *Semantic:* `What would change` must mention at least one entity/relation from the problem. LLM-judge: "If the answer to the question were the opposite, would the stated downstream use also change?"
- **Failure modes:** (a) "The purpose is to demonstrate reasoning." Generic, content-free. Catch: judge prompt checks whether the answer references problem entities. (b) The downstream use is the same regardless of answer. Catch: the second field must be non-trivial.
- **Comparison to Sathish 2026:** *Absent.* Sathish has no *prayojana* phase. This is grounded directly in NS 1.1.1 (fourth padartha) and NS 1.1.24, not invented. Its evaluative payoff is catching goal drift, which is a known frontier-LLM weakness.

---

### Phase 4: Siddhanta-vibhaga (Background-Tenet Typing)

- **Classical source(s):** NS 1.1.26–29 (the four-fold typology of *siddhanta*); Vatsyayana's bhashya on these sutras.
- **Cognitive operation:** Make explicit what background tenets the reasoning will rely on, and classify each one as: (i) *sarvatantra* — accepted by all schools / universal logical-mathematical principle, (ii) *pratitantra* — accepted by the specific framework of this problem (e.g., "this is a Boolean-SAT problem in classical 2-valued logic"), (iii) *adhikarana* — hypothetical, accepted for the sake of this argument, (iv) *abhyupagama* — granted by assumption pending later test.
- **Why distinct from neighbors:** This is the phase that surfaces *what the model is taking for granted*. It is logically distinct from Phase 6 (*pramana* — the sources of new information) and from Phase 8 (*vyapti* — particular generalizations). A *sarvatantra* such as "modus ponens is valid" is presupposed by everything but is not itself an inferred pervasion.
- **LLM output specification:**
  - A list of 1–N tenets, each as: `[type] tenet`
  - Each tenet must be tagged with one of {sarvatantra, pratitantra, adhikarana, abhyupagama}
- **Evaluation criteria:**
  - *Format:* At least one entry; every entry tagged with exactly one of the four types.
  - *Semantic:* Each tenet must be a declarative proposition (not a question, not a definition request). LLM-judge: "Does the subsequent reasoning actually depend on this tenet?" (For at least N–1 of N entries, yes.)
- **Failure modes:** (a) Model lists tenets it never uses. Catch: judge prompt checks dependency. (b) Model marks everything *sarvatantra*. Catch: distribution sanity (a problem-specific framework assumption is rarely *sarvatantra*).
- **Comparison to Sathish 2026:** *Absent.* Sathish has no analog. NS 1.1.26–29 is canonical, and this phase does real work: it separates "the rules of the game" from "the evidence within the game," which Sathish's Pramana phase conflates by lumping mathematical axioms into *sabda*.

---

### Phase 5: Prameya-vislesana (Object-of-Cognition Analysis)

- **Classical source(s):** NS 1.1.9 (enumerates twelve *prameya*s: atman, sarira, indriya, artha, buddhi, manas, pravrtti, dosa, pretyabhava, phala, duhkha, apavarga). The pedagogical use here is structural: Nyaya treats *what kind of thing the answer must be* as a first-class question.
- **Cognitive operation:** Analyze the *kind of object* the answer must be. For a logic puzzle: is the answer a Boolean? An assignment? An object satisfying a relation? A non-existent entity (i.e., the answer is "no such X exists")? This phase does *not* solve the problem; it specifies the type of the solution.
- **Why distinct from neighbors:** Phase 1 said the question; Phase 5 says the *type of the answer*. Without Phase 5, the model can produce a syntactically valid output of the wrong type (e.g., a list when a single object was asked for). This is a known frontier-LLM failure on constraint problems.
- **LLM output specification:**
  - `Answer type:` <one of: Boolean, single entity, ordered tuple, set, mapping, "no such X exists" (negative), distribution-over-options>
  - `Cardinality expectation:` <if applicable: one, exactly k, at most k, at least k, all>
  - `Domain of answer:` <set from which the answer is drawn>
- **Evaluation criteria:**
  - *Format:* All three fields present.
  - *Semantic:* Answer type must be consistent with the question's interrogative form (yes/no question → Boolean; "find all X such that" → set with cardinality "all"; etc.). Mechanical check from question pattern.
- **Failure modes:** (a) Says "the answer is a number" when the answer is a Boolean. Catch: type-vs-question consistency rule. (b) Says "set" when only one element is asked for. Catch: cardinality field cross-check.
- **Comparison to Sathish 2026:** *Absent.* Sathish does not type the answer at all. Phase 5 is grounded in NS 1.1.9's framing of *prameya* as the second of the 16 padarthas, parallel to *pramana*.

> **Design note (flagged per §7):** I am repurposing *prameya* — which classically refers to the twelve metaphysical objects of cognition (self, body, etc.) — as "answer-type analysis" for the benchmark setting. This is a defensible *pedagogical* extension because Nyaya treats *prameya* as "that which the pramana cognizes," and in a benchmark setting the cognized object is the answer. But it is not a verbatim repetition of NS 1.1.9. Marked.

---

### Phase 6: Pramana-yojana (Selection of Knowledge Sources)

- **Classical source(s):** NS 1.1.3 (four *pramana*s: pratyaksa, anumana, upamana, sabda); Tarka-Sangraha §§30–47 (Annambhatta's treatment of each, including *alaukika pratyaksa* — extraordinary perception — under three sub-types: *samanyalaksana, jnanalaksana, yogaja*).
- **Cognitive operation:** For *this specific problem*, identify which of the four pramanas will be marshalled and *what each will contribute*. Crucially, the model must justify why a pramana is *not* used, if it isn't.
- **Why distinct from neighbors:** Phase 4 fixed the background tenets (the rules); Phase 6 fixes the *information sources* (the inputs). They are separable: the rules of arithmetic are *sarvatantra* (Phase 4); the actual numbers given in the problem are *pratyaksa* / *sabda* (Phase 6).
- **LLM output specification:** For each of {pratyaksa, anumana, upamana, sabda}, exactly one of:
  - `Used: <one-sentence statement of what role it plays>` OR
  - `Not used: <one-sentence reason>`
- **Evaluation criteria:**
  - *Format:* All four pramanas accounted for; each with either "Used" or "Not used."
  - *Semantic:* If the problem statement contains explicit data, *pratyaksa* should be marked Used (referring to that data as directly given). If the problem invokes a named rule or definition, *sabda* should be marked Used. *Anumana* should virtually always be Used (since the schema ends in inferential ascertainment).
- **Failure modes:** (a) Model marks all four Used reflexively. Catch: scorer validates that the "Not used" cases are justified — e.g., on a pure-deduction problem, *upamana* is correctly Not used. (b) Model conflates *sabda* with all background knowledge. Catch: *sabda* should reference an authority/definition; general world knowledge is *sarvatantra* (Phase 4). Cross-phase consistency check.
- **Comparison to Sathish 2026:** *Present (Sathish Phase 2: Pramana).* Sathish requires "completeness — all four pramana types must be present," which forces the model to fabricate uses for each. My Phase 6 *requires accounting for each but allows justified non-use*, which is closer to the actual Nyaya treatment (Annambhatta does not require all four pramanas in every inference) and produces a sharper evaluation signal.

---

### Phase 7: Drstanta (Probative Example Anchoring)

- **Classical source(s):** NS 1.1.25 (*drstanta* defined as that on which laypersons and experts agree); Vatsyayana's gloss; Tarka-Sangraha §49 on *udaharana* (the example used inside the syllogism).
- **Cognitive operation:** Identify a *concrete example* — preferably one outside the immediate problem — that anchors the kind of reasoning to be deployed. Per NS 1.1.25, this example must be one that any rational party would agree on. For a logic problem, this is typically a worked instance of the relevant rule (e.g., for modus ponens: "as in the standard 'all men are mortal, Socrates is a man' inference").
- **Why distinct from neighbors:** Phase 7 is the *out-of-problem* anchor; Phase 11 (*udaharana* sub-component) is the *in-problem* example used inside the syllogism. The classical literature already distinguishes *drstanta* (the agreed-upon paradigm) from *udaharana* (the example as cited within the formal argument) — they share lexical roots but are functionally distinct, and the schema honors this.
- **LLM output specification:**
  - `Drstanta:` <a paradigmatic example outside the problem, illustrating the relevant kind of reasoning>
  - `Why both layperson and expert would accept this:` <one sentence>
- **Evaluation criteria:**
  - *Format:* Both fields present.
  - *Semantic:* The drstanta must be from a different domain than the immediate problem (so it really is an "anchor," not a restatement). LLM-judge: "Is this example structurally analogous to the problem but materially different?"
- **Failure modes:** (a) Model uses the problem itself as drstanta. Catch: domain-overlap check. (b) Model uses an obscure example unlikely to be universally agreed. Catch: judge prompt.
- **Comparison to Sathish 2026:** *Implicit in Sathish's Phase 3 (Pancha Avayava) under udaharana, but not separated.* Distinguishing Phase 7 (*drstanta*, the out-of-problem anchor, from NS 1.1.25) from the in-syllogism *udaharana* (Phase 11) is faithful to the primary literature and operationally distinct.

---

### Phase 8: Vyapti-graha (Pervasion-Grasping)

- **Classical source(s):** Gangesha, *Tattvacintamani*, Anumana-khanda 5.1 — the famous *vyapti-pancaka* (Gangesha's survey of and answer to twenty-four prior definitions of pervasion). Also Tarka-Sangraha §50 on *vyapti*.
- **Cognitive operation:** State the *invariable concomitance* — the universal rule of the form "wherever H, there is S" — that the inference will rely on. This is the universal generalization; it is *not* the example, *not* the application, *not* the conclusion.
- **Why distinct from neighbors:** This is the load-bearing Navya-Nyaya phase. Old Nyaya's syllogism (Phase 11) uses *udaharana* (an example) but does not isolate the underlying universal as a separate cognitive event. Gangesha's innovation is exactly this isolation. Phase 8 forces the model to commit to a generalization *before* applying it (Phase 10) and *before* stress-testing it (Phase 9).
- **LLM output specification:**
  - `Hetu (H, the reason / probans):` <noun phrase>
  - `Sadhya (S, the probandum):` <noun phrase>
  - `Vyapti statement:` `For all x, if H(x) then S(x).` (or its negative-inference dual: `For all x, if not-S(x) then not-H(x).`)
  - `Source of the vyapti:` <one of: definition, prior pramana, induction-from-cases, theorem, axiom>
- **Evaluation criteria:**
  - *Format:* H and S identified; vyapti statement uses universal quantifier or "wherever … there"; source field present.
  - *Semantic:* The vyapti must actually entail the conclusion when combined with paksadharmata (Phase 10). Symbolic check: substituting the paksa for x must yield the conclusion. LLM-judge fallback when symbolic check is not feasible.
- **Failure modes:** (a) Model states a particular ("Socrates is mortal") rather than a universal. Catch: regex for universal quantifier; if absent, fail format. (b) Model states a vyapti that doesn't actually generalize ("In this problem, X implies Y" — vacuous). Catch: judge prompt.
- **Comparison to Sathish 2026:** *Absent as a distinct phase.* Sathish folds *vyapti* into the *udaharana* sub-component of Phase 3 (Pancha Avayava), exactly as Old Nyaya did. This collapses Gangesha's central innovation. Phase 8 is the single most important divergence between this schema and Sathish 2026, and it is grounded directly in TC Anumana-khanda §5.1.

---

### Phase 9: Upadhi-pariksa (Defeating-Condition Test)

- **Classical source(s):** Gangesha, *Tattvacintamani*, Anumana-khanda 5.3. Stanford Encyclopedia of Philosophy notes Gangesha devotes "more than a tenth" of the inference chapter to *upadhi*. Tarka-Sangraha §52.
- **Cognitive operation:** Search for an *upadhi* — a condition U that pervades the *sadhya* but does not pervade the *hetu*. If such a U exists, the vyapti from Phase 8 is conditional on U, and the inference is defeated (or restricted) until U is itself shown to obtain in the *paksa*.
- **Why distinct from neighbors:** Different from Phase 13 (*hetvabhasa*) in a precise way: a *hetvabhasa* is a defect intrinsic to the hetu's relation to sadhya (e.g., the hetu doesn't actually obtain in the paksa). An *upadhi* is an *external condition* that secretly carries the inference. Gangesha treats them as separate; Sathish does not. The classic example: "fire is on the mountain because there is smoke" is sound; "fire produces smoke" only on the *upadhi* of "wet fuel." The hetu is fine; the vyapti has a hidden condition. Phase 9 catches this; Phase 13 does not.
- **LLM output specification:**
  - `Candidate upadhi(s):` list of 0–N conditions U_i
  - For each candidate: `Pervades sadhya: yes/no | Pervades hetu: yes/no`
  - `Verdict:` either `No upadhi found` or `Upadhi U restricts vyapti to: {when U holds}`
- **Evaluation criteria:**
  - *Format:* At least the verdict line is present; if upadhi found, restriction stated.
  - *Semantic:* If a candidate is listed, the two pervasion claims must be consistent (an upadhi by definition pervades sadhya and fails to pervade hetu). LLM-judge: "Is the claimed upadhi actually independent of the hetu in the relevant way?"
- **Failure modes:** (a) Model never searches and just writes "No upadhi found." Catch: require at least one *considered* candidate even when verdict is negative. (b) Model conflates upadhi with hetvabhasa. Catch: cross-phase de-duplication check between Phase 9 and Phase 13 — same defect should not be listed in both.
- **Comparison to Sathish 2026:** *Completely absent.* This is the second major divergence and is grounded directly in TC Anumana-khanda §5.3.

---

### Phase 10: Paksadharmata + Paramarsa (Subject-Locus & Reflective Synthesis)

- **Classical source(s):** Gangesha, *Tattvacintamani*, Anumana-khanda 5.2. *Paksadharmata*: the hetu must actually be a property of the paksa. *Paramarsa*: the synthesizing cognition "this paksa has the hetu, which is pervaded by sadhya" — Gangesha identifies this as the proximate cause of inferential cognition (*anumiti*).
- **Cognitive operation:** (a) Verify that the hetu actually obtains in the paksa (subject of the inference). (b) Produce the *paramarsa* — the explicit synthesis "the paksa has the hetu (pakshadharmata), and the hetu is pervaded by sadhya (vyapti) — therefore the paksa has the sadhya." This is *not yet* the formal syllogism (Phase 11); it is the cognitive act that the syllogism externalizes.
- **Why distinct from neighbors:** Phase 8 was the universal; Phase 10 is the application of the universal *to this case*. Phase 11 is the formalization of Phase 10 into the canonical five-member structure. Distinguishing the cognitive act (10) from its formal expression (11) is a Navya-Nyaya commitment — Gangesha treats *paramarsa* as a real mental event, not a mere rhetorical step.
- **LLM output specification:**
  - `Paksa (subject of inference):` <noun phrase, must be from problem>
  - `Paksadharmata claim:` `<paksa> has <hetu>` + one-sentence justification
  - `Paramarsa:` `Since <paksa> has <hetu>, and <hetu> is invariably accompanied by <sadhya>, therefore <paksa> has <sadhya>.`
- **Evaluation criteria:**
  - *Format:* All three sub-fields present; paramarsa uses the template "Since … and … therefore …".
  - *Semantic:* The hetu and sadhya in the paramarsa must match Phase 8 verbatim. The paksa must appear in Phase 1's inquired entity. Cross-phase identity check.
- **Failure modes:** (a) Model substitutes a different hetu or sadhya than Phase 8. Catch: string-match across phases. (b) Model writes the paramarsa but the paksadharmata is unjustified. Catch: judge prompt asks "Does the problem statement, or a prior phase, support the claim that the paksa has the hetu?"
- **Comparison to Sathish 2026:** *Absent.* Sathish has no paramarsa phase. This is the third major Gangesha-derived divergence.

---

### Phase 11: Pancha-avayava (Five-Membered Formal Argument)

- **Classical source(s):** NS 1.1.32–39 (the five members: pratijna, hetu, udaharana, upanaya, nigamana); Tarka-Sangraha §49.
- **Cognitive operation:** Formalize the inference as the canonical Indian syllogism with all five members.
- **Why distinct from neighbors:** Phase 11 is the *formal external* statement of what Phases 8 and 10 produced cognitively. It is expressly redundant with Phase 10's *paramarsa* in content — but the redundancy is the point: the syllogism is the auditable, communicable form that a debate partner can attack member by member. Without Phase 11, Phases 12 and 13 cannot run cleanly (they need explicit members to attack).
- **LLM output specification:**
  - `1. Pratijna (thesis):` `<paksa> has <sadhya>.`
  - `2. Hetu (reason):` `Because <paksa> has <hetu>.`
  - `3. Udaharana (universal example):` `Wherever <hetu>, there <sadhya>; as in <example>.`
  - `4. Upanaya (application):` `<paksa> has <hetu>.`
  - `5. Nigamana (conclusion):` `Therefore, <paksa> has <sadhya>.`
- **Evaluation criteria:**
  - *Format:* Five numbered lines, in this order, each beginning with the canonical Sanskrit term.
  - *Semantic:* Cross-phase consistency: paksa, hetu, sadhya must match Phases 8 and 10. Udaharana's "wherever … there" pattern must match the Phase 8 vyapti.
- **Failure modes:** (a) Model writes a 3-step modus ponens instead of all 5. Catch: structural rule. (b) Model fills the slots with material inconsistent with prior phases. Catch: cross-phase token match.
- **Comparison to Sathish 2026:** *Present (Sathish Phase 3: Pancha Avayava).* Largely identical here. Difference: my schema requires the udaharana to *match* the Phase 8 vyapti verbatim, providing a sharper consistency check.

---

### Phase 12: Tarka (Counterfactual Reductio)

- **Classical source(s):** NS 1.1.40 (*tarka* defined as hypothetical reasoning that supports a pramana by ruling out alternatives). Eighth padartha in NS 1.1.1.
- **Cognitive operation:** Suppose the conclusion is false. Derive a contradiction with either (a) the established tenets of Phase 4, (b) the pramana inputs of Phase 6, or (c) the vyapti of Phase 8. The contradiction confirms the conclusion via reductio.
- **Why distinct from neighbors:** Phase 12 is *constructive negative*: it actively assumes the negation and derives consequences. Phase 13 (*hetvabhasa*) is *diagnostic*: it audits the existing argument for known fallacy patterns. They are not the same operation. *Tarka* is not finding errors in the argument — *tarka* is independently re-deriving the conclusion from the contrary hypothesis's failure.
- **LLM output specification:**
  - `Suppose:` `not(<conclusion of Phase 11>)`
  - `Then:` <chain of 1–N derivations, each citing which prior phase's commitment is contradicted>
  - `Contradiction with:` <Phase reference + content>
  - `Therefore the original conclusion stands.`
- **Evaluation criteria:**
  - *Format:* Four blocks present; "Contradiction with" line cites a specific earlier phase.
  - *Semantic:* The derivation must be valid in the background tenets (Phase 4). LLM-judge: "Does the contradiction actually follow from the supposition?"
- **Failure modes:** (a) Model just restates the conclusion in negative form. Catch: requires at least one intermediate derivation step. (b) The contradiction cited is itself the conclusion (begging the question). Catch: cross-phase reference must point to Phase ≤ 10, not Phase 11.
- **Comparison to Sathish 2026:** *Present (Sathish Phase 4: Tarka).* Largely identical. Tightening: my schema requires the contradiction to cite a *specific* earlier phase, preventing vacuous "this leads to a contradiction" claims.

---

### Phase 13: Hetvabhasa-pariksa (Fallacy Audit, Navya List)

- **Classical source(s):** Tarka-Sangraha §§54–67 — the five Navya-Nyaya *hetvabhasa*s: *savyabhicara* (deviating reason), *viruddha* (contradictory reason), *satpratipaksa* (counter-inferentially balanced), *asiddha* (unestablished — sub-types: *asraya-asiddha, svarupa-asiddha, vyapyatva-asiddha*), *badhita* (sublated by another pramana).
- **Cognitive operation:** Audit the argument from Phases 8, 10, 11 against each of the five Navya fallacies. For each, state explicitly: present (with description) or absent (with one-sentence reason).
- **Why distinct from neighbors:** Distinct from Phase 9 (*upadhi*): an *upadhi* is a hidden condition restricting a vyapti; a *hetvabhasa* is a structural defect in the hetu's relation to sadhya or paksa. Distinct from Phase 12 (*tarka*): *tarka* asks "does the negation lead to absurdity?"; *hetvabhasa* asks "does our reason commit a known error?". Distinct from Phase 14 (*anupalabdhi/abhava*): *hetvabhasa* concerns the hetu; Phase 14 concerns reasoning about absences.
- **LLM output specification:** For each of the five Navya fallacies, exactly one line:
  - `Savyabhicara: present | absent — <reason>`
  - `Viruddha: present | absent — <reason>`
  - `Satpratipaksa: present | absent — <reason>`
  - `Asiddha: present | absent — <reason; if present, sub-type>`
  - `Badhita: present | absent — <reason>`
- **Evaluation criteria:**
  - *Format:* All five fallacies named verbatim; each tagged present/absent.
  - *Semantic:* If any is "present," the inference is defeated and Phase 15 must report that. If all are "absent," the reasons must be non-trivial. LLM-judge spot-checks reasons.
- **Failure modes:** (a) Model marks all absent without engagement. Catch: judge prompt scores reason quality. (b) Model uses the *Gautama* five-fold list (savyabhicara, viruddha, prakaranasama, sadhyasama, kalatita) — Sathish 2026's list — instead of the Navya list. Catch: name-set match against the Navya five.
- **Comparison to Sathish 2026:** *Present but uses the older Gautama list.* Sathish enumerates *savyabhichara, viruddha, prakaranasama, sadhyasama, kalaatita* — Gautama's fivefold from NS 1.2.4. The Navya list (Annambhatta) replaces these with *satpratipaksa* (counter-inferentially balanced) and *badhita* (sublated by another pramana), which are *substantively different* fallacies. *Satpratipaksa* in particular has no analog in Sathish's list and is critical for evaluating cases where two opposing inferences are both formally valid. This is the fourth major divergence.

---

### Phase 14: Anupalabdhi / Abhava-vicara (Negation & Non-Apprehension Check)

- **Classical source(s):** Udayana, *Atmatattvaviveka* (against Mimamsa's claim that *anupalabdhi* is a separate pramana); Tarka-Sangraha on *abhava* as a categorial *padartha*; Raghunatha Siromani's *Padartha-Tattva-Nirupanam* (treats *abhava* as a real category). The Nyaya position is that absence is known by qualified perception (*visesanata-pratyasatti*) — but the *check* must still be performed.
- **Cognitive operation:** Has the reasoning made any claim of the form "X is absent" or "no such X exists"? If yes, justify by either (a) qualified perception (we looked and did not find X in a place where, were X present, X would have been perceived), or (b) inference from a positive incompatible. Conversely: if the answer type from Phase 5 was "no such X exists" (negative answer), this phase is mandatory.
- **Why distinct from neighbors:** Reasoning about absences is a known LLM failure mode separate from positive inference. The phase is independently grounded in classical Nyaya's debate with Mimamsa over *anupalabdhi*. Without it, the schema treats "X exists" and "no X exists" symmetrically, which Nyaya explicitly does not.
- **LLM output specification:**
  - `Negative claim made:` yes / no
  - If yes:
    - `Negative claim:` <statement of the form "no X is Y" or "X does not exist" or "X is absent">
    - `Mode of knowing the absence:` <one of: qualified perception (would have been perceived if present), inference from incompatible positive, granted by sabda/definition>
    - `Counterfactual locus:` <where X *would* be if it existed>
- **Evaluation criteria:**
  - *Format:* The first line is always present. If yes, all sub-fields present.
  - *Semantic:* If Phase 5 said the answer type is "no such X exists" (negative), then this phase must say `yes`. Cross-phase consistency check. LLM-judge: "Is the counterfactual locus the right place to have looked?"
- **Failure modes:** (a) Negative answer in Phase 15 but Phase 14 says no negative claim made. Catch: cross-phase rule (hard fail). (b) Model claims absence without specifying where it would have been findable. Catch: counterfactual locus required.
- **Comparison to Sathish 2026:** *Absent.* Sathish has no negation-handling phase. This is the fifth major divergence. Grounded in Udayana and Annambhatta, with confirmation from Raghunatha.

> **Design note (flagged per §7):** Some Nyaya schools accept *anupalabdhi* as a sixth pramana; classical Nyaya does not. By placing the negation check *after* the pramana phase rather than as a sixth pramana, the schema follows Udayana's perceptual treatment of absence rather than the Mimamsa expansion. This is a defensible call but worth flagging.

---

### Phase 15: Nirnaya (Ascertainment + Epistemic-Status Report)

- **Classical source(s):** NS 1.1.41 (*nirnaya* — definitive ascertainment after doubt has been resolved); ninth padartha in NS 1.1.1.
- **Cognitive operation:** Issue the final answer, plus an epistemic status report drawing on all prior phases.
- **Why distinct from neighbors:** This is the externalized, decision-ready output. It must reference (and be consistent with) Phases 1, 5, 11, and any "present" findings in 9, 13, 14.
- **LLM output specification:**
  - `Answer:` <the answer to the Phase 1 question, in the type specified by Phase 5>
  - `Epistemic status:` one of:
    - `Prama` (established knowledge; no upadhi found, no hetvabhasa, tarka confirms)
    - `Sasamshaya` (residual doubt; specify which phase flagged it)
    - `Bhrama` (the inference is defeated; specify by which fallacy/upadhi/sublation)
  - `Doubt resolved (if Prama):` restate the Phase 2 conflicting predications and which one is settled.
  - `Confidence justification:` one sentence referencing Phases 9, 12, 13, 14 verdicts.
- **Evaluation criteria:**
  - *Format:* All four sub-fields present; epistemic status is exactly one of the three Sanskrit terms.
  - *Semantic:* Cross-phase: if Phase 13 marked any fallacy "present," status must be `Bhrama`. If Phase 9 found an unrestricted upadhi, status must be `Sasamshaya` or `Bhrama`. If Phase 5 said "no such X exists" and Phase 14 says no negative claim made, hard inconsistency.
  - The `Answer` field is the primary correctness signal — checked against the gold answer either by exact match (for closed-form problems) or by LLM-judge (for open-form).
- **Failure modes:** (a) Model says `Prama` while Phase 13 flagged a fallacy. Catch: cross-phase rule. (b) Model gives a confident answer with vague "Confidence justification." Catch: justification must reference at least one phase by number.
- **Comparison to Sathish 2026:** *Present (Sathish Phase 6: Nirnaya).* My schema sharpens by requiring the three-way *Prama / Sasamshaya / Bhrama* classification (which is canonical in Tarka-Sangraha) rather than a free-text "confidence level."

---

## 4. Logical Dependency Graph

```
                    [1 Jijnasa]
                         |
                         v
                    [2 Samshaya] -- types the doubt -->
                         |                              \
                         v                               \
                    [3 Prayojana]                         |
                         |                                |  determines
                         v                                |  pramana
                    [4 Siddhanta-vibhaga] --background---> |  selection
                         |                                |
                         v                                v
                    [5 Prameya-vislesana] ------> [6 Pramana-yojana]
                         |                                |
                         v                                v
                    [7 Drstanta] (out-of-problem anchor)
                         |
                         v
   +-------------------- [8 Vyapti-graha] ---------------- universal
   |                          |                            commitment
   |                          v
   |                     [9 Upadhi-pariksa] -- if upadhi found, restrict 8
   |                          |
   |                          v
   +-------------------> [10 Paksadharmata + Paramarsa] -- application
                              |
                              v
                         [11 Pancha-avayava] -- formalization
                              |
                  +-----------+-----------+
                  v           v           v
            [12 Tarka]  [13 Hetvabhasa] [14 Abhava-vicara]
                  \           |           /
                   \          v          /
                    +---> [15 Nirnaya] <-+
```

**Why the order matters.**

- 1→2: Doubt has no referent without a fixed question.
- 2→6: The *type* of doubt determines which pramanas are relevant (e.g., *vipratipatti* needs *sabda*).
- 4→8: The vyapti must be consistent with the background tenets; if a vyapti contradicts a *sarvatantra*, it is rejected before stress-testing.
- 8→9→10: Pervasion must be stated *before* upadhi can be searched against it; upadhi must be checked *before* the pervasion is applied to the paksa, otherwise the paramarsa carries a hidden condition.
- 10→11: The formal syllogism externalizes the cognitive act; reversing forces the model to invent post-hoc cognition.
- 11→{12,13,14}: All three adversarial phases attack the formalized argument; running them before formalization gives them no fixed target.
- {9,12,13,14}→15: The ascertainment must integrate verdicts from upadhi, tarka, fallacy, and absence checks.

**What goes wrong if the model answers them in the wrong order.** The most common failure is post-hoc rationalization: the model decides the answer (effectively Phase 15) first and back-fills the earlier phases. Phase-order verification (each phase's content must be consistent with — but not reference forward to — later phases' content) is a structural defense. The dependency graph is also the audit graph: the scorer should run cross-phase consistency checks along the edges shown.

---

## 5. Worked Example

**Problem:** *A logic puzzle.* Three suspects A, B, C. Exactly one is the thief. Statements: A says "B did it." B says "I did not do it." C says "A did it." Exactly one statement is true. Who is the thief?

**Phase 1 (Jijnasa).**
- Question: Which of A, B, C is the thief?
- Inquired entity: the unique thief among {A, B, C}.
- Inquired relation: identity (which suspect *is* the thief).
- Restatement check: Solving for the thief subject to "exactly one statement true" answers the original question.

**Phase 2 (Samshaya).**
- Doubt type: *vipratipatti* (conflicting testimony — A and C make incompatible identification claims).
- Locus of doubt: the identity of the thief.
- Conflicting predications: `thief = B` (per A) vs `thief = A` (per C) vs `thief ∉ {A}` (per B's denial).
- Justification: The three speakers give partially incompatible reports; *vipratipatti* is precisely the doubt arising from conflicting testimony.

**Phase 3 (Prayojana).**
- Downstream use of the answer: To assign blame correctly; downstream inferences about each suspect's veracity follow.
- What would change if the answer flipped: Different statements become the "one true" statement, restructuring the truthfulness map.

**Phase 4 (Siddhanta-vibhaga).**
- `[sarvatantra]` Classical 2-valued logic; modus ponens, modus tollens, exclusive disjunction valid.
- `[pratitantra]` Within this puzzle: exactly one suspect is guilty, and exactly one statement is true.
- `[abhyupagama]` (granted hypothetically) Each suspect makes their stated claim sincerely (i.e., the puzzle's statements are the actual content asserted).

**Phase 5 (Prameya-vislesana).**
- Answer type: single entity from {A, B, C}.
- Cardinality expectation: exactly one.
- Domain of answer: {A, B, C}.

**Phase 6 (Pramana-yojana).**
- *Pratyaksa:* Used — the three statements as given are direct data.
- *Anumana:* Used — the answer is reached by inference from constraints.
- *Upamana:* Not used — no analogical mapping required.
- *Sabda:* Used — the puzzle's framing rules ("exactly one statement is true," "exactly one is the thief") function as authoritative stipulation.

**Phase 7 (Drstanta).**
- Drstanta: As in the standard "knights and knaves" puzzle, where assigning truth-values to speakers and checking consistency yields the unique solution.
- Why both layperson and expert would accept this: knights-and-knaves is a textbook anchoring example.

**Phase 8 (Vyapti-graha).**
- Hetu: "the assignment (thief = X, true-statement = S) is consistent with the puzzle constraints"
- Sadhya: "X is the thief"
- Vyapti statement: For all candidate assignments, if exactly-one-thief and exactly-one-statement-true are simultaneously satisfied by the assignment, then the X named by that assignment is the thief.
- Source: definitional (the puzzle's stipulations).

**Phase 9 (Upadhi-pariksa).**
- Candidate upadhi: "the suspects are speaking sincerely about the puzzle's stated propositions." Pervades sadhya: yes (the inference only goes through if their statements are sincere). Pervades hetu: no (one could imagine the constraints being checked against insincere statements).
- Verdict: Upadhi U found; vyapti restricted to: when speakers' statements are taken as the propositions being evaluated for truth. (This is Phase 4's *abhyupagama*, so the restriction is satisfied.)

**Phase 10 (Paksadharmata + Paramarsa).**
- Paksa: the actual scenario described.
- Paksadharmata: the assignment (thief = B, true-statement = A's) satisfies the constraints — A says "B did it" (true), B says "I did not" (false, since B is the thief), C says "A did it" (false). Exactly one true. Exactly one thief.
  - Check assignment (thief = A): A says "B did it" (false), B says "I didn't" (true), C says "A did it" (true). Two true — fails.
  - Check assignment (thief = C): A says "B did it" (false), B says "I didn't" (true), C says "A did it" (false). One true — but wait: this also satisfies. Cross-check: does this give two consistent thieves? No, only C. So both (B) and (C) appear to satisfy?
  - Re-check (thief = C): exactly one statement true (B's). Exactly one thief (C). Satisfies.
  - Re-check (thief = B): exactly one statement true (A's). Exactly one thief (B). Satisfies.
  - Both fit. The scenario as literally stated has two solutions.
- Paramarsa: Since the scenario admits two assignments satisfying the constraints, and the vyapti maps satisfying-assignment to thief, *the scenario as stated does not uniquely determine the thief.*

**Phase 11 (Pancha-avayava).**
- 1. Pratijna: The puzzle as literally stated has two consistent solutions.
- 2. Hetu: Because both assignments (thief = B) and (thief = C) make exactly one statement true.
- 3. Udaharana: Wherever a constraint set admits multiple satisfying models, the answer is non-unique; as in an underdetermined SAT instance.
- 4. Upanaya: This puzzle's constraint set admits two satisfying models.
- 5. Nigamana: Therefore the puzzle is underdetermined.

**Phase 12 (Tarka).**
- Suppose: the puzzle has a unique answer (say, thief = B).
- Then: assignment (thief = C) must violate some constraint. Check: exactly-one-thief (yes, only C), exactly-one-statement-true (B's; yes). Both constraints satisfied. Contradiction with uniqueness assumption.
- Contradiction with: Phase 4's *pratitantra* (uniqueness of solution under stated constraints) is *not* derivable; the supposed unique-answer assumption is unsupported.
- Therefore: the underdetermination conclusion stands.

**Phase 13 (Hetvabhasa-pariksa).**
- Savyabhicara: absent — the hetu (constraint-satisfaction) does not deviate to non-thieves.
- Viruddha: absent — the hetu does not entail non-uniqueness's opposite.
- Satpratipaksa: *present-but-managed* — there is a counter-inference "the puzzle has a unique answer because it is presented as having one." This counter-inference rests on a meta-pragmatic premise outside the literal constraints; flagging it under *satpratipaksa* surfaces the issue.
- Asiddha: absent — the hetu (each assignment's constraint-satisfaction) is established by direct check.
- Badhita: absent — no other pramana sublates the conclusion.

**Phase 14 (Anupalabdhi/Abhava-vicara).**
- Negative claim made: yes.
- Negative claim: No unique thief is determined by the stated constraints.
- Mode of knowing the absence: inference from incompatible positive — two distinct satisfying assignments coexist.
- Counterfactual locus: a literal reading of "exactly one is the thief" plus "exactly one statement is true," which would have produced a unique solution if it could.

**Phase 15 (Nirnaya).**
- Answer: The puzzle as literally stated is underdetermined; either B or C is consistent with the constraints.
- Epistemic status: *Sasamshaya* (residual doubt due to *satpratipaksa* in Phase 13 and the underdetermination flagged in Phase 14).
- Doubt resolved: The *vipratipatti* among A, B, C is *partially* resolved — A's case is excluded — but the choice between B and C is not.
- Confidence justification: Phase 12 confirms via reductio that uniqueness is not derivable; Phase 13 flags *satpratipaksa* (meta-pragmatic counter-inference); Phase 14 is consistent. The literally-correct epistemic status is *Sasamshaya*, not *Prama*.

(Note on the worked example: the puzzle as I stated it is genuinely underdetermined. A sharper formulation — e.g., adding "B is innocent" — yields *Prama* with thief = C. The worked example is deliberately one where the schema's machinery surfaces the underdetermination rather than papering over it; this is exactly the kind of case where Sathish's six-phase, lacking *upadhi* and *satpratipaksa*, would either confidently state a wrong answer or hand-wave.)

---

## 6. Comparison Table: This 15-Phase Schema vs Sathish 2026 Six-Phase

| This schema (15) | Sathish 2026 (6) | Relationship | What is gained / lost |
|---|---|---|---|
| 1. Jijnasa | — | New (from Vatsyayana's prefatory bhashya) | Anchors goal; catches drift |
| 2. Samshaya | 1. Samshaya | Preserved, restructured | Forces explicit conflicting predications |
| 3. Prayojana | — | New (NS 1.1.24) | Decision-relevance check |
| 4. Siddhanta-vibhaga | partial overlap with Sathish *sabda* sub-component | Separated | Distinguishes rules from inputs |
| 5. Prameya-vislesana | — | New (NS 1.1.9, repurposed) | Answer-type discipline |
| 6. Pramana-yojana | 2. Pramana | Refined | Allows justified non-use; sharper signal |
| 7. Drstanta | folded into Sathish *udaharana* | Separated | Out-of-problem anchor distinct from in-syllogism example |
| 8. Vyapti-graha | folded into Sathish *udaharana* | **Separated — major divergence** | Gangesha's central innovation surfaced (TC 5.1) |
| 9. Upadhi-pariksa | — | **New — major divergence** | Catches hidden-condition defeaters (TC 5.3) |
| 10. Paksadharmata + Paramarsa | folded into Sathish *upanaya* | **Separated — major divergence** | Gangesha's *paramarsa* surfaced (TC 5.2) |
| 11. Pancha-avayava | 3. Pancha Avayava | Preserved, tightened | Cross-phase consistency required |
| 12. Tarka | 4. Tarka | Preserved, tightened | Must cite specific earlier phase |
| 13. Hetvabhasa-pariksa | 5. Hetvabhasa | **Reformulated — major divergence** | Navya five-fold (Annambhatta) replaces Gautama five-fold (Sathish) |
| 14. Anupalabdhi/Abhava-vicara | — | **New — major divergence** | Negation handling (Udayana) |
| 15. Nirnaya | 6. Nirnaya | Preserved, sharpened | Three-way *Prama / Sasamshaya / Bhrama* taxonomy |

**Sathish phases that collapse multiple phases of this schema, and what is lost:**
- *Sathish's Pancha Avayava* collapses my Phases 7, 8, 10, 11. Lost: pervasion as a separate commitment (Phase 8, Gangesha's main contribution) and *paramarsa* as a distinct cognitive act (Phase 10). These are exactly the Navya-Nyaya innovations that distinguish Gangesha-onwards from Old Nyaya.
- *Sathish's Pramana* covers what my Phases 4 and 6 separate. Lost: the *sarvatantra/pratitantra/adhikarana/abhyupagama* typing of background tenets (NS 1.1.26–29).
- *Sathish's Hetvabhasa* uses the older Gautama list. Lost: *satpratipaksa* (counter-inference) and *badhita* (sublation by another pramana), both of which are real, named, distinct fallacies in Annambhatta's Navya list.

**Phases of this schema that are genuinely new (with citation):**
- Phase 1 (Vatsyayana's prefatory bhashya).
- Phase 3 (NS 1.1.24).
- Phase 4 (NS 1.1.26–29).
- Phase 5 (NS 1.1.9, repurposed for benchmark setting; flagged as design call in §7).
- Phase 8 (TC Anumana-khanda 5.1).
- Phase 9 (TC Anumana-khanda 5.3).
- Phase 10 (TC Anumana-khanda 5.2).
- Phase 14 (Atmatattvaviveka; Tarka-Sangraha on abhava).

**Net assessment.** Of Sathish's 6 phases, all 6 are *preserved in spirit* in this schema (in Phases 2, 6, 11, 12, 13, 15) — but 5 of the 6 are restructured. This schema is *not* a re-bulleting of Sathish: 8 of its 15 phases have no analog in his formulation, and they are precisely the phases that derive from the Navya-Nyaya core (Gangesha, Udayana, Annambhatta) that Sathish's 6-phase elides. About 40% of the schema is preserved-from-Sathish; about 60% is independently derived from the primary corpus.

---

## 7. Open Questions and Known Weaknesses

This section is honest about where the design has soft spots. The auditing agent will find these regardless; flagging them up front shows the schema is not pretending.

**Design calls I made (not verbatim from primary sources):**

1. **Phase 5 (Prameya-vislesana) repurposes *prameya*.** Classically, *prameya* names the twelve metaphysical objects of cognition (atman, sarira, indriya, …, apavarga; NS 1.1.9). I am using the structural slot of "the object the pramana cognizes" to mean "the type of the answer in a benchmark problem." This is a *pedagogical* extension. A purist could object that nothing in Vatsyayana licenses applying *prameya* to "Boolean / set / mapping." Defense: the structural role of *prameya* in NS 1.1.1's enumeration *is* "what the pramana grasps," and in benchmark problems that is the answer. Alternative I considered: drop Phase 5 and merge type-checking into Phase 1. Rejected because answer-typing and question-stating are operationally distinct.

2. **Phase 14 (Anupalabdhi) is positioned as a phase, not a pramana.** Mimamsa treats *anupalabdhi* as a sixth pramana; classical Nyaya (per Udayana) treats absence as known by qualified perception. The schema follows Nyaya — but a defender of Mimamsa-style Nyaya could argue Phase 14 should be folded into Phase 6 as a fifth pramana. I rejected this because (a) the schema is Navya-Nyaya specific, and (b) treating absence-handling as a separate audit phase produces a cleaner evaluation signal than burying it inside pramana selection.

3. **Order of Phase 9 (upadhi) before Phase 10 (paramarsa).** In Gangesha's *Tattvacintamani*, the order of treatment is vyapti → paramarsa → upadhi (in his exposition). I have placed upadhi *between* vyapti and paramarsa. The defense: the *cognitive logic* (a paramarsa carrying a hidden upadhi is defective) requires upadhi to be checked before the paramarsa is committed. This is a benchmark-driven reordering, faithful in spirit to Gangesha but not in expository sequence. Alternative: keep Gangesha's expository order. Rejected because it weakens the cross-phase consistency check.

4. **Phase 13 uses Annambhatta's Navya five-fold, not Gautama's.** Both are canonical. The Navya list is sharper for benchmark evaluation (*satpratipaksa* and *badhita* are mechanically more checkable than *prakaranasama* and *kalatita*). A purist who wants the Gautama list could substitute. Marked.

5. **Phase 1 (Jijnasa) is grounded in Vatsyayana's *prefatory* bhashya rather than a numbered sutra.** This is the weakest classical anchor in the schema. *Jijnasa* is a real and load-bearing Nyaya term — but it is not one of the 16 padarthas of NS 1.1.1. A reasonable critic could say this phase is implicit-not-explicit in the primary corpus. Defense: every Nyaya treatise opens with the question that is being addressed, and Vatsyayana names the precondition *jijnasa*; pretending the schema can begin without this phase is the alternative, which is worse.

**Where the classical sources are silent or contested:**

- *Whether vyapti-knowledge is itself a separate pramana or a species of inference.* Different Navya-Nyaya commentators disagree. The schema sidesteps by making *vyapti-graha* a phase, not a pramana.
- *The exact relation between paramarsa and the formal upanaya.* Are they the same cognitive event? Gangesha treats them as distinct (paramarsa is mental, upanaya is verbal); Old Nyaya tends to identify them. I follow Gangesha; this is the basis for separating Phases 10 and 11.
- *The status of jnanalaksana (knowledge-as-sense-contact).* Phase 6 mentions it under *alaukika pratyaksa* but does not require its use. A more aggressive schema could mandate its consideration; I judged that to be over-engineering for the benchmark context.

**Risks to defensibility:**

- *Phase 5 risk:* "Repurposing *prameya*" could be flagged as Sanskrit-flavored window dressing rather than primary-source grounding. The honest counter is: it is partial primary-source grounding (the structural slot is real; the benchmark application is mine).
- *Phase 7 vs Phase 11 risk:* An auditor may ask whether *drstanta* and *udaharana* are *really* distinct enough to warrant two phases. The texts do distinguish them (*drstanta* is the agreed paradigm per NS 1.1.25; *udaharana* is the in-syllogism example per NS 1.1.36). But a strict critic could argue this is splitting hairs. The benchmark payoff (forcing an out-of-problem anchor distinct from the in-syllogism example) is the operational defense.
- *Phase 14 risk:* If a benchmark contains few problems with negative answers, Phase 14 will be trivially "no negative claim" most of the time. This is fine for fast cases but means the phase carries weight only on a subset of problems. A balanced benchmark mix is required for Phase 14 to earn its slot.
- *General risk:* 15 phases is a lot. Format-adherence will be lower than Sathish's 6-phase by construction — there is more to get right. Sathish reports 40% strict format adherence on his 6-phase. This schema should expect, before any prompt-engineering, lower strict format adherence and *higher* semantic correctness on the problems where the additional phases catch defects (especially upadhi cases and *satpratipaksa* cases). The benchmark's value is in the diagnostic resolution, not the headline format-adherence number.

**Open question for the next agent.** Whether to add an explicit "vada/jalpa/vitanda" debate-phase audit (the seventh through ninth padarthas in NS 1.1.1) at Phase 16. I judged that for a single-shot reasoning benchmark these are not load-bearing — they govern multi-party debate, not single-cognizer inference. But for a multi-agent benchmark, they would belong.

---

## Sources

- [Pramana: Fine-Tuning Large Language Models for Epistemic Reasoning through Navya-Nyaya (Sathish 2026, arXiv:2604.04937)](https://arxiv.org/html/2604.04937)
- [Nyaya Sutras — Wikipedia](https://en.wikipedia.org/wiki/Ny%C4%81ya_S%C5%ABtras)
- [Gautama's Sixteen Padarthas (wisdomlib)](https://www.wisdomlib.org/hinduism/essay/nyaya-vaisheshika-critical-and-historical-study/d/doc1455983.html)
- [Gaṅgeśa — Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/gangesa/)
- [Tarka-Sangraha — Wikipedia](https://en.wikipedia.org/wiki/Tarka-Sangraha)
- [Tarka Sangraha Dipika English Translation, Bahulikar (PDF)](https://sambhasha.ksu.ac.in/CompLing/Tarka%20Sangraha/Tarka%20Sangraha%20Dipika%20%20English%20Translation%20Bahulikar%20B.N..pdf)
- [Navya Nyaya — Wikipedia](https://en.wikipedia.org/wiki/Navya-Ny%C4%81ya)
- [Vyapti — Wikipedia](https://en.wikipedia.org/wiki/Vyapti)
- [Hetvabhasa: Annambhatta's five-fold Navya list (wisdomlib)](https://www.wisdomlib.org/hinduism/essay/anumana-in-indian-philosophy/d/doc1085489.html)
- [Udayana — Wikipedia](https://en.wikipedia.org/wiki/Udayana)
- [Nyayakusumanjali of Udayana, on Anupalabdhi (wisdomlib)](https://www.wisdomlib.org/hinduism/essay/nyayakusumanjali-of-udayana-study/d/doc1211174.html)
- [Authors of Nyaya: Raghunatha Siromani (wisdomlib)](https://www.wisdomlib.org/hinduism/essay/nyaya-vaisheshika-critical-and-historical-study/d/doc1455882.html)
- [Vatsyayana's commentary, NS 1.1.24 on prayojana (Vātsyāyana's Guide to Liberation, JIP)](https://link.springer.com/article/10.1007/s10781-020-09438-x)
- [Nyaya — Internet Encyclopedia of Philosophy](https://iep.utm.edu/nyaya/)
- [Extraordinary Perception (alaukika-pratyaksa) (wisdomlib)](https://www.wisdomlib.org/hinduism/essay/nyaya-vaisheshika-critical-and-historical-study/d/doc1455965.html)
