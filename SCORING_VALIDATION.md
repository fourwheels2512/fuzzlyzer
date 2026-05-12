# Scoring validation

This document describes what the `pramana-eval` scorers actually check, what they do not check, the known false-pass and false-fail risks, and a list of recommended improvements for v0.2. The goal is to be honest about the strictness and the gaps so contributors can make informed decisions about reading or extending the results.

## Format scorer

**What it checks.**

- YAML frontmatter is present at the top of the response and parses.
- The frontmatter includes a `ground_truth` field.
- All six section headers exist: Samshaya (Doubt Analysis), Pramana (Evidence Sources), Pancha Avayava (Five-Member Syllogism), Tarka (Counterfactual Testing), Hetvabhasa (Fallacy Detection), and Nirnaya (Definitive Conclusion).
- The six headers appear in the prescribed order.
- The Pramana section names all four evidence sub-types (Pratyaksha, Anumana, Upamana, Shabda).
- The Pancha Avayava section names all five syllogism parts (Pratijna, Hetu, Udaharana, Upanaya, Nigamana).
- The Udaharana line uses the literal "Wherever ... there is" template phrasing.
- The Hetvabhasa section names all five fallacy types (Savyabhichara, Viruddha, Prakaranasama, Sadhyasama, Kalaatita).
- The Nirnaya section has Answer, Confidence, and Epistemic Status fields.
- Each section has a minimum amount of non-empty content (a header alone is not a pass).

All checks must pass for the response to count as format-passing.

**What it does not check.**

- Whether the content of each section is actually about the right concept. A model could write nonsense under a correctly-named header and still format-pass.
- Whether the Tarka counterfactual is a real reductio or just a vague paragraph.
- Whether the Confidence value is sensible. `0.99` on a wrong answer still passes the format check.
- Whether the Hetvabhasa entries reflect a real check or are boilerplate "absent" lines.
- Whether the YAML `ground_truth` field matches the body of the response.

The format scorer measures structural adherence, not reasoning quality.

## Semantic scorer

**What it checks.**

- Extracts the Nirnaya section text and matches it against the problem's `ground_truth_keywords`.
- Keyword shape is `list[list[str]]` (AND-of-OR): the outer list is required answer components, each inner list is a synonym set within that component.
- Matching is case-insensitive, uses word boundaries (so `cat` does not match `category`), and normalizes digit/word forms when both are listed in the same synonym group (e.g. `10` and `ten`).
- A response is semantic-pass if every outer-list group has at least one inner-list keyword present in the Nirnaya text.

**Known false-pass risks.**

- *Negation blindness.* The scorer does not parse logical structure. A response that says "Alice does not have the fish" still matches the keyword `fish`, so a wrong answer can semantic-pass.
- *Multi-answer ambiguity.* For problems with multiple possible solutions where the keyword list happens to match either, a wrong-but-listed alternative passes.
- *Keyword leakage in scaffolding.* If a model echoes the question or premises into the Nirnaya section, keywords from the question may appear without the model actually answering.

**Known false-fail risks.**

- *Synonyms not enumerated.* If the answer uses `correct` and the keyword group only lists `right`, the response fails.
- *Unicode hyphen variations.* `warm-blooded` (ASCII hyphen) vs `warm‐blooded` (unicode hyphen) can mismatch; we normalize ASCII but not all unicode dash variants.
- *Phrasing mismatches.* "ten students" vs "10 students" only match if both forms are in the synonym group.
- *Scoping.* Only the Nirnaya section is scored. A correct final answer placed in Pratijna or Tarka but not restated in Nirnaya fails.

## Recommended improvements for v0.2

- *LLM-judge for semantic.* Replace or augment keyword scoring with a frontier-model judge that is shown the problem, the ground truth, and the model's Nirnaya, and asked for a binary semantic-pass with a justification. Cross-check against keyword scores so silent disagreement is visible.
- *Multi-run consistency check.* Run each problem N=3 to N=5 times at temperature=0 and report whether responses agree, since most providers retain small non-determinism even at temp=0.
- *Expanded test set.* n>=50, ideally n>=100, with held-out problems written before any model run. Pre-register keyword groups.
- *Per-section content check.* A lightweight content-relevance check that the Pramana section actually discusses evidence rather than restating the question.

## A note on strictness

The format scorer is intentionally strict and conservative. A response that drifts from the schema in any of the structural ways listed above counts as a format failure, even if the reasoning is sound. We prefer this stance because the whole point of the test is to measure adherence to a schema.

The semantic scorer is intentionally simple. Contributors adding test problems are encouraged to supply rich keyword groups - both the AND structure (every required component) and the OR synonyms within each component (digit/word forms, common phrasings). Pre-register your groups before running the model so the score reflects schema adherence, not keyword tuning.
