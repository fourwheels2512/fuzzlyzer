# Sample output

Here is a representative response from Gemini 2.5 Pro on test-001 (paper-verified, Alice/Bob/Carol pets). The model produced YAML frontmatter and all six sections, with the Udaharana template phrasing and an explicit confidence score. This response was scored format-pass and semantic-pass by the harness.

**Problem:** Alice, Bob, and Carol each have one pet: a cat, a dog, or a fish. Constraints: (1) Alice does not have the dog. (2) Bob has the cat. (3) Carol does not have the fish. Question: Who has which pet?

**Model:** `gemini-2.5-pro`, zero-shot, `temperature=0`. Wall time: 16.7 s.

````markdown
---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 6
ground_truth: "Alice has the fish, Bob has the cat, Carol has the dog."
---
## Samshaya (Doubt Analysis)
Aneka Dharma Upapatti: The doubt arises from considering multiple possible pets (properties) for each person (subject) under the given constraints.

## Pramana (Evidence Sources)
- Pratyaksha: Bob has the cat; Alice does not have the dog; Carol does not have the fish.
- Anumana: Since Bob has the cat, the remaining pets for Alice and Carol are the dog and the fish; since Alice does not have the dog, she must have the fish; therefore, Carol must have the last remaining pet, the dog.
- Upamana: (none in this problem)
- Shabda: The implicit axioms that each person has exactly one pet and each pet is owned by exactly one person.

## Pancha Avayava (Five-Member Syllogism)
- Pratijna: Alice has the fish.
- Hetu: Because the only pets available for her are the dog and the fish, and it is given that she does not have the dog.
- Udaharana: Wherever a subject must possess one of two exclusive properties and is known not to possess the first, there is the possession of the second. Example: If a coin lands either heads or tails, and it is not heads, it is tails.
- Upanaya: In this case, Alice must have either the dog or the fish, and it is stated she does not have the dog.
- Nigamana: Therefore, Alice has the fish.

## Tarka (Counterfactual Testing)
Suppose the final conclusion is false, for instance, that Carol does not have the dog. Since Bob has the cat and Alice is explicitly denied the dog, this would mean no one has the dog, which contradicts the premise that each pet is owned by one of the three people.

## Hetvabhasa (Fallacy Detection)
- Savyabhichara: absent
- Viruddha: absent
- Prakaranasama: absent
- Sadhyasama: absent
- Kalaatita: absent

## Nirnaya (Definitive Conclusion)
Answer: Alice has the fish, Bob has the cat, Carol has the dog.
Confidence: 1.0
Epistemic Status: Definitive Knowledge
````

Full per-problem traces for all 10 problems are in `pramana_test_results.json`.
