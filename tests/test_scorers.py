"""Unit tests for pramana_test scorers.

These tests do NOT call any LLM API. They exercise score_format and
score_semantic against canned response strings only.
"""

from __future__ import annotations

import pytest

from pramana_test import Problem, score_format, score_semantic


# ---------------------------------------------------------------------------
# Canned response fixtures
# ---------------------------------------------------------------------------

# A fully-correct response: YAML frontmatter with ground_truth, all six
# section headers in order, all required Pramana subtypes / Avayava parts /
# Hetvabhasa fallacy types / Nirnaya fields populated with >= 3 real tokens.
FULL_RESPONSE = """\
---
id: test-canned
problem_type: deduction
difficulty: easy
variable_count: 1
ground_truth: "yes"
---

## Samshaya (Doubt Analysis)
The doubt under examination is whether the proposition holds, given the stated
premises. This is a doubt of definitive determination requiring deductive
inference from the given facts.

## Pramana (Evidence Sources)
- Pratyaksha: direct perception is not applicable to this abstract problem.
- Anumana: inference from the stated premises is the primary evidence source.
- Upamana: analogy with similar logical patterns supports the inference.
- Shabda: testimony of the premises themselves is taken as authoritative.

## Pancha Avayava (Five-Member Syllogism)
- Pratijna: the proposition under test follows from the premises.
- Hetu: because the premises chain via modus ponens and are consistent.
- Udaharana: Wherever the antecedent holds, there is the consequent, as in similar deductive cases.
- Upanaya: the present case satisfies the antecedent in the same way.
- Nigamana: therefore the proposition follows from the premises.

## Tarka (Counterfactual Testing)
Suppose the proposition were false. Then one of the premises would have to be
false, contradicting the given facts. The counterfactual reduces to absurdity.

## Hetvabhasa (Fallacy Detection)
- Savyabhichara: no irregular middle term is present in the chain.
- Viruddha: no contradiction between premises and conclusion is observed.
- Prakaranasama: the reasoning does not beg the question under examination.
- Sadhyasama: the proof does not assume what it is trying to establish.
- Kalaatita: the temporal ordering of premises and conclusion is consistent.

## Nirnaya (Definitive Conclusion)
Answer: yes
Confidence: 0.95
Epistemic Status: Definitive Knowledge
"""


# Placeholder-only response: every required bullet is "x".
PLACEHOLDER_X_RESPONSE = """\
---
id: test-canned
ground_truth: "yes"
---

## Samshaya (Doubt Analysis)
x

## Pramana (Evidence Sources)
- Pratyaksha: x
- Anumana: x
- Upamana: x
- Shabda: x

## Pancha Avayava (Five-Member Syllogism)
- Pratijna: x
- Hetu: x
- Udaharana: Wherever x there is x.
- Upanaya: x
- Nigamana: x

## Tarka (Counterfactual Testing)
x

## Hetvabhasa (Fallacy Detection)
- Savyabhichara: x
- Viruddha: x
- Prakaranasama: x
- Sadhyasama: x
- Kalaatita: x

## Nirnaya (Definitive Conclusion)
Answer: x
Confidence: x
Epistemic Status: x
"""


# Out-of-order response: Nirnaya appears before Tarka and Hetvabhasa.
OUT_OF_ORDER_RESPONSE = """\
---
id: test-canned
ground_truth: "yes"
---

## Samshaya (Doubt Analysis)
The doubt under examination is whether the proposition holds.

## Pramana (Evidence Sources)
- Pratyaksha: direct perception is not applicable here.
- Anumana: inference from the premises is the primary evidence source.
- Upamana: analogy with similar deductive patterns supports the inference.
- Shabda: the premises themselves are taken as authoritative testimony.

## Pancha Avayava (Five-Member Syllogism)
- Pratijna: the proposition follows from the premises.
- Hetu: because the chain of premises is consistent and complete.
- Udaharana: Wherever the antecedent holds, there is the consequent.
- Upanaya: the present case satisfies that antecedent in the same way.
- Nigamana: therefore the proposition holds.

## Nirnaya (Definitive Conclusion)
Answer: yes
Confidence: 0.95
Epistemic Status: Definitive Knowledge

## Tarka (Counterfactual Testing)
The counterfactual reduces to absurdity by contradicting the given premises.

## Hetvabhasa (Fallacy Detection)
- Savyabhichara: no irregular middle term is present.
- Viruddha: no contradiction is observed.
- Prakaranasama: the reasoning is not circular.
- Sadhyasama: the proof does not assume its conclusion.
- Kalaatita: the temporal ordering is consistent.
"""


EMPTY_RESPONSE = ""


# ---------------------------------------------------------------------------
# score_format tests
# ---------------------------------------------------------------------------

def test_score_format_full_response_passes() -> None:
    report = score_format(FULL_RESPONSE)
    assert report.passes, (
        f"Expected fully-correct response to pass; report={report!r}"
    )


def test_score_format_placeholder_x_response_fails() -> None:
    report = score_format(PLACEHOLDER_X_RESPONSE)
    assert not report.passes
    # The specific failure mode: bullets full of "x" should be rejected.
    assert not report.placeholders_clean


def test_score_format_out_of_order_response_fails() -> None:
    report = score_format(OUT_OF_ORDER_RESPONSE)
    assert not report.passes
    # The specific failure mode: section ordering is violated.
    assert not report.headers_in_order


def test_score_format_empty_response_fails() -> None:
    report = score_format(EMPTY_RESPONSE)
    assert not report.passes


# ---------------------------------------------------------------------------
# score_semantic tests
# ---------------------------------------------------------------------------

def _make_problem(keywords: list[list[str]]) -> Problem:
    return Problem(
        id="test-unit",
        question="(unused in scorer tests)",
        ground_truth_keywords=keywords,
        source="constructed",
    )


def _nirnaya_only(answer_text: str) -> str:
    """Wrap an answer in just enough scaffold for score_semantic to extract it.

    score_semantic only needs the `## Nirnaya` section text (or the YAML
    ground_truth fallback) to find keywords.
    """
    return f"## Nirnaya (Definitive Conclusion)\nAnswer: {answer_text}\n"


def test_score_semantic_unsatisfiable_does_not_match_satisfiable() -> None:
    """Word-boundary fix: "satisfiable" must NOT match inside "unsatisfiable"."""
    problem = _make_problem([["satisfiable"]])
    response = _nirnaya_only("The formula is unsatisfiable.")
    assert score_semantic(response, problem) is False


def test_score_semantic_yes_matches_yes() -> None:
    problem = _make_problem([["yes"]])
    response = _nirnaya_only("Yes, the proposition holds.")
    assert score_semantic(response, problem) is True


def test_score_semantic_digit_matches_word_synonym() -> None:
    """Number normalization: an answer of "10" should match a ["10","ten"] group."""
    problem = _make_problem([["10", "ten"]])
    response = _nirnaya_only("10 students passed history.")
    assert score_semantic(response, problem) is True


def test_score_semantic_empty_keywords_raises() -> None:
    problem = _make_problem([])
    response = _nirnaya_only("anything")
    with pytest.raises(ValueError):
        score_semantic(response, problem)
