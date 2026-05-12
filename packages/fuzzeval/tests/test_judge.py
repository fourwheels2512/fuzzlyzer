from dataclasses import FrozenInstanceError

import pytest

from fuzzeval.judge import BinaryScore, ContinuousScore, JudgeResult


def test_continuous_score_holds_value():
    s = ContinuousScore(value=0.7)
    assert s.value == 0.7


def test_continuous_score_is_frozen():
    s = ContinuousScore(value=0.7)
    with pytest.raises(FrozenInstanceError):
        s.value = 0.8  # type: ignore[misc]


def test_binary_score_holds_rationale():
    b = BinaryScore(passed=True, rationale="meets rubric")
    assert b.passed
    assert b.rationale == "meets rubric"


def test_binary_score_is_frozen():
    b = BinaryScore(passed=False, rationale="off topic")
    with pytest.raises(FrozenInstanceError):
        b.passed = True  # type: ignore[misc]


def test_judge_result_union_accepts_both():
    results: list[JudgeResult] = [
        ContinuousScore(value=0.5),
        BinaryScore(passed=False, rationale="off-topic"),
    ]
    assert len(results) == 2


def test_judge_result_supports_runtime_isinstance():
    """PEP 604 union — isinstance(x, JudgeResult) must work at runtime so
    consumers can narrow Judge.score() returns without typing.TypeGuard tricks.
    """
    assert isinstance(ContinuousScore(value=0.3), JudgeResult)
    assert isinstance(BinaryScore(passed=True, rationale="ok"), JudgeResult)
    assert not isinstance("a string", JudgeResult)
    assert not isinstance(0.5, JudgeResult)


def test_judge_protocol_structural_conformance():
    """Any class with `score(payload: dict) -> JudgeResult` satisfies the
    Judge Protocol at the type-check level. Verified by a fake implementer
    constructed without `Judge` in its MRO.
    """
    from fuzzeval.judge import Judge

    class FakeJudge:
        def score(self, payload: dict) -> JudgeResult:
            return ContinuousScore(value=0.42)

    # Structural typing: a Judge variable can hold FakeJudge.
    j: Judge = FakeJudge()
    result = j.score({"any": "payload"})
    assert isinstance(result, ContinuousScore)
    assert result.value == 0.42
