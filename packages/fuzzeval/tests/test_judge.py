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
