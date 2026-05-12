"""Judge protocol and result types.

Implementations (SingleJudge, JuryOf3) land in Weekend 2 step 7. This module
provides only the type contract from BLUEPRINT_v5plus.md §17a so that other
modules (notably `cache`) can reference `JudgeResult` without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union


@dataclass(frozen=True)
class ContinuousScore:
    """A real-valued judge score in [0, 1]."""

    value: float


@dataclass(frozen=True)
class BinaryScore:
    """A pass/fail judge score with a rationale string."""

    passed: bool
    rationale: str


JudgeResult = Union[ContinuousScore, BinaryScore]


class Judge(Protocol):
    """Judge protocol. Implementations land in step 7.

    `payload` is a rubric-specific dict. Topofuzz will pass:
        {"variant_id", "eval_row", "agent_output"}
    Pramana (future) will pass:
        {"problem", "ground_truth", "candidate_answer", "nirnaya_section"}
    """

    def score(self, payload: dict) -> JudgeResult: ...
