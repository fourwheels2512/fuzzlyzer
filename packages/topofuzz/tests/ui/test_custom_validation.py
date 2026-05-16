# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for the custom-workflow form validator.

Covers every BLOCKING constraint from the BYOP spec:
  * agent_id grammar (regex)
  * model allow-list
  * agent count bounds (1..10)
  * label / domain / verb length bounds
  * rubric length bounds
  * verdict_agent_id must match LAST agent
  * duplicate agent_ids rejected
"""

from __future__ import annotations

import pytest

from topofuzz.ui.custom_validation import (
    CustomFormError,
    validate_custom_workflow_form,
)


def _ok_form(**overrides) -> dict:
    """A minimal valid form-input dict; tests override one field at a time."""
    base = {
        "label": "My pipeline",
        "domain_noun": "loan application",
        "decision_verb": "approve / deny",
        "agent_ids": ["intake_clerk", "final_decider"],
        "agent_models": [
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash",
        ],
        "verdict_agent_id": "final_decider",
        "rubric_paragraph": "x" * 100,
    }
    base.update(overrides)
    return base


class TestHappyPath:
    def test_valid_form_succeeds(self) -> None:
        cw = validate_custom_workflow_form(**_ok_form())
        assert cw.label == "My pipeline"
        assert cw.verdict_agent_id == "final_decider"
        assert len(cw.agents) == 2

    def test_workflow_id_generated_server_side(self) -> None:
        cw = validate_custom_workflow_form(**_ok_form())
        # Format: YYYYMMDDThhmmss-<8hex>
        assert len(cw.workflow_id) == 8 + 1 + 6 + 1 + 8
        assert "T" in cw.workflow_id
        assert "-" in cw.workflow_id

    def test_explicit_workflow_id_preserved(self) -> None:
        cw = validate_custom_workflow_form(
            workflow_id="20260515T120000-abcdef12", **_ok_form()
        )
        assert cw.workflow_id == "20260515T120000-abcdef12"

    def test_blank_verdict_defaults_to_last(self) -> None:
        cw = validate_custom_workflow_form(
            **_ok_form(verdict_agent_id="")
        )
        assert cw.verdict_agent_id == "final_decider"


class TestAgentIdValidation:
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "A",  # too short
            "Has-Hyphen",
            "UPPER",
            "1starts_with_digit",
            "_starts_with_underscore",
            "has spaces",
            "has.dot",
            "x" * 41,  # too long
        ],
    )
    def test_bad_agent_id_rejected(self, bad: str) -> None:
        form = _ok_form(
            agent_ids=[bad, "final_decider"],
            verdict_agent_id="final_decider",
        )
        with pytest.raises(CustomFormError) as ctx:
            validate_custom_workflow_form(**form)
        assert any("agent" in k for k in ctx.value.errors)

    def test_duplicate_agent_id_rejected(self) -> None:
        form = _ok_form(
            agent_ids=["dup_name", "dup_name"],
            verdict_agent_id="dup_name",
        )
        with pytest.raises(CustomFormError) as ctx:
            validate_custom_workflow_form(**form)
        assert any("duplicate" in v for v in ctx.value.errors.values())


class TestModelAllowlist:
    def test_disallowed_model_rejected(self) -> None:
        form = _ok_form(
            agent_models=["openai/gpt-4o", "google/gemini-2.5-flash"]
        )
        with pytest.raises(CustomFormError) as ctx:
            validate_custom_workflow_form(**form)
        assert any("model" in k for k in ctx.value.errors)

    def test_blank_model_rejected(self) -> None:
        form = _ok_form(agent_models=["", "google/gemini-2.5-flash"])
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(**form)


class TestAgentCount:
    def test_zero_agents_rejected(self) -> None:
        form = _ok_form(
            agent_ids=[], agent_models=[], verdict_agent_id=""
        )
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(**form)

    def test_eleven_agents_rejected(self) -> None:
        ids = [f"agent_{i:02d}" for i in range(11)]
        models = ["google/gemini-2.5-flash"] * 11
        form = _ok_form(
            agent_ids=ids, agent_models=models, verdict_agent_id=ids[-1]
        )
        with pytest.raises(CustomFormError) as ctx:
            validate_custom_workflow_form(**form)
        assert "agents" in ctx.value.errors

    def test_ten_agents_accepted(self) -> None:
        ids = [f"agent_{i:02d}" for i in range(10)]
        models = ["google/gemini-2.5-flash"] * 10
        form = _ok_form(
            agent_ids=ids, agent_models=models, verdict_agent_id=ids[-1]
        )
        cw = validate_custom_workflow_form(**form)
        assert len(cw.agents) == 10

    def test_blank_rows_dropped(self) -> None:
        # Editor shows 4 empty rows by default — submitting some blank
        # must drop them, not register as 4 invalid agents.
        form = _ok_form(
            agent_ids=["intake_clerk", "", "final_decider", ""],
            agent_models=[
                "google/gemini-2.5-flash",
                "",
                "google/gemini-2.5-flash",
                "",
            ],
            verdict_agent_id="final_decider",
        )
        cw = validate_custom_workflow_form(**form)
        assert [a.agent_id for a in cw.agents] == [
            "intake_clerk",
            "final_decider",
        ]


class TestVerdictAgent:
    def test_verdict_not_in_agents_rejected(self) -> None:
        form = _ok_form(verdict_agent_id="not_an_agent")
        with pytest.raises(CustomFormError) as ctx:
            validate_custom_workflow_form(**form)
        assert "verdict_agent_id" in ctx.value.errors

    def test_verdict_not_last_rejected(self) -> None:
        form = _ok_form(
            agent_ids=["agent_a", "agent_b", "agent_c"],
            agent_models=["google/gemini-2.5-flash"] * 3,
            verdict_agent_id="agent_b",  # NOT last
        )
        with pytest.raises(CustomFormError) as ctx:
            validate_custom_workflow_form(**form)
        assert "verdict_agent_id" in ctx.value.errors


class TestLabelDomainVerbLengths:
    def test_short_label_rejected(self) -> None:
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(**_ok_form(label="x"))

    def test_long_label_rejected(self) -> None:
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(**_ok_form(label="x" * 61))

    def test_short_domain_rejected(self) -> None:
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(**_ok_form(domain_noun="x"))

    def test_long_domain_rejected(self) -> None:
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(
                **_ok_form(domain_noun="x" * 31)
            )

    def test_short_verb_rejected(self) -> None:
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(**_ok_form(decision_verb="x"))


class TestRubricLength:
    def test_short_rubric_rejected(self) -> None:
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(
                **_ok_form(rubric_paragraph="too short")
            )

    def test_long_rubric_rejected(self) -> None:
        with pytest.raises(CustomFormError):
            validate_custom_workflow_form(
                **_ok_form(rubric_paragraph="x" * 2001)
            )

    def test_minimum_length_accepted(self) -> None:
        cw = validate_custom_workflow_form(
            **_ok_form(rubric_paragraph="x" * 50)
        )
        assert cw.rubric_paragraph.startswith("x")

    def test_maximum_length_accepted(self) -> None:
        cw = validate_custom_workflow_form(
            **_ok_form(rubric_paragraph="x" * 2000)
        )
        assert len(cw.rubric_paragraph) == 2000
