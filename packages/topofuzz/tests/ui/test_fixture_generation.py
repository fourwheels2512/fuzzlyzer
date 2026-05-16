# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for the auto-generated fixture.py + rubric.txt templates.

A regression here would silently break every custom run: either the
runner fails to import the fixture, or it imports but produces the wrong
chain. Both are caught by:
  * Determinism: same input ⇒ byte-identical output.
  * compile() success: the generated source must parse + bytecode-compile.
  * Module attribute presence: __topofuzz_name__, AGENT_IDS, LLM_CONFIGS,
    build_graph all exist.
  * Verdict node = last in chain.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

from topofuzz.ui.custom_templates import render_fixture_py, render_rubric_txt
from topofuzz.ui.presets import (
    CustomAgent,
    CustomWorkflow,
    StarterScenario,
)


def _cw(
    *,
    agent_specs: list[tuple[str, str]] | None = None,
    label: str = "Demo",
    rubric: str = "x" * 100,
    workflow_id: str = "20260515T120000-deadbeef",
    created_at: float = 1715760000.0,
) -> CustomWorkflow:
    if agent_specs is None:
        agent_specs = [
            ("intake_clerk", "google/gemini-2.5-flash"),
            ("final_decider", "google/gemini-2.5-flash"),
        ]
    agents = tuple(CustomAgent(aid, mdl) for aid, mdl in agent_specs)
    return CustomWorkflow(
        workflow_id=workflow_id,
        label=label,
        domain_noun="loan application",
        decision_verb="approve / deny",
        agents=agents,
        verdict_agent_id=agents[-1].agent_id,
        rubric_paragraph=rubric,
        created_at=created_at,
    )


class TestFixtureDeterminism:
    def test_byte_identical_for_same_input(self) -> None:
        cw = _cw()
        a = render_fixture_py(cw)
        b = render_fixture_py(cw)
        assert a == b


class TestFixtureCompiles:
    def test_compile_succeeds(self, tmp_path: Path) -> None:
        cw = _cw()
        src = render_fixture_py(cw)
        # compile() catches any syntax error from a template bug.
        compile(src, "<custom-fixture>", "exec")

    def test_module_loads_and_has_required_attrs(
        self, tmp_path: Path
    ) -> None:
        cw = _cw(
            agent_specs=[
                ("intake_clerk", "google/gemini-2.5-flash"),
                ("skeptic", "google/gemini-2.5-pro"),
                ("final_decider", "google/gemini-2.5-flash"),
            ]
        )
        src = render_fixture_py(cw)
        path = tmp_path / "custom_fixture.py"
        path.write_text(src, encoding="utf-8")
        # Import as a module.
        spec = importlib.util.spec_from_file_location(
            "topofuzz_test_custom_fixture", path
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            assert getattr(module, "__topofuzz_name__") == (
                f"custom:{cw.workflow_id}"
            )
            assert hasattr(module, "AGENT_IDS")
            assert tuple(module.AGENT_IDS) == tuple(
                a.agent_id for a in cw.agents
            )
            assert hasattr(module, "LLM_CONFIGS")
            llm_configs = module.LLM_CONFIGS
            assert set(llm_configs.keys()) == {
                a.agent_id for a in cw.agents
            }
            for a in cw.agents:
                assert llm_configs[a.agent_id].model == a.model
            assert callable(getattr(module, "build_graph", None))
        finally:
            sys.modules.pop(spec.name, None)


class TestFixtureChainShape:
    def test_verdict_node_is_last(self) -> None:
        cw = _cw()
        assert cw.agents[-1].agent_id == cw.verdict_agent_id

    def test_render_rejects_verdict_not_last(self) -> None:
        # Hand-craft an inconsistent CustomWorkflow (skip validation
        # layer) and confirm the renderer refuses.
        agents = (
            CustomAgent("a", "google/gemini-2.5-flash"),
            CustomAgent("b", "google/gemini-2.5-flash"),
        )
        cw = CustomWorkflow(
            workflow_id="20260515T120000-deadbeef",
            label="x",
            domain_noun="case",
            decision_verb="approve / deny",
            agents=agents,
            verdict_agent_id="a",  # NOT last
            rubric_paragraph="x" * 100,
            created_at=time.time(),
        )
        with pytest.raises(ValueError):
            render_fixture_py(cw)

    def test_render_rejects_zero_agents(self) -> None:
        cw = CustomWorkflow(
            workflow_id="20260515T120000-deadbeef",
            label="x",
            domain_noun="case",
            decision_verb="approve / deny",
            agents=(),
            verdict_agent_id="",
            rubric_paragraph="x" * 100,
            created_at=time.time(),
        )
        with pytest.raises(ValueError):
            render_fixture_py(cw)


class TestFixtureRespectsModelChoice:
    def test_pro_vs_flash_round_trip(self) -> None:
        cw = _cw(
            agent_specs=[
                ("a", "google/gemini-2.5-flash"),
                ("b", "google/gemini-2.5-pro"),
                ("c", "google/gemini-2.5-flash"),
            ]
        )
        src = render_fixture_py(cw)
        assert "'google/gemini-2.5-flash'" in src
        assert "'google/gemini-2.5-pro'" in src


class TestFixtureSafeFromUserInput:
    def test_label_with_quotes_escapes(self) -> None:
        # A label with embedded quotes must NOT break the source — repr()
        # turns it into a safe string literal.
        cw = _cw(label='Bad "label" with \\backslash')
        src = render_fixture_py(cw)
        compile(src, "<test>", "exec")  # must still parse


class TestRubricTemplate:
    def test_user_paragraph_appears(self) -> None:
        cw = _cw(rubric="MARKER " + "x" * 60)
        out = render_rubric_txt(cw)
        assert "MARKER" in out

    def test_runtime_placeholders_preserved(self) -> None:
        cw = _cw()
        out = render_rubric_txt(cw)
        # The runner fills these at scoring time — they must reach the
        # output file unmangled.
        assert "{applicant_blurb}" in out
        assert "{agent_output}" in out

    def test_score_rationale_format_present(self) -> None:
        cw = _cw()
        out = render_rubric_txt(cw)
        assert "Score: N" in out
        assert "Rationale:" in out

    def test_user_braces_escaped(self) -> None:
        # If the user types "{applicant_blurb}" inside their rubric (e.g.
        # they think they need to reference it), it must NOT eat the real
        # placeholder substitution.
        cw = _cw(
            rubric=(
                "Treat {applicant_blurb} as the case profile. " * 3
            )
        )
        out = render_rubric_txt(cw)
        # The user's literal "{applicant_blurb}" survives (not replaced
        # by Python format) — it shows up multiple times AND the genuine
        # placeholder also exists exactly once outside their paragraph.
        assert out.count("{applicant_blurb}") >= 2
        assert "{agent_output}" in out

    def test_domain_noun_threaded_through(self) -> None:
        cw = _cw()
        out = render_rubric_txt(cw)
        assert "loan application" in out
