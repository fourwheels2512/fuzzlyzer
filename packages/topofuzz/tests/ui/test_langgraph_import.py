"""Tests for ``topofuzz.ui.langgraph_import``.

Covers:
  * Linear-chain happy path → CustomWorkflow with per-node models.
  * Branching, cycles, missing START, and non-compiled inputs all reject
    with :class:`LangGraphImportError`.
  * BYOP validation (agent_id grammar, model allow-list) routes
    through ``validate_custom_workflow_form`` and surfaces as
    :class:`CustomFormError`.
  * ``custom_workflow_from_module_spec`` resolves a ``path:attr`` spec
    and auto-detects ``LLM_CONFIGS`` from the module.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from topofuzz.ir import LLMConfig
from topofuzz.ui.custom_validation import CustomFormError
from topofuzz.ui.langgraph_import import (
    LangGraphImportError,
    custom_workflow_from_langgraph,
    custom_workflow_from_module_spec,
)
from topofuzz.ui.presets import CustomWorkflow


# A reusable rubric of 50+ chars — the validator enforces MIN_RUBRIC_LEN.
_RUBRIC = (
    "Approve when the final agent recommends approval, deny when it "
    "recommends against. Borderline cases are common."
)


class _S(TypedDict, total=False):
    notes: str


def _noop(state: _S) -> _S:
    return state


def _linear_graph(*node_ids: str):
    """Build a compiled linear-chain LangGraph with the given node ids."""
    if not node_ids:
        raise ValueError("at least one node required")
    g = StateGraph(_S)
    for nid in node_ids:
        g.add_node(nid, _noop)
    g.add_edge(START, node_ids[0])
    for src, dst in zip(node_ids, node_ids[1:]):
        g.add_edge(src, dst)
    g.add_edge(node_ids[-1], END)
    return g.compile()


class TestLinearHappyPath:
    def test_three_nodes_round_trip(self) -> None:
        compiled = _linear_graph("intake", "reviewer", "decider")
        cw = custom_workflow_from_langgraph(
            compiled,
            label="My Linear Pipeline",
            domain_noun="loan application",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        assert isinstance(cw, CustomWorkflow)
        assert [a.agent_id for a in cw.agents] == ["intake", "reviewer", "decider"]
        # Every agent gets the default fallback model.
        assert {a.model for a in cw.agents} == {"google/gemini-2.5-flash"}
        assert cw.verdict_agent_id == "decider"
        assert cw.label == "My Linear Pipeline"

    def test_per_node_llm_configs_override_fallback(self) -> None:
        compiled = _linear_graph("intake", "advocate", "decider")
        cfgs = {
            "advocate": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),
            "decider": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),
        }
        cw = custom_workflow_from_langgraph(
            compiled,
            label="Mixed Pro/Flash",
            domain_noun="insurance claim",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
            llm_configs=cfgs,
        )
        models = {a.agent_id: a.model for a in cw.agents}
        assert models["intake"] == "google/gemini-2.5-flash"  # fallback
        assert models["advocate"] == "google/gemini-2.5-pro"
        assert models["decider"] == "google/gemini-2.5-pro"

    def test_single_node_chain(self) -> None:
        compiled = _linear_graph("only_node")
        cw = custom_workflow_from_langgraph(
            compiled,
            label="Solo",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        assert [a.agent_id for a in cw.agents] == ["only_node"]
        assert cw.verdict_agent_id == "only_node"


class TestRejectsNonLinear:
    def test_branching_rejected(self) -> None:
        g = StateGraph(_S)
        g.add_node("router", _noop)
        g.add_node("yes", _noop)
        g.add_node("no", _noop)
        g.add_edge(START, "router")
        g.add_conditional_edges(
            "router", lambda s: "yes", {"yes": "yes", "no": "no"}
        )
        g.add_edge("yes", END)
        g.add_edge("no", END)
        with pytest.raises(LangGraphImportError, match="branches"):
            custom_workflow_from_langgraph(
                g.compile(),
                label="Branchy",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )

    def test_cycle_rejected(self) -> None:
        g = StateGraph(_S)
        g.add_node("a", _noop)
        g.add_node("b", _noop)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", "a")  # cycle
        # Note: LangGraph permits cycles at the builder level — terminal
        # detection happens at compile. Some versions raise at .compile();
        # if so, the test exercises the import path through the underlying
        # walk and still rejects.
        try:
            compiled = g.compile()
        except Exception:  # pragma: no cover — depends on langgraph version
            pytest.skip("langgraph rejected the cycle at compile()")
        with pytest.raises(LangGraphImportError):
            custom_workflow_from_langgraph(
                compiled,
                label="Cyclic",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )

    def test_missing_compiled_api_rejected(self) -> None:
        class _NotAGraph:
            pass

        with pytest.raises(LangGraphImportError, match="compiled LangGraph"):
            custom_workflow_from_langgraph(
                _NotAGraph(),
                label="bogus",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )


class TestValidationRouting:
    def test_bad_agent_id_surfaces_as_form_error(self) -> None:
        # CamelCase node id fails the AGENT_ID_RE — the BYOP form
        # validator owns that rule, not the importer.
        compiled = _linear_graph("intakeStep", "reviewer", "decider")
        with pytest.raises(CustomFormError) as exc_info:
            custom_workflow_from_langgraph(
                compiled,
                label="bad names",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )
        # The validator reports per-field errors keyed by index.
        assert any("agent" in k for k in exc_info.value.errors)

    def test_unknown_llm_configs_key_rejected(self) -> None:
        compiled = _linear_graph("intake", "decider")
        cfgs = {"nope": LLMConfig(model="google/gemini-2.5-pro")}
        with pytest.raises(LangGraphImportError, match="node id"):
            custom_workflow_from_langgraph(
                compiled,
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
                llm_configs=cfgs,
            )

    def test_fallback_model_must_be_allowlisted(self) -> None:
        compiled = _linear_graph("intake", "decider")
        with pytest.raises(LangGraphImportError, match="allow-list"):
            custom_workflow_from_langgraph(
                compiled,
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
                fallback_model="openai/gpt-4o",
            )

    def test_rubric_too_short_rejected(self) -> None:
        compiled = _linear_graph("intake", "decider")
        with pytest.raises(CustomFormError):
            custom_workflow_from_langgraph(
                compiled,
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph="too short",
            )


class TestModuleSpecLoader:
    """``custom_workflow_from_module_spec`` resolves ``path:attr`` + LLM_CONFIGS."""

    def _write_fixture(
        self, tmp_path: Path, *, with_llm_configs: bool = True
    ) -> Path:
        fixture = tmp_path / "fixture.py"
        body = textwrap.dedent(
            """
            from typing import TypedDict
            from langgraph.graph import END, START, StateGraph
            from topofuzz.ir import LLMConfig

            class _S(TypedDict, total=False):
                notes: str

            def _noop(state: _S) -> _S:
                return state

            def build_graph():
                g = StateGraph(_S)
                g.add_node("intake", _noop)
                g.add_node("reviewer", _noop)
                g.add_node("decider", _noop)
                g.add_edge(START, "intake")
                g.add_edge("intake", "reviewer")
                g.add_edge("reviewer", "decider")
                g.add_edge("decider", END)
                return g.compile()
            """
        )
        if with_llm_configs:
            body += textwrap.dedent(
                """
                LLM_CONFIGS = {
                    "reviewer": LLMConfig(model="google/gemini-2.5-pro"),
                }
                """
            )
        fixture.write_text(body, encoding="utf-8")
        return fixture

    def test_loads_module_and_picks_up_llm_configs(
        self, tmp_path: Path
    ) -> None:
        fixture = self._write_fixture(tmp_path, with_llm_configs=True)
        cw = custom_workflow_from_module_spec(
            f"{fixture}:build_graph",
            label="From file",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        models = {a.agent_id: a.model for a in cw.agents}
        # reviewer was overridden via the module's LLM_CONFIGS.
        assert models["reviewer"] == "google/gemini-2.5-pro"
        # intake + decider fall back.
        assert models["intake"] == "google/gemini-2.5-flash"
        assert models["decider"] == "google/gemini-2.5-flash"

    def test_no_llm_configs_means_all_fallback(self, tmp_path: Path) -> None:
        fixture = self._write_fixture(tmp_path, with_llm_configs=False)
        cw = custom_workflow_from_module_spec(
            f"{fixture}:build_graph",
            label="No configs",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        assert {a.model for a in cw.agents} == {"google/gemini-2.5-flash"}

    def test_missing_attr_errors(self, tmp_path: Path) -> None:
        fixture = self._write_fixture(tmp_path)
        with pytest.raises(LangGraphImportError, match="no attribute"):
            custom_workflow_from_module_spec(
                f"{fixture}:does_not_exist",
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )

    def test_missing_file_errors(self, tmp_path: Path) -> None:
        with pytest.raises(LangGraphImportError, match="not found"):
            custom_workflow_from_module_spec(
                f"{tmp_path / 'nope.py'}:build_graph",
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )

    def test_malformed_spec_errors(self) -> None:
        with pytest.raises(LangGraphImportError, match="form"):
            custom_workflow_from_module_spec(
                "no_colon_here.py",
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )

    def test_malformed_llm_configs_in_module_errors(
        self, tmp_path: Path
    ) -> None:
        fixture = tmp_path / "bad.py"
        fixture.write_text(
            textwrap.dedent(
                """
                from typing import TypedDict
                from langgraph.graph import END, START, StateGraph

                class _S(TypedDict, total=False):
                    notes: str

                def _noop(s): return s

                def build_graph():
                    g = StateGraph(_S)
                    g.add_node("only", _noop)
                    g.add_edge(START, "only")
                    g.add_edge("only", END)
                    return g.compile()

                LLM_CONFIGS = "not a dict"
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(LangGraphImportError, match="dict"):
            custom_workflow_from_module_spec(
                f"{fixture}:build_graph",
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )

    def test_callable_llm_configs_rejected(self, tmp_path: Path) -> None:
        """Module exports `LLM_CONFIGS` as a callable, not a dict.

        Documented in the importer as a defense-in-depth shape check on
        the module's contribution; closes the code-review MINOR gap.
        """
        fixture = tmp_path / "bad_callable.py"
        fixture.write_text(
            textwrap.dedent(
                """
                from typing import TypedDict
                from langgraph.graph import END, START, StateGraph

                class _S(TypedDict, total=False):
                    notes: str

                def _noop(s): return s

                def build_graph():
                    g = StateGraph(_S)
                    g.add_node("only", _noop)
                    g.add_edge(START, "only")
                    g.add_edge("only", END)
                    return g.compile()

                def LLM_CONFIGS():
                    return {}
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(LangGraphImportError, match="dict"):
            custom_workflow_from_module_spec(
                f"{fixture}:build_graph",
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )


class TestImporterCoverageGaps:
    """Cases the first pass missed — added per code-review audit."""

    def test_parallel_duplicate_edges_tolerated(self) -> None:
        """`add_edge(src, dst)` called twice is a single edge.

        Documented in `_extract_linear_order` as tolerated via
        `dict.fromkeys` dedup; this pins the behavior so a future
        refactor that drops the dedup breaks loudly.
        """
        g = StateGraph(_S)
        for nid in ("aa", "bb", "cc"):
            g.add_node(nid, _noop)
        g.add_edge(START, "aa")
        g.add_edge("aa", "bb")
        g.add_edge("aa", "bb")  # duplicate — must not trip the branch check
        g.add_edge("bb", "cc")
        g.add_edge("cc", END)
        cw = custom_workflow_from_langgraph(
            g.compile(),
            label="dup edges",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        assert [a.agent_id for a in cw.agents] == ["aa", "bb", "cc"]

    def test_start_to_end_only_rejected(self) -> None:
        """A graph with no agent nodes between START and END.

        LangGraph accepts START -> END constructions via
        ``add_conditional_edges`` (the empty-chain case is otherwise
        not buildable). The importer must reject when the walk produces
        zero nodes.
        """
        g = StateGraph(_S)
        # The only way to build an empty linear graph is to set a
        # conditional entrypoint that routes straight to END. Use a
        # one-branch path_map.
        g.add_conditional_edges(START, lambda s: "end", {"end": END})
        with pytest.raises(LangGraphImportError):
            custom_workflow_from_langgraph(
                g.compile(),
                label="empty",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
            )

    def test_mtime_cache_invalidates_on_edit(self, tmp_path: Path) -> None:
        """Editing a fixture between two imports in the same process.

        Closes RT-audit MAJOR S1-M1. The previous cache key was
        path-only and silently returned the stale module.
        """
        import os
        fixture = tmp_path / "evolving.py"
        v1 = textwrap.dedent(
            """
            from typing import TypedDict
            from langgraph.graph import END, START, StateGraph

            class _S(TypedDict, total=False):
                notes: str

            def _noop(s): return s

            def build_graph():
                g = StateGraph(_S)
                g.add_node("v1_first", _noop)
                g.add_node("v1_second", _noop)
                g.add_edge(START, "v1_first")
                g.add_edge("v1_first", "v1_second")
                g.add_edge("v1_second", END)
                return g.compile()
            """
        )
        fixture.write_text(v1, encoding="utf-8")
        cw1 = custom_workflow_from_module_spec(
            f"{fixture}:build_graph",
            label="evolving",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        assert [a.agent_id for a in cw1.agents] == ["v1_first", "v1_second"]

        v2 = v1.replace("v1_first", "v2_first").replace(
            "v1_second", "v2_second"
        )
        fixture.write_text(v2, encoding="utf-8")
        # Bump mtime by 2 seconds — some filesystems (FAT, certain SMB
        # mounts) round to 2s, so the write_text alone may not change
        # st_mtime_ns. Belt-and-suspenders.
        future = fixture.stat().st_mtime + 2
        os.utime(fixture, (future, future))

        cw2 = custom_workflow_from_module_spec(
            f"{fixture}:build_graph",
            label="evolving",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        assert [a.agent_id for a in cw2.agents] == ["v2_first", "v2_second"]

    def test_workflow_id_matches_storage_regex(self) -> None:
        """The id minted by the validator must satisfy the storage layer's regex.

        `save_custom_workflow` rejects a workflow whose id does not match
        `_WORKFLOW_ID_RE` (YYYYMMDDThhmmss-<8hex>); a stale id format
        produced upstream would surface as a confusing late-binding
        error. Pin the handshake here.
        """
        from topofuzz.ui.custom_storage import is_valid_workflow_id

        compiled = _linear_graph("intake", "decider")
        cw = custom_workflow_from_langgraph(
            compiled,
            label="id check",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
        )
        assert is_valid_workflow_id(cw.workflow_id), (
            f"minted workflow_id {cw.workflow_id!r} fails storage regex"
        )

    def test_llm_config_overrides_win_over_module(self, tmp_path: Path) -> None:
        """`llm_config_overrides` merge on top of the module's `LLM_CONFIGS`.

        The CLI uses this to surface its repeatable `--llm-config NODE=MODEL`
        flag without editing user source.
        """
        fixture = tmp_path / "with_configs.py"
        fixture.write_text(
            textwrap.dedent(
                """
                from typing import TypedDict
                from langgraph.graph import END, START, StateGraph
                from topofuzz.ir import LLMConfig

                class _S(TypedDict, total=False):
                    notes: str

                def _noop(s): return s

                def build_graph():
                    g = StateGraph(_S)
                    g.add_node("intake", _noop)
                    g.add_node("decider", _noop)
                    g.add_edge(START, "intake")
                    g.add_edge("intake", "decider")
                    g.add_edge("decider", END)
                    return g.compile()

                LLM_CONFIGS = {
                    "intake": LLMConfig(model="google/gemini-2.5-flash"),
                    "decider": LLMConfig(model="google/gemini-2.5-flash"),
                }
                """
            ),
            encoding="utf-8",
        )
        # Override `decider` -> pro. The override must win over the
        # module's flash binding.
        cw = custom_workflow_from_module_spec(
            f"{fixture}:build_graph",
            label="overridden",
            domain_noun="case",
            decision_verb="approve / deny",
            rubric_paragraph=_RUBRIC,
            llm_config_overrides={
                "decider": LLMConfig(model="google/gemini-2.5-pro"),
            },
        )
        models = {a.agent_id: a.model for a in cw.agents}
        assert models["intake"] == "google/gemini-2.5-flash"
        assert models["decider"] == "google/gemini-2.5-pro"

    def test_llm_config_overrides_malformed_rejected(
        self, tmp_path: Path
    ) -> None:
        """Non-LLMConfig values in `llm_config_overrides` raise at the importer layer.

        The `custom_workflow_from_module_spec` path validates override
        shape as defense-in-depth — the CLI's NODE=MODEL parser already
        produces correct types, but a library caller could pass garbage.
        """
        fixture = tmp_path / "f.py"
        fixture.write_text(
            textwrap.dedent(
                """
                from typing import TypedDict
                from langgraph.graph import END, START, StateGraph

                class _S(TypedDict, total=False):
                    notes: str

                def _noop(s): return s

                def build_graph():
                    g = StateGraph(_S)
                    g.add_node("intake", _noop)
                    g.add_node("decider", _noop)
                    g.add_edge(START, "intake")
                    g.add_edge("intake", "decider")
                    g.add_edge("decider", END)
                    return g.compile()
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(LangGraphImportError, match="LLMConfig"):
            custom_workflow_from_module_spec(
                f"{fixture}:build_graph",
                label="x",
                domain_noun="case",
                decision_verb="approve / deny",
                rubric_paragraph=_RUBRIC,
                llm_config_overrides={
                    "intake": "not a config",  # type: ignore[dict-item]
                },
            )
