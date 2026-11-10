# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for the ``topofuzz langgraph-import`` CLI subcommand.

Covers:
  * Success path: a saved CustomWorkflow appears under
    ``TOPOFUZZ_HOME/custom-workflows/``.
  * Branching graphs reject with exit 2 and an actionable stderr.
  * Rubric flag plumbing: --rubric / --rubric-file / both / neither.
  * --llm-config NODE=MODEL parser: happy path, malformed input,
    duplicate node.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

# `topofuzz langgraph-import` loads topofuzz.ui, which imports the
# FastAPI app eagerly; without the `ui` extra skip instead of failing.
pytest.importorskip("fastapi")

from typer.testing import CliRunner  # noqa: E402

from topofuzz.cli import app  # noqa: E402


# Rubric long enough to satisfy MIN_RUBRIC_LEN (50 chars).
_RUBRIC = (
    "Approve when the final agent recommends approval, deny when it "
    "recommends against. Borderline cases are common."
)


def _linear_fixture(path: Path) -> Path:
    """Write a 3-node linear LangGraph fixture without LLM_CONFIGS."""
    path.write_text(
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
                g.add_node("reviewer", _noop)
                g.add_node("decider", _noop)
                g.add_edge(START, "intake")
                g.add_edge("intake", "reviewer")
                g.add_edge("reviewer", "decider")
                g.add_edge("decider", END)
                return g.compile()
            """
        ),
        encoding="utf-8",
    )
    return path


def _branchy_fixture(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """
            from typing import TypedDict
            from langgraph.graph import END, START, StateGraph

            class _S(TypedDict, total=False):
                notes: str

            def _noop(s): return s

            def build_graph():
                g = StateGraph(_S)
                g.add_node("router", _noop)
                g.add_node("yes", _noop)
                g.add_node("no", _noop)
                g.add_edge(START, "router")
                g.add_conditional_edges(
                    "router", lambda s: "yes",
                    {"yes": "yes", "no": "no"},
                )
                g.add_edge("yes", END)
                g.add_edge("no", END)
                return g.compile()
            """
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def topofuzz_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect ~/.topofuzz/ to a tmp dir for the test."""
    monkeypatch.setenv("TOPOFUZZ_HOME", str(tmp_path))
    return tmp_path


class TestSuccessPath:
    def test_imports_linear_graph_and_writes_meta(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "Smoke imported",
                "--rubric",
                _RUBRIC,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Imported 'Smoke imported'" in result.output
        # 3 agents, ASCII arrow (cp1252 safe).
        assert "intake -> reviewer -> decider" in result.output

        workflows_dir = topofuzz_home / "custom-workflows"
        saved = list(workflows_dir.iterdir())
        assert len(saved) == 1
        meta = json.loads((saved[0] / "meta.json").read_text(encoding="utf-8"))
        assert meta["label"] == "Smoke imported"
        assert [a["agent_id"] for a in meta["agents"]] == [
            "intake",
            "reviewer",
            "decider",
        ]
        # Default fallback model for every node when no LLM_CONFIGS.
        assert {a["model"] for a in meta["agents"]} == {
            "google/gemini-2.5-flash"
        }
        assert meta["verdict_agent_id"] == "decider"

    def test_llm_config_overrides_show_up_in_meta(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "Mixed models",
                "--rubric",
                _RUBRIC,
                "--llm-config",
                "reviewer=google/gemini-2.5-pro",
                "--llm-config",
                "decider=google/gemini-2.5-pro",
            ],
        )
        assert result.exit_code == 0, result.output

        saved = list((topofuzz_home / "custom-workflows").iterdir())
        meta = json.loads((saved[0] / "meta.json").read_text(encoding="utf-8"))
        models = {a["agent_id"]: a["model"] for a in meta["agents"]}
        assert models["intake"] == "google/gemini-2.5-flash"  # fallback
        assert models["reviewer"] == "google/gemini-2.5-pro"  # override
        assert models["decider"] == "google/gemini-2.5-pro"  # override

    def test_rubric_file_reads_text_from_disk(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        rubric_path = tmp_path / "rubric.txt"
        rubric_path.write_text(_RUBRIC, encoding="utf-8")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "From file",
                "--rubric-file",
                str(rubric_path),
            ],
        )
        assert result.exit_code == 0, result.output
        saved = list((topofuzz_home / "custom-workflows").iterdir())
        meta = json.loads((saved[0] / "meta.json").read_text(encoding="utf-8"))
        assert meta["rubric_paragraph"] == _RUBRIC


class TestRubricFlagPlumbing:
    def test_neither_rubric_flag_is_usage_error(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "x",
            ],
        )
        assert result.exit_code != 0
        # Typer CliRunner mixes stderr into output.
        assert "rubric" in result.output.lower()

    def test_both_rubric_flags_is_usage_error(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        rubric_path = tmp_path / "rubric.txt"
        rubric_path.write_text(_RUBRIC, encoding="utf-8")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "x",
                "--rubric",
                _RUBRIC,
                "--rubric-file",
                str(rubric_path),
            ],
        )
        assert result.exit_code != 0
        assert "not both" in result.output.lower()


class TestLlmConfigParsing:
    def test_no_equals_sign_is_usage_error(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "x",
                "--rubric",
                _RUBRIC,
                "--llm-config",
                "no_equals_sign",
            ],
        )
        assert result.exit_code != 0
        assert "NODE=MODEL" in result.output

    def test_empty_side_is_usage_error(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "x",
                "--rubric",
                _RUBRIC,
                "--llm-config",
                "intake=",  # empty model
            ],
        )
        assert result.exit_code != 0

    def test_duplicate_node_is_usage_error(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "x",
                "--rubric",
                _RUBRIC,
                "--llm-config",
                "reviewer=google/gemini-2.5-flash",
                "--llm-config",
                "reviewer=google/gemini-2.5-pro",
            ],
        )
        assert result.exit_code != 0
        assert "more than once" in result.output.lower()


class TestBranchingRejected:
    def test_branchy_graph_rejected_with_actionable_message(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _branchy_fixture(tmp_path / "branchy.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "Branchy",
                "--rubric",
                _RUBRIC,
            ],
        )
        assert result.exit_code != 0
        # Importer error message must surface to stderr (the CLI prefixes
        # it with the command name).
        assert "langgraph-import" in result.output
        assert "branches" in result.output


class TestModelAllowlist:
    def test_unknown_override_model_rejected(
        self, tmp_path: Path, topofuzz_home: Path
    ) -> None:
        fixture = _linear_fixture(tmp_path / "linear.py")
        result = CliRunner().invoke(
            app,
            [
                "langgraph-import",
                f"{fixture}:build_graph",
                "--label",
                "Bad model",
                "--rubric",
                _RUBRIC,
                "--llm-config",
                "reviewer=openai/gpt-4o",  # not in allow-list
            ],
        )
        assert result.exit_code != 0
        assert "allow" in result.output.lower() or "validation" in result.output.lower()
