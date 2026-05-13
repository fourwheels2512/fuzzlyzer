"""Tests for ``topofuzz.loaders`` (step 6 fixture + eval I/O)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from topofuzz.loaders import load_eval_rows, load_topology  # noqa: E402


# ---------------------------------------------------------------------------
# load_topology
# ---------------------------------------------------------------------------


_FIXTURE_BODY = '''
from langgraph.graph import END, START, StateGraph
from topofuzz.ir import LLMConfig
from typing import TypedDict


class _S(TypedDict, total=False):
    x: str


def _agent(state):
    return {**state, "x": "ok"}


def build_graph():
    g = StateGraph(_S)
    g.add_node("a", _agent)
    g.add_node("b", _agent)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile()


LLM_CONFIGS = {
    "a": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),
    "b": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),
}
'''


def _write_fixture(tmp_path: Path, body: str = _FIXTURE_BODY) -> Path:
    f = tmp_path / "fx.py"
    f.write_text(body, encoding="utf-8")
    return f


def test_load_topology_parses_callable_attr(tmp_path):
    f = _write_fixture(tmp_path)
    topo = load_topology(f"{f}:build_graph")
    assert {n.id for n in topo.nodes} == {"a", "b"}


def test_load_topology_uses_llm_configs_dict(tmp_path):
    f = _write_fixture(tmp_path)
    topo = load_topology(f"{f}:build_graph")
    for node in topo.nodes:
        assert node.llm_config is not None
        assert node.llm_config.model == "google/gemini-2.5-pro"


def test_load_topology_missing_colon_raises(tmp_path):
    f = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="missing ':'"):
        load_topology(str(f))


def test_load_topology_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_topology(f"{tmp_path / 'nope.py'}:graph")


def test_load_topology_bad_attr_raises(tmp_path):
    f = _write_fixture(tmp_path)
    with pytest.raises(AttributeError):
        load_topology(f"{f}:not_an_attr")


_FIXTURE_BAD_SHAPE = '''
from langgraph.graph import END, START, StateGraph
from typing import TypedDict


class _S(TypedDict, total=False):
    x: str


def _agent(state):
    return state


def build_graph():
    g = StateGraph(_S)
    g.add_node("a", _agent)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    return g.compile()


LLM_CONFIGS = "this is wrong"
'''


_FIXTURE_BAD_VALUES = '''
from langgraph.graph import END, START, StateGraph
from typing import TypedDict


class _S(TypedDict, total=False):
    x: str


def _agent(state):
    return state


def build_graph():
    g = StateGraph(_S)
    g.add_node("a", _agent)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    return g.compile()


LLM_CONFIGS = {"a": "not an LLMConfig"}
'''


def test_load_topology_rejects_bad_llm_configs_shape(tmp_path):
    f = _write_fixture(tmp_path, _FIXTURE_BAD_SHAPE)
    with pytest.raises(TypeError, match="LLM_CONFIGS"):
        load_topology(f"{f}:build_graph")


def test_load_topology_rejects_bad_llm_configs_values(tmp_path):
    f = _write_fixture(tmp_path, _FIXTURE_BAD_VALUES)
    with pytest.raises(TypeError, match="LLM_CONFIGS"):
        load_topology(f"{f}:build_graph")


# ---------------------------------------------------------------------------
# load_eval_rows — JSONL
# ---------------------------------------------------------------------------


def test_load_eval_rows_jsonl_happy(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"id": "a", "input": "x"}\n'
        '{"id": "b", "input": "y", "expected": "B"}\n',
        encoding="utf-8",
    )
    rows = load_eval_rows(p)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[1]["expected"] == "B"


def test_load_eval_rows_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"id": "a", "input": "x"}\n'
        "\n"
        '{"id": "b", "input": "y"}\n'
        "   \n",
        encoding="utf-8",
    )
    assert len(load_eval_rows(p)) == 2


def test_load_eval_rows_jsonl_bad_json_reports_line(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"id": "a", "input": "x"}\n'
        "not-json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":2:"):
        load_eval_rows(p)


def test_load_eval_rows_jsonl_non_object_row_raises(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_eval_rows(p)


# ---------------------------------------------------------------------------
# load_eval_rows — CSV
# ---------------------------------------------------------------------------


def test_load_eval_rows_csv_happy(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("id,input\na,x\nb,y\n", encoding="utf-8")
    rows = load_eval_rows(p)
    assert [r["id"] for r in rows] == ["a", "b"]


def test_load_eval_rows_csv_no_header_raises(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_eval_rows(p)


# ---------------------------------------------------------------------------
# load_eval_rows — validation
# ---------------------------------------------------------------------------


def test_load_eval_rows_missing_id_raises(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"id": "a", "input": "x"}\n'
        '{"input": "y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row 2"):
        load_eval_rows(p)


def test_load_eval_rows_missing_input_raises(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'input'"):
        load_eval_rows(p)


def test_load_eval_rows_duplicate_id_raises(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"id": "a", "input": "x"}\n'
        '{"id": "a", "input": "y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_eval_rows(p)


def test_load_eval_rows_unsupported_format_raises(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("id: a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported eval format"):
        load_eval_rows(p)


def test_load_eval_rows_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_rows(tmp_path / "nope.jsonl")


def test_load_eval_rows_csv_strips_utf8_bom(tmp_path):
    """Excel-exported UTF-8 CSVs carry a BOM. Without ``utf-8-sig`` the
    first header would be ``\\ufeffid`` and every row's ``id`` lookup
    would fail with the confusing "got None" message (audit M-7)."""
    p = tmp_path / "e.csv"
    p.write_bytes(b"\xef\xbb\xbfid,input\na,x\nb,y\n")
    rows = load_eval_rows(p)
    assert [r["id"] for r in rows] == ["a", "b"]


def test_load_eval_rows_csv_strips_header_whitespace(tmp_path):
    """A CSV header like ``id, input`` (one space after the comma —
    extremely common from spreadsheet exports) used to produce a key
    `` input`` that the validator reported as missing. The loader now
    strips header whitespace (audit M4)."""
    p = tmp_path / "e.csv"
    p.write_text("id, input\na,x\nb,y\n", encoding="utf-8")
    rows = load_eval_rows(p)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["input"] == "x"


# ---------------------------------------------------------------------------
# Fixture mtime-aware caching (audit M-6)
# ---------------------------------------------------------------------------


def test_load_topology_picks_up_edited_fixture(tmp_path):
    """Re-importing the same path after the file changes must return
    the new content. The previous path-only cache key returned stale
    modules (audit M-6).
    """
    import os

    f = _write_fixture(tmp_path)
    topo1 = load_topology(f"{f}:build_graph")
    assert "intake" not in {n.id for n in topo1.nodes}

    # Mutate the file: add a third node. Advance mtime explicitly so
    # the test doesn't rely on filesystem resolution being sub-second.
    edited = _FIXTURE_BODY.replace(
        'g.add_node("b", _agent)\n    g.add_edge(START, "a")',
        'g.add_node("b", _agent)\n    g.add_node("c", _agent)\n    g.add_edge(START, "a")',
    ).replace(
        'g.add_edge("a", "b")\n    g.add_edge("b", END)',
        'g.add_edge("a", "b")\n    g.add_edge("b", "c")\n    g.add_edge("c", END)',
    ).replace(
        '"b": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),\n}',
        '"b": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),\n'
        '    "c": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),\n}',
    )
    f.write_text(edited, encoding="utf-8")
    now = os.stat(f).st_mtime + 1.0  # bump mtime to be safe
    os.utime(f, (now, now))

    topo2 = load_topology(f"{f}:build_graph")
    assert {n.id for n in topo2.nodes} == {"a", "b", "c"}
