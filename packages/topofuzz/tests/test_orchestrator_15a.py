"""Step 15a orchestrator-side tests.

Covers the §8 + ground-truth + agent-vendors plumbing introduced by
BLUEPRINT §14 step 15a:

* ``_binarize_ground_truth`` — string → 0/1/None
* ``_agent_vendors_from_topology`` — IR → sorted vendor list
* ``_build_judge_independence_dict`` — None judge / single judge / jury-like
  judge → full §8 dict (or None when judge absent)
* ``write_manifest`` — persists ``eval_ground_truth`` + ``agent_vendors``
  parallel to existing ``eval_ids``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from topofuzz.ir import Edge, LLMConfig, Node, Topology
from topofuzz.orchestrator import (
    RunPlan,
    _agent_vendors_from_topology,
    _binarize_ground_truth,
    _build_judge_independence_dict,
    _vendor_from_model_id,
    write_manifest,
)


# ---------------------------------------------------------------------------
# _binarize_ground_truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("approve", 1),
        ("deny", 0),
        ("APPROVE", 1),
        ("  Deny ", 0),
        ("pass", 1),
        ("fail", 0),
        ("1", 1),
        ("0", 0),
        ("true", 1),
        ("false", 0),
        (1, 1),
        (0, 0),
        (True, 1),
        (False, 0),
        (None, None),
        ("unknown", None),
        ("yes", None),  # not in the map by design — let the user be explicit
        (2, None),  # out-of-range int → None
        (1.0, None),  # floats not accepted (avoid silent NaN paths)
        ({}, None),
    ],
)
def test_binarize_ground_truth(raw: Any, expected: int | None) -> None:
    assert _binarize_ground_truth(raw) == expected


# ---------------------------------------------------------------------------
# _vendor_from_model_id + _agent_vendors_from_topology
# ---------------------------------------------------------------------------


def test_vendor_from_model_id_normalizes():
    assert _vendor_from_model_id("google/gemini-2.5-pro") == "google"
    assert _vendor_from_model_id("  Anthropic /claude") == "anthropic"
    assert _vendor_from_model_id("openai/gpt-5") == "openai"


def test_vendor_from_model_id_returns_none_without_slash():
    assert _vendor_from_model_id("jury_of_3:anthropic+google") is None
    assert _vendor_from_model_id("") is None
    assert _vendor_from_model_id(None) is None


def _make_topology(
    nodes: list[tuple[str, str | None]],
) -> Topology:
    """Build a minimal Topology where each node has ``(id, model_id)``.
    A model_id of ``None`` produces a node without an llm_config."""
    ir_nodes = []
    for nid, model in nodes:
        if model is None:
            ir_nodes.append(Node(id=nid, role="agent"))
        else:
            ir_nodes.append(
                Node(
                    id=nid,
                    role="agent",
                    llm_config=LLMConfig(model=model, temperature=0.0),
                )
            )
    edges = [
        Edge(from_id=ir_nodes[i].id, to_id=ir_nodes[i + 1].id)
        for i in range(len(ir_nodes) - 1)
    ]
    return Topology(
        name="t",
        nodes=tuple(ir_nodes),
        edges=tuple(edges),
        entrypoint=ir_nodes[0].id,
        terminal=(ir_nodes[-1].id,),
    )


def test_agent_vendors_sorted_unique():
    topo = _make_topology([
        ("a", "google/gemini-2.5-flash"),
        ("b", "google/gemini-2.5-pro"),
        ("c", "anthropic/claude"),
    ])
    assert _agent_vendors_from_topology(topo) == ["anthropic", "google"]


def test_agent_vendors_skips_no_llm_config():
    topo = _make_topology([
        ("a", "google/gemini"),
        ("b", None),  # router or human — no llm_config
    ])
    assert _agent_vendors_from_topology(topo) == ["google"]


def test_agent_vendors_skips_malformed_model_id():
    topo = _make_topology([
        ("a", "google/gemini"),
        ("b", "no-slash-here"),  # malformed → skipped
    ])
    assert _agent_vendors_from_topology(topo) == ["google"]


# ---------------------------------------------------------------------------
# _build_judge_independence_dict
# ---------------------------------------------------------------------------


def test_build_judge_independence_returns_none_for_no_judge():
    topo = _make_topology([("a", "google/gemini")])
    assert _build_judge_independence_dict(None, topo) is None


class _FakeSingleJudge:
    """Stand-in for fuzzeval.judge.SingleJudge — exposes only ``.model``."""
    def __init__(self, model: str) -> None:
        self.model = model


def test_build_judge_independence_single_judge_overlap():
    topo = _make_topology([("a", "google/gemini-2.5-flash")])
    judge = _FakeSingleJudge("google/gemini-2.5-pro")
    out = _build_judge_independence_dict(judge, topo)
    assert out is not None
    assert out["judge_protocol"] == "single"
    assert out["judges"] == [
        {"vendor": "google", "model_id": "google/gemini-2.5-pro"}
    ]
    assert out["vendor_overlap"] == ["google"]
    assert out["vendor_disjoint"] is False
    assert "google" in out["self_preference_warning"]
    # SingleJudge has no vendor-call-level failure tracking.
    assert out["vendor_call_failure_rate"] == 0.0


def test_build_judge_independence_single_judge_disjoint():
    """Agent on Google, judge on Anthropic → no overlap, no warning."""
    topo = _make_topology([("a", "google/gemini")])
    judge = _FakeSingleJudge("anthropic/claude")
    out = _build_judge_independence_dict(judge, topo)
    assert out["vendor_overlap"] == []
    assert out["vendor_disjoint"] is True
    assert out["self_preference_warning"] is None


class _FakeJuryJudge:
    """Stand-in for fuzzeval.judge.JuryJudge — exposes ``vendors``,
    ``_judges`` dict, ``judge_failure_rate``, ``judge_independence``."""
    def __init__(self, vendor_models: dict[str, str], fail_rate: float = 0.0) -> None:
        self.vendors = tuple(vendor_models)
        self._judges = {v: _FakeSingleJudge(m) for v, m in vendor_models.items()}
        self.judge_failure_rate = fail_rate
        self.judge_independence = {"vendor_overlap": sorted(self.vendors)}


def test_build_judge_independence_jury_overlap():
    topo = _make_topology([
        ("a", "anthropic/claude"),
        ("b", "google/gemini"),
    ])
    judge = _FakeJuryJudge(
        {
            "anthropic": "anthropic/claude-opus",
            "openai": "openai/gpt-5",
            "google": "google/gemini-pro",
        },
        fail_rate=0.05,
    )
    out = _build_judge_independence_dict(judge, topo)
    assert out["judge_protocol"] == "jury_of_3"
    assert out["judges"] == [
        {"vendor": "anthropic", "model_id": "anthropic/claude-opus"},
        {"vendor": "openai", "model_id": "openai/gpt-5"},
        {"vendor": "google", "model_id": "google/gemini-pro"},
    ]
    # Agents: anthropic + google. Judges: anthropic + openai + google.
    # Overlap: anthropic + google.
    assert out["vendor_overlap"] == ["anthropic", "google"]
    assert out["vendor_disjoint"] is False
    assert out["self_preference_warning"] is not None
    assert "anthropic" in out["self_preference_warning"]
    assert out["vendor_call_failure_rate"] == 0.05


def test_build_judge_independence_jury_disjoint_no_agents_with_llm():
    """Topology has no llm_configs → agent_vendors is empty → vendor_disjoint."""
    topo = _make_topology([("a", None)])
    judge = _FakeJuryJudge({"anthropic": "anthropic/c", "google": "google/g",
                            "openai": "openai/o"})
    out = _build_judge_independence_dict(judge, topo)
    assert out["vendor_overlap"] == []
    assert out["vendor_disjoint"] is True
    assert out["self_preference_warning"] is None


class _FakeSingleJudgeWithJudgeIndependence:
    """Audit RT-M3: a SingleJudge that gained a ``judge_independence``
    property — only one of the three jury attributes. Must NOT be
    classified as jury_of_3."""
    def __init__(self, model: str) -> None:
        self.model = model
        self.judge_independence = {"vendor_overlap": []}  # stub


def test_build_judge_independence_partial_jury_attrs_classified_single():
    """A judge with judge_independence but no vendors / _judges is a
    single judge with stubbed disclosure. The strict 3-attr check
    avoids misclassifying it as jury_of_3 and shipping an empty
    disclosure that hides self-preference risk."""
    topo = _make_topology([("a", "google/gemini")])
    judge = _FakeSingleJudgeWithJudgeIndependence("google/gemini-pro")
    out = _build_judge_independence_dict(judge, topo)
    assert out["judge_protocol"] == "single"
    # Single-judge path correctly populates the judges entry.
    assert out["judges"] == [
        {"vendor": "google", "model_id": "google/gemini-pro"}
    ]
    assert out["vendor_overlap"] == ["google"]


class _FakeSingleJudgeMissingModel:
    """SingleJudge with no model attr — represents a buggy stub or a
    test mock that forgot to set ``model``."""
    pass


def test_build_judge_independence_single_judge_missing_model_dispatches_unknown():
    """Audit RT-M4: a single judge whose model is None / empty must
    NOT silently emit empty-string vendor / model_id fields. The
    disclosure should explicitly flag the unintrospectable state and
    assume self-preference risk (the conservative posture for an
    auditor)."""
    topo = _make_topology([("a", "google/gemini")])
    judge = _FakeSingleJudgeMissingModel()
    out = _build_judge_independence_dict(judge, topo)
    assert out["judge_protocol"] == "single"
    assert out["judges"] == [{"vendor": "unknown", "model_id": "unknown"}]
    # vendor_disjoint=False under the conservative posture: we don't
    # KNOW the judge is disjoint from agents; assume not.
    assert out["vendor_disjoint"] is False
    assert "not introspectable" in (out["self_preference_warning"] or "")


def test_build_judge_independence_single_judge_empty_model_string():
    """An empty-string model id triggers the same unknown-disclosure path."""
    topo = _make_topology([("a", "google/gemini")])
    judge = _FakeSingleJudge("")
    out = _build_judge_independence_dict(judge, topo)
    assert out["judges"] == [{"vendor": "unknown", "model_id": "unknown"}]


def test_binarize_ground_truth_warns_on_unknown_string(caplog):
    """Audit RT-M5: a non-None ``ground_truth_decision`` string that
    isn't in the binary map must emit a WARNING so a label-mapping
    bug doesn't silently degrade signed BA-drop to ``None``."""
    import logging
    with caplog.at_level(logging.WARNING):
        result = _binarize_ground_truth("recommend")
    assert result is None
    assert any(
        "not in known binary map" in record.message for record in caplog.records
    ), f"expected WARNING log for unrecognised string; got {caplog.records!r}"


# ---------------------------------------------------------------------------
# write_manifest persists new fields
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path, eval_rows: list[dict]) -> RunPlan:
    topo = _make_topology([
        ("a", "google/gemini-flash"),
        ("b", "anthropic/claude"),
    ])
    return RunPlan(
        topology=topo,
        variants=(topo,),
        eval_rows=tuple(eval_rows),
        run_id="rmanifest",
        run_dir=tmp_path / "run",
        cache_dir=None,
        max_cost_usd=None,
        judge_model="single",
        rubric_hash="rh",
    )


def test_write_manifest_persists_eval_ground_truth(tmp_path: Path):
    plan = _make_plan(
        tmp_path,
        eval_rows=[
            {"id": "e1", "input": "x", "ground_truth_decision": "approve"},
            {"id": "e2", "input": "y", "ground_truth_decision": "deny"},
            {"id": "e3", "input": "z"},  # no ground_truth_decision → null
        ],
    )
    path = write_manifest(plan, cli_args={})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["eval_ids"] == ["e1", "e2", "e3"]
    assert manifest["eval_ground_truth"] == [1, 0, None]


def test_write_manifest_persists_agent_vendors(tmp_path: Path):
    plan = _make_plan(tmp_path, eval_rows=[{"id": "e1", "input": "x"}])
    path = write_manifest(plan, cli_args={})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["agent_vendors"] == ["anthropic", "google"]


def test_write_manifest_eval_ground_truth_alignment(tmp_path: Path):
    """eval_ground_truth must align positionally with eval_ids."""
    plan = _make_plan(
        tmp_path,
        eval_rows=[
            {"id": "e2", "input": "x", "ground_truth_decision": "deny"},
            {"id": "e1", "input": "y", "ground_truth_decision": "approve"},
        ],
    )
    path = write_manifest(plan, cli_args={})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["eval_ids"] == ["e2", "e1"]
    assert manifest["eval_ground_truth"] == [0, 1]


# ---------------------------------------------------------------------------
# single_agent fixture sanity
# ---------------------------------------------------------------------------


def test_single_agent_fixture_builds():
    """Step 15a (a): the negative-control fixture imports + compiles."""
    from topofuzz.fixtures.single_agent import (
        AGENT_IDS,
        LLM_CONFIGS,
        build_graph,
    )
    assert AGENT_IDS == ("solo",)
    assert "solo" in LLM_CONFIGS
    graph = build_graph()
    # LangGraph compiles to something with a ``get_graph`` accessor; we
    # only need to verify a callable graph object came back.
    assert graph is not None
