# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""langgraph-supervisor-research: 4-agent supervisor pattern.

Inspired by the LangGraph "hierarchical agent teams" tutorial
(https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/),
which succeeded the older "agent_supervisor" tutorial on the docs site.
A supervisor LLM dispatches to one of three workers (researcher,
reviewer, critic) to produce a publication-readiness verdict on a draft
article. The reviewer role is review-framed on purpose: the runner's
default prompt hard-prepends "You are the '{role}' agent", so a role
named "writer" would commission a *draft* instead of a *review* (caught
by the step-20a gallery audit — see BLUEPRINT v5plus dogfooding note).

Topology shape: supervisor + add_conditional_edges with an explicit
path_map to the three workers {researcher, reviewer, critic}. Only
``critic`` wires to END, so the topology is SINGLE-TERMINAL and the
runner unambiguously judges ``critic``'s output as the verdict. (An
earlier revision also mapped ``FINISH`` -> END, which made ``supervisor``
a second terminal node; the runner judges the declared terminal node, so
a stray second terminal is a real measurement hazard — caught by the
step-20a gallery audit.) The conditional edges define the IR shape that
variant generators see; the topofuzz runner walks ``variant_ir.nodes``
linearly at execution time (see ``topofuzz.runner.run_variant``), so
every node runs once per cell regardless of routing.

All four nodes share gemini-2.5-flash to keep the gallery run cost
modest while still giving BLUEPRINT v5plus §6 ``role_swap`` a
maximal number of swappable pairs (since role_swap requires matching
models). Observed cost on the v0.5 reference run: ~$1.25 per entry at
``--n 8 --variants 4`` (BLUEPRINT 19a's ``<$1`` target was missed by
~25%; the cost estimator under-estimates real Gemini Flash output cost
by ~3-5x). All four start at temperature=0.0 so ``decoding_param_swap``
produces N=4 variants.
"""

from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


class _ArticleState(TypedDict, total=False):
    article: str
    supervisor_plan: str
    research_notes: str
    review_notes: str
    critic_verdict: str
    decision: str


AGENT_IDS: tuple[str, ...] = ("supervisor", "researcher", "reviewer", "critic")

LLM_CONFIGS: dict[str, LLMConfig] = {
    "supervisor": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "researcher": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "reviewer": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "critic": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
}


def _stub(name: str):
    def agent(state: _ArticleState) -> _ArticleState:
        # Stubs are placeholders. The topofuzz runner replaces them with
        # litellm.acompletion calls keyed off LLM_CONFIGS at run time
        # (see packages/topofuzz/src/topofuzz/runner.py:_call_llm).
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def _route_after_supervisor(state: _ArticleState) -> str:
    # Stub router. Always returns "researcher" so the compiled graph has
    # a deterministic execution path under LangGraph's own .invoke() —
    # the topofuzz runner ignores this and walks IR-order.
    return "researcher"


def build_graph():
    """Return a compiled LangGraph for the 4-agent supervisor pattern."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_ArticleState)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _stub(agent_id))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "researcher": "researcher",
            "reviewer": "reviewer",
            "critic": "critic",
        },
    )
    g.add_edge("researcher", "reviewer")
    g.add_edge("reviewer", "critic")
    g.add_edge("critic", END)
    return g.compile()
