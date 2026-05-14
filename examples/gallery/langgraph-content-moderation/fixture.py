"""langgraph-content-moderation: 4-agent linear moderation pipeline.

Inspired by the LangGraph multi-agent pipeline patterns for content
moderation: extract claims → classify category → policy check → final
decision. Each agent passes notes forward to the next; the final
decision is read by the judge.

All four nodes share gemini-2.5-flash to keep the gallery run cost
modest while still giving BLUEPRINT v5plus §6 ``role_swap`` a maximal
number of swappable pairs (since role_swap requires matching models).
Observed cost on the v0.5 reference run (which was discarded from the
gallery on the curation gate): ~$0.79 at ``--n 8 --variants 4``. All
four start at temperature=0.0 so ``decoding_param_swap`` produces N=4
variants.
"""

from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


class _PostState(TypedDict, total=False):
    post: str
    extract_notes: str
    classify_notes: str
    policy_check_notes: str
    decision_notes: str


AGENT_IDS: tuple[str, ...] = ("extract", "classify", "policy_check", "decision")

LLM_CONFIGS: dict[str, LLMConfig] = {
    "extract": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=1500),
    "classify": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=1500),
    "policy_check": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=1500),
    "decision": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=1500),
}


def _stub(name: str):
    def agent(state: _PostState) -> _PostState:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent content-moderation chain."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_PostState)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _stub(agent_id))
    g.add_edge(START, AGENT_IDS[0])
    for src, dst in zip(AGENT_IDS, AGENT_IDS[1:]):
        g.add_edge(src, dst)
    g.add_edge(AGENT_IDS[-1], END)
    return g.compile()
