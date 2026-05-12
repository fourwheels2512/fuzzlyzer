"""lending_4role: 4-agent linear loan-decision chain.

Canonical demo fixture for v0.5: ``intake -> risk -> policy -> decision``.
Referenced by BLUEPRINT_v5plus.md §10a (worst-case viewer) to illustrate
ordering-induced decision flips. Node callables are stubs — real runs
swap them for LLM calls.
"""

from __future__ import annotations

from typing import TypedDict


class LendingState(TypedDict, total=False):
    application: dict
    intake_notes: str
    risk_notes: str
    policy_notes: str
    decision_notes: str
    decision: str


AGENT_IDS: tuple[str, ...] = ("intake", "risk", "policy", "decision")


def _make_stub_agent(name: str):
    def agent(state: LendingState) -> LendingState:
        # Dynamic key by design: every agent writes `<name>_notes`. All
        # four resulting keys are declared on LendingState above so this
        # is type-consistent with `total=False`.
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent lending chain."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(LendingState)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _make_stub_agent(agent_id))
    g.add_edge(START, AGENT_IDS[0])
    for src, dst in zip(AGENT_IDS, AGENT_IDS[1:]):
        g.add_edge(src, dst)
    g.add_edge(AGENT_IDS[-1], END)
    return g.compile()
