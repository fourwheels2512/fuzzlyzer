"""lending_4role: 4-agent linear loan-decision chain.

Canonical demo fixture for v0.5: ``intake -> risk -> policy -> decision``.
Referenced by BLUEPRINT_v5plus.md §10a (worst-case viewer) to illustrate
ordering-induced decision flips. Node callables are stubs — real runs
swap them for LLM calls.

The fixture also exposes ``LLM_CONFIGS``: a realistic per-node mapping
suitable for ``adapter.from_langgraph(..., llm_configs=LLM_CONFIGS)``.
Two nodes share each model so the BLUEPRINT §6 ``role_swap`` generator
fires (pairs: ``intake/decision`` on gemini-flash, ``risk/policy`` on
gemini-pro), and all four start at ``temperature=0.0`` so
``decoding_param_swap`` produces N=4 variants.
"""

from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


class LendingState(TypedDict, total=False):
    application: dict
    intake_notes: str
    risk_notes: str
    policy_notes: str
    decision_notes: str
    decision: str


AGENT_IDS: tuple[str, ...] = ("intake", "risk", "policy", "decision")

# Two model groups so BLUEPRINT §6 role_swap finds at least one matching pair
# per group (intake↔decision on gemini-flash; risk↔policy on gemini-pro).
# All four start at temperature=0.0 so decoding_param_swap produces N variants
# for N nodes per §6.
LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0),
    "risk": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),
    "policy": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0),
    "decision": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0),
}


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
