"""single_agent: 1-node negative-control fixture for BLUEPRINT §14 step 15a.

A provably-stable baseline: one agent, one role, no role overlap, so §6
variant generators have nothing to permute. The §9
``variance_across_orderings`` headline should round to 0 on this
fixture; if it doesn't, topofuzz is fabricating instability and step 22
(CEO outreach) must not ship.

Step 22 test 1 asserts:

    voo.point ≤ 0.01 AND voo.ci_upper ≤ 0.05

against this fixture. Eval set is a ≤10-row subset of
``lending_eval.csv`` so the negative control rides the same rubric as
the demo run (apples-to-apples — the recipient can compare directly).

The single node uses ``google/gemini-2.5-flash`` to match
``lending_4role.py``'s cheap-side model — same provider, same cost
basin, same dry-run determinism.

Mirrors ``lending_4role.py``'s shape: a TypedDict state, a stub agent
callable, and ``build_graph()`` returning a compiled LangGraph. Public
callable is ``build_graph`` so ``topofuzz run single_agent.py:build_graph``
resolves identically to the lending demo.
"""

from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


class SoloState(TypedDict, total=False):
    application: dict
    solo_notes: str
    decision: str


AGENT_IDS: tuple[str, ...] = ("solo",)

# One vendor, one model. With a single node, §6's order_swap /
# role_swap / decoding_param_swap all yield the topology itself — no
# semantically-equivalent permutations exist, which is precisely the
# negative-control property we want.
LLM_CONFIGS: dict[str, LLMConfig] = {
    "solo": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0),
}


def _make_stub_agent(name: str):
    def agent(state: SoloState) -> SoloState:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the single-agent fixture."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(SoloState)
    g.add_node(AGENT_IDS[0], _make_stub_agent(AGENT_IDS[0]))
    g.add_edge(START, AGENT_IDS[0])
    g.add_edge(AGENT_IDS[0], END)
    return g.compile()
