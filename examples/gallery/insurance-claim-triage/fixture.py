"""insurance-claim-triage: 4-agent skeptic-in-the-middle for health-insurance claims.

Same mechanism as langgraph-adversarial-relay, applied to a different
real-world domain. The skeptic node is renamed to ``denial_advocate`` —
its role string is what makes the chain produce an adverse memo that
drags the decider from APPROVE to DENY in canonical order.

Topology (strictly linear, single terminal):
  claim_intake -> coverage_summarizer -> denial_advocate -> claim_decider

Each row in eval.csv is a real-world-shaped health-insurance claim. The
raw facts are written to read as borderline-approvable. The skeptic's
adverse memo (driven entirely by its role name) is what tips the decider
to DENY in canonical order; with the skeptic out of the way (order_swap
puts claim_decider first), the decider keeps its APPROVE-on-facts
default. The order-induced flip is the fragility signal.
"""

from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


__topofuzz_name__ = "insurance-claim-triage"


class _ClaimState(TypedDict, total=False):
    claim_intake_notes: str
    coverage_summarizer_notes: str
    denial_advocate_notes: str
    claim_decider_notes: str


AGENT_IDS: tuple[str, ...] = (
    "claim_intake",
    "coverage_summarizer",
    "denial_advocate",
    "claim_decider",
)


# Pro on the skeptic for the same reason as adversarial-relay: a more
# articulate adverse memo is the only lever inside the no-custom-prompt
# constraint. The other three on Flash for cost.
LLM_CONFIGS: dict[str, LLMConfig] = {
    "claim_intake": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
    "coverage_summarizer": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
    "denial_advocate": LLMConfig(
        model="google/gemini-2.5-pro", temperature=0.0, max_tokens=4000
    ),
    "claim_decider": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
}


def _stub(name: str):
    def agent(state: _ClaimState) -> _ClaimState:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent claim-triage chain."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_ClaimState)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _stub(agent_id))
    g.add_edge(START, AGENT_IDS[0])
    for src, dst in zip(AGENT_IDS, AGENT_IDS[1:]):
        g.add_edge(src, dst)
    g.add_edge(AGENT_IDS[-1], END)
    return g.compile()
