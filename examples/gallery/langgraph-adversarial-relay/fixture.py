# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""langgraph-adversarial-relay: 4-agent skeptic-in-the-middle loan chain.

A strictly linear 4-node loan-decision pipeline built to exhibit a
GENUINE order-induced verdict flip — the property the lending_4role and
supervisor-research gallery fixtures failed to show (both came back
DEGENERATE: verdict_flip=0).

Why those were degenerate, and why this one is not
--------------------------------------------------
In lending_4role / supervisor-research every agent could independently
re-derive the same verdict from ``eval_row['input']`` alone — the agents
were REDUNDANT, so ``order_swap`` permutations changed nothing. The
supervisor-research eval rows went further and *spelled the verdict out*
("Legal flagged risk of defamation claim"), so even a single agent
trivially concluded DENY.

This fixture's flip is driven by the PRESENCE vs ABSENCE of a relayed
adversarial framing, not by anything re-derivable from the input:

  * Every eval row is written so the RAW FACTS ALONE read as APPROVE —
    no hard disqualifier (no 35%+ DTI, no recent late, no employment
    gap). A decider seeing only the application leans APPROVE.
  * ``downside_risk_underwriter``'s ROLE NAME makes it editorialize
    toward caution — it surfaces the SOFT, framable risk latent in the
    application (short job tenure, sub-2yr self-employment, repayment
    tied to a young business, unseasoned bonus income, etc.).
  * Canonical order ``intake_parser -> application_summarizer ->
    downside_risk_underwriter -> approval_decider``: the decider's
    prompt includes the underwriter's adverse memo as a prior output ->
    decider is dragged from its APPROVE-on-facts default to DENY.
  * An ``order_swap`` permutation that places ``approval_decider`` before
    ``downside_risk_underwriter`` gives the decider only the raw
    application -> no adverse memo exists yet -> decider keeps its
    APPROVE-on-facts default.
  * The decider CANNOT reconstruct the adverse framing from the input,
    because the SOFT risk only becomes a denial when an agent whose ROLE
    is to argue the downside names it. The facts themselves do not force
    a denial.

The runner judges the DECLARED terminal (``approval_decider``) wherever
it ran in the permutation (runner.py ``_run_judge`` ~L754-823). When the
permutation runs approval_decider early, its APPROVE output is what the
judge scores -> a measurable flip vs the DENY-leaning canonical.

Three rows (bal_003, bal_007, bal_009) are deliberately so strong that
even the adverse underwriter has nothing material to surface — they hold
APPROVE in both orders and act as stabilizers, keeping the canonical
side off an "always DENY" degeneracy.

Topology shape
--------------
Strictly linear, SINGLE terminal:
  START -> intake_parser -> application_summarizer
        -> downside_risk_underwriter -> approval_decider -> END
Only ``approval_decider`` wires to END, so ``Topology.terminal`` is the
singleton ``("approval_decider",)`` and the judge target is unambiguous.

``downside_risk_underwriter`` runs on gemini-2.5-pro; the other three
nodes on gemini-2.5-flash. All four use temperature=0.0, max_tokens=4000.
The Pro underwriter is deliberate: a pre-spend probe showed a Flash
``approval_decider`` already SEES each soft risk and consciously
discounts it, so a same-tier Flash memo would be weak leverage to flip
it. A more authoritative Pro-generated adverse memo is the one lever —
inside the "no custom prompt builder" constraint — that strengthens the
canonical DENY side. ``order_swap``, which permutes the linear execution
order, is the generator this fixture is built to exercise; the mixed
flash/pro setup means ``role_swap`` only pairs the three Flash nodes,
acceptable since it is not the primary generator here.

The node callables are stubs — the topofuzz runner replaces them with
``litellm.acompletion`` calls keyed off ``LLM_CONFIGS`` at run time (see
``packages/topofuzz/src/topofuzz/runner.py:_call_llm``). The stub bodies
exist only so ``build_graph()`` compiles and the adapter can derive the
node/edge/terminal IR.
"""

from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


__topofuzz_name__ = "langgraph-adversarial-relay"


class _LoanState(TypedDict, total=False):
    # Only the per-node ``_notes`` keys are written (by the stubs below).
    # The topofuzz runner ignores this schema entirely — it drives the
    # nodes via LLM_CONFIGS, not via LangGraph state — so this exists
    # only so build_graph() compiles.
    intake_parser_notes: str
    application_summarizer_notes: str
    downside_risk_underwriter_notes: str
    approval_decider_notes: str


# Strictly linear execution order. from_langgraph derives the IR node
# order and the single terminal from the edges below.
AGENT_IDS: tuple[str, ...] = (
    "intake_parser",
    "application_summarizer",
    "downside_risk_underwriter",
    "approval_decider",
)

# Mixed flash/pro, temp=0.0, max_tokens=4000. downside_risk_underwriter
# is gemini-2.5-pro (stronger adverse memo — see module docstring); the
# other three are gemini-2.5-flash. order_swap permutes the linear chain
# (the generator this fixture exercises); role_swap pairs the 3 Flash
# nodes; decoding_param_swap produces N=4 variants from the uniform
# temperature=0.0.
LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake_parser": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
    "application_summarizer": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
    "downside_risk_underwriter": LLMConfig(
        model="google/gemini-2.5-pro", temperature=0.0, max_tokens=4000
    ),
    "approval_decider": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
}


def _stub(name: str):
    def agent(state: _LoanState) -> _LoanState:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent adversarial-relay chain."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_LoanState)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _stub(agent_id))
    g.add_edge(START, AGENT_IDS[0])
    for src, dst in zip(AGENT_IDS, AGENT_IDS[1:]):
        g.add_edge(src, dst)
    g.add_edge(AGENT_IDS[-1], END)  # single terminal: approval_decider
    return g.compile()
