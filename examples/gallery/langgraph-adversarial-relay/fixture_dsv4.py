"""langgraph-adversarial-relay-dsv4: identical to fixture.py, skeptic on DeepSeek V4 Pro.

The ONLY difference from ``fixture.py`` is the ``downside_risk_underwriter``
model: ``deepseek/deepseek-v4-pro`` instead of ``google/gemini-2.5-pro``.
Everything else — topology, role names, the other three Flash nodes,
temperature, max_tokens — is held constant so the run is a clean A/B on
the skeptic-model lever alone.

Why DeepSeek V4 Pro for the skeptic
-----------------------------------
Two reasons, both empirical:

1. **Reasoning strength.** Published 2026 benchmarks put V4 Pro
   materially ahead of Gemini 2.5 Pro on the reasoning axes that matter
   for an adversarial-underwriting role — Humanity's Last Exam 37.7% vs
   21.6%, Math Arena Apex 38.3% vs 0.5%, long-context reasoning (MRCR
   v2, 1M) 83.5% vs 16.4%, GPQA Diamond 90.1% vs ~84%. The skeptic's
   job is to construct the strongest possible case against the loan; a
   model that reasons harder produces a harder-to-dismiss memo, which
   is the exact lever this fixture runs on.

2. **Vendor-disjoint credibility.** The all-Gemini fixture trips the
   ``judge_independence.self_preference_warning`` flag ("Agents and
   judges share vendor(s): google. Same-vendor scoring can inflate
   apparent agreement"). Mixing DeepSeek agent + Gemini judge breaks
   that flag — a stronger story for a buyer who would otherwise discount
   the result as same-vendor inflated.

Cost side-effect: V4 Pro output is ~$3.48/M vs Gemini Pro's $10/M — the
skeptic node was the cost driver, so this swap roughly halves the real
spend per run, even though topofuzz's cost estimator will wildly
overshoot (its rate table has no DeepSeek entry → falls back to the
$40/M default, ~25× hot for V4). The ``max_tokens=4000`` per-call cap
is the real cost bound.
"""

from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


__topofuzz_name__ = "langgraph-adversarial-relay-dsv4"


class _LoanState(TypedDict, total=False):
    intake_parser_notes: str
    application_summarizer_notes: str
    downside_risk_underwriter_notes: str
    approval_decider_notes: str


AGENT_IDS: tuple[str, ...] = (
    "intake_parser",
    "application_summarizer",
    "downside_risk_underwriter",
    "approval_decider",
)


# Mixed flash/deepseek-v4-pro, temp=0.0, max_tokens=4000.
# downside_risk_underwriter is deepseek/deepseek-v4-pro (the swapped node
# — see module docstring for the empirical reasoning-benchmark
# justification); the other three remain gemini-2.5-flash.
# order_swap permutes the linear chain (the generator this fixture
# exercises); role_swap only finds same-model pairs (the 3 Flash nodes),
# acceptable since order_swap is the primary generator;
# decoding_param_swap produces N=4 variants from the uniform temp=0.0.
LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake_parser": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
    "application_summarizer": LLMConfig(
        model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000
    ),
    "downside_risk_underwriter": LLMConfig(
        model="deepseek/deepseek-v4-pro", temperature=0.0, max_tokens=4000
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
    """Return a compiled LangGraph for the 4-agent adversarial-relay chain (DSv4 skeptic)."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_LoanState)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _stub(agent_id))
    g.add_edge(START, AGENT_IDS[0])
    for src, dst in zip(AGENT_IDS, AGENT_IDS[1:]):
        g.add_edge(src, dst)
    g.add_edge(AGENT_IDS[-1], END)
    return g.compile()
