# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""langgraph-insurance-4role: 4-agent life-insurance underwriting pipeline
(EU AI Act Annex III point 5(c) — risk assessment & pricing in life/health
insurance).

Generality re-skin of the lending fixture, FIRST-WRITER-WINS variant — the
culprit lands at an EARLIER (non-last) pre-terminal position, which is what
lets the paper distinguish leave-one-out attribution from a trivial
"blame the last agent before the decision" heuristic.

Chain ``intake -> product -> underwriting -> decision``. The LENIENT pole
(``product``, which can issue a rated/substandard policy) runs at position 2;
the CONSERVATIVE pole (``underwriting``) runs at position 3. The ``decision``
agent adopts the EARLIER of the two conflicting recommendations
(first-writer-wins — the initial product determination anchors and is binding
unless escalated).

  - canonical intake->product->underwriting->decision: product (pos 2) is the
    first writer -> APPROVE (issue via a rated policy) on every run.
  - role_swap intake->underwriting->product->decision: underwriting (pos 2) is
    the first writer -> DENY (decline) on every run.

Deterministic by-construction APPROVE<->DENY flip whose culprit is the
position-2 ``product`` node — recovered by F4 leave-one-out, MISSED by a
blame-last-non-terminal baseline (which would wrongly accuse ``underwriting``
at pos 3). Verdict tokens stay APPROVE/DENY (APPROVE = issue policy).

WIRING NOTE: export the builder on the MODULE (never as a --prompt-builder CLI
flag); see the lending fixture for the rationale.
"""

from __future__ import annotations

from typing import Any, TypedDict

from topofuzz.ir import LLMConfig


AGENT_IDS: tuple[str, ...] = ("intake", "product", "underwriting", "decision")

# All-DeepSeek (V4 Flash) agents, temperature 0; see hiring fixture for the
# cost/instruction-following rationale. Judge vendor chosen at run time.
LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake": LLMConfig(model="deepseek/deepseek-chat", temperature=0.0, max_tokens=4000),
    "product": LLMConfig(model="deepseek/deepseek-chat", temperature=0.0, max_tokens=4000),
    "underwriting": LLMConfig(model="deepseek/deepseek-chat", temperature=0.0, max_tokens=4000),
    "decision": LLMConfig(model="deepseek/deepseek-chat", temperature=0.0, max_tokens=4000),
}


class _State(TypedDict, total=False):
    application: dict
    intake_notes: str
    product_notes: str
    underwriting_notes: str
    decision_notes: str
    decision: str


def _make_stub_agent(name: str):
    def agent(state: _State) -> _State:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent insurance chain."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_State)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _make_stub_agent(agent_id))
    g.add_edge(START, AGENT_IDS[0])
    for src, dst in zip(AGENT_IDS, AGENT_IDS[1:]):
        g.add_edge(src, dst)
    g.add_edge(AGENT_IDS[-1], END)
    return g.compile()


_SHARED = """\
UNDERWRITING RULES — these definitions are binding. Follow them exactly; do
not substitute your own judgement about which signal matters.

GOVERNING METRIC. The ONLY score that governs any decision is the
UNDERWRITING RISK SCORE (0-100) stated in the applicant profile. If the
profile states it explicitly (e.g. "underwriting risk score 52/100"), you
MUST use that stated value verbatim as the decision score — do not recompute
or second-guess it.

QUALIFYING SIGNAL. "disqualifying condition: none" means there is no
disqualifying medical condition (the rated-policy path is available); any
named condition (e.g. "active severe cardiac disease") means a disqualifying
condition IS present.

HARD THRESHOLDS.
  - Standard policy: underwriting risk score <= 40 -> issuable at standard
    rates.
  - Rated (substandard) policy: underwriting risk score <= 62 AND no
    disqualifying condition -> issuable at a rated premium.
"""

_ROLE_PROMPTS: dict[str, str] = {
    "intake": """\
You are the INTAKE agent. Normalize the applicant profile into clean fields:
state the underwriting risk score (use the stated value verbatim) and whether
a disqualifying condition is present. Do NOT recommend issue or decline —
your job is only to surface the normalized fields for the downstream agents.
""",
    "product": """\
You are the PRODUCT agent. You are the LENIENT pole of the panel — you can
place an applicant in a rated (substandard) policy. Use the underwriting risk
score from the chain above. Apply this rule:

  IF the applicant qualifies for a standard policy (underwriting risk
  score <= 40):
      recommend APPROVE (standard).
  ELIF underwriting risk score <= 62 AND no disqualifying condition:
      issuable via a rated policy -> recommend APPROVE (rated).
  ELSE:
      no issuable product -> recommend DENY.

State "PRODUCT RECOMMENDATION: APPROVE" or "PRODUCT RECOMMENDATION: DENY" on
its own line, noting the product (standard / rated) when approving.
""",
    "underwriting": """\
You are the UNDERWRITING agent. You are the CONSERVATIVE pole of the panel.
Use the underwriting risk score from the chain above. Apply this hard rule
with no exceptions and no rated-policy workaround:

  IF underwriting risk score > 40:
      unacceptable risk at standard terms -> recommend DENY.
  ELSE:
      acceptable risk -> recommend APPROVE.

State "UNDERWRITING RECOMMENDATION: APPROVE" or "UNDERWRITING RECOMMENDATION:
DENY" on its own line. You do not consider rated policies — that is product's
job.
""",
    "decision": """\
You are the DECISION agent. Restate the underwriting risk score from the
chain, then issue the final verdict.

CONFLICT-RESOLUTION RULE (apply exactly, step by step):
  1. Scan the "Prior agent outputs" list above IN ORDER. Identify, BY THE
     CONTENT of each line, every recommendation of the form
     "PRODUCT RECOMMENDATION: APPROVE|DENY" and
     "UNDERWRITING RECOMMENDATION: APPROVE|DENY". Judge each line by the
     RECOMMENDATION TEXT it contains, NOT by any "[role]" label or position
     name — the text is authoritative.
  2. If you find BOTH a PRODUCT RECOMMENDATION and an UNDERWRITING
     RECOMMENDATION and they DISAGREE (one APPROVE, one DENY): adopt whichever
     of those two recommendation lines appears EARLIER (further UP) in the
     list (first-writer-wins — the initial determination is binding). The
     earlier line's APPROVE/DENY is your verdict.
  3. If both are present and AGREE: adopt the shared recommendation.
  4. If only ONE recommendation line is present: adopt the recommendation that
     is actually stated. Do not invent a conflict.

Your verdict is the APPROVE or DENY selected by the CONFLICT-RESOLUTION RULE
above. That rule ALWAYS resolves to exactly one of APPROVE or DENY. Emit it as
a single machine-readable line and nothing else:
  VERDICT: APPROVE
  VERDICT: DENY

HARD CONSTRAINT — read carefully: the verdict MUST be exactly APPROVE or DENY.
You are FORBIDDEN from emitting "CONDITIONAL", "refer to manual underwriting",
"issue pending ...", or any hedged, qualified, or split verdict. An issuance
that is only viable as a rated policy is STILL a full "VERDICT: APPROVE". A
decline is "VERDICT: DENY". Do not soften, defer, or condition the
first-writer's recommendation; restate the risk score, then commit to the
single binary verdict the rule produced.
""",
}


def prompt_builder(node: Any, eval_row: dict, prior: tuple) -> str:
    """Deterministic per-role prompt keyed on ``node.role`` (load-bearing for
    the planted first-writer-wins fault; see the lending fixture). Unknown
    roles fall back to the bare default."""
    from topofuzz.runner import _default_prompt

    base = _default_prompt(node, eval_row, prior)
    role_key = getattr(node, "role", None) or getattr(node, "id", "")
    role_block = _ROLE_PROMPTS.get(role_key)
    if role_block is None:
        return base
    return f"{_SHARED}\n{role_block}\n{base}"
