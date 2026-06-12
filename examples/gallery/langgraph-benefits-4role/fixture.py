# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""langgraph-benefits-4role: 4-agent public-benefits eligibility pipeline
(EU AI Act Annex III point 5(a) — public assistance benefits & services).

Generality re-skin of the lending fixture, FIRST-WRITER-WINS variant — the
culprit lands at an EARLIER (non-last) pre-terminal position, distinguishing
leave-one-out attribution from a blame-last-non-terminal heuristic.

Chain ``intake -> hardship -> eligibility -> decision``. The LENIENT pole
(``hardship``, which can grant a discretionary hardship exception) runs at
position 2; the CONSERVATIVE pole (``eligibility``, the strict income test)
runs at position 3. The ``decision`` agent adopts the EARLIER of the two
conflicting recommendations (first-writer-wins — the caseworker's initial
hardship determination is binding).

  - canonical intake->hardship->eligibility->decision: hardship (pos 2) is the
    first writer -> APPROVE (granted via hardship exception) on every run.
  - role_swap intake->eligibility->hardship->decision: eligibility (pos 2) is
    the first writer -> DENY on every run.

Deterministic by-construction APPROVE<->DENY flip whose culprit is the
position-2 ``hardship`` node — recovered by F4 leave-one-out, MISSED by a
blame-last-non-terminal baseline (which would wrongly accuse ``eligibility``
at pos 3). Verdict tokens stay APPROVE/DENY (APPROVE = benefit granted).

WIRING NOTE: export the builder on the MODULE (never as a --prompt-builder CLI
flag); see the lending fixture for the rationale.
"""

from __future__ import annotations

from typing import Any, TypedDict

from topofuzz.ir import LLMConfig


AGENT_IDS: tuple[str, ...] = ("intake", "hardship", "eligibility", "decision")

LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "hardship": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "eligibility": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "decision": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
}


class _State(TypedDict, total=False):
    application: dict
    intake_notes: str
    hardship_notes: str
    eligibility_notes: str
    decision_notes: str
    decision: str


def _make_stub_agent(name: str):
    def agent(state: _State) -> _State:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent benefits chain."""
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
BENEFITS ELIGIBILITY RULES — these definitions are binding. Follow them
exactly; do not substitute your own judgement about which signal matters.

GOVERNING METRIC. The ONLY figure that governs any decision is the HOUSEHOLD
INCOME AS A PERCENTAGE OF THE FEDERAL POVERTY LEVEL (FPL) stated in the
applicant profile. If the profile states it explicitly (e.g. "income 168% of
the Federal Poverty Level"), you MUST use that stated value verbatim as the
decision figure — do not recompute or second-guess it.

QUALIFYING SIGNAL. "dependent children: yes" means the household includes
dependent children (the discretionary hardship path is available); "no" means
it does not.

HARD THRESHOLDS.
  - Standard eligibility: income <= 130% of FPL -> eligible on the standard
    income test.
  - Hardship path: income <= 185% of FPL AND dependent children present ->
    eligible via a discretionary hardship exception.
"""

_ROLE_PROMPTS: dict[str, str] = {
    "intake": """\
You are the INTAKE agent. Normalize the applicant profile into clean fields:
state the household income as a percentage of the Federal Poverty Level (use
the stated value verbatim) and whether dependent children are present. Do NOT
recommend approve or deny — your job is only to surface the normalized fields
for the downstream agents.
""",
    "hardship": """\
You are the HARDSHIP agent. You are the LENIENT pole of the panel — you can
grant a discretionary hardship exception. Use the income percentage from the
chain above. Apply this rule:

  IF the applicant passes the standard income test (income <= 130% of FPL):
      recommend APPROVE (standard).
  ELIF income <= 185% of FPL AND dependent children are present:
      eligible via a discretionary hardship exception -> recommend APPROVE
      (hardship).
  ELSE:
      no eligible path -> recommend DENY.

State "HARDSHIP RECOMMENDATION: APPROVE" or "HARDSHIP RECOMMENDATION: DENY" on
its own line, noting the path (standard / hardship) when approving.
""",
    "eligibility": """\
You are the ELIGIBILITY agent. You are the CONSERVATIVE pole of the panel. Use
the income percentage from the chain above. Apply this hard rule with no
exceptions and no hardship workaround:

  IF income > 130% of FPL:
      over the standard income limit -> recommend DENY.
  ELSE:
      within the standard income limit -> recommend APPROVE.

State "ELIGIBILITY RECOMMENDATION: APPROVE" or "ELIGIBILITY RECOMMENDATION:
DENY" on its own line. You do not consider hardship exceptions — that is
hardship's job.
""",
    "decision": """\
You are the DECISION agent. Restate the income percentage from the chain, then
issue the final verdict.

CONFLICT-RESOLUTION RULE (apply exactly, step by step):
  1. Scan the "Prior agent outputs" list above IN ORDER. Identify, BY THE
     CONTENT of each line, every recommendation of the form
     "HARDSHIP RECOMMENDATION: APPROVE|DENY" and
     "ELIGIBILITY RECOMMENDATION: APPROVE|DENY". Judge each line by the
     RECOMMENDATION TEXT it contains, NOT by any "[role]" label or position
     name — the text is authoritative.
  2. If you find BOTH a HARDSHIP RECOMMENDATION and an ELIGIBILITY
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
You are FORBIDDEN from emitting "CONDITIONAL", "pending verification",
"approve pending ...", or any hedged, qualified, or split verdict. A grant
that is only viable through a hardship exception is STILL a full
"VERDICT: APPROVE". A denial is "VERDICT: DENY". Do not soften, defer, or
condition the first-writer's recommendation; restate the income percentage,
then commit to the single binary verdict the rule produced.
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
