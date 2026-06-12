# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""langgraph-admissions-4role: 4-agent admissions pipeline (EU AI Act
Annex III point 3 — education / access to admission).

Generality re-skin of the lending fixture, LAST-WRITER-WINS variant (the
culprit lands at the LAST pre-terminal position, like lending). Chain
``intake -> academic -> holistic -> decision``: the CONSERVATIVE pole
(``academic``) and the LENIENT pole (``holistic``) disagree on a
near-boundary band; ``decision`` adopts the LATER of the two conflicting
recommendations.

  - canonical intake->academic->holistic->decision: holistic is last ->
    APPROVE (admit via holistic review) on every run.
  - role_swap intake->holistic->academic->decision: academic is last ->
    DENY (reject) on every run.

Deterministic by-construction APPROVE<->DENY flip; F4 leave-one-out recovers
the culprit = the last-position conflicting node (``holistic``). Verdict
tokens stay APPROVE/DENY (APPROVE = admit) for the shared verdict rubric.

WIRING NOTE: export the builder on the MODULE (never as a --prompt-builder
CLI flag); see the lending fixture for the full rationale.
"""

from __future__ import annotations

from typing import Any, TypedDict

from topofuzz.ir import LLMConfig


AGENT_IDS: tuple[str, ...] = ("intake", "academic", "holistic", "decision")

LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "academic": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "holistic": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "decision": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
}


class _State(TypedDict, total=False):
    application: dict
    intake_notes: str
    academic_notes: str
    holistic_notes: str
    decision_notes: str
    decision: str


def _make_stub_agent(name: str):
    def agent(state: _State) -> _State:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent admissions chain."""
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
ADMISSIONS RULES — these definitions are binding. Follow them exactly; do not
substitute your own judgement about which signal matters.

GOVERNING METRIC. The ONLY score that governs any decision is the COMPOSITE
ACADEMIC INDEX (0-100) stated in the applicant profile. If the profile states
it explicitly (e.g. "composite academic index 64/100"), you MUST use that
stated value verbatim as the decision index — do not recompute or
second-guess it.

QUALIFYING SIGNAL. "strong contextual factors: yes" means the applicant
presents strong contextual / holistic factors (first-generation,
under-resourced school, significant responsibilities, low-opportunity area)
that open the holistic-review path; "no" means they do not.

HARD THRESHOLDS.
  - Standard track: composite academic index >= 70 -> admissible on academic
    merit alone.
  - Holistic track: composite academic index >= 58 AND strong contextual
    factors -> admissible via holistic review.
"""

_ROLE_PROMPTS: dict[str, str] = {
    "intake": """\
You are the INTAKE agent. Normalize the applicant profile into clean fields:
state the composite academic index (use the stated value verbatim) and
whether strong contextual factors are present. Do NOT recommend admit or
reject — your job is only to surface the normalized fields for the downstream
agents.
""",
    "academic": """\
You are the ACADEMIC agent. You are the CONSERVATIVE pole of the panel. Use
the composite academic index from the chain above. Apply this hard rule with
no exceptions and no holistic workaround:

  IF composite academic index < 70:
      below the academic bar -> recommend DENY.
  ELSE:
      meets the academic bar -> recommend APPROVE.

State "ACADEMIC RECOMMENDATION: APPROVE" or "ACADEMIC RECOMMENDATION: DENY" on
its own line. You do not consider holistic factors — that is holistic's job.
""",
    "holistic": """\
You are the HOLISTIC agent. You are the LENIENT pole of the panel. Use the
composite academic index from the chain above. Apply this rule:

  IF the applicant meets the standard track (composite academic index >= 70):
      recommend APPROVE (standard).
  ELIF strong contextual factors are present AND composite academic index
  >= 58:
      admissible via holistic review -> recommend APPROVE (holistic).
  ELSE:
      no admissible track -> recommend DENY.

State "HOLISTIC RECOMMENDATION: APPROVE" or "HOLISTIC RECOMMENDATION: DENY" on
its own line, noting the track (standard / holistic) when approving.
""",
    "decision": """\
You are the DECISION agent. Restate the composite academic index from the
chain, then issue the final verdict.

CONFLICT-RESOLUTION RULE (apply exactly, step by step):
  1. Scan the "Prior agent outputs" list above IN ORDER. Identify, BY THE
     CONTENT of each line, every recommendation of the form
     "ACADEMIC RECOMMENDATION: APPROVE|DENY" and
     "HOLISTIC RECOMMENDATION: APPROVE|DENY". Judge each line by the
     RECOMMENDATION TEXT it contains, NOT by any "[role]" label or position
     name — the text is authoritative.
  2. If you find BOTH an ACADEMIC RECOMMENDATION and a HOLISTIC RECOMMENDATION
     and they DISAGREE (one APPROVE, one DENY): adopt whichever of those two
     recommendation lines appears LATER (further down) in the list
     (last-writer-wins). The later line's APPROVE/DENY is your verdict.
  3. If both are present and AGREE: adopt the shared recommendation.
  4. If only ONE recommendation line is present: adopt the recommendation that
     is actually stated. Do not invent a conflict.

Your verdict is the APPROVE or DENY selected by the CONFLICT-RESOLUTION RULE
above. That rule ALWAYS resolves to exactly one of APPROVE or DENY. Emit it as
a single machine-readable line and nothing else:
  VERDICT: APPROVE
  VERDICT: DENY

HARD CONSTRAINT — read carefully: the verdict MUST be exactly APPROVE or DENY.
You are FORBIDDEN from emitting "CONDITIONAL", "waitlist", "admit pending ...",
or any hedged, qualified, or split verdict. An admit that is only viable
through holistic review is STILL a full "VERDICT: APPROVE". A reject is
"VERDICT: DENY". Do not soften, defer, or condition the last-writer's
recommendation; restate the academic index, then commit to the single binary
verdict the rule produced.
""",
}


def prompt_builder(node: Any, eval_row: dict, prior: tuple) -> str:
    """Deterministic per-role prompt keyed on ``node.role`` (load-bearing for
    the planted last-writer-wins fault; see the lending fixture). Unknown roles
    fall back to the bare default."""
    from topofuzz.runner import _default_prompt

    base = _default_prompt(node, eval_row, prior)
    role_key = getattr(node, "role", None) or getattr(node, "id", "")
    role_block = _ROLE_PROMPTS.get(role_key)
    if role_block is None:
        return base
    return f"{_SHARED}\n{role_block}\n{base}"
