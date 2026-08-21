# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""langgraph-hiring-4role: 4-agent hiring-decision pipeline (EU AI Act
Annex III point 4 — employment / recruitment).

A generality re-skin of the canonical lending fixture. Identical attribution
mechanism, different regulated domain: a candidate-screening chain
``screening -> standards -> potential -> decision`` where a CONSERVATIVE
pole (``standards``) and a LENIENT pole (``potential``) disagree on a
near-boundary band, and the ``decision`` agent resolves the disagreement by
LAST-WRITER-WINS over the prior outputs (which arrive in execution order).

  - canonical screening->standards->potential->decision: potential is last ->
    APPROVE (hire via the equivalent-experience path) on every run.
  - role_swap screening->potential->standards->decision: standards is last ->
    DENY (no-hire) on every run.

Same input + same order -> identical verdict (reproducible); swap the order
-> a guaranteed, by-construction APPROVE<->DENY flip whose cause is known
(the last-position conflicting agent). ``order_swap`` is a no-op on a linear
chain, so ``role_swap`` is the live perturbation axis; F4 leave-one-out
output-substitution recovers the culprit.

Verdict tokens stay APPROVE/DENY (APPROVE = hire) so the shared
``rubric-verdict-generic.txt`` verdict-extraction judge applies unchanged.

CRITICAL WIRING NOTE — do NOT pass this builder as a ``--prompt-builder`` CLI
flag. It must be exported on the fixture MODULE so the loader attaches it to
the Topology (kickoff AND the F4/F7 terminal re-execution reconstruct it from
one source of truth). A ``cli_args`` builder trips the attribution refusal
guards in ablation.py / counterfactual_audit.py.
"""

from __future__ import annotations

from typing import Any, TypedDict

from topofuzz.ir import LLMConfig


AGENT_IDS: tuple[str, ...] = ("screening", "standards", "potential", "decision")

# Per-node models are chosen so role_swap permutes ONLY the two reviewer
# poles. role_swap pairs same-model nodes in declared-index order; if all four
# shared one model it would also pair the normalizer (index 0) with a pole,
# and at small --variants the SELECTED flipping variant routes through the
# normalizer — attributing the flip to `screening` instead of the reviewer
# that caused it (verified by replicating the CLI's _round_robin_select).
# Giving the two poles ONE model and the normalizer + decision two OTHER
# distinct models makes (standards, potential) the ONLY swappable pair, so the
# role_swap flip axis is exactly the conservative<->lenient reviewer swap.
# The two poles run the heavy threshold reasoning on DeepSeek V4 Flash
# ($0.14/$0.28 per 1M, cheap + strong deterministic instruction-following);
# the light bookend agents use Gemini. Run with --variants 2 (canonical + the
# single flip axis). Judge is a single Gemini-Flash extraction pass at run time.
LLM_CONFIGS: dict[str, LLMConfig] = {
    "screening": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "standards": LLMConfig(model="deepseek/deepseek-chat", temperature=0.0, max_tokens=4000),
    "potential": LLMConfig(model="deepseek/deepseek-chat", temperature=0.0, max_tokens=4000),
    "decision": LLMConfig(model="google/gemini-2.5-pro", temperature=0.0, max_tokens=4000),
}


class _State(TypedDict, total=False):
    application: dict
    screening_notes: str
    standards_notes: str
    potential_notes: str
    decision_notes: str
    decision: str


def _make_stub_agent(name: str):
    def agent(state: _State) -> _State:
        return {**state, f"{name}_notes": f"[stub from {name}]"}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for the 4-agent hiring chain."""
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
HIRING RULES — these definitions are binding. Follow them exactly; do not
substitute your own judgement about which signal matters.

GOVERNING METRIC. The ONLY score that governs any decision is the STRUCTURED
COMPETENCY ASSESSMENT SCORE (0-100) stated in the applicant profile. If the
profile states it explicitly (e.g. "competency assessment score 68/100"), you
MUST use that stated value verbatim as the decision score — do not recompute
or second-guess it.

QUALIFYING SIGNAL. "verified equivalent-work portfolio: yes" means the
applicant has a verified portfolio of equivalent professional work (the
equivalent-experience path is available to them); "no" means it is not.

HARD THRESHOLDS.
  - Standard track: competency score >= 75 -> meets the role on merit alone.
  - Equivalent-experience track: competency score >= 62 AND a verified
    equivalent-work portfolio -> meets the role via demonstrated equivalent
    work.
"""

_ROLE_PROMPTS: dict[str, str] = {
    "screening": """\
You are the SCREENING agent. Normalize the applicant profile into clean
fields: state the candidate's competency assessment score (use the stated
value verbatim), years of relevant experience, and whether a verified
equivalent-work portfolio is present. Do NOT recommend hire or no-hire — your
job is only to surface the normalized fields for the downstream agents.
""",
    "standards": """\
You are the STANDARDS agent. You are the CONSERVATIVE pole of the panel. Use
the competency score from the chain above. Apply this hard rule with no
exceptions and no equivalent-experience workaround:

  IF competency score < 75:
      does NOT meet the standard bar -> recommend DENY.
  ELSE:
      meets the bar -> recommend APPROVE.

State "STANDARDS RECOMMENDATION: APPROVE" or "STANDARDS RECOMMENDATION: DENY"
on its own line. You do not consider the equivalent-experience path — that is
potential's job.
""",
    "potential": """\
You are the POTENTIAL agent. You are the LENIENT pole of the panel. Use the
competency score from the chain above. Apply this rule:

  IF the applicant meets the standard track (competency score >= 75):
      recommend APPROVE (standard).
  ELIF the applicant has a verified equivalent-work portfolio AND competency
  score >= 62:
      eligible via the equivalent-experience track -> recommend APPROVE
      (equivalent-experience).
  ELSE:
      no eligible track -> recommend DENY.

State "POTENTIAL RECOMMENDATION: APPROVE" or "POTENTIAL RECOMMENDATION: DENY"
on its own line, noting the track (standard / equivalent-experience) when
approving.
""",
    "decision": """\
You are the DECISION agent. Restate the candidate's competency score from the
chain, then issue the final verdict.

CONFLICT-RESOLUTION RULE (apply exactly, step by step):
  1. Scan the "Prior agent outputs" list above IN ORDER. Identify, BY THE
     CONTENT of each line, every recommendation of the form
     "STANDARDS RECOMMENDATION: APPROVE|DENY" and
     "POTENTIAL RECOMMENDATION: APPROVE|DENY". Judge each line by the
     RECOMMENDATION TEXT it contains, NOT by any "[role]" label or position
     name — the text is authoritative.
  2. If you find BOTH a STANDARDS RECOMMENDATION and a POTENTIAL
     RECOMMENDATION and they DISAGREE (one APPROVE, one DENY): adopt whichever
     of those two recommendation lines appears LATER (further down) in the
     list (last-writer-wins). The later line's APPROVE/DENY is your verdict.
  3. If both are present and AGREE: adopt the shared recommendation.
  4. If only ONE recommendation line is present: adopt the recommendation that
     is actually stated. Do not invent a conflict.

Your verdict is the APPROVE or DENY selected by the CONFLICT-RESOLUTION RULE
above. That rule ALWAYS resolves to exactly one of APPROVE or DENY. Emit it as
a single machine-readable line and nothing else:
  VERDICT: APPROVE
  VERDICT: DENY

HARD CONSTRAINT — read carefully: the verdict MUST be exactly APPROVE or DENY.
You are FORBIDDEN from emitting "CONDITIONAL", "conditional offer", "hire
pending ...", or any hedged, qualified, or split verdict. A hire that is only
viable through the equivalent-experience track is STILL a full
"VERDICT: APPROVE" — route it through that track, but commit to APPROVE. A
no-hire is "VERDICT: DENY". Do not soften, defer, or condition the
last-writer's recommendation; restate the competency score, then commit to the
single binary verdict the rule produced.
""",
}


def prompt_builder(node: Any, eval_row: dict, prior: tuple) -> str:
    """Deterministic per-role prompt, keyed on ``node.role`` (load-bearing for
    the planted last-writer-wins fault; see the lending fixture for the full
    rationale). Unknown roles fall back to the bare default so the builder is
    safe on any topology."""
    from topofuzz.runner import _default_prompt

    base = _default_prompt(node, eval_row, prior)
    role_key = getattr(node, "role", None) or getattr(node, "id", "")
    role_block = _ROLE_PROMPTS.get(role_key)
    if role_block is None:
        return base
    return f"{_SHARED}\n{role_block}\n{base}"
