# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""langgraph-lending-4role: gallery wrapper around the home demo fixture.

Reuses the build_graph from ``topofuzz.fixtures.lending_4role`` (the
canonical 4-agent lending-decision pipeline shipped with topofuzz)
but overrides LLM_CONFIGS to all-Flash to keep the gallery run cost
modest. The packaged ``topofuzz.fixtures.lending_4role`` uses a mixed
flash + pro setup tuned for §6 role_swap pair coverage; for the
gallery, all-flash gives more swappable pairs AND ~25x lower per-call
cost. Observed cost on the v0.5 reference run: ~$1.20 at
``--n 8 --variants 4`` (BLUEPRINT 19a's ``<$1`` target was missed by
~20%; the cost estimator under-estimates real Gemini Flash output cost
by ~3-5x).

The eval set + rubric live in this directory so the gallery entry is
self-contained. ``eval.csv`` is a byte-copy of the packaged
``topofuzz.fixtures.lending_eval.csv``. ``rubric.txt`` is a
**verdict-extraction** rubric (the judge reports the pipeline's
APPROVE/DENY verdict, score 10/0) — it deliberately DIVERGES from the
packaged reasoning-quality ``topofuzz.prompts.lending_judge.txt``,
which scored reasoning quality 0-10 and so could not feed the
blueprint's verdict-flip metric (§9/§10a/§22 expect the score to
encode the verdict). See ``.planning/verdict-flip-rootcause/`` and the
step-20a measurement-integrity discipline note.
"""

from __future__ import annotations

from typing import Any

from topofuzz.fixtures.lending_4role import build_graph as _build_graph
from topofuzz.ir import LLMConfig


AGENT_IDS: tuple[str, ...] = ("intake", "risk", "policy", "decision")

LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "risk": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "policy": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "decision": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
}


def build_graph():
    """Return the canonical lending_4role graph (delegated to the package)."""
    return _build_graph()


# ---------------------------------------------------------------------------
# Deterministic role prompts + planted attribution fault
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS (2026-05-30). The flagship attribution run originally
# executed every agent on ``runner._default_prompt`` — a bare
# "You are the 'X' agent ... respond" with NO lending rules. With nothing
# pinning which debt ratio governs, upstream agents nondeterministically
# decided on the applicant's *current* DTI vs their *projected
# post-mortgage* DTI, and the decision node faithfully propagated whichever
# it happened to see. Re-running the SAME input flipped the verdict — noise
# masquerading as a topology effect. (See
# ``.planning/2026-05-30-fix-design-prompts-and-fault.md``.)
#
# The fix is two-fold:
#   1. Pin the decision DTI to a single deterministic definition (the
#      shared block below) so a given input + ordering yields the SAME
#      verdict on every re-run.
#   2. Plant a KNOWN, deterministic ordering fault so the flagship paper
#      has a ground-truth culprit to validate per-agent attribution
#      against: the conservative ``risk`` agent and the lenient ``policy``
#      agent disagree on the marginal band, and the ``decision`` agent
#      resolves the disagreement by LAST-WRITER-WINS over its prior
#      outputs (which arrive in execution order). So:
#        - canonical intake->risk->policy->decision: policy is last ->
#          APPROVE (via FHA) on every run.
#        - swapped intake->policy->risk->decision: risk is last -> DENY
#          on every run.
#      Same input + same order -> identical verdict (reproducible);
#      swap the order -> a guaranteed, by-construction flip whose cause
#      is known (the last-position conflicting agent).
#
# CRITICAL WIRING NOTE — do NOT pass this builder as a ``--prompt-builder``
# CLI flag. That lands in the manifest ``cli_args`` and trips the F4/F7
# attribution refusal guards (``ablation.py`` / ``counterfactual_audit.py``),
# silently disabling attribution. The builder is exported here on the
# fixture MODULE and the loader attaches it to the Topology, so kickoff AND
# the F4/F7 terminal re-execution reconstruct it from one source of truth.

_SHARED = """\
LENDING RULES — these definitions are binding. Follow them exactly; do not
substitute your own judgement about which ratio matters.

DEBT-TO-INCOME (DTI). The ONLY DTI that governs any decision is the
PROJECTED post-mortgage back-end DTI — the applicant's debt load AFTER this
new mortgage is added.

If the applicant profile STATES a projected post-mortgage DTI explicitly
(e.g. "projected post-mortgage DTI 46%"), you MUST use that stated value
verbatim as the decision DTI — do not recompute or second-guess it. Only
when no projected DTI is stated do you compute it deterministically:

  gross_monthly_income = annual_income / 12
  current_monthly_debts = stated_current_DTI * gross_monthly_income
      (the "DTI NN%" quoted in the applicant profile is the CURRENT,
       pre-mortgage figure — it is NOT the decision DTI.)
  new_PITI = monthly principal+interest (amortize the loan amount over the
             stated term at 7.0% APR)
           + property tax (1.1% of home value per year / 12)
           + insurance ($1,200 per year / 12)
           + PMI (0.5% of loan amount per year / 12, ONLY if down payment
             < 20% of home value)
  projected_DTI = (current_monthly_debts + new_PITI) / gross_monthly_income

Always show this arithmetic and state the projected_DTI as a percentage.
Never decide on the stated current DTI.

HARD THRESHOLDS.
  - Conventional loan: projected DTI must be <= 43% AND credit score >= 620.
  - FHA loan: projected DTI must be <= 50% AND credit score >= 580.
  - PMI is required whenever the loan-to-value ratio exceeds 80%
    (i.e. down payment < 20%).
"""

_ROLE_PROMPTS: dict[str, str] = {
    "intake": """\
You are the INTAKE agent. Normalize the applicant profile into clean fields
and compute BOTH the current DTI (as stated) and the PROJECTED post-mortgage
DTI using the formula in the LENDING RULES above. Show the PITI arithmetic
explicitly. State: gross monthly income, current monthly debts, new PITI,
and the projected_DTI percentage. Do NOT recommend approve or deny — your
job is only to surface the normalized numbers for the downstream agents.
""",
    "risk": """\
You are the RISK agent. You are the CONSERVATIVE pole of the panel. Use the
INTAKE agent's PROJECTED DTI (recompute it yourself if intake did not state
it). Apply this hard rule with no exceptions and no FHA workaround:

  IF projected_DTI > 43% OR credit score < 620:
      this is HIGH RISK -> recommend DENY.
  ELSE:
      acceptable risk -> recommend APPROVE.

State "RISK RECOMMENDATION: APPROVE" or "RISK RECOMMENDATION: DENY" on its
own line. You do not consider FHA eligibility — that is policy's job.
""",
    "policy": """\
You are the POLICY agent. You are the LENIENT pole of the panel. Use the
PROJECTED DTI from the chain above. Apply this rule:

  IF the applicant FAILS conventional (projected_DTI > 43%) BUT still
  qualifies for FHA (projected_DTI <= 50% AND credit score >= 580):
      they are ELIGIBLE via an FHA loan -> recommend APPROVE (FHA).
  ELIF the applicant passes conventional (projected_DTI <= 43% AND
  credit >= 620):
      recommend APPROVE (conventional).
  ELSE:
      no eligible product -> recommend DENY.

State "POLICY RECOMMENDATION: APPROVE" or "POLICY RECOMMENDATION: DENY" on
its own line, noting the product (conventional / FHA) when approving.
""",
    "decision": """\
You are the DECISION agent. Restate the PROJECTED DTI from the chain, then
issue the final verdict.

CONFLICT-RESOLUTION RULE (apply exactly, step by step):
  1. Scan the "Prior agent outputs" list above IN ORDER. Identify, BY THE
     CONTENT of each line, every recommendation of the form
     "RISK RECOMMENDATION: APPROVE|DENY" and
     "POLICY RECOMMENDATION: APPROVE|DENY". Judge each line by the
     RECOMMENDATION TEXT it contains, NOT by any "[role]" label or position
     name — the text is authoritative.
  2. If you find BOTH a RISK RECOMMENDATION and a POLICY RECOMMENDATION and
     they DISAGREE (one APPROVE, one DENY): adopt whichever of those two
     recommendation lines appears LATER (further down) in the list
     (last-writer-wins). The later line's APPROVE/DENY is your verdict.
  3. If both are present and AGREE: adopt the shared recommendation.
  4. If only ONE recommendation line is present (e.g. two POLICY lines, or
     one of each role-position emitting the same recommendation text): adopt
     the recommendation that is actually stated. Do not invent a conflict.

Your verdict is the APPROVE or DENY selected by the CONFLICT-RESOLUTION RULE
above. That rule ALWAYS resolves to exactly one of APPROVE or DENY (the
last-writer's recommendation, or the shared one when they agree). Emit it as
a single machine-readable line and nothing else:
  VERDICT: APPROVE
  VERDICT: DENY

HARD CONSTRAINT — read carefully: the verdict MUST be exactly APPROVE or
DENY. You are FORBIDDEN from emitting "CONDITIONAL", "conditional approval",
"approve pending ...", "FHA path", "decline conventional / recommend FHA", or
any hedged, qualified, or split verdict. An approval that is only viable
through an FHA product is STILL a full "VERDICT: APPROVE" — route it through
FHA, but commit to APPROVE. A denial is "VERDICT: DENY". Do not soften, defer,
or condition the last-writer's recommendation; restate the projected DTI, then
commit to the single binary verdict the rule produced.
""",
}


def prompt_builder(node: Any, eval_row: dict, prior: tuple) -> str:
    """Build the deterministic per-role prompt for one node.

    Prepends the binding LENDING RULES block + the role-specific
    instructions to the runner's default prompt body (which carries the
    task input and the prior agent outputs in execution order).

    Keyed on ``node.role`` (NOT ``node.id``) — this is load-bearing for the
    planted fault. ``role_swap`` (the live variant axis for this linear-chain
    fixture; ``order_swap`` is a no-op on a chain) permutes ``node.role``
    while keeping ``node.id`` and the execution order fixed. Keying on role
    means a role-swapped node actually RECEIVES the other role's prompt and
    behaves like it. So the canonical chain runs risk(conservative→DENY) at
    position 2 and policy(lenient→APPROVE) at position 3 → decision's
    last-writer-wins adopts policy → APPROVE; the risk↔policy role_swap
    variant runs policy at position 2 and risk at position 3 → last writer is
    risk → DENY. That is a deterministic, by-construction APPROVE↔DENY flip
    whose culprit is the swapped conflict-resolution node — recoverable by
    F4's output-substitution LOO. (In the canonical topology role == id, so
    the kickoff run is unaffected; only role-permuted variants differ.)
    Unknown roles fall back to the bare default so the builder is safe on
    any topology.

    Imported lazily inside the function so importing this fixture module
    (which the loader does at IR-build time) never pulls the runner's
    heavier dependency surface at module-import time.
    """
    from topofuzz.runner import _default_prompt

    base = _default_prompt(node, eval_row, prior)
    role_key = getattr(node, "role", None) or getattr(node, "id", "")
    role_block = _ROLE_PROMPTS.get(role_key)
    if role_block is None:
        return base
    return f"{_SHARED}\n{role_block}\n{base}"
