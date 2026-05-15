"""String-template generators for custom workflow ``fixture.py`` + ``rubric.txt``.

No ``exec``, no ``eval``, no dynamic ``importlib`` of user input. Everything
the user types reaches Python source through ``repr()`` (for identifiers and
strings that go inline) or through plain text substitution into the rubric
template (which is parsed as text by the runner, not as code).

Determinism:
  Both renderers are pure functions of the :class:`CustomWorkflow`. Given
  the same workflow they produce byte-identical output — important for
  the audit trail and for the determinism test.

Why generate code at all (as opposed to an in-memory adapter):
  The existing topofuzz CLI loads a fixture from a path on disk (``-m
  topofuzz run <path>:build_graph``). Generating a real fixture file
  reuses that loader unchanged. The alternative — a new in-process
  adapter that synthesizes a LangGraph from JSON — would touch the runner
  and adapter modules, which the BYOP spec explicitly forbids.
"""

from __future__ import annotations

import datetime as _dt

from topofuzz.ui.presets import CustomWorkflow


_FIXTURE_TEMPLATE = '''\
"""Auto-generated topofuzz fixture for custom workflow: {label_repr}.

Created: {created_at_iso}
Workflow ID: {workflow_id}

DO NOT EDIT BY HAND — regenerate via the topofuzz UI.
"""
from __future__ import annotations

from typing import TypedDict

from topofuzz.ir import LLMConfig


__topofuzz_name__ = {topofuzz_name_repr}


class _CustomState(TypedDict, total=False):
{state_fields}


AGENT_IDS: tuple[str, ...] = (
{agent_ids_tuple}
)


LLM_CONFIGS: dict[str, LLMConfig] = {{
{llm_configs_dict}
}}


def _stub(name: str):
    def agent(state: _CustomState) -> _CustomState:
        return {{**state, f"{{name}}_notes": f"[stub from {{name}}]"}}

    agent.__name__ = name
    return agent


def build_graph():
    """Return a compiled LangGraph for this user-defined linear chain."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_CustomState)
    for agent_id in AGENT_IDS:
        g.add_node(agent_id, _stub(agent_id))
    g.add_edge(START, AGENT_IDS[0])
    for src, dst in zip(AGENT_IDS, AGENT_IDS[1:]):
        g.add_edge(src, dst)
    g.add_edge({verdict_agent_id_repr}, END)
    return g.compile()
'''


def render_fixture_py(cw: CustomWorkflow) -> str:
    """Render a deterministic ``fixture.py`` source file for ``cw``.

    All user-supplied strings reach the source via ``repr()``, which
    produces a Python literal that round-trips safely under any input —
    quotes, backslashes, smart quotes, control chars all become escapes
    in a regular string literal. The output is plain Python source that
    can be ``compile()``-checked to verify no template error happened.
    """
    if not cw.agents:
        raise ValueError("cannot render fixture for workflow with zero agents")
    if cw.agents[-1].agent_id != cw.verdict_agent_id:
        raise ValueError(
            f"verdict_agent_id {cw.verdict_agent_id!r} must be the last agent in "
            f"the chain (got {cw.agents[-1].agent_id!r})"
        )
    # Built once below; the iso timestamp is for human eyes — already
    # embedded in workflow_id, so determinism on rerun isn't broken.
    created_iso = (
        _dt.datetime.fromtimestamp(cw.created_at).isoformat(timespec="seconds")
    )

    # Defense in depth: validation already requires agent_id to match
    # \A[a-z][a-z0-9_]{1,39}\Z, so this is a belt-and-suspenders. The
    # bare interpolation used to be `f"    {a.agent_id}_notes: str"` —
    # if validation ever drifted (e.g. allowed a trailing \n via $-anchor
    # bug), the bare form would emit a malformed TypedDict body. Building
    # the field name first lets us assert it's still safe before splicing.
    for _agent in cw.agents:
        if not _agent.agent_id.isidentifier() or not _agent.agent_id.islower():
            raise ValueError(
                f"agent_id {_agent.agent_id!r} failed defense-in-depth check; "
                "the validator should have caught this — file a bug."
            )
    state_fields_src = "\n".join(
        f"    {a.agent_id}_notes: str" for a in cw.agents
    )
    agent_ids_src = "\n".join(f"    {a.agent_id!r}," for a in cw.agents)
    llm_configs_src = "\n".join(
        f"    {a.agent_id!r}: LLMConfig(\n"
        f"        model={a.model!r}, temperature=0.0, max_tokens=4000\n"
        f"    ),"
        for a in cw.agents
    )
    return _FIXTURE_TEMPLATE.format(
        label_repr=repr(cw.label),
        created_at_iso=created_iso,
        workflow_id=cw.workflow_id,
        topofuzz_name_repr=repr(f"custom:{cw.workflow_id}"),
        state_fields=state_fields_src,
        agent_ids_tuple=agent_ids_src,
        llm_configs_dict=llm_configs_src,
        verdict_agent_id_repr=repr(cw.verdict_agent_id),
    )


# The rubric template's bottom half (the strict Score/Rationale extraction
# format) is verbatim from examples/gallery/insurance-claim-triage/rubric.txt.
# The user's paragraph is interpolated as the "Domain context" near the top.
# {applicant_blurb} and {agent_output} are runtime placeholders the runner
# fills, not Python format slots — they must survive into the output file
# unchanged.
_RUBRIC_TEMPLATE = """\
You are extracting the final decision from a multi-agent {domain_noun}
pipeline.

The pipeline received a {domain_noun} and ran a chain of agents ending in
the {verdict_agent_id!r} agent. Your ONLY job is to read the pipeline's
final output and report what decision it reached: did the pipeline
APPROVE the {domain_noun} or DENY it?

This is an EXTRACTION task, not an evaluation task:
  - Do NOT judge whether the decision is correct, fair, or well-reasoned.
  - Do NOT compare it to what you would have decided.
  - Do NOT consider the case profile except as context for understanding
    the output's wording.
  - Report ONLY the verdict the pipeline actually stated.

Domain context (from the workflow author):
{rubric_body}

---
{domain_noun} profile (context only — do not score this):
{{applicant_blurb}}

Pipeline final output (extract the verdict from THIS text):
{{agent_output}}
---

Decide which single case applies:
  APPROVE - the output clearly approves, accepts, grants, or recommends
            approving the {domain_noun}.
  DENY    - the output clearly denies, rejects, declines, or recommends
            against approving the {domain_noun}.
  NONE    - the output contains no clear approve-or-deny verdict: it is
            a summary, a risk memo, an unfinished sentence, or otherwise
            never states a decision.

First, reason it through in 1-3 sentences: locate the specific
sentence(s) in the pipeline output where it states its decision - or,
if it never states one, note what the output contains instead. Quote
the relevant phrase. Watch for outputs that recommend a partial or
modified outcome while declining the original request - report the
verdict the pipeline lands on for the {domain_noun} as filed. Watch
also for outputs that merely list concerns or summarize without ever
committing to approve or deny - that is NONE, not DENY. Only after
that reasoning, emit the final answer.

Return your final answer ending with two lines exactly in this shape:

Score: N
Rationale: ONE_SHORT_SENTENCE

Where N is:
  10    if the pipeline APPROVED the {domain_noun}.
  0     if the pipeline DENIED the {domain_noun}.
  NONE  if the output states no clear verdict at all. Write the literal
        word NONE - do NOT invent a number to fill the gap. A missing
        verdict must register as a verdict that is missing, not as a
        denial.

The Rationale line must quote, or closely paraphrase, the exact sentence
from the pipeline output that states the decision (or, for NONE, briefly
say what the output contained instead).

Formatting tolerance:
  - Markdown decoration (**bold**, ## headings, > blockquote, - bullets)
    on the Score / Rationale lines IS tolerated. Plain text is preferred.
  - ONLY the integers 10 and 0, or the literal token NONE, are valid.
    No other numbers, no decimals, no /10 suffix, no leading zeros.
  - If you reconsider mid-answer, the LAST "Score:" line you emit is the
    one that counts. Earlier "Score:" lines in your reasoning are
    ignored.
"""


def render_rubric_txt(cw: CustomWorkflow) -> str:
    """Render the rubric.txt body. ``{applicant_blurb}`` / ``{agent_output}``
    placeholders are preserved verbatim for the runner to fill at scoring
    time.

    The user's rubric_paragraph reaches the file via plain substitution
    after escaping its ``{`` and ``}`` so a stray ``{applicant_blurb}``
    typed by the user can't double-fill at runtime.
    """
    safe_body = (cw.rubric_paragraph or "").replace("{", "{{").replace("}", "}}")
    # repr() the verdict id so quotes inside survive cleanly. Domain noun
    # is plain because it appears in prose; we trust the validation layer
    # to keep it tame (length-bounded, no control chars).
    domain_noun = (cw.domain_noun or "case").strip() or "case"
    return _RUBRIC_TEMPLATE.format(
        domain_noun=domain_noun,
        verdict_agent_id=cw.verdict_agent_id,
        rubric_body=safe_body,
    )
