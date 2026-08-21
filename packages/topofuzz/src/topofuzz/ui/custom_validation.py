# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Form-input validation for the custom-workflow editor.

Single entry point: :func:`validate_custom_workflow_form`. Returns a
:class:`CustomWorkflow` on success, or raises :class:`CustomFormError`
with a list of human-readable messages keyed to the form field that
failed. The route handler turns those into inline form errors.

Why a hand-rolled validator and not Pydantic:
  * Field-level error messages need to be plain English ("workflow
    name is too short") not machine error names. Pydantic's defaults
    leak class names ("string_too_short") into the form.
  * The server-side allow-list (model, agent_id regex) must be enforced
    EVEN IF the form itself was bypassed by a hand-crafted POST. A
    Pydantic schema in the route doesn't help with that.
"""

from __future__ import annotations

import re
import time

from topofuzz.ui.presets import (
    ALLOWED_CUSTOM_MODEL_KEYS,
    CustomAgent,
    CustomWorkflow,
    StarterScenario,
)
from topofuzz.ui.custom_storage import new_workflow_id


# Limits — match the spec.
MIN_AGENTS = 1
MAX_AGENTS = 10
MIN_LABEL_LEN = 2
MAX_LABEL_LEN = 60
MIN_DOMAIN_NOUN_LEN = 2
MAX_DOMAIN_NOUN_LEN = 30
MIN_DECISION_VERB_LEN = 2
MAX_DECISION_VERB_LEN = 40
MIN_RUBRIC_LEN = 50
MAX_RUBRIC_LEN = 2000

# agent_id grammar: starts with a lowercase letter, then [a-z0-9_]{1,39}.
# Total length 2..40. Same shape as Python identifiers, but stricter
# (no Unicode, no uppercase, no leading digit/underscore).
#
# \A and \Z anchor at start-of-string and end-of-string. Plain ^/$ in re.match
# would let a trailing newline through (e.g. agent_id="ab\n") because $ matches
# before final \n by default. That trailing newline used to slip into the
# generated fixture.py state-fields template; the template now also repr()s
# the agent_id (defense in depth), but anchoring properly closes the door at
# the validation layer.
AGENT_ID_RE = re.compile(r"\A[a-z][a-z0-9_]{1,39}\Z")

# SECURITY: character allow-list for the free-text fields (``label``,
# ``domain_noun``, ``decision_verb``) that get spliced into GENERATED
# PYTHON SOURCE via repr() (custom_templates.py's module docstring +
# rubric template). Length bounds alone don't stop the content from
# being dangerous: a label containing an embedded ``"""`` can terminate
# a triple-quoted docstring early if repr()'s escaping is ever bypassed
# or the template changes shape — today that's a DoS (the generated
# fixture fails to compile), and it is only ONE stray edit away from
# becoming a code-injection primitive. Reject the character classes
# outright rather than relying solely on repr() as the last line of
# defense. Double quotes and backslashes are the two characters repr()
# must escape to keep a string literal well-formed; ASCII control chars
# (and non-printable Unicode, checked separately below) have no
# legitimate place in a workflow name/noun/verb either.
_FREE_TEXT_FORBIDDEN_RE = re.compile(r'["\\\x00-\x1f\x7f]')


def _check_free_text_chars(value: str, field_display_name: str) -> str | None:
    """Return an error message if ``value`` contains a disallowed
    character, else None.

    Disallows double quotes, backslashes, ASCII control/DEL characters,
    and (via ``str.isprintable()``) broader Unicode non-printables —
    without rejecting the ASCII space, which ``isprintable()`` already
    treats as printable.
    """
    if _FREE_TEXT_FORBIDDEN_RE.search(value):
        return (
            f"{field_display_name} may only contain letters, numbers, "
            "spaces, and basic punctuation — not quotes or backslashes"
        )
    if not value.isprintable():
        return (
            f"{field_display_name} contains a non-printable character; "
            "please use plain text"
        )
    return None


class CustomFormError(ValueError):
    """One or more form fields failed validation.

    ``errors`` is an ordered dict of field_name → message.
    """

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def _check_agent_id(agent_id: str) -> str | None:
    """Return None if valid, otherwise an error message."""
    if not agent_id:
        return "agent name is required"
    if len(agent_id) < 2:
        return "agent name must be at least 2 characters"
    if len(agent_id) > 40:
        return "agent name must be 40 characters or fewer"
    if not AGENT_ID_RE.match(agent_id):
        return (
            "agent name must be lowercase letters, digits, and underscores "
            "only, and must start with a letter"
        )
    return None


def _check_model(model: str) -> str | None:
    if not model:
        return "model is required"
    if model not in ALLOWED_CUSTOM_MODEL_KEYS:
        return (
            f"model {model!r} is not in the allow-list. Allowed: "
            + ", ".join(sorted(ALLOWED_CUSTOM_MODEL_KEYS))
        )
    return None


def validate_custom_workflow_form(
    *,
    label: str,
    domain_noun: str,
    decision_verb: str,
    agent_ids: list[str],
    agent_models: list[str],
    verdict_agent_id: str,
    rubric_paragraph: str,
    workflow_id: str | None = None,
    starter_scenarios: list[StarterScenario] | None = None,
) -> CustomWorkflow:
    """Validate parsed form fields and return a :class:`CustomWorkflow`.

    Parameters mirror the form fields exactly. ``workflow_id`` is None
    on create and an existing id on edit (caller decides which); either
    way, the returned object's id is server-generated if input was None.
    """
    errors: dict[str, str] = {}

    label = (label or "").strip()
    if len(label) < MIN_LABEL_LEN:
        errors["label"] = (
            f"workflow name must be at least {MIN_LABEL_LEN} characters"
        )
    elif len(label) > MAX_LABEL_LEN:
        errors["label"] = (
            f"workflow name must be {MAX_LABEL_LEN} characters or fewer"
        )
    else:
        char_err = _check_free_text_chars(label, "workflow name")
        if char_err:
            errors["label"] = char_err

    domain_noun = (domain_noun or "").strip()
    if len(domain_noun) < MIN_DOMAIN_NOUN_LEN:
        errors["domain_noun"] = (
            f"domain noun must be at least {MIN_DOMAIN_NOUN_LEN} characters"
        )
    elif len(domain_noun) > MAX_DOMAIN_NOUN_LEN:
        errors["domain_noun"] = (
            f"domain noun must be {MAX_DOMAIN_NOUN_LEN} characters or fewer"
        )
    else:
        char_err = _check_free_text_chars(domain_noun, "domain noun")
        if char_err:
            errors["domain_noun"] = char_err

    decision_verb = (decision_verb or "").strip()
    if len(decision_verb) < MIN_DECISION_VERB_LEN:
        errors["decision_verb"] = (
            f"decision verb must be at least {MIN_DECISION_VERB_LEN} characters"
        )
    elif len(decision_verb) > MAX_DECISION_VERB_LEN:
        errors["decision_verb"] = (
            f"decision verb must be {MAX_DECISION_VERB_LEN} characters or fewer"
        )
    else:
        char_err = _check_free_text_chars(decision_verb, "decision verb")
        if char_err:
            errors["decision_verb"] = char_err

    rubric_paragraph = (rubric_paragraph or "").strip()
    if len(rubric_paragraph) < MIN_RUBRIC_LEN:
        errors["rubric_paragraph"] = (
            f"rubric paragraph must be at least {MIN_RUBRIC_LEN} characters"
        )
    elif len(rubric_paragraph) > MAX_RUBRIC_LEN:
        errors["rubric_paragraph"] = (
            f"rubric paragraph must be {MAX_RUBRIC_LEN} characters or fewer"
        )

    # Pair-and-trim agent rows. The form posts parallel agent_ids[] and
    # agent_models[] lists; tolerate length mismatch but never silently
    # drop content — the user sees a clear "row N: model required" error.
    cleaned_ids = [(s or "").strip() for s in agent_ids]
    cleaned_models = [(s or "").strip() for s in agent_models]
    n_rows = max(len(cleaned_ids), len(cleaned_models))
    paired: list[tuple[str, str]] = []
    for i in range(n_rows):
        aid = cleaned_ids[i] if i < len(cleaned_ids) else ""
        mdl = cleaned_models[i] if i < len(cleaned_models) else ""
        # Drop a fully blank row (the editor auto-includes blanks for
        # convenience). A row with EITHER field set is treated as
        # intentional and validated as a real entry.
        if not aid and not mdl:
            continue
        paired.append((aid, mdl))

    if len(paired) < MIN_AGENTS:
        errors["agents"] = f"add at least {MIN_AGENTS} agent"
    elif len(paired) > MAX_AGENTS:
        errors["agents"] = f"at most {MAX_AGENTS} agents allowed"

    # Per-row validation + duplicate detection. Stack errors — never
    # `continue` past a sibling check, so the user sees every problem in
    # one round-trip instead of fixing one error only to reveal the next.
    seen: set[str] = set()
    agents: list[CustomAgent] = []
    for idx, (aid, mdl) in enumerate(paired, start=1):
        row_ok = True
        id_err = _check_agent_id(aid)
        if id_err:
            errors[f"agent_id_{idx}"] = f"agent {idx}: {id_err}"
            row_ok = False
        elif aid in seen:
            errors[f"agent_id_{idx}"] = (
                f"agent {idx}: duplicate name {aid!r}; each agent must be unique"
            )
            row_ok = False
        else:
            seen.add(aid)
        model_err = _check_model(mdl)
        if model_err:
            errors[f"agent_model_{idx}"] = f"agent {idx}: {model_err}"
            row_ok = False
        if row_ok:
            agents.append(CustomAgent(agent_id=aid, model=mdl))

    # Verdict agent must be one of the agents above (and the linear chain
    # requires it be the LAST one — the fixture-generation step asserts
    # that). Default to the last agent if the form was submitted blank.
    verdict_agent_id = (verdict_agent_id or "").strip()
    if agents and not verdict_agent_id:
        verdict_agent_id = agents[-1].agent_id
    if agents:
        if verdict_agent_id not in {a.agent_id for a in agents}:
            errors["verdict_agent_id"] = (
                f"verdict agent {verdict_agent_id!r} is not one of the agent names"
            )
        elif agents and verdict_agent_id != agents[-1].agent_id:
            errors["verdict_agent_id"] = (
                "verdict agent must be the LAST agent in the chain "
                "(reorder agents so the chosen verdict agent is last)"
            )

    if errors:
        raise CustomFormError(errors)

    cw_workflow_id = workflow_id or new_workflow_id()
    return CustomWorkflow(
        workflow_id=cw_workflow_id,
        label=label,
        domain_noun=domain_noun,
        decision_verb=decision_verb,
        agents=tuple(agents),
        verdict_agent_id=verdict_agent_id,
        rubric_paragraph=rubric_paragraph,
        created_at=time.time(),
        starter_scenarios=tuple(starter_scenarios or ()),
    )
