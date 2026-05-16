# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Import a user's existing compiled LangGraph as a topofuzz custom workflow.

Walks a ``StateGraph.compile()`` result, asserts the topology is a linear
chain (START -> n1 -> n2 -> ... -> END with exactly one out-edge per
node), and produces a :class:`CustomWorkflow` ready to persist via
:func:`topofuzz.ui.custom_storage.save_custom_workflow`.

Why this exists:
  Topofuzz already accepts a path-on-disk fixture via the runner's
  ``topofuzz run path/to/fixture.py:build_graph`` form. But a coder with
  an existing LangGraph wants two things that path doesn't give:

    1. Their graph surfaced as a tile in the UI alongside the gallery
       workflows, so a stakeholder can click "Run" without invoking
       the CLI.
    2. Their per-node model bindings carried through automatically,
       not transcribed by hand into a separate LLM_CONFIGS dict.

  This module is the 2-hour shim that closes both gaps. The CLI
  command ``topofuzz langgraph-import`` is the user-facing entry.

Out of scope (will raise :class:`LangGraphImportError`):
  * Branching / conditional edges (BYOP is linear-chain only).
  * Multiple START edges (single entrypoint required).
  * Cycles (must terminate at END).
  * Compiled objects that aren't a LangGraph 1.x ``CompiledStateGraph``
    (no ``.builder`` or ``.get_graph`` attribute).
  * Node ids that don't match the BYOP ``\\A[a-z][a-z0-9_]{1,39}\\Z``
    grammar — surfaced as a normal :class:`CustomFormError` from
    :func:`validate_custom_workflow_form`, asking the user to rename
    in their source.
  * Models that aren't in :data:`ALLOWED_CUSTOM_MODEL_KEYS` — same
    routing as above.

Threading note: the importer is pure (no I/O, no global state). The
caller saves via ``save_custom_workflow`` separately, so two parallel
imports of the same graph would produce two CustomWorkflows with
distinct workflow_ids.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from topofuzz.ir import LLMConfig
from topofuzz.loaders import _WARNED_FIXTURE_PATHS  # noqa: PLC2701 — shared one-shot set
from topofuzz.ui.custom_validation import validate_custom_workflow_form
from topofuzz.ui.presets import (
    ALLOWED_CUSTOM_MODEL_KEYS,
    CustomWorkflow,
    StarterScenario,
)


_log = logging.getLogger(__name__)

# LangGraph 1.x sentinels. Mirrored verbatim in adapter.py — kept as
# local constants here so this module has no import-time dependency on
# adapter's private constants.
_START_NODE = "__start__"
_END_NODE = "__end__"


class LangGraphImportError(ValueError):
    """Raised when an imported LangGraph cannot be expressed as a BYOP workflow."""


def _extract_linear_order(compiled: Any) -> list[str]:
    """Return node ids in execution order, START -> ... -> END.

    Raises :class:`LangGraphImportError` if the graph is not a linear
    chain. The walk is deterministic: at each step, the current node
    must have exactly one unique outgoing edge (duplicate edges to the
    same target are tolerated — LangGraph allows repeated ``add_edge``
    calls with the same args).
    """
    if not hasattr(compiled, "get_graph") or not hasattr(compiled, "builder"):
        raise LangGraphImportError(
            "expected a compiled LangGraph (the result of "
            "StateGraph.compile()); got "
            f"{type(compiled).__name__} which does not expose `.builder` / "
            "`.get_graph`. If you have a StateGraph builder, call "
            "`.compile()` on it first."
        )

    # xray=1 flattens depth-1 subgraphs into their inner nodes. We
    # tolerate the flatten, but the result must still be linear after
    # flattening — branching INSIDE a subgraph will surface here.
    drawable = compiled.get_graph(xray=1)
    outgoing: dict[str, list[str]] = {}
    for edge in drawable.edges:
        outgoing.setdefault(edge.source, []).append(edge.target)

    entrypoints = list(dict.fromkeys(outgoing.get(_START_NODE, [])))
    if not entrypoints:
        raise LangGraphImportError(
            "graph has no START edge — every LangGraph must declare "
            "`add_edge(START, <first_node>)`."
        )
    if len(entrypoints) > 1:
        raise LangGraphImportError(
            "graph has multiple START edges to "
            f"{entrypoints!r}; BYOP requires a single entrypoint "
            "(linear chain)."
        )

    order: list[str] = []
    seen: set[str] = set()
    cur = entrypoints[0]
    while cur != _END_NODE:
        if cur in seen:
            raise LangGraphImportError(
                f"cycle detected — node {cur!r} reached twice. BYOP "
                "requires a finite linear chain."
            )
        seen.add(cur)
        order.append(cur)
        nexts = list(dict.fromkeys(outgoing.get(cur, [])))
        if not nexts:
            raise LangGraphImportError(
                f"node {cur!r} has no outgoing edge; the chain must "
                "terminate at END."
            )
        if len(nexts) > 1:
            raise LangGraphImportError(
                f"node {cur!r} branches to {nexts!r}; BYOP supports "
                "linear chains only. Either flatten the branch in your "
                "LangGraph source or import a linear sub-chain."
            )
        cur = nexts[0]

    if not order:
        raise LangGraphImportError(
            "graph has no agent nodes between START and END."
        )
    return order


def custom_workflow_from_langgraph(
    compiled: Any,
    *,
    label: str,
    domain_noun: str,
    decision_verb: str,
    rubric_paragraph: str,
    llm_configs: Mapping[str, LLMConfig] | None = None,
    fallback_model: str = "google/gemini-2.5-flash",
    starter_scenarios: tuple[StarterScenario, ...] = (),
) -> CustomWorkflow:
    """Build a validated :class:`CustomWorkflow` from a compiled LangGraph.

    The returned workflow is NOT yet saved — caller passes it to
    :func:`topofuzz.ui.custom_storage.save_custom_workflow`.

    Args:
        compiled: A ``StateGraph.compile()`` result.
        label: User-facing workflow name (2-60 chars).
        domain_noun: e.g. ``"loan application"`` (2-30 chars).
        decision_verb: e.g. ``"approve / deny"`` (2-40 chars).
        rubric_paragraph: 50-2000 char rubric.
        llm_configs: Optional ``{node_id: LLMConfig}`` mapping. Models
            must be in :data:`ALLOWED_CUSTOM_MODEL_KEYS`. Nodes absent
            from this mapping fall back to ``fallback_model``.
        fallback_model: Used for any node not covered by ``llm_configs``.
            Must also be in :data:`ALLOWED_CUSTOM_MODEL_KEYS`.
        starter_scenarios: Optional pre-filled scenarios shown in the
            UI when the workflow is selected. Empty by default; the
            user can add them via the UI editor after import.

    Raises:
        LangGraphImportError: Graph is not a linear chain, or
            ``llm_configs`` references a node not in the walked graph,
            or ``fallback_model`` is not allow-listed.
        CustomFormError: A node id, model, or one of ``label`` /
            ``domain_noun`` / ``decision_verb`` / ``rubric_paragraph``
            fails BYOP validation.
    """
    if fallback_model not in ALLOWED_CUSTOM_MODEL_KEYS:
        raise LangGraphImportError(
            f"fallback_model {fallback_model!r} is not in the allow-list. "
            f"Allowed: {sorted(ALLOWED_CUSTOM_MODEL_KEYS)}."
        )

    order = _extract_linear_order(compiled)

    configs = llm_configs or {}
    unknown_keys = set(configs) - set(order)
    if unknown_keys:
        raise LangGraphImportError(
            f"llm_configs references node id(s) not in the graph: "
            f"{sorted(unknown_keys)!r}. Graph nodes: {order!r}."
        )

    agent_models: list[str] = []
    for node_id in order:
        cfg = configs.get(node_id)
        agent_models.append(cfg.model if cfg is not None else fallback_model)

    # Delegate field-shape + length-cap validation to the existing
    # BYOP form validator so import and form-submit converge to the
    # same set of rules. A graph node id that violates the agent_id
    # regex surfaces here as CustomFormError, not LangGraphImportError.
    return validate_custom_workflow_form(
        label=label,
        domain_noun=domain_noun,
        decision_verb=decision_verb,
        agent_ids=list(order),
        agent_models=agent_models,
        verdict_agent_id=order[-1],
        rubric_paragraph=rubric_paragraph,
        starter_scenarios=list(starter_scenarios),
    )


def _resolve_module_attr(spec: str) -> tuple[Any, str]:
    """Parse a ``path/to/file.py:attr`` spec, import, return ``(module, attr)``.

    Mirrors the spec format used by ``topofuzz run`` /
    :func:`topofuzz.loaders.load_topology` so users only have to learn
    one form. Windows drive letters (``C:\\foo.py:attr``) are tolerated
    by splitting on the LAST colon after stripping any 2-char drive
    prefix.
    """
    drive_prefix = ""
    rest = spec
    if len(spec) >= 2 and spec[1] == ":" and spec[0].isalpha():
        drive_prefix = spec[:2]
        rest = spec[2:]
    if ":" not in rest:
        raise LangGraphImportError(
            f"spec {spec!r} must be of the form 'path/to/file.py:attr'."
        )
    path_tail, _, attr = rest.rpartition(":")
    path_str = drive_prefix + path_tail
    if not path_str or not attr:
        raise LangGraphImportError(
            f"spec {spec!r} must have both a path and an attribute."
        )
    path = Path(path_str).resolve()
    if not path.exists():
        raise LangGraphImportError(f"fixture file not found: {path!s}")

    # One-shot stderr warning shared with `topofuzz.loaders._import_module_from_path`
    # so a user who already loaded this path via `topofuzz run` doesn't see the
    # warning twice. Same threat surface (exec_module on user-supplied code), one
    # log channel. (RT-audit S1-B1.)
    resolved_str = str(path)
    if resolved_str not in _WARNED_FIXTURE_PATHS:
        _WARNED_FIXTURE_PATHS.add(resolved_str)
        print(  # noqa: T201 — intentional stderr surface
            f"topofuzz: loading fixture {path!s} -- this executes "
            f"arbitrary Python. Only run trusted fixtures.",
            file=sys.stderr,
        )

    # Mix the file's mtime into the cache key so an in-process edit
    # (pytest reruns, long-lived UI server) reloads the fixture instead
    # of returning a stale module. Mirrors loaders._import_module_from_path
    # (audit M-6). sha1 with usedforsecurity=False — collision-resistance
    # is not the property we need, but it's stronger than the salted
    # hash() and the digest is path+mtime deterministic across processes.
    mtime_ns = path.stat().st_mtime_ns
    digest = hashlib.sha1(
        f"{resolved_str}|{mtime_ns}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    mod_name = f"topofuzz._lg_import_{digest}"
    if mod_name in sys.modules:
        return sys.modules[mod_name], attr

    importlib_spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if importlib_spec is None or importlib_spec.loader is None:
        raise LangGraphImportError(
            f"could not load module from {path!s} (spec_from_file_location "
            "returned None)."
        )
    module = importlib.util.module_from_spec(importlib_spec)
    sys.modules[mod_name] = module
    try:
        importlib_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return module, attr


def custom_workflow_from_module_spec(
    spec: str,
    *,
    label: str,
    domain_noun: str,
    decision_verb: str,
    rubric_paragraph: str,
    fallback_model: str = "google/gemini-2.5-flash",
    starter_scenarios: tuple[StarterScenario, ...] = (),
    llm_config_overrides: Mapping[str, LLMConfig] | None = None,
) -> CustomWorkflow:
    """Load a ``path:attr`` spec and import the resulting compiled graph.

    If the same module also exports ``LLM_CONFIGS: dict[str, LLMConfig]``
    (the gallery convention), it is auto-detected and threaded through
    so per-node models survive the import. Otherwise every node falls
    back to ``fallback_model``.

    ``llm_config_overrides`` are merged on top of the module's
    ``LLM_CONFIGS`` (override wins). This is how the CLI surfaces its
    repeatable ``--llm-config NODE=MODEL`` flag — users with a graph
    that has no exported ``LLM_CONFIGS`` can still pick per-node models
    without editing source.

    The ``path:attr`` form matches the runner's ``topofuzz run`` so a
    user can preview their import with the same spec they'd pass to
    the runner.
    """
    module, attr = _resolve_module_attr(spec)
    if not hasattr(module, attr):
        available = sorted(n for n in dir(module) if not n.startswith("_"))[:10]
        raise LangGraphImportError(
            f"module loaded from {spec!r} has no attribute {attr!r}. "
            f"Available: {available!r}"
        )
    target = getattr(module, attr)
    compiled = target() if callable(target) else target

    raw_configs = getattr(module, "LLM_CONFIGS", None)
    merged: dict[str, LLMConfig] = {}
    if raw_configs is not None:
        if not isinstance(raw_configs, dict):
            raise LangGraphImportError(
                f"{spec}:LLM_CONFIGS must be dict[str, LLMConfig]; got "
                f"{type(raw_configs).__name__}."
            )
        for k, v in raw_configs.items():
            if not isinstance(k, str) or not isinstance(v, LLMConfig):
                raise LangGraphImportError(
                    f"{spec}:LLM_CONFIGS has a malformed entry "
                    f"{k!r} -> {type(v).__name__}; keys must be str, "
                    "values must be LLMConfig."
                )
        merged.update(raw_configs)
    if llm_config_overrides:
        # Validate override shape too — the CLI builds these from a
        # NODE=MODEL string and an `LLMConfig(model=...)` constructor,
        # so the type should already be correct; this is defense in
        # depth for direct library callers.
        for k, v in llm_config_overrides.items():
            if not isinstance(k, str) or not isinstance(v, LLMConfig):
                raise LangGraphImportError(
                    f"llm_config_overrides has a malformed entry "
                    f"{k!r} -> {type(v).__name__}; keys must be str, "
                    "values must be LLMConfig."
                )
        merged.update(llm_config_overrides)
    llm_configs: dict[str, LLMConfig] | None = merged or None

    return custom_workflow_from_langgraph(
        compiled,
        label=label,
        domain_noun=domain_noun,
        decision_verb=decision_verb,
        rubric_paragraph=rubric_paragraph,
        llm_configs=llm_configs,
        fallback_model=fallback_model,
        starter_scenarios=starter_scenarios,
    )
