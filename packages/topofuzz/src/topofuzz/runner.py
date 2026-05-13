"""Runner: execute a Topology variant against an eval row → Cell.

Per BLUEPRINT_v5plus.md §7. The runner walks ``variant_ir.nodes`` in tuple
order (variants reorder the tuple → reorder execution), calls each node's
LLM via ``litellm.acompletion``, threads results through the §7 6-tuple
``LLMCache``, calls the injected judge, threads through ``JudgeCache``,
and returns a single :class:`Cell`.

LiteLLM is imported lazily (same pattern as ``adapter.py`` for
``langgraph``) so ``topofuzz[verify]`` users don't pull ~400 MB of LLM
SDKs. A ``llm_callable`` injection point lets tests bypass LiteLLM
entirely.

**Topology is frozen — runner reads, never mutates.** Pydantic frozen
models catch direct attribute set; tuple immutability protects collection
fields. Tests still verify ``variant.canonical_hash()`` is unchanged
before/after as a belt-and-braces check on the §11 signing chain.

**Concurrency model (v0.5).** Inside one ``run_variant`` call nodes run
**sequentially** (the §7 contract — agent N's prompt depends on agent
N-1's output). The semaphore is acquired per LLM call (not per variant)
so the cap bounds simultaneous **LLM calls** across the whole orchestrator
— this is what bounds rate-limit headroom and cost-per-second, the things
operators actually care about. Variant N+1's first node may interleave
between variant N's nodes; that's intended. The orchestrator (step 6 CLI)
must share one ``asyncio.Semaphore`` across all parallel ``run_variant``
calls; the per-call default of ``Semaphore(8)`` is **useful only in
tests** — production callers without a shared semaphore get no global cap.

**Cache served-version contract (v0.5 deviation, documented).** Per
``fuzzeval/cache.py:23-46`` the §7 cache key keys on
``model_served_version`` — the post-call ``response.model``, not the
request id, because providers alias-route. The runner cannot look up
the cache before calling without knowing the served id, so the **resolver
is the single source of truth** for the served-version slot. Cache writes
go under the resolver's prediction (which is also what subsequent lookups
use), making the cache effective across runs. The actual ``response.model``
is recorded in :class:`NodeOutput` for forensics and a warning fires on
mismatch. Net effect:

* Providers that don't alias-route (Gemini direct, OpenAI direct):
  identity resolver is correct → §7-compliant.
* Providers that alias-route (Anthropic, multi-region OpenAI): caller
  MUST inject a resolver that returns the served id. Without one, cache
  is consistent and effective across runs but is technically keyed on
  the request id (deviation from §7 spec text — accepted for v0.5,
  surface in step 6 CLI's first-run warning).

**Cancellation (step 6 readiness).** ``run_variant`` accepts an optional
``cancel: asyncio.Event`` checked between nodes and before the judge call.
On cancellation the runner returns a :class:`Cell` with
``error="cancelled"`` and whatever ``node_outputs`` were already produced;
the caller can persist a partial cell so resume picks up where it stopped.
This is what step 6's ``--max-cost-usd`` hard-stop and Ctrl+C handling
will use. **Granularity is per-node-boundary, not per-LLM-call**: an
in-flight ``_call_llm`` runs to completion (and its response is cached)
before cancel takes effect. Step 6's cost cap may overrun by one
LLM call per active variant — acceptable because cancelling mid-HTTP
would orphan the response without caching it.

**Step-6 readiness gaps still open** (tracked in PLAN.md for step 6 prep):

* Per-node cost / token telemetry. ``NodeOutput`` records ``latency_ms``
  but no ``input_tokens`` / ``output_tokens`` / ``cost_usd``. Step 6's
  ``--max-cost-usd`` cap will need these — add a 6.0 prep step that
  threads ``response.usage`` through ``_extract_text_and_served`` before
  the orchestrator code starts.
* Per-process resolver-mismatch warning dedup. Currently warns once per
  cell; a 1000-cell sweep with a misconfigured Anthropic resolver
  produces ~1000 identical lines. Step 6's CLI should install a
  ``logging.Filter`` that dedupes by ``(predicted, actual)`` pair.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fuzzeval.cache import JudgeCache, LLMCache
from fuzzeval.canonical import canonical_hash
from fuzzeval.costest import cost_for_call
from fuzzeval.judge import BinaryScore, ContinuousScore, Judge, JudgeResult
from topofuzz.ir import Node, Topology

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeOutput:
    """One agent's contribution to a Cell.

    ``model_served_version`` is the LiteLLM ``response.model`` value (or
    the resolver's prediction on a cache hit). ``cache_hit=True`` means
    the response came from ``LLMCache`` and was not billed.

    **Token / cost fields are optional.** They are populated only when the
    underlying ``llm_callable`` returned a ``usage`` payload (LiteLLM
    ``ModelResponse.usage`` / dict ``"usage"``). They are ``None`` on:

    * cache hits (the response was not billed — see :class:`fuzzeval.cache.LLMCache`),
    * responses without ``usage`` (older mocks, providers that don't report it),
    * models absent from :data:`fuzzeval.costest._RATE_USD_PER_1K`
      (``cost_usd`` is ``None`` even when token counts are present).

    Step 6's ``--max-cost-usd`` cap aggregates non-``None`` ``cost_usd`` values
    across cells. A run that mixes priced + un-priced models will under-count;
    surface this in the CLI's first-run warning.
    """

    node_id: str
    role: str
    response_text: str
    model_served_version: str
    cache_hit: bool
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class Cell:
    """One (variant, eval) result row.

    ``variant_id`` IS ``ir_canonical_hash`` — duplicated in the schema for
    ergonomic SQL filtering ("WHERE variant_id = ...") without forcing
    callers to remember which column to use. The §9 metrics consume
    ``scores[variant_id][eval_id]``; downstream code keys on ``variant_id``.

    ``judge_result`` is None when no judge was supplied OR when the
    judge raised. Distinguish via ``judge_failure``: True means a judge
    was supplied and failed (count toward §8 ``judge_failure_rate >
    5% → degraded``); False means no judge was supplied at all.

    ``error`` is None on success. Populated on cancellation
    (``"cancelled"``). The runner does NOT swallow LLM-side exceptions;
    those propagate so the orchestrator can decide retry policy. Caching
    a malformed response is worse than re-billing on a deterministic
    response-shape failure — fix the driver, not the runner.
    """

    variant_id: str
    eval_id: str
    ir_canonical_hash: str
    node_outputs: tuple[NodeOutput, ...]
    judge_result: JudgeResult | None
    judge_failure: bool
    judge_failure_reason: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Injection-point types
# ---------------------------------------------------------------------------

PromptBuilder = Callable[[Node, dict, tuple[NodeOutput, ...]], str]
"""Build the prompt text for one node. Receives the node, the eval row,
and the tuple of prior NodeOutputs in execution order.

The default builder (:func:`_default_prompt`) only reads
``eval_row['input']``; if your eval rows carry additional context
(``expected_answer``, ``category``, etc.) supply a custom builder so the
prompt — and therefore the §7 ``prompt_hash`` cache key slot — reflects
the full input."""

ServedVersionResolver = Callable[[str], str]
"""Map a requested model id to the expected served-version id. Default
is the identity function — correct for providers that don't alias-route
(Gemini direct, OpenAI direct). Anthropic / multi-region OpenAI users
should inject a real resolver to satisfy the §7 served-version cache
contract."""

LLMCallable = Callable[..., Awaitable[Any]]
"""Async LLM call; default is ``litellm.acompletion``. Tests inject a
fake to avoid network. Must accept ``model``, ``messages``, ``temperature``,
``max_tokens`` and optionally ``seed``; must return an object with
``.choices[0].message.content`` and ``.model`` (or the dict equivalent)."""


def _default_prompt(
    node: Node, eval_row: dict, prior: tuple[NodeOutput, ...]
) -> str:
    """Minimal default prompt; reads ONLY ``eval_row['input']``.

    Real workloads should override via the ``prompt_builder`` argument so
    additional eval-row fields (context, expected output, category) flow
    into the §7 ``prompt_hash`` and don't silently collide on cache.
    """
    parts: list[str] = [
        f"You are the {node.role!r} agent in a multi-agent pipeline.",
        f"Task input: {eval_row.get('input', '')!s}",
    ]
    if prior:
        parts.append("Prior agent outputs:")
        for p in prior:
            parts.append(f"- [{p.role}] {p.response_text}")
    parts.append("Respond with your analysis.")
    return "\n".join(parts)


def _identity_resolver(model: str) -> str:
    return model


def _safe_resolve(
    resolver: ServedVersionResolver, requested_model: str
) -> str:
    """Run a user-supplied resolver under a try/except.

    A buggy resolver shouldn't take down a multi-hour sweep. On error,
    fall back to identity (request model) and log a warning. The cache
    key for that node will then key on the request id (consistent with
    the no-resolver default), accepted for v0.5.
    """
    try:
        result = resolver(requested_model)
    except Exception as e:  # noqa: BLE001 — boundary: untrusted user code
        _log.warning(
            "model_served_version_resolver raised %s for model %r: %s; "
            "falling back to request id.",
            type(e).__name__, requested_model, e,
        )
        return requested_model
    if not isinstance(result, str) or not result:
        _log.warning(
            "model_served_version_resolver returned %r (expected non-empty "
            "str) for model %r; falling back to request id.",
            result, requested_model,
        )
        return requested_model
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _cancelled_cell(
    *,
    ir_hash: str,
    eval_id: str,
    outputs: list[NodeOutput],
    reason: str,
) -> Cell:
    return Cell(
        variant_id=ir_hash,
        eval_id=eval_id,
        ir_canonical_hash=ir_hash,
        node_outputs=tuple(outputs),
        judge_result=None,
        judge_failure=False,
        judge_failure_reason=None,
        error=reason,
    )


async def run_variant(
    variant_ir: Topology,
    eval_row: dict,
    *,
    llm_cache: LLMCache | None = None,
    judge: Judge | None = None,
    judge_cache: JudgeCache | None = None,
    rubric_hash: str = "",
    judge_model_version: str = "",
    semaphore: asyncio.Semaphore | None = None,
    prompt_builder: PromptBuilder = _default_prompt,
    judge_prompt_renderer: Callable[[dict], str] | None = None,
    model_served_version_resolver: ServedVersionResolver = _identity_resolver,
    llm_callable: LLMCallable | None = None,
    cancel: asyncio.Event | None = None,
) -> Cell:
    """Execute ``variant_ir`` against ``eval_row`` → :class:`Cell`.

    See module docstring for cache, concurrency, resolver, and
    cancellation contracts. ``tenant_id`` is intentionally NOT a parameter
    here — it lives on the ``LLMCache`` / ``JudgeCache`` constructors so
    that "this whole cache is tenant X" is the single source of truth.
    Supply the right cache; do not try to override per call.

    Args:
        variant_ir: The (frozen) Topology to execute.
        eval_row: Dict with required key ``id`` (str) and ``input`` (any).
            Extra keys are passed through to ``prompt_builder`` and the
            judge payload unchanged. The full row (sorted-key JCS hash)
            contributes to the JudgeCache key — adding/removing fields
            invalidates cached judgements for that row.
        llm_cache: Optional. Skipped when None (no caching, every call
            hits the LLM). Production callers always supply one.
        judge: Optional. Skipped when None (Cell.judge_result stays None
            and judge_failure=False).
        judge_cache: Optional. Only consulted when ``judge`` is supplied.
        rubric_hash: Hash of the rubric prompt; goes into the JudgeCache key.
            v0.5 callers compute this once at run start.
        judge_model_version: Goes into the JudgeCache key. Same model
            string judges use to advertise themselves (e.g.
            ``"google/gemini-2.5-pro@2026-04"``).
        semaphore: Optional shared concurrency cap on simultaneous LLM
            calls. Default fresh ``Semaphore(8)`` — useful in tests, wrong
            for production (each call would get its own cap).
        prompt_builder: See type alias docstring.
        judge_prompt_renderer: Optional ``(payload: dict) -> str`` used to
            compute the §7 JudgeCache ``prompt_hash`` slot. Step 7+
            callers pass :meth:`fuzzeval.judge.SingleJudge.render_prompt`
            so the cache key is the byte-exact prompt the judge saw.
            When ``None`` (step-5 callers / no judge) the runner falls
            back to hashing the full ``eval_row`` — preserves step-5
            behavior but produces a different cache row than the
            renderer version. A renderer that raises produces a
            ``judge_failure`` cell rather than falling back, so a buggy
            renderer is loud, not silent.
        model_served_version_resolver: See type alias docstring.
        llm_callable: See type alias docstring.
        cancel: Optional :class:`asyncio.Event` checked between nodes and
            before the judge call. When set, returns a Cell with
            ``error="cancelled"`` and the partial ``node_outputs``.

    Returns:
        A :class:`Cell` with ``variant_id == variant_ir.canonical_hash()``.

    Raises:
        ValueError: ``eval_row`` missing ``id``; or a node has
            ``llm_config=None``.
        ImportError: Default ``llm_callable`` requested but ``litellm``
            isn't installed (install ``topofuzz[full]``).
    """
    eval_id = eval_row.get("id")
    if not isinstance(eval_id, str) or not eval_id:
        raise ValueError(
            f"eval_row['id'] must be a non-empty str; got {eval_id!r}"
        )

    ir_hash = variant_ir.canonical_hash()
    sem = semaphore if semaphore is not None else asyncio.Semaphore(8)

    outputs: list[NodeOutput] = []
    for node in variant_ir.nodes:
        if cancel is not None and cancel.is_set():
            return _cancelled_cell(
                ir_hash=ir_hash, eval_id=eval_id,
                outputs=outputs, reason="cancelled",
            )
        if node.llm_config is None:
            raise ValueError(
                f"Node {node.id!r} has llm_config=None; runner requires every "
                "node to have an LLMConfig. Pass llm_configs to "
                "from_langgraph(), or set node.llm_config before running."
            )

        prompt = prompt_builder(node, eval_row, tuple(outputs))
        if not isinstance(prompt, str):
            raise TypeError(
                f"prompt_builder returned {type(prompt).__name__} for "
                f"node {node.id!r}; must return str."
            )
        prompt_hash = canonical_hash(prompt)
        predicted_served = _safe_resolve(
            model_served_version_resolver, node.llm_config.model
        )

        identity = {
            "ir_canonical_hash": ir_hash,
            "node_id": node.id,
            "llm_config_hash": node.llm_config.canonical_hash(),
            "prompt_hash": prompt_hash,
            "model_served_version": predicted_served,
        }

        cached_bytes: bytes | None = None
        if llm_cache is not None:
            cached_bytes = llm_cache.get(identity)

        input_tokens: int | None = None
        output_tokens: int | None = None
        cost_usd: float | None = None
        if cached_bytes is not None:
            response_text = cached_bytes.decode("utf-8")
            served_version = predicted_served
            cache_hit = True
            latency_ms = 0
        else:
            async with sem:
                started = time.monotonic()
                response = await _call_llm(node, prompt, llm_callable)
                latency_ms = int((time.monotonic() - started) * 1000)
            response_text, actual_served = _extract_text_and_served(response)
            input_tokens, output_tokens = _extract_usage(response)
            cost_usd = cost_for_call(
                node.llm_config.model, input_tokens, output_tokens
            )

            # Cache under the resolver's prediction (also the lookup key) so
            # subsequent runs hit. The §7 spec text says "key on served
            # version"; here the resolver is the single source of truth
            # for that slot — see module docstring "Cache served-version
            # contract" for the v0.5 deviation rationale. The actual
            # served id is still recorded in NodeOutput for forensics.
            served_version = actual_served or predicted_served
            if actual_served and actual_served != predicted_served:
                _log.warning(
                    "model_served_version_resolver returned %r for request "
                    "%r but provider served %r. Cache is keyed on the "
                    "resolver's prediction; to get §7-compliant served-"
                    "version caching, update the resolver to return the "
                    "actual served id.",
                    predicted_served,
                    node.llm_config.model,
                    actual_served,
                )

            cache_hit = False
            if llm_cache is not None:
                llm_cache.set(identity, response_text.encode("utf-8"))

        outputs.append(
            NodeOutput(
                node_id=node.id,
                role=node.role,
                response_text=response_text,
                model_served_version=served_version,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        )

    if cancel is not None and cancel.is_set():
        return _cancelled_cell(
            ir_hash=ir_hash, eval_id=eval_id,
            outputs=outputs, reason="cancelled",
        )

    judge_result, judge_failure, judge_failure_reason = await _run_judge(
        judge=judge,
        judge_cache=judge_cache,
        outputs=outputs,
        eval_row=eval_row,
        variant_id=ir_hash,
        rubric_hash=rubric_hash,
        judge_model_version=judge_model_version,
        judge_prompt_renderer=judge_prompt_renderer,
    )

    return Cell(
        variant_id=ir_hash,
        eval_id=eval_id,
        ir_canonical_hash=ir_hash,
        node_outputs=tuple(outputs),
        judge_result=judge_result,
        judge_failure=judge_failure,
        judge_failure_reason=judge_failure_reason,
        error=None,
    )


# ---------------------------------------------------------------------------
# LLM + judge plumbing
# ---------------------------------------------------------------------------


async def _call_llm(
    node: Node, prompt: str, llm_callable: LLMCallable | None
) -> Any:
    cfg = node.llm_config
    assert cfg is not None  # checked in run_variant
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.seed is not None:
        kwargs["seed"] = cfg.seed

    if llm_callable is not None:
        return await llm_callable(**kwargs)

    try:
        from litellm import acompletion
    except ImportError as e:
        raise ImportError(
            "topofuzz.runner requires litellm for the default llm_callable; "
            "install via `pip install topofuzz[full]` or inject a custom "
            "llm_callable into run_variant()."
        ) from e
    return await acompletion(**kwargs)


def _extract_text_and_served(response: Any) -> tuple[str, str]:
    """Pull (text, served_model_id) from a LiteLLM-shaped response.

    Accepts either the LiteLLM ``ModelResponse`` object or a dict (some
    providers / mocks return one). Raises TypeError if either field is
    missing or not a string — those failures should not silently corrupt
    the cache.
    """
    try:
        if isinstance(response, dict):
            text = response["choices"][0]["message"]["content"]
            served = response.get("model", "")
        else:
            text = response.choices[0].message.content
            served = getattr(response, "model", "") or ""
    except (KeyError, IndexError, AttributeError) as e:
        raise TypeError(
            f"LLM response shape unexpected: {type(response).__name__}; "
            f"could not extract choices[0].message.content / .model: {e}"
        ) from e

    if not isinstance(text, str):
        raise TypeError(
            f"LLM response content is {type(text).__name__}, expected str. "
            "Refusing to cache non-string content."
        )
    return text, str(served or "")


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) from a LiteLLM-shaped response.

    Returns ``(None, None)`` when ``usage`` is absent or malformed — the
    runner treats missing usage as "not billable from telemetry" and
    leaves :class:`NodeOutput.cost_usd` ``None``. This is intentional: a
    cost-cap that silently substitutes zeros for missing token counts
    would overrun the cap on un-instrumented runs.

    Accepts the LiteLLM names (``prompt_tokens`` / ``completion_tokens``)
    and the OpenAI-Python-SDK aliases (``input_tokens`` / ``output_tokens``).
    Both dict envelopes and attribute-bearing objects are supported.
    """
    usage: Any = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None, None

    def _get(field_pair: tuple[str, str]) -> int | None:
        for name in field_pair:
            if isinstance(usage, dict):
                if name in usage:
                    val = usage[name]
                    break
            else:
                if hasattr(usage, name):
                    val = getattr(usage, name)
                    break
        else:
            return None
        if val is None or isinstance(val, bool):
            return None
        # Some older provider proxies wrap usage values as strings.
        # Coerce; reject anything that fails int() — strings like "100"
        # become 100, "1.5" or "abc" become None (audit m-8).
        if isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                return None
        if not isinstance(val, int):
            return None
        if val < 0:
            return None
        return val

    return _get(("prompt_tokens", "input_tokens")), _get(
        ("completion_tokens", "output_tokens")
    )


def _eval_row_hash(eval_row: dict) -> str:
    """Stable hash of the entire eval_row.

    Covers ``input`` plus any other field a custom prompt builder or judge
    might reference (``expected_answer``, ``category``, ``ground_truth``).
    JCS canonicalisation in :func:`canonical_hash` normalises dict
    insertion order, so two logically-equal rows produce the same hash.
    """
    try:
        return canonical_hash(eval_row)
    except TypeError as e:
        raise TypeError(
            f"eval_row {eval_row.get('id')!r} contains a non-JSON-"
            f"serialisable value; the JudgeCache cannot key on it. "
            f"Convert sets / bytes / numpy scalars to JSON-native types "
            f"before passing to run_variant. Underlying error: {e}"
        ) from e


async def _run_judge(
    *,
    judge: Judge | None,
    judge_cache: JudgeCache | None,
    outputs: list[NodeOutput],
    eval_row: dict,
    variant_id: str,
    rubric_hash: str,
    judge_model_version: str,
    judge_prompt_renderer: Callable[[dict], str] | None = None,
) -> tuple[JudgeResult | None, bool, str | None]:
    if judge is None or not outputs:
        return None, False, None

    final_output = outputs[-1].response_text
    payload = {
        "variant_id": variant_id,
        "eval_row": eval_row,
        "agent_output": final_output,
    }

    # §7 ``prompt_hash`` slot.
    #
    # Step 7 refinement: when the caller supplies ``judge_prompt_renderer``
    # (typically ``SingleJudge.render_prompt``), hash the rendered text
    # exactly as the judge would see it. Two cells whose rendered prompts
    # are byte-equal therefore share a JudgeCache row — the right
    # invalidation behavior when the rubric or the per-row substitution
    # changes.
    #
    # Step 5/6 fallback: hash the full eval_row. This keeps step-5
    # behavior intact for callers (mostly tests) that supply a Judge but
    # no renderer. A renderer that itself raises (bad payload, template
    # error) is caught here and converted into a judge_failure cell —
    # the eval_row hash is NOT used as a fallback after the renderer
    # raised, because that would silently let a buggy renderer share a
    # cache row with the step-5 stopgap behavior.
    #
    # We pass a ``copy.deepcopy`` of the payload to the renderer so a
    # buggy user-supplied renderer that mutates ``payload['eval_row']``
    # in place can't poison the judge's view downstream (audit MINOR:
    # renderer mutation risk).
    if judge_prompt_renderer is not None:
        try:
            rendered = judge_prompt_renderer(copy.deepcopy(payload))
        except Exception as e:  # noqa: BLE001 — boundary: untrusted user code
            return None, True, (
                f"judge_prompt_renderer raised "
                f"{type(e).__name__}: {e}"
            )
        if not isinstance(rendered, str):
            return None, True, (
                "judge_prompt_renderer returned "
                f"{type(rendered).__name__}; expected str."
            )
        prompt_hash = canonical_hash(rendered)
    else:
        prompt_hash = _eval_row_hash(eval_row)

    judge_identity = {
        "rubric_hash": rubric_hash,
        "prompt_hash": prompt_hash,
        "response_hash": canonical_hash(final_output),
        "judge_model_version": judge_model_version,
    }

    if judge_cache is not None:
        cached = judge_cache.get(judge_identity)
        if cached is not None:
            return cached, False, None
    try:
        # Judge.score is sync per the Protocol; offload to a worker thread
        # so a slow judge doesn't block the event loop's other run_variant
        # coroutines from making progress on cached LLM lookups. Note:
        # asyncio.to_thread uses the default executor, which does NOT
        # support task cancellation — a hung judge wedges its worker
        # thread for the run lifetime. Acceptable for v0.5 (judges are
        # HTTP-bounded in practice); revisit if a real user reports it.
        result = await asyncio.to_thread(judge.score, payload)
    except (KeyboardInterrupt, SystemExit):
        # Don't swallow user-initiated cancellation as a judge failure.
        raise
    except Exception as e:  # noqa: BLE001 — runner is the failure boundary
        return None, True, f"{type(e).__name__}: {e}"

    if not isinstance(result, (ContinuousScore, BinaryScore)):
        return (
            None,
            True,
            f"Judge returned non-JudgeResult type {type(result).__name__}",
        )

    if judge_cache is not None:
        judge_cache.set(judge_identity, result)
    return result, False, None


# ---------------------------------------------------------------------------
# Persistence (RunWriter)
# ---------------------------------------------------------------------------


# Schema version bumps require a migration path. v0.5.0 step 5 = 1.
# Step 6 = 2 (cost telemetry: cells.cell_cost_usd). Any column add or
# rename must increment and either (a) ship a migration in `topofuzz
# migrate`, or (b) inline a one-shot ALTER in :func:`_migrate` for a
# small step bump pre-launch. The full migrate command is tracked as a
# step-6 readiness gap; pre-launch we ALTER inline.
#
# Schema DDL deliberately omits ``PRAGMA user_version``. The PRAGMA is
# written by :func:`_migrate` after CREATE/ALTER finishes, so the
# pre-existing user_version (0 for fresh DBs, 1 for legacy step-5 DBs,
# 99+ for a hypothetical future DB) is observable when migrate decides
# whether to ALTER or refuse.
_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    topology_canonical_hash TEXT NOT NULL,
    topology_name TEXT NOT NULL,
    tds_version TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cells (
    run_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    eval_id TEXT NOT NULL,
    ir_canonical_hash TEXT NOT NULL,
    node_outputs_json TEXT NOT NULL,
    error TEXT,
    cell_cost_usd REAL,
    PRIMARY KEY (run_id, variant_id, eval_id)
);
CREATE TABLE IF NOT EXISTS judge_results (
    run_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    eval_id TEXT NOT NULL,
    judge_kind TEXT NOT NULL,
    score_value REAL,
    passed INTEGER,
    rationale TEXT,
    failure_reason TEXT,
    PRIMARY KEY (run_id, variant_id, eval_id)
);
"""


def _check_supported_schema(db: sqlite3.Connection) -> None:
    """Refuse to open a run.sqlite written by a newer topofuzz.

    Run BEFORE :data:`_SCHEMA` so a forward-compat user_version isn't
    overwritten before we look at it.
    """
    current = db.execute("PRAGMA user_version").fetchone()[0]
    if current > _SCHEMA_VERSION:
        raise RuntimeError(
            f"run.sqlite was written by a newer topofuzz "
            f"(schema {current} > supported {_SCHEMA_VERSION}). "
            "Downgrade refused; upgrade topofuzz to read this run."
        )


def _migrate(db: sqlite3.Connection) -> None:
    """Run any pre-launch in-place migrations.

    Pre-launch (v0.5.0.dev*) we inline column adds rather than ship a
    full ``topofuzz migrate`` command — there are no shipped DBs in the
    wild. Each migration is idempotent so a fresh DB (where CREATE TABLE
    already produced the v2 column set) is a no-op.

    Called AFTER :data:`_SCHEMA` runs CREATE TABLE IF NOT EXISTS. For
    fresh DBs the table already has the v2 column set; for legacy v1
    DBs the table is the v1 shape and needs ALTER.
    """
    current = db.execute("PRAGMA user_version").fetchone()[0]
    # _check_supported_schema already refused current > _SCHEMA_VERSION
    # before CREATE TABLE ran; assert defends against caller drift.
    assert current <= _SCHEMA_VERSION, "forward-version slipped past check"
    if current < 2:
        cols = {
            row[1]
            for row in db.execute("PRAGMA table_info(cells)").fetchall()
        }
        if "cell_cost_usd" not in cols:
            db.execute("ALTER TABLE cells ADD COLUMN cell_cost_usd REAL")
    db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    db.commit()


def _judge_row(
    cell: Cell,
) -> tuple[str, float | None, int | None, str | None, str | None]:
    """Pack a Cell's judge state into the judge_results column tuple.

    judge_kind:
        "continuous" — ContinuousScore
        "binary"     — BinaryScore
        "failure"    — judge raised
        "absent"     — no judge supplied
    """
    if cell.judge_failure:
        return "failure", None, None, None, cell.judge_failure_reason
    if isinstance(cell.judge_result, ContinuousScore):
        return "continuous", cell.judge_result.value, None, None, None
    if isinstance(cell.judge_result, BinaryScore):
        passed = 1 if cell.judge_result.passed else 0
        return "binary", None, passed, cell.judge_result.rationale, None
    return "absent", None, None, None, None


def _cell_cost_usd(cell: Cell) -> float | None:
    """Sum non-``None`` ``cost_usd`` across a cell's NodeOutputs.

    Returns ``None`` only when **every** node lacks cost telemetry (all
    cache hits, all un-priced models, or all without ``usage``). Returns
    ``0.0`` for a degenerate empty-outputs cell (e.g. cancellation before
    the first node) so downstream "sum cell costs" math stays in float
    space without a None branch per cell. A cell with one priced node
    and one un-priced node returns the priced node's cost — the un-priced
    node contributes 0; surface this in the CLI's first-run warning.
    """
    if not cell.node_outputs:
        return None
    contributions = [o.cost_usd for o in cell.node_outputs if o.cost_usd is not None]
    if not contributions:
        return None
    return float(sum(contributions))


def _judge_to_jsonable(cell: Cell) -> dict | None:
    if cell.judge_failure:
        return {"kind": "failure", "reason": cell.judge_failure_reason}
    if isinstance(cell.judge_result, ContinuousScore):
        return {"kind": "continuous", "value": cell.judge_result.value}
    if isinstance(cell.judge_result, BinaryScore):
        return {
            "kind": "binary",
            "passed": cell.judge_result.passed,
            "rationale": cell.judge_result.rationale,
        }
    return None


def _validate_run_dir(run_dir: Path) -> Path:
    """Refuse a filesystem-root ``run_dir`` to prevent ``run.sqlite`` at /.

    A buggy step-6 CLI that defaults ``--run-dir`` to ``Path()`` and
    appends an empty fragment can land at the drive root on Windows or
    the filesystem root on Unix. Both writes succeed silently, both are
    catastrophic. Refuse at construction.
    """
    resolved = run_dir.resolve()
    if str(resolved) == str(resolved.anchor):
        raise ValueError(
            f"run_dir {run_dir!r} resolves to filesystem root "
            f"{resolved!s}; refusing to write run artifacts there. "
            "Pass a project-scoped path like ./runs/<id>."
        )
    return resolved


class RunWriter:
    """Append cells to ``runs/<id>/traces.jsonl`` and ``runs/<id>/run.sqlite``.

    Single-writer design (v0.5). The orchestrator (step 6 CLI) is expected
    to call :meth:`write` from one coroutine — typically draining a queue
    fed by parallel ``run_variant`` calls. The SQLite connection uses
    ``check_same_thread=True`` (the default) to enforce this contract at
    runtime — a multi-thread call surface would silently corrupt the
    journal.

    ``INSERT OR REPLACE`` semantics make :meth:`write` idempotent — a
    crash mid-run is recovered by re-running the same orchestrator,
    which queries :meth:`completed_cells` for already-persisted
    ``(variant_id, eval_id)`` tuples and skips them.

    **Resume policy.** :meth:`completed_cells` returns successful cells
    only (``error IS NULL``). Cells with a recorded ``error`` (e.g.
    ``"cancelled"``) are NOT returned, so the orchestrator will re-run
    them. This destroys the original error trail on retry — if you want
    forensic preservation, copy ``run.sqlite`` before resuming. Judge
    failures (``judge_kind = 'failure'``) are persisted on the
    ``judge_results`` row but are NOT in scope of the resume filter —
    they count as "completed" because the cell-level row itself is
    error-free; re-judging requires manual row deletion.

    **Durability.** ``run.sqlite`` is the authoritative resume source —
    every :meth:`write` commits. ``traces.jsonl`` is best-effort: only
    flushed (not fsync'd) per record, so a crash mid-line can leave a
    half-written record that breaks ``cat traces.jsonl | jq``. Recover by
    rewriting the JSONL from the SQLite tables.

    **Judge implementations MUST NOT call back into this writer.** The
    judge runs in a worker thread (``asyncio.to_thread``) and the SQLite
    connection is bound to the main thread (``check_same_thread=True``).
    Persist all writes from the orchestrator coroutine.
    """

    def __init__(
        self,
        run_dir: Path | str,
        run_id: str,
        topology: Topology,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self._run_dir = _validate_run_dir(Path(run_dir))
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id

        self._db: sqlite3.Connection | None = None
        self._jsonl: Any = None
        try:
            self._db = sqlite3.connect(str(self._run_dir / "run.sqlite"))
            _check_supported_schema(self._db)
            self._db.executescript(_SCHEMA)
            _migrate(self._db)
            self._init_runs_row(run_id, topology)
            self._jsonl = (self._run_dir / "traces.jsonl").open(
                "a", encoding="utf-8"
            )
        except BaseException:
            # Roll back the partial init so a Windows file lock doesn't
            # break the next attempt. close() tolerates a half-built state.
            self._safe_close()
            raise

    def _init_runs_row(self, run_id: str, topology: Topology) -> None:
        """Insert (or verify) the runs-table row for this run_id.

        On resume — when a runs row already exists for this run_id —
        verify the topology hash matches. A mismatch means the caller is
        resuming a different topology under the same run_id, which would
        silently mix variant_ids in the cells table (the runs row keeps
        the original hash; new cells get the new variant_ids). Refuse
        with a clear error so the orchestrator picks a fresh run_id or
        reconciles the topology.
        """
        assert self._db is not None
        existing = self._db.execute(
            "SELECT topology_canonical_hash FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != topology.canonical_hash():
                raise ValueError(
                    f"Resume topology mismatch for run_id={run_id!r}: "
                    f"existing run was kicked off with topology hash "
                    f"{existing[0]!r}, but this RunWriter was given a "
                    f"topology with hash {topology.canonical_hash()!r}. "
                    f"Pick a fresh run_id, or pass the original topology."
                )
            return
        # First kickoff. INSERT (no OR IGNORE — we already proved no row
        # exists for this run_id, so a conflict here is a real bug).
        self._db.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                topology.canonical_hash(),
                topology.name,
                topology.tds_version,
                time.time(),
            ),
        )
        self._db.commit()

    def _safe_close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:  # noqa: BLE001
                pass
            self._db = None
        if self._jsonl is not None:
            try:
                self._jsonl.close()
            except Exception:  # noqa: BLE001
                pass
            self._jsonl = None

    def write(self, cell: Cell) -> None:
        if self._db is None or self._jsonl is None:
            raise RuntimeError("RunWriter is closed")

        node_dicts = [asdict(o) for o in cell.node_outputs]
        node_outputs_json = json.dumps(node_dicts, ensure_ascii=False)

        cell_cost = _cell_cost_usd(cell)

        record = {
            "run_id": self._run_id,
            "variant_id": cell.variant_id,
            "eval_id": cell.eval_id,
            "ir_canonical_hash": cell.ir_canonical_hash,
            "node_outputs": node_dicts,
            "judge_result": _judge_to_jsonable(cell),
            "judge_failure": cell.judge_failure,
            "judge_failure_reason": cell.judge_failure_reason,
            "error": cell.error,
            "cell_cost_usd": cell_cost,
        }
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()

        self._db.execute(
            "INSERT OR REPLACE INTO cells "
            "(run_id, variant_id, eval_id, ir_canonical_hash, "
            " node_outputs_json, error, cell_cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self._run_id,
                cell.variant_id,
                cell.eval_id,
                cell.ir_canonical_hash,
                node_outputs_json,
                cell.error,
                cell_cost,
            ),
        )
        kind, score_value, passed, rationale, failure_reason = _judge_row(cell)
        self._db.execute(
            "INSERT OR REPLACE INTO judge_results VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._run_id,
                cell.variant_id,
                cell.eval_id,
                kind,
                score_value,
                passed,
                rationale,
                failure_reason,
            ),
        )
        self._db.commit()

    def completed_cells(self) -> set[tuple[str, str]]:
        """Return the set of ``(variant_id, eval_id)`` already persisted.

        See class docstring **Resume policy** for the exact filter.
        """
        if self._db is None:
            raise RuntimeError("RunWriter is closed")
        rows = self._db.execute(
            "SELECT variant_id, eval_id FROM cells "
            "WHERE run_id = ? AND error IS NULL",
            (self._run_id,),
        ).fetchall()
        return {(v, e) for v, e in rows}

    def close(self) -> None:
        self._safe_close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()


__all__ = [
    "Cell",
    "LLMCallable",
    "NodeOutput",
    "PromptBuilder",
    "RunWriter",
    "ServedVersionResolver",
    "run_variant",
]
