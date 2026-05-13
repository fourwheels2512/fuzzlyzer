"""Step 6 orchestrator: drive parallel ``run_variant`` calls into one RunWriter.

Single coroutine (the "writer drain") owns the SQLite connection and
flushes JSONL. Multiple variant workers compete for the cap on LLM
calls via one shared :class:`asyncio.Semaphore`. Each worker enqueues
its produced :class:`Cell` and continues; the writer drains the queue
in arrival order.

Cancellation is a single :class:`asyncio.Event` shared across all
workers and the cost-cap monitor. Causes that set ``cancel``:

* Ctrl+C (the CLI installs a signal handler — see
  :func:`topofuzz.cli.run`).
* Hard cost cap reached (``--max-cost-usd``).
* A worker raised an unrecoverable exception.

The runner's cancel contract is per-node-boundary, not per-LLM-call —
so the cost cap may overrun by **one in-flight LLM call per active
variant** (`packages/topofuzz/src/topofuzz/runner.py` module docstring,
section "Cancellation"). The cap therefore trips at the *first* overrun
boundary, not exactly at the threshold. Worth surfacing in the CLI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from fuzzeval.cache import JudgeCache, LLMCache

from topofuzz.ir import Topology
from topofuzz.metrics import classify_run_status
from topofuzz.runner import (
    Cell,
    LLMCallable,
    PromptBuilder,
    RunWriter,
    ServedVersionResolver,
    _default_prompt,
    _identity_resolver,
    run_variant,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolver-mismatch warning dedup (step 6.5)
# ---------------------------------------------------------------------------


class _ResolverMismatchDedupFilter(logging.Filter):
    """Drop duplicate resolver-mismatch warnings.

    The runner emits one warning per cell when its served-version
    resolver disagrees with the provider. A 1000-cell sweep with a
    misconfigured resolver produces 1000 identical lines (handoff
    memo step-6 gap #2). This filter dedups by the *whole formatted
    message* — the message includes both the resolver's prediction
    and the served id, so distinct mismatches still get one log each.

    Process-scope by design: a Cell whose resolver flips behavior
    mid-run is rare enough that we accept the false-negative dedup.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if "resolver" not in record.msg.lower():
            return True
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        if message in self._seen:
            return False
        self._seen.add(message)
        return True


def install_resolver_mismatch_dedup() -> _ResolverMismatchDedupFilter:
    """Attach (or reset) the dedup filter on the runner's logger.

    Returns the filter so a caller (typically tests) can detach it.
    On every call the filter's ``_seen`` set is cleared so back-to-back
    ``execute_run`` invocations in the same process don't inherit each
    other's deduped messages (audit m-2). The runner logger is module-
    scoped (``topofuzz.runner``) so the filter does not leak to
    unrelated loggers.
    """
    runner_log = logging.getLogger("topofuzz.runner")
    for existing in runner_log.filters:
        if isinstance(existing, _ResolverMismatchDedupFilter):
            existing._seen.clear()
            return existing
    f = _ResolverMismatchDedupFilter()
    runner_log.addFilter(f)
    return f


# ---------------------------------------------------------------------------
# Configuration + result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPlan:
    """Static inputs for one ``topofuzz run`` invocation.

    ``rubric_hash`` is the JCS-canonical SHA-256 of the rubric template
    text (computed once at run start by the caller — typically
    :class:`fuzzeval.judge.SingleJudge.rubric_hash`). Threaded through
    :func:`execute_run` into every cell's JudgeCache key so a rubric
    edit cleanly invalidates the cache without touching agent
    rollouts. Empty string when no judge is supplied — :class:`Cell`
    rows still have judge fields, but the cache slot is unused.
    """

    topology: Topology
    variants: tuple[Topology, ...]
    eval_rows: tuple[dict, ...]
    run_id: str
    run_dir: Path
    cache_dir: Path | None
    max_cost_usd: float | None
    judge_model: str
    semaphore_size: int = 8
    rubric_hash: str = ""


@dataclass(frozen=True)
class RunResult:
    """What the orchestrator returns to the CLI for human-friendly summary.

    ``status`` is the §A run-status stamped on the ``runs`` row, one of
    ``"succeeded"``, ``"degraded"``, ``"failed"``, ``"cost_cap_hit"``, or
    ``"interrupted"``. ``metrics_exit_code`` is the §A-mapped exit code
    the CLI should use when the run was NOT cost-cap-tripped — the cost-
    cap branch still owns its own exit code (3) at the CLI layer.
    """

    run_id: str
    run_dir: Path
    total_cells: int           # variants * len(eval_rows)
    completed_cells: int       # actually persisted (success OR cancelled)
    successful_cells: int      # error IS NULL
    total_cost_usd: float      # sum of cell_cost_usd; un-priced cells contribute 0
    cap_tripped: bool          # True if --max-cost-usd triggered cancel
    elapsed_s: float
    manifest_path: Path
    status: str = "succeeded"
    metrics_exit_code: int = 0


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(plan: RunPlan, *, cli_args: dict[str, Any]) -> Path:
    """Write ``runs/<id>/manifest.json`` at kickoff.

    The manifest records (a) what was run and (b) how it was invoked, so
    a post-mortem reader can reproduce the invocation. Token / cost
    totals are NOT in the manifest — they live in the SQLite rollup.

    ``rubric_hash`` is sourced from ``plan.rubric_hash`` (step 7) and
    surfaced at the top level so post-mortem tooling (and step 11's
    report HTML) can read it without parsing ``cli_args``. The empty-
    string default for "no rubric supplied" is preserved verbatim.
    """
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": plan.run_id,
        "topology_canonical_hash": plan.topology.canonical_hash(),
        "topology_name": plan.topology.name,
        "tds_version": plan.topology.tds_version,
        "tenant_id": "local",  # §16 #4 forward-compat placeholder
        "n_variants": len(plan.variants),
        "n_eval_rows": len(plan.eval_rows),
        "variant_canonical_hashes": [
            v.canonical_hash() for v in plan.variants
        ],
        # Per-variant node_id sequence in slot order — needed by §6a
        # per-agent influence (step 15). Persisted at kickoff so the
        # post-hoc report path can recompute without needing the live
        # Topology objects. Parallel to ``variant_canonical_hashes``;
        # rows are ordered identically.
        "variant_orderings": [
            [n.id for n in v.nodes] for v in plan.variants
        ],
        "eval_ids": [r["id"] for r in plan.eval_rows],
        "max_cost_usd": plan.max_cost_usd,
        "judge_model": plan.judge_model,
        "rubric_hash": plan.rubric_hash,
        "semaphore_size": plan.semaphore_size,
        "cli_args": cli_args,
        "kickoff_ts": time.time(),
    }
    path = plan.run_dir / "manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Orchestration loop
# ---------------------------------------------------------------------------


async def _writer_drain(
    queue: asyncio.Queue[Cell | None],
    writer: RunWriter,
    *,
    cost_state: dict,
    max_cost_usd: float | None,
    cancel: asyncio.Event,
    on_cap_trip: Callable[[], None],
) -> None:
    """Single consumer that drains ``Cell`` objects into ``RunWriter``.

    A ``None`` sentinel terminates the loop. Cost-cap evaluation lives
    here (not in the workers) so cap tripping has a single arbiter
    and races between workers don't double-count.
    """
    soft_warn_emitted = False
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        try:
            writer.write(item)
            # Aggregate cost; un-priced cells contribute 0.
            cell_cost = _sum_cost(item)
            cost_state["total_cost_usd"] += cell_cost
            cost_state["completed_cells"] += 1
            if item.error is None:
                cost_state["successful_cells"] += 1
            # Step 9: track judge_failures separately so §A status
            # classification can treat them as failed without changing
            # the existing ``successful_cells`` semantic (which is
            # ``error IS NULL`` per ``RunResult`` docstring + Cell
            # judge_failure rows have ``error=None``).
            if item.judge_failure:
                cost_state["judge_failure_cells"] += 1
            if max_cost_usd is not None:
                total = cost_state["total_cost_usd"]
                if not soft_warn_emitted and total >= 0.80 * max_cost_usd:
                    _log.warning(
                        "Cost soft-warn: $%.4f / $%.4f cap (80%% threshold).",
                        total, max_cost_usd,
                    )
                    soft_warn_emitted = True
                if total >= max_cost_usd and not cancel.is_set():
                    _log.warning(
                        "Cost hard-stop: $%.4f >= $%.4f cap. Cancelling run; "
                        "in-flight LLM calls finish but no new ones start.",
                        total, max_cost_usd,
                    )
                    cost_state["cap_tripped"] = True
                    cancel.set()
                    on_cap_trip()
        finally:
            queue.task_done()


def _sum_cost(cell: Cell) -> float:
    """Sum of NodeOutput.cost_usd, treating ``None`` as 0.0.

    Mirrors :func:`topofuzz.runner._cell_cost_usd` but always returns a
    float (the orchestrator wants a running total even when no node
    had telemetry). The runner's variant returns ``None`` for "no
    information" so the SQLite column can carry NULL.
    """
    return sum(
        (o.cost_usd or 0.0) for o in cell.node_outputs
    )


async def _worker(
    variant_ir: Topology,
    eval_row: dict,
    *,
    queue: asyncio.Queue[Cell | None],
    llm_cache: LLMCache | None,
    judge: Any,
    judge_cache: JudgeCache | None,
    rubric_hash: str,
    judge_model_version: str,
    semaphore: asyncio.Semaphore,
    prompt_builder: PromptBuilder,
    judge_prompt_renderer: Callable[[dict], str] | None,
    resolver: ServedVersionResolver,
    llm_callable: LLMCallable | None,
    cancel: asyncio.Event,
) -> None:
    """Run one variant × one eval row, enqueue the resulting Cell.

    Exceptions are caught and converted to an error-marked Cell so one
    bad row doesn't take down the run. The runner already returns a
    cancellation cell when ``cancel`` was set before kickoff; we only
    catch the residual exception surface.
    """
    try:
        cell = await run_variant(
            variant_ir,
            eval_row,
            llm_cache=llm_cache,
            judge=judge,
            judge_cache=judge_cache,
            rubric_hash=rubric_hash,
            judge_model_version=judge_model_version,
            semaphore=semaphore,
            prompt_builder=prompt_builder,
            judge_prompt_renderer=judge_prompt_renderer,
            model_served_version_resolver=resolver,
            llm_callable=llm_callable,
            cancel=cancel,
        )
    except asyncio.CancelledError:
        # Supervisor-initiated cancel after the drain dies must
        # propagate, not be persisted as a fake "CancelledError" cell.
        # See execute_run's supervisor block: the drain's exception
        # is the real cause, and reaching the except-Exception arm
        # below would record a misleading error string.
        raise
    except Exception as e:  # noqa: BLE001 — boundary: keep the run alive
        # Reconstruct a minimal failed Cell. Use empty node_outputs +
        # the exception text as ``error``; cost is None for this cell.
        cell = Cell(
            variant_id=variant_ir.canonical_hash(),
            eval_id=str(eval_row.get("id", "<unknown>")),
            ir_canonical_hash=variant_ir.canonical_hash(),
            node_outputs=(),
            judge_result=None,
            judge_failure=False,
            judge_failure_reason=None,
            error=f"{type(e).__name__}: {e}",
        )
    await queue.put(cell)


async def execute_run(
    plan: RunPlan,
    *,
    llm_callable: LLMCallable | None = None,
    judge: Any = None,
    prompt_builder: PromptBuilder = _default_prompt,
    judge_prompt_renderer: Callable[[dict], str] | None = None,
    resolver: ServedVersionResolver = _identity_resolver,
    cancel: asyncio.Event | None = None,
) -> RunResult:
    """Run all (variant, eval_row) pairs into one ``runs/<id>/`` directory.

    See module docstring for concurrency and cancellation semantics.
    Resume-aware: cells already persisted under this run_id are skipped.

    ``rubric_hash`` is sourced from ``plan.rubric_hash`` (step-7 refinement
    — was a kwarg in step 6). Threaded into every worker's JudgeCache key.

    ``judge_prompt_renderer`` is the step-7 refinement of the §7 cache
    contract. When supplied (typically ``SingleJudge.render_prompt``), the
    runner uses ``canonical_hash(rendered_judge_prompt)`` for the
    JudgeCache ``prompt_hash`` slot instead of step-5's hash-the-whole-
    eval_row stopgap. When ``None`` (no judge, or judge has no renderer),
    the runner falls back to the eval-row hash so step-5 callers continue
    working unchanged.
    """
    install_resolver_mismatch_dedup()

    # Surface the legacy fallback combination so callers see it instead
    # of silently keying the JudgeCache on the eval_row stopgap
    # (audit M6). Step-5 callers (tests, mostly) that intentionally
    # skip the renderer can ignore the warning; step-7+ orchestration
    # paths always supply both.
    if judge is not None and judge_prompt_renderer is None:
        _log.warning(
            "execute_run: judge supplied without judge_prompt_renderer. "
            "JudgeCache will key on the full eval_row (step-5 stopgap). "
            "For the step-7 contract, pass judge_prompt_renderer="
            "<judge>.render_prompt or equivalent."
        )

    cancel_event = cancel if cancel is not None else asyncio.Event()
    semaphore = asyncio.Semaphore(plan.semaphore_size)
    queue: asyncio.Queue[Cell | None] = asyncio.Queue()

    llm_cache: LLMCache | None = None
    judge_cache: JudgeCache | None = None
    if plan.cache_dir is not None:
        llm_cache = LLMCache(plan.cache_dir / "llm")
        if judge is not None:
            judge_cache = JudgeCache(plan.cache_dir / "judge")

    cost_state = {
        "total_cost_usd": 0.0,
        "completed_cells": 0,
        "successful_cells": 0,
        "judge_failure_cells": 0,
        "cap_tripped": False,
    }

    started = time.monotonic()
    already_successful_count = 0
    status = "succeeded"
    metrics_exit_code = 0
    try:
        with RunWriter(
            plan.run_dir,
            plan.run_id,
            plan.topology,
            rubric_hash=plan.rubric_hash,
        ) as writer:
            already_successful = writer.completed_cells()
            already_successful_count = len(already_successful)
            pending_pairs = _pending_pairs(plan, already_successful)

            drain_task = asyncio.create_task(
                _writer_drain(
                    queue,
                    writer,
                    cost_state=cost_state,
                    max_cost_usd=plan.max_cost_usd,
                    cancel=cancel_event,
                    on_cap_trip=lambda: None,
                ),
                name="topofuzz-runwriter-drain",
            )

            workers = [
                asyncio.create_task(
                    _worker(
                        variant_ir,
                        eval_row,
                        queue=queue,
                        llm_cache=llm_cache,
                        judge=judge,
                        judge_cache=judge_cache,
                        rubric_hash=plan.rubric_hash,
                        judge_model_version=plan.judge_model,
                        semaphore=semaphore,
                        prompt_builder=prompt_builder,
                        judge_prompt_renderer=judge_prompt_renderer,
                        resolver=resolver,
                        llm_callable=llm_callable,
                        cancel=cancel_event,
                    ),
                    name=f"topofuzz-worker-{variant_ir.canonical_hash()[:8]}-{eval_row['id']}",
                )
                for variant_ir, eval_row in pending_pairs
            ]

            # Supervisor pattern: watch BOTH the worker pool and the
            # drain. If the drain dies first (e.g. SQLite locked, disk
            # full, cell shape unexpected) we MUST cancel the workers
            # — otherwise they keep running and billing while the drain
            # exception is silently logged-and-swallowed by the
            # original M3 fix (audit B-1, second-pass). The drain
            # exception is re-raised so the caller sees the real cause.
            if workers:
                async def _wait_all_workers() -> None:
                    await asyncio.gather(*workers, return_exceptions=True)

                workers_done = asyncio.create_task(
                    _wait_all_workers(),
                    name="topofuzz-workers-gather",
                )
                done, pending = await asyncio.wait(
                    {workers_done, drain_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if drain_task in done and not workers_done.done():
                    # Drain died before workers — cancel everyone.
                    cancel_event.set()
                    for t in workers:
                        t.cancel()
                    # Wait for workers to settle so the loop teardown
                    # doesn't see un-awaited tasks.
                    try:
                        await workers_done
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                    # Surface the real drain exception.
                    drain_task.result()
                else:
                    # Workers finished (success or each-handled-error).
                    # Signal drain to terminate and wait for it.
                    await queue.put(None)
                    try:
                        await drain_task
                    except Exception:  # noqa: BLE001
                        _log.exception(
                            "RunWriter drain task failed after workers finished"
                        )
            else:
                # No pending pairs (resume of a fully-completed run).
                # Drain still needs to terminate cleanly.
                await queue.put(None)
                try:
                    await drain_task
                except Exception:  # noqa: BLE001
                    _log.exception("RunWriter drain task failed on empty plan")

            # Stamp end-of-run state on the runs row per §A error
            # taxonomy. Step 9 wires the cell-failure-rate branches
            # ('succeeded' / 'degraded' / 'failed') via
            # :func:`topofuzz.metrics.classify_run_status`; cost-cap
            # and interrupted paths keep their fixed strings.
            #
            # Priority order (highest first):
            #   1. cost_cap_hit  — cap_tripped flag (cap_tripped → cancel)
            #   2. interrupted   — cancel_event set, NOT cap_tripped (Ctrl+C)
            #   3. succeeded/degraded/failed — by failure rate
            #
            # Failure-rate counting (§A row): "judge_failure or LLM error"
            # both count. Judge-failure cells have ``error IS NULL`` so
            # the existing ``successful_cells`` counter alone underreports
            # failures; ``judge_failure_cells`` (step 9) is added in.
            n_planned = len(plan.variants) * len(plan.eval_rows)
            n_errored = (
                cost_state["completed_cells"]
                - cost_state["successful_cells"]
            )
            n_failed = n_errored + cost_state["judge_failure_cells"]

            if cost_state["cap_tripped"]:
                status = "cost_cap_hit"
                metrics_exit_code = 0  # CLI uses _COST_CAP_EXIT (3)
            elif cancel_event.is_set():
                # Cancel without cap-trip ⇒ user-initiated SIGINT (the
                # signal handler in :func:`topofuzz.cli._run_with_signal_handling`
                # sets the event on first Ctrl+C). §A maps this to
                # status='interrupted', exit code 130.
                status = "interrupted"
                metrics_exit_code = 130
            else:
                status, metrics_exit_code = classify_run_status(
                    n_failed_cells=n_failed,
                    n_total_cells=n_planned,
                    n_judge_failure_cells=cost_state["judge_failure_cells"],
                )

            # Polish-audit DRIFT-2 fix: write the actual judge protocol
            # so report HTML + §A consumers can see which judging mode
            # was used. JuryJudge is recognized structurally by the
            # presence of the ``judge_independence`` attribute — avoids
            # a fuzzeval.judge import here (the orchestrator keeps the
            # CLI's lazy-import policy intact).
            if judge is None:
                judge_protocol = None
            elif hasattr(judge, "judge_independence"):
                judge_protocol = "jury_of_3"
            else:
                judge_protocol = "single"
            writer.finalize(
                status=status,
                cost_actual_usd=cost_state["total_cost_usd"],
                n_variants=len(plan.variants),
                n_evals=len(plan.eval_rows),
                judge_protocol=judge_protocol,
            )
    finally:
        if llm_cache is not None:
            llm_cache.close()
        if judge_cache is not None:
            judge_cache.close()

    elapsed = time.monotonic() - started
    final_successful = cost_state["successful_cells"] + already_successful_count
    final_total = len(plan.variants) * len(plan.eval_rows)
    _maybe_emit_attestation(
        plan,
        status=status,
        successful_cells=final_successful,
        total_cells=final_total,
        total_cost_usd=cost_state["total_cost_usd"],
        elapsed_s=elapsed,
    )
    return RunResult(
        run_id=plan.run_id,
        run_dir=plan.run_dir,
        total_cells=final_total,
        # ``completed_cells()`` filters on ``error IS NULL`` per its
        # docstring contract, so prior-run successes contribute to both
        # counters here. If that filter ever broadens, the rename
        # (already_done → already_successful) flags the spot to update.
        completed_cells=cost_state["completed_cells"] + already_successful_count,
        successful_cells=final_successful,
        total_cost_usd=cost_state["total_cost_usd"],
        cap_tripped=cost_state["cap_tripped"],
        elapsed_s=elapsed,
        manifest_path=plan.run_dir / "manifest.json",
        status=status,
        metrics_exit_code=metrics_exit_code,
    )


def _maybe_emit_attestation(
    plan: RunPlan,
    *,
    status: str,
    successful_cells: int,
    total_cells: int,
    total_cost_usd: float,
    elapsed_s: float,
) -> None:
    """Best-effort emit of ``<run_dir>/attestation.json`` post-finalize.

    Step 10a (BLUEPRINT_v5plus.md §11). Two log levels distinguish the
    failure modes:

    * INFO — no dev key on disk. This is the common path for first-run
      users who haven't run ``topofuzz keygen``. Not an error; the run
      itself succeeded.
    * WARNING — key exists but something downstream failed: key load,
      manifest read, or sign/write. The run still succeeded, but the
      operator should investigate (corrupted key, disk-full, etc.).

    Either way this function never propagates an exception — attestation
    is additive metadata, not a precondition for a valid run.
    """
    log = logging.getLogger(__name__)
    try:
        # Imports are inside the function to avoid pulling cryptography
        # / rfc8785 at orchestrator import time for callers that don't
        # actually run (e.g. test_orchestrator's pure-helper tests).
        from fuzzeval.attestation import Ed25519Signer
        from fuzzeval.keys import (
            KeyNotFoundError,
            default_dev_priv_path,
            load_private_key,
        )

        from topofuzz.attest import emit_run_attestation
    except ImportError as e:
        log.warning("attestation imports unavailable: %s — run not signed", e)
        return

    priv_path = default_dev_priv_path()
    try:
        priv_key = load_private_key(priv_path)
    except KeyNotFoundError:
        log.info(
            "no dev key at %s — run not signed "
            "(run `topofuzz keygen` to enable attestation)",
            priv_path,
        )
        return
    except (ValueError, OSError) as e:
        log.warning(
            "dev key at %s could not be loaded: %s — run not signed",
            priv_path,
            e,
        )
        return

    manifest_path = plan.run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning(
            "manifest.json read failed at %s: %s — run not signed",
            manifest_path,
            e,
        )
        return

    signer = Ed25519Signer(priv_key)
    try:
        out_path = emit_run_attestation(
            run_dir=plan.run_dir,
            run_id=plan.run_id,
            manifest=manifest,
            status=status,
            successful_cells=successful_cells,
            total_cells=total_cells,
            n_variants=len(plan.variants),
            n_evals=len(plan.eval_rows),
            total_cost_usd=total_cost_usd,
            elapsed_s=elapsed_s,
            signer=signer,
        )
    except (OSError, ValueError, TypeError) as e:
        # OSError covers disk-full / permission-denied. ValueError +
        # TypeError cover JCS / encoding failures from rfc8785. We
        # deliberately do NOT catch broader exceptions so a real bug
        # (e.g. an AttributeError from a refactor) surfaces in tests
        # rather than being silently swallowed.
        log.warning("attestation emit failed: %s — run not signed", e)
        return
    log.info("wrote attestation: %s", out_path)


def _pending_pairs(
    plan: RunPlan, already_done: set[tuple[str, str]]
) -> list[tuple[Topology, dict]]:
    """Return the (variant, eval_row) pairs the orchestrator still owes."""
    pairs: list[tuple[Topology, dict]] = []
    for variant in plan.variants:
        vid = variant.canonical_hash()
        for row in plan.eval_rows:
            if (vid, row["id"]) in already_done:
                continue
            pairs.append((variant, row))
    return pairs


# ---------------------------------------------------------------------------
# Cap-trip helper: counted as a separate function so the CLI can test it
# ---------------------------------------------------------------------------


def cells_in_plan(plan_or_iters: Iterable[Topology] | RunPlan, n_rows: int | None = None) -> int:
    """Convenience: total cells implied by a plan or (variants, n_rows) pair.

    Used by the cost-estimator pre-flight to size the dry-run.
    """
    if isinstance(plan_or_iters, RunPlan):
        return len(plan_or_iters.variants) * len(plan_or_iters.eval_rows)
    if n_rows is None:
        raise TypeError(
            "cells_in_plan needs n_rows when given a plain variants iterable."
        )
    return sum(1 for _ in plan_or_iters) * n_rows
