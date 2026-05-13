"""topofuzz CLI entrypoint.

Step 6: ``topofuzz run`` is wired. ``scan`` / ``report`` / ``verify``
remain stubs until later weekends per BLUEPRINT_v5plus.md §14.

CLI surface for ``run`` is the blueprint-literal shape (user decision,
2026-05-12):

    topofuzz run FIXTURE_PATH:graph \\
        --eval path/to/eval.jsonl \\
        --n 50 \\
        --variants 8 \\
        --max-cost-usd 50 \\
        --judge google/gemini-2.5-pro \\
        --out runs/<id> \\
        --cache .cache/

Reserved-flag namespace per BLUEPRINT §16 #2 raises a friendly error
pointing at v1.0. The exit code for those is the same as ``--help``-on-
error so CI checks don't have to special-case it.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

import typer

from topofuzz import __version__
from topofuzz.loaders import load_eval_rows, load_topology
from topofuzz.orchestrator import RunPlan, execute_run, write_manifest
from topofuzz.runids import new_run_id
from topofuzz.variants.generators import (
    decoding_param_swap,
    order_swap,
    role_swap,
)

# Reserved flag namespace per BLUEPRINT_v5plus.md §16 #2. The CLI must
# refuse these now so user habits do not form around squatting on a
# name that v1.0 will repurpose.
RESERVED_FLAGS: tuple[str, ...] = (
    "--conformance",
    "--signoff",
    "--judge-pool",
    "--sandbox",
    "--key-uri",
    "--kms",
    "--allow-side-effects",
    "--tenant-id",
    "--per-cell-entropy",
)

_DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro"
_DEFAULT_SEMAPHORE_SIZE = 8


def _round_robin_select(
    *,
    original: Any,
    family_pools: tuple[list, ...],
    total: int,
) -> list:
    """Pick at most ``total`` variants — the original plus a round-robin
    sample across family pools — so a small ``--variants N`` gives the
    user a diverse mix rather than draining one family at a time.

    ``total`` is the upper bound on the returned list length (including
    the original at index 0). With ``total=1`` the result is exactly
    ``[original]``; with ``total=4`` and three populated families it's
    ``[original, fam0_0, fam1_0, fam2_0]``. When a pool runs out we
    keep cycling the others.
    """
    if total <= 0:
        return []
    selected: list = [original]
    if total == 1:
        return selected
    pool_iters = [iter(pool) for pool in family_pools]
    exhausted = [False] * len(pool_iters)
    while len(selected) < total and not all(exhausted):
        for i, it in enumerate(pool_iters):
            if exhausted[i]:
                continue
            try:
                selected.append(next(it))
            except StopIteration:
                exhausted[i] = True
                continue
            if len(selected) >= total:
                break
    return selected
_RESERVED_FLAG_EXIT = 2
_NOT_IMPLEMENTED_EXIT = 2
_BAD_USAGE_EXIT = 2

app = typer.Typer(
    name="topofuzz",
    help="Topology Safety Fuzzer for multi-agent LLM systems (TSF v5+).",
    no_args_is_help=True,
    add_completion=False,
)


def _reject_reserved_flags(argv: list[str]) -> None:
    """Raise typer.Exit if any reserved flag is present in argv.

    Called from each subcommand's body with that subcommand's
    ``ctx.args`` (the leftover tokens after Click parses declared
    options). Subcommands enabling reserved-flag rejection MUST also
    set ``context_settings={"ignore_unknown_options": True,
    "allow_extra_args": True}`` on the command — otherwise Click
    refuses the unknown option before this function ever runs.
    """
    reserved = set(RESERVED_FLAGS)
    for tok in argv:
        bare = tok.split("=", 1)[0]
        if bare in reserved:
            typer.echo(
                f"topofuzz: {bare} is reserved for v1.0+ and not accepted in "
                f"v0.5. See BLUEPRINT_v5plus.md §16 #2.",
                err=True,
            )
            raise typer.Exit(code=_RESERVED_FLAG_EXIT)


def _reject_unknown_extras(argv: list[str]) -> None:
    """Raise typer.Exit on any non-reserved leftover token.

    With ``ignore_unknown_options=True`` Click also drops genuinely-
    unknown flags (e.g. typos like ``--variantss``). Surface those as
    a usage error rather than silently accepting them — UX regression
    risk noted in step 6.1 design.
    """
    if not argv:
        return
    typer.echo(
        f"topofuzz: unknown option(s): {' '.join(argv)}",
        err=True,
    )
    raise typer.Exit(code=_BAD_USAGE_EXIT)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Print version and exit."
    ),
) -> None:
    if version:
        typer.echo(f"topofuzz {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _not_yet(command: str, lands: str) -> None:
    typer.echo(
        f"topofuzz {command}: not yet implemented (lands {lands}).",
        err=True,
    )
    raise typer.Exit(code=_NOT_IMPLEMENTED_EXIT)


@app.command()
def scan(fixture: str) -> None:
    """Static risk badge for a LangGraph topology. (Not yet implemented.)"""
    _not_yet("scan", "Weekend 5 step 14")


@app.command(
    context_settings={
        # Reserved-flag rejection (BLUEPRINT §16 #2) needs the unknown
        # flag to reach the command body. Click's default would refuse
        # ``--per-cell-entropy`` as "no such option" before we can
        # produce the v1.0-reserved message. The body manually
        # re-emits "unknown option" for anything that isn't reserved
        # so genuine typos still surface as a usage error.
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def run(
    ctx: typer.Context,
    fixture: str = typer.Argument(
        ..., help="Fixture spec 'path/to/file.py:attr' (attr = compiled graph or no-arg callable)."
    ),
    eval_path: Path = typer.Option(
        ...,
        "--eval",
        help="Eval rows in .jsonl or .csv format. Each row needs an 'id' (unique) and 'input'.",
    ),
    n: int = typer.Option(
        50,
        "--n",
        min=1,
        help="Number of eval rows to use (sampled from the head of --eval).",
    ),
    variants: int = typer.Option(
        8,
        "--variants",
        min=1,
        help="Max variants per generator family (§6 caps still apply).",
    ),
    max_cost_usd: float | None = typer.Option(
        None,
        "--max-cost-usd",
        min=0.0,
        help="Soft-warn at 80%%, hard-stop at 100%%. None = uncapped.",
    ),
    judge_model: str = typer.Option(
        _DEFAULT_JUDGE_MODEL,
        "--judge",
        help="Judge model id (advertised via judge_model_version in the cache).",
    ),
    jury_of_3: bool = typer.Option(
        False,
        "--jury-of-3",
        help="Opt-in jury-of-3 judging. Requires Anthropic + OpenAI + Gemini keys. v0.5 default judge is single Gemini.",
    ),
    out_dir: Path = typer.Option(
        Path("runs"),
        "--out",
        help="Parent directory for run artifacts. The run id is appended.",
    ),
    cache_dir: Path | None = typer.Option(
        Path(".cache/topofuzz"),
        "--cache",
        help="LLMCache + JudgeCache root.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable LLM + judge caches (every call hits the provider).",
    ),
    rng_seed: int = typer.Option(
        0,
        "--rng-seed",
        help="Variant-generator seed; same value → same variant list.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the cost-estimator confirmation prompt.",
    ),
) -> None:
    """Run topology fuzzing against an eval set.

    The cost estimator runs first (§12 contract). Confirm to proceed.
    """
    # Reserved-flag + unknown-flag enforcement. Order matters: reserved
    # gets its specific message, then any remaining tokens are unknown.
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    if jury_of_3:
        typer.echo(
            "topofuzz: --jury-of-3 requires Anthropic + OpenAI + Gemini keys; "
            "v0.5 has Gemini only. File a v0.5.1 issue or supply the missing keys.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    try:
        topology = load_topology(fixture)
    except (
        FileNotFoundError,
        ValueError,
        AttributeError,
        TypeError,
        NotImplementedError,
    ) as e:
        typer.echo(f"topofuzz: failed to load fixture: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    # Every node MUST have an llm_config for cost estimation + the runner
    # to function. Catching this here gives a clean usage error instead of
    # (a) a misleading cost estimate computed against the wrong rate
    # (audit B-2), and (b) a per-cell runner crash that wastes preflight
    # time. The loader accepts partial LLM_CONFIGS (it only validates
    # types) so we re-check here at the use site.
    missing = [n.id for n in topology.nodes if n.llm_config is None]
    if missing:
        typer.echo(
            f"topofuzz: fixture nodes have no llm_config: {missing!r}. "
            "Define LLM_CONFIGS on the fixture module with an entry per node.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    try:
        all_rows = load_eval_rows(eval_path)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"topofuzz: failed to load eval rows: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    eval_rows = all_rows[:n]
    if len(eval_rows) < n:
        typer.echo(
            f"topofuzz: --n {n} requested but only {len(eval_rows)} rows in "
            f"{eval_path}; proceeding with {len(eval_rows)}.",
            err=True,
        )

    # ``--variants N`` = max TOTAL variants (including the original).
    # Round-robin across the three §6 families so a small N gives a
    # *diverse* sample rather than "8 order-swap perms and nothing
    # else". B-3 (first pass) tried `[:variants]` after concatenating
    # — that silently dropped every variation when N=1. B-2 (second
    # pass) caught that the family-independent cap version produced
    # `[original] + 4 role + 4 decoding` = 9 variants on a 4-shared-
    # model topology when N=1. This round-robin keeps the user-visible
    # contract: N variants means at most N variants.
    ord_list = list(order_swap(topology, rng_seed=rng_seed))
    role_list = list(role_swap(topology))
    dec_list = list(decoding_param_swap(topology))
    original = ord_list[0]  # order_swap always emits the original first
    ord_perms = ord_list[1:]
    variant_list = _round_robin_select(
        original=original,
        family_pools=(ord_perms, role_list, dec_list),
        total=variants,
    )

    run_id = new_run_id()
    # Resolve --out at parse time (audit O-3) so the manifest records an
    # absolute path even if the orchestrator's cwd changes mid-run.
    run_dir = (out_dir / run_id).resolve()

    effective_cache_dir = None if no_cache else cache_dir

    plan = RunPlan(
        topology=topology,
        variants=tuple(variant_list),
        eval_rows=tuple(eval_rows),
        run_id=run_id,
        run_dir=run_dir,
        cache_dir=effective_cache_dir,
        max_cost_usd=max_cost_usd,
        judge_model=judge_model,
        semaphore_size=_DEFAULT_SEMAPHORE_SIZE,
    )

    _preflight_cost(plan, eval_rows=eval_rows, judge_model=judge_model, yes=yes)

    cli_args = {
        "fixture": fixture,
        "eval": str(eval_path),
        "n": n,
        "variants": variants,
        "max_cost_usd": max_cost_usd,
        "judge": judge_model,
        "jury_of_3": jury_of_3,
        "out": str(out_dir),
        "cache": str(effective_cache_dir) if effective_cache_dir else None,
        "no_cache": no_cache,
        "rng_seed": rng_seed,
    }
    write_manifest(plan, cli_args=cli_args)

    result = _run_with_signal_handling(plan)

    typer.echo("")
    typer.echo(f"Run id: {result.run_id}")
    typer.echo(f"Run dir: {result.run_dir}")
    typer.echo(
        f"Cells: {result.successful_cells} ok / {result.completed_cells} "
        f"completed / {result.total_cells} planned"
    )
    typer.echo(f"Cost (priced cells only): ${result.total_cost_usd:.4f}")
    if result.cap_tripped:
        # Resume is intentionally not exposed from the CLI in v0.5 —
        # every invocation mints a fresh run_id (audit RT M-3). Partial
        # results survive in the run dir for forensic inspection.
        typer.echo(
            "Cost cap tripped; partial results in the run dir. "
            "Resume is not exposed in v0.5 — file a v0.5.1 issue if you need it."
        )
    typer.echo(f"Elapsed: {result.elapsed_s:.2f}s")


def _preflight_cost(
    plan: RunPlan,
    *,
    eval_rows: list[dict],
    judge_model: str,
    yes: bool,
) -> None:
    """Cost estimate + ``Continue? [y/N]`` prompt before kickoff.

    The current estimator is a static placeholder (§costest) — call it
    once, print, prompt. When the measured-rate update lands the call
    site stays the same.
    """
    from fuzzeval.costest import estimate as _estimate

    sample = eval_rows[: min(5, len(eval_rows))]
    est = _estimate(
        sample_rows=sample,
        judge_mode="single",
        n_total_rows=len(eval_rows),
        n_variants=len(plan.variants),
        agent_model=plan.topology.nodes[0].llm_config.model
        if plan.topology.nodes and plan.topology.nodes[0].llm_config
        else _DEFAULT_JUDGE_MODEL,
        judge_model=judge_model,
    )
    typer.echo("")
    typer.echo("Estimated cost:")
    for line in est.lines:
        typer.echo(
            f"  {line.name:<14} "
            f"cold=${line.cold_estimate_usd:.4f} ± ${line.cold_ci_half_width_usd:.4f} / "
            f"warm=${line.warm_estimate_usd:.4f} ± ${line.warm_ci_half_width_usd:.4f}"
        )
    typer.echo(
        f"  Total:         cold=${est.total_cold_usd:.4f} / "
        f"warm=${est.total_warm_usd:.4f} (90% CI)"
    )
    typer.echo(f"Estimated wallclock: {est.wallclock_estimate_s:.0f}s")
    if plan.max_cost_usd is not None:
        typer.echo(f"Hard cap: ${plan.max_cost_usd:.2f}")
    typer.echo("")
    if yes:
        return
    if not typer.confirm("Continue?", default=False):
        typer.echo("Aborted.")
        raise typer.Exit(code=0)


def _run_with_signal_handling(plan: RunPlan) -> Any:
    """Run the orchestrator with SIGINT → cancel-event wiring.

    On Ctrl+C the orchestrator's cancel event is set; the runner stops
    starting new LLM calls and the writer drains whatever already
    completed. A *second* Ctrl+C lets the default handler interrupt
    abruptly — we don't try to swallow it, in case the orchestrator
    itself is wedged.
    """
    loop_cancel = asyncio.Event()

    def _signal_handler(sig: int, frame: Any) -> None:
        if loop_cancel.is_set():
            # Second Ctrl+C: restore the default handler AND immediately
            # raise KeyboardInterrupt so this signal aborts the process
            # right now. Without the explicit raise the next SIGINT
            # would abort, but this one would just return — requiring
            # *three* Ctrl+C presses to actually abort (audit M1).
            signal.signal(sig, signal.SIG_DFL)
            raise KeyboardInterrupt
        typer.echo(
            "\nReceived interrupt; cancelling. Press Ctrl+C again to "
            "abort immediately.",
            err=True,
        )
        loop_cancel.set()

    previous = signal.signal(signal.SIGINT, _signal_handler)
    try:
        return asyncio.run(execute_run(plan, cancel=loop_cancel))
    finally:
        signal.signal(signal.SIGINT, previous)


@app.command()
def report(run_dir: str) -> None:
    """Open the HTML report for a run. (Not yet implemented.)"""
    _not_yet("report", "Weekend 3 step 11")


@app.command()
def verify(run_dir: str) -> None:
    """Verify DSSE signature on a run's report. (Not yet implemented.)"""
    _not_yet("verify", "Weekend 3 step 10a")


def main() -> None:
    """Module entry-point used by ``python -m topofuzz``."""
    logging.basicConfig(level=logging.WARNING)
    app()


if __name__ == "__main__":
    main()
