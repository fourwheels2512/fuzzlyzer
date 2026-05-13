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
import atexit
import logging
import signal
from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import Any

import typer

from fuzzeval.keys import (
    KeyAlreadyExistsError,
    KeyNotFoundError,
    default_dev_priv_path,
    default_dev_pub_path,
    generate_dev_keys,
    load_public_key,
)

from topofuzz import __version__
from topofuzz.attest import load_and_verify_attestation

# Heavy imports (orchestrator, loaders, fuzzeval.judge, topofuzz.runids,
# topofuzz.variants) are lazy — pulled inside :func:`run` and the kickoff
# helpers only. BLUEPRINT_v5plus.md §11a calls out verify-only installs
# (``pip install topofuzz[verify]``, no langgraph / LLM SDKs); the
# ``topofuzz verify`` and ``topofuzz keygen`` codepaths must NOT
# transitively pull the run-time stack. Step 10b ships an import-guard
# test that asserts this.

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


# Module-scoped ExitStack holds open any resource handles created by
# :func:`_packaged_default_rubric_path`. For editable / wheel installs
# `importlib.resources.as_file` returns a real fs path with no cleanup
# callback (no-op exit). For zip-imported / frozen-executable cases it
# extracts to a tempdir and the exit callback unlinks the file —
# **after** the path has been returned to the caller, which would leave
# :class:`fuzzeval.judge.SingleJudge` reading a dangling path (audit M3).
# Holding the stack at module scope keeps every resource alive for the
# process lifetime. ``atexit`` closes it on interpreter shutdown.
_RESOURCE_STACK = ExitStack()
atexit.register(_RESOURCE_STACK.close)


def _packaged_default_rubric_path() -> Path:
    """Resolve the bundled ``lending_judge.txt`` rubric path.

    Uses :mod:`importlib.resources` so the path is correct regardless of
    whether topofuzz was installed editable from the repo or as a wheel.
    Hatch (see ``packages/topofuzz/pyproject.toml``) ships every file
    under ``src/topofuzz/`` so the ``prompts/`` directory rides along.

    The returned :class:`pathlib.Path` is valid for the lifetime of the
    Python interpreter — see :data:`_RESOURCE_STACK` for the cleanup
    contract. Repeated calls re-enter ``as_file`` (cheap for fs-resident
    resources; one extra temp-file extraction for zip-imported resources).
    """
    resource = resources.files("topofuzz").joinpath("prompts/lending_judge.txt")
    return Path(_RESOURCE_STACK.enter_context(resources.as_file(resource)))


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
# `topofuzz verify` returns 1 on signature/JCS/digest failure per
# BLUEPRINT_v5plus.md §15 DoD line 436 ("Tampering with one byte of the
# payload returns exit 1"). Distinct from _BAD_USAGE_EXIT (=2) so CI
# scripts can tell a tampered bundle apart from a missing one.
_VERIFY_FAIL_EXIT = 1
# §A error-taxonomy exit codes (BLUEPRINT lines 634-643). Step 6 wired
# the codes observable then (cost cap, SIGINT, unsupported graph). Step 9
# adds the 0/1/2 success/degraded/failed split — the metrics layer
# computes ``status`` + ``metrics_exit_code`` and stamps both onto the
# ``runs`` row + ``RunResult``; we just translate to ``typer.Exit``.
_COST_CAP_EXIT = 3
_UNSUPPORTED_GRAPH_EXIT = 64
# 130 (POSIX SIGINT exit) is emitted by :class:`topofuzz.orchestrator.RunResult`
# via ``metrics_exit_code`` when ``cancel_event`` was set without a cap-trip.
# No CLI-side constant — the value flows through ``result.metrics_exit_code``
# and reaches ``typer.Exit`` at the bottom of :func:`run`.

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


@app.command(
    context_settings={
        # Reserved-flag rejection (§16 #2) is a global policy. ``scan``
        # opts into the same context-settings as ``run`` so a user
        # typing ``topofuzz scan --sandbox foo`` gets the v1.0-reserved
        # message instead of a "no such option" from Click.
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def scan(ctx: typer.Context, fixture: str) -> None:
    """Static risk badge for a LangGraph topology (BLUEPRINT §5a).

    Walks the Topology IR with a fixed rule table and emits a HIGH /
    MEDIUM / LOW badge plus a single-line rationale. Zero LLM calls,
    runs in milliseconds. Use it BEFORE ``topofuzz run`` to flag
    obviously ordering-sensitive topologies for free.

    Output:

    * Human-readable banner → **stderr** (so terminal users see it
      inline but pipes don't have to strip it). The :mod:`topofuzz.loaders`
      one-time "executes arbitrary Python" warning also lands here.
    * Machine-readable JSON → **stdout**. Schema:
      ``{badge, matched_patterns, rationale}``. Pipe with ``2>/dev/null``
      to feed ``jq`` cleanly (``topofuzz scan ... | jq`` works; merging
      streams via ``2>&1 | jq`` does not — the banner is plain prose).

    Exit codes:

    * 0 — scan completed (regardless of badge color; the badge is
      diagnostic, not a gate). Per BLUEPRINT DoD #11.
    * 2 — usage error (bad fixture spec, file not found, LLM_CONFIGS
      malformed, fixture module fails to import/exec). Same code as
      ``topofuzz run`` for symmetry.
    * 64 — fixture uses an out-of-scope LangGraph construct (same
      ``_UNSUPPORTED_GRAPH_EXIT`` as ``run``).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    # Lazy imports to preserve §11a verify-path safety: ``topofuzz.cli``
    # MUST NOT pull langgraph at module import time (loaders.load_topology
    # transitively imports adapter → langgraph). ``topofuzz.scan`` itself
    # is verify-path-safe but loading topologies needs the run-time stack.
    import json

    from topofuzz.loaders import load_topology
    from topofuzz.scan import scan as scan_topology

    # Audit fix CR-MAJOR-1 (pass-1): ``load_topology`` exec_modules an
    # arbitrary Python file, which can raise ``ImportError`` /
    # ``SyntaxError`` / ``ModuleNotFoundError`` (nested import failure
    # inside the fixture) in addition to the originally-enumerated set.
    # Surface all of those as exit-2 with the documented "failed to load
    # fixture" prefix — a raw traceback from the gateway-drug feature
    # contradicts the §5a "instant, friendly" UX contract. We do NOT
    # catch ``Exception`` blindly because a topofuzz internal bug should
    # still surface as a traceback; the enumerated set covers every
    # exception ``load_topology`` is documented (and observed) to raise.
    try:
        topology = load_topology(fixture)
    except NotImplementedError as e:
        typer.echo(f"topofuzz: unsupported graph construct: {e}", err=True)
        raise typer.Exit(code=_UNSUPPORTED_GRAPH_EXIT) from e
    except (
        FileNotFoundError,
        ValueError,
        AttributeError,
        TypeError,
        ImportError,
        SyntaxError,
    ) as e:
        # Audit fix RT-M2 (pass-2): prefix the message with the
        # exception class so an ``ImportError`` is visually
        # distinguishable from a ``SyntaxError`` without re-running.
        # ``ImportError`` covers a fixture's own internal-import bugs,
        # which would otherwise look identical to a missing-fixture-
        # file message — the class prefix tells a debugger "the file
        # loaded but its imports failed."
        typer.echo(
            f"topofuzz: failed to load fixture: {type(e).__name__}: {e}",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    result = scan_topology(topology)

    # Banner → stderr. Two-line layout: headline (badge + rationale)
    # followed by the heuristic-only disclaimer and the next-step nudge.
    # Wrap is implicit on terminal; we do NOT pre-wrap because terminal
    # widths vary and `rich`/`textwrap` is overkill for ~3 lines.
    typer.echo(f"{result.badge} RISK — {result.rationale}", err=True)
    typer.echo(
        "Heuristic only — false positives and false negatives both exist. "
        f"Run `topofuzz run {fixture} --eval <file>` to measure actual "
        "variance.",
        err=True,
    )

    # JSON → stdout. ``sort_keys=True`` keeps the byte sequence stable
    # across Python dict-insertion-order changes so CI snapshot tests
    # don't flake. No trailing newline beyond ``typer.echo``'s default —
    # one ``\n`` is conventional and pipe-friendly.
    typer.echo(json.dumps(result.to_json_dict(), sort_keys=True))


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
        None,
        "--cache",
        help=(
            "LLMCache + JudgeCache root. Defaults to .cache/topofuzz "
            "when omitted; passing a value here lets the CLI distinguish "
            "user-supplied from default for jury-disable warnings."
        ),
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
    rubric_path: Path | None = typer.Option(
        None,
        "--rubric",
        help=(
            "Path to a rubric template (str.format placeholders). "
            "Default: packaged topofuzz/prompts/lending_judge.txt."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Skip the real LLM judge call; every cell gets a deterministic "
            "ContinuousScore(0.5). Judge costs $0. Implies --no-cache. "
            "Agent rollouts STILL bill real money (use --max-cost-usd to cap)."
        ),
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
    # Lazy imports of the runtime stack (orchestrator + loaders +
    # variant generators + judge). Keeping these out of module scope
    # is the §11a "verify-path-safe" contract: ``topofuzz verify``
    # and ``topofuzz keygen`` must NOT pull langgraph / LLM SDKs at
    # import time. Step 10b ships the regression-guard test.
    from fuzzeval.judge import JuryJudge, SingleJudge

    from topofuzz.loaders import load_eval_rows, load_topology
    from topofuzz.orchestrator import RunPlan, write_manifest
    from topofuzz.runids import new_run_id
    from topofuzz.variants.generators import (
        decoding_param_swap,
        order_swap,
        role_swap,
    )

    # Reserved-flag + unknown-flag enforcement. Order matters: reserved
    # gets its specific message, then any remaining tokens are unknown.
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    if jury_of_3:
        # Step 12b: wire the opt-in jury per BLUEPRINT_v5plus.md §12b.
        # The runtime check that ANTHROPIC_API_KEY + OPENAI_API_KEY +
        # GEMINI_API_KEY env vars exist happens lazily at first call —
        # LiteLLM raises an authentication error which JuryJudge.score
        # accumulates as a failed vendor call. The cell ends up
        # judge_failure=True if all three vendors fail; §A's degraded
        # badge surfaces it in the report.
        #
        # We do NOT preflight-check env keys here because the soft DoD
        # (BLUEPRINT line 445) explicitly allows jury infra to have
        # known issues without blocking ship — file a v0.5.1 issue
        # instead. Emitting the hint to stderr is the right surface.
        typer.echo(
            "topofuzz: --jury-of-3 invokes Anthropic + OpenAI + Gemini "
            "judges per BLUEPRINT §6. Missing env keys "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY) will "
            "surface as judge_failure cells, not a kickoff error.",
            err=True,
        )

    try:
        topology = load_topology(fixture)
    except NotImplementedError as e:
        # §A: adapter saw a graph construct it can't translate. Exit 64
        # so CI can distinguish "your fixture uses an unsupported
        # langgraph feature" from generic bad-usage.
        typer.echo(f"topofuzz: unsupported graph construct: {e}", err=True)
        raise typer.Exit(code=_UNSUPPORTED_GRAPH_EXIT) from e
    except (
        FileNotFoundError,
        ValueError,
        AttributeError,
        TypeError,
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

    # --dry-run implies --no-cache. Without this, the judge's deterministic
    # ContinuousScore(0.5) sentinel gets written to the JudgeCache; a
    # subsequent real-mode run with the same cache_dir would hit the cached
    # 0.5 and silently skip the real judge call, corrupting the §9 score
    # matrix (audit B2). Forcing no-cache in dry-run also matches user
    # intent — a dry-run is throwaway by definition.
    #
    # --jury-of-3 ALSO implies --no-cache. The v0.5 JudgeCache keys on a
    # single ``judge_model_version`` string, which can't distinguish a
    # cached single-judge row from a cached jury row (and per-vendor
    # cache slicing is deferred to v0.5.1 per the JuryJudge docstring).
    # Disabling the cache for jury runs sidesteps the foot-gun; jury
    # runs are expensive enough that re-running uncached is acceptable.
    #
    # Pass-2 CR-Critical / RT-MAJOR-1 audit fix: distinguish a
    # user-supplied --cache value from the typer-applied default by
    # checking ``cache_dir is not None`` AFTER the option default is set
    # to ``None``. Then resolve the default in the body. Without this
    # split, every --jury-of-3 invocation tripped the "you supplied" hint
    # for a path the user never typed.
    user_supplied_cache = cache_dir is not None
    resolved_cache_dir = (
        cache_dir if cache_dir is not None else Path(".cache/topofuzz")
    )
    effective_cache_dir = (
        None if (no_cache or dry_run or jury_of_3) else resolved_cache_dir
    )

    # Pass-1 CR-M1 / RT-M3 audit fix: when --jury-of-3 forces the cache
    # off but the user ALSO supplied --cache (i.e., they had a positive
    # intent to cache), echo a hint so the disabled cache isn't a silent
    # surprise. Compared to the dry-run code path this hint is more
    # important — dry-run is throwaway by definition; jury is a billed
    # run the user expected to cache. We surface the hint even under
    # --dry-run because the user's --cache value is still being silently
    # ignored — only --no-cache is interpreted as positive consent.
    if jury_of_3 and user_supplied_cache and not no_cache:
        typer.echo(
            f"topofuzz: --jury-of-3 disables the JudgeCache in v0.5; the "
            f"--cache {str(cache_dir)!r} you supplied is ignored. Per-vendor "
            "cache slicing is tracked for v0.5.1.",
            err=True,
        )

    # Construct the judge. Falls back to the packaged rubric if --rubric
    # was not supplied. Both SingleJudge and JuryJudge read the rubric
    # file at construction so bad paths / empty files surface here,
    # before the cost-estimator prompt runs.
    effective_rubric_path = (
        rubric_path if rubric_path is not None else _packaged_default_rubric_path()
    )
    try:
        if jury_of_3:
            judge = JuryJudge(
                rubric_path=effective_rubric_path,
                dry_run=dry_run,
            )
        else:
            judge = SingleJudge(
                model=judge_model,
                rubric_path=effective_rubric_path,
                dry_run=dry_run,
            )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"topofuzz: bad rubric: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    # When jury is on, the user-supplied ``--judge`` value is ignored
    # (the trio comes from BLUEPRINT §6). Reflect that in the manifest's
    # ``judge_model`` field so downstream readers (report HTML, audits)
    # see the jury identity rather than a stale single-judge id. The raw
    # ``--judge`` value still lands in ``cli_args["judge"]`` for
    # forensic completeness — it preserves what the user actually typed.
    effective_judge_model = judge.model if jury_of_3 else judge_model

    plan = RunPlan(
        topology=topology,
        variants=tuple(variant_list),
        eval_rows=tuple(eval_rows),
        run_id=run_id,
        run_dir=run_dir,
        cache_dir=effective_cache_dir,
        max_cost_usd=max_cost_usd,
        judge_model=effective_judge_model,
        semaphore_size=_DEFAULT_SEMAPHORE_SIZE,
        rubric_hash=judge.rubric_hash,
    )

    _preflight_cost(
        plan, eval_rows=eval_rows, judge_model=judge_model,
        yes=yes, dry_run=dry_run, jury_of_3=jury_of_3,
    )

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
        "rubric": str(effective_rubric_path),
        "rubric_hash": judge.rubric_hash,
        "dry_run": dry_run,
    }
    write_manifest(plan, cli_args=cli_args)

    if dry_run:
        typer.echo("[DRY-RUN] No real judge LLM calls — every cell gets ContinuousScore(0.5).")

    result = _run_with_signal_handling(plan, judge=judge)

    typer.echo("")
    typer.echo(f"Run id: {result.run_id}")
    typer.echo(f"Run dir: {result.run_dir}")
    typer.echo(f"Status: {result.status}")
    typer.echo(
        f"Cells: {result.successful_cells} ok / {result.completed_cells} "
        f"completed / {result.total_cells} planned"
    )
    typer.echo(f"Cost (priced cells only): ${result.total_cost_usd:.4f}")
    # Step 12b: surface §B DoD #14 jury accounting from the JuryJudge
    # instance. Done here (CLI scope) rather than via manifest plumbing
    # to keep the step-12b surface minimal — promoting to manifest is a
    # v0.5.1 schema bump per the JuryJudge class docstring.
    #
    # Polish-audit DRIFT-3 fix: this rate is the VENDOR-CALL failure
    # rate (across all three vendors, across all cells), not the
    # BLUEPRINT §8 cell-level ``judge_failure_rate`` (= fraction of
    # cells where ALL three vendors failed). The §8 cell-level gate
    # lives in :func:`topofuzz.metrics.classify_run_status`; the label
    # here is explicit so a user can't confuse the two numbers.
    if isinstance(judge, JuryJudge):
        typer.echo(
            f"Jury vendor-call failure rate: {judge.judge_failure_rate:.2%} "
            f"({judge.failed_vendor_calls}/{judge.total_vendor_calls} "
            "vendor calls; cell-level §8 rate reported separately by "
            "metrics)"
        )
        typer.echo(
            "Jury judge_independence.vendor_overlap: "
            f"{judge.judge_independence['vendor_overlap']!r}"
        )
    if result.cap_tripped:
        # Resume is intentionally not exposed from the CLI in v0.5 —
        # every invocation mints a fresh run_id (audit RT M-3). Partial
        # results survive in the run dir for forensic inspection.
        typer.echo(
            "Cost cap tripped; partial results in the run dir. "
            "Resume is not exposed in v0.5 — file a v0.5.1 issue if you need it."
        )
    typer.echo(f"Elapsed: {result.elapsed_s:.2f}s")

    # §A error-taxonomy exit codes (BLUEPRINT lines 634-643).
    # Cap-tripped takes precedence (exit 3) so CI can distinguish a
    # budget-tripped run from success. Otherwise use the §A status
    # mapping from the metrics layer (0 = succeeded / degraded-low,
    # 1 = degraded-high, 2 = failed, 130 = interrupted).
    if result.cap_tripped:
        raise typer.Exit(code=_COST_CAP_EXIT)
    if result.metrics_exit_code != 0:
        raise typer.Exit(code=result.metrics_exit_code)


def _preflight_cost(
    plan: RunPlan,
    *,
    eval_rows: list[dict],
    judge_model: str,
    yes: bool,
    dry_run: bool = False,
    jury_of_3: bool = False,
) -> None:
    """Cost estimate + ``Continue? [y/N]`` prompt before kickoff.

    The current estimator is a static placeholder (§costest) — call it
    once, print, prompt. When the measured-rate update lands the call
    site stays the same.

    ``dry_run=True`` zeros the judge column in the printed estimate so
    the user sees the run's *actual* expected billing (agent calls
    still cost real money; the judge stub does not). The estimator
    itself is judge-mode agnostic; we annotate downstream.

    ``jury_of_3=True`` selects ``judge_mode="jury_of_3"`` in the
    underlying estimator so the judge column reflects three vendor
    calls per cell instead of one (BLUEPRINT §6 trio). Polish-audit
    improvement: this drops the prior ``~3×`` disclaimer in favor of
    an honest displayed total — the estimator's
    ``_JUDGE_CALLS_PER_CELL["jury_of_3"] == 3`` already pices the trio.
    """
    from fuzzeval.costest import estimate as _estimate

    sample = eval_rows[: min(5, len(eval_rows))]
    est = _estimate(
        sample_rows=sample,
        judge_mode="jury_of_3" if jury_of_3 else "single",
        n_total_rows=len(eval_rows),
        n_variants=len(plan.variants),
        agent_model=plan.topology.nodes[0].llm_config.model
        if plan.topology.nodes and plan.topology.nodes[0].llm_config
        else _DEFAULT_JUDGE_MODEL,
        judge_model=judge_model,
    )
    # Compute dry-run-aware totals BEFORE printing so the displayed
    # bottom-line matches reality (audit M4 — without this, the user
    # saw the judge column zeroed but the Total still summed in the
    # original judge cost, contradicting the disclaimer).
    judge_cold_subtracted = 0.0
    judge_warm_subtracted = 0.0
    if dry_run:
        for line in est.lines:
            if "judg" in line.name.lower():
                judge_cold_subtracted += line.cold_estimate_usd
                judge_warm_subtracted += line.warm_estimate_usd
    effective_total_cold = est.total_cold_usd - judge_cold_subtracted
    effective_total_warm = est.total_warm_usd - judge_warm_subtracted

    typer.echo("")
    typer.echo("Estimated cost:")
    for line in est.lines:
        # In dry-run, the judge does not bill — surface zeros for the
        # judge-priced line so the displayed total matches reality.
        # Other lines (agent rollouts) still bill.
        if dry_run and "judg" in line.name.lower():
            typer.echo(
                f"  {line.name:<14} "
                f"cold=$0.0000 ± $0.0000 / warm=$0.0000 ± $0.0000  "
                f"[DRY-RUN judge — no LLM call]"
            )
            continue
        typer.echo(
            f"  {line.name:<14} "
            f"cold=${line.cold_estimate_usd:.4f} ± ${line.cold_ci_half_width_usd:.4f} / "
            f"warm=${line.warm_estimate_usd:.4f} ± ${line.warm_ci_half_width_usd:.4f}"
        )
    typer.echo(
        f"  Total:         cold=${effective_total_cold:.4f} / "
        f"warm=${effective_total_warm:.4f} (90% CI)"
    )
    if dry_run:
        typer.echo(
            "  (dry-run total reflects agent rollouts only; "
            "judge column subtracted)"
        )
    if jury_of_3 and not dry_run:
        typer.echo(
            "  (--jury-of-3: judge column already prices the 3-vendor "
            "trio — Anthropic + OpenAI + Gemini per BLUEPRINT §6.)"
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


def _run_with_signal_handling(plan: RunPlan, *, judge: Any = None) -> Any:
    """Run the orchestrator with SIGINT → cancel-event wiring.

    On Ctrl+C the orchestrator's cancel event is set; the runner stops
    starting new LLM calls and the writer drains whatever already
    completed. A *second* Ctrl+C lets the default handler interrupt
    abruptly — we don't try to swallow it, in case the orchestrator
    itself is wedged.

    When ``judge`` is supplied, its :meth:`render_prompt` is passed as
    the orchestrator's ``judge_prompt_renderer`` so the §7 JudgeCache
    ``prompt_hash`` slot keys on the rendered prompt rather than the
    full eval_row (step-5 stopgap).
    """
    # Lazy import of the orchestrator (mirrors the lazy-import policy
    # in :func:`run` — see comment there about §11a verify-path safety).
    from topofuzz.orchestrator import execute_run

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
        return asyncio.run(
            execute_run(
                plan,
                cancel=loop_cancel,
                judge=judge,
                judge_prompt_renderer=judge.render_prompt if judge is not None else None,
            )
        )
    finally:
        signal.signal(signal.SIGINT, previous)


@app.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def report(ctx: typer.Context, run_dir: str) -> None:
    """Render ``<run_dir>/report.html`` and print its ``file://`` URL.

    Reads ``manifest.json`` + ``run.sqlite`` from ``run_dir`` (both
    required) and, when present, surfaces a ``[DEV-KEY]`` banner from
    ``attestation.json``. Two inline matplotlib SVGs render per-eval
    variance + the score matrix per BLUEPRINT_v5plus.md §11.

    An existing ``<run_dir>/report.html`` is silently overwritten —
    every invocation regenerates from scratch.

    Exit codes:

    * 0 — report written.
    * 2 — usage error (run_dir not a directory; manifest.json or
      run.sqlite missing; sqlite has zero / multiple runs;
      matplotlib / numpy / jinja2 missing — install ``topofuzz[full]``).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    run_dir_path = Path(run_dir)
    if not run_dir_path.is_dir():
        typer.echo(
            f"topofuzz report: run_dir is not a directory: {run_dir}",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    # Lazy import: pulls matplotlib + jinja2 + numpy. Keeping this out
    # of module scope is the §11a verify-path-safety contract — the
    # import-guard test (``tests/test_import_guard.py``) only allows
    # ``topofuzz.cli`` to load without the run-time stack.
    from topofuzz.report import (
        ReportDependencyError,
        ReportInputError,
        render_report,
    )

    try:
        out_path = render_report(run_dir_path)
    except ReportInputError as e:
        typer.echo(f"topofuzz report: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e
    except ReportDependencyError as e:
        typer.echo(f"topofuzz report: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e
    except ValueError as e:
        typer.echo(f"topofuzz report: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    typer.echo(out_path.resolve().as_uri())


@app.command(
    context_settings={
        # The reserved-flag policy applies to every subcommand, so verify
        # opts into the same context-settings even though it declares its
        # own --bundle / --pubkey options below. Click parses the declared
        # options first; any leftover unknown flags reach _reject_*.
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def verify(
    ctx: typer.Context,
    run_dir: str = typer.Argument(
        ...,
        help=(
            "Run directory containing attestation.json + manifest.json. "
            "Pass '-' as run_dir when also using --bundle."
        ),
    ),
    bundle: Path | None = typer.Option(
        None,
        "--bundle",
        help=(
            "Path to a standalone attestation.json bundle. When given, "
            "OVERRIDES run_dir entirely: run_dir is ignored, and the "
            "file-vs-envelope checks (manifest_match, run_sqlite_sha256_match) "
            "are skipped. Useful for verify-from-stdin CI pipelines and "
            "public-key-only verification."
        ),
    ),
    pubkey: Path | None = typer.Option(
        None,
        "--pubkey",
        help=(
            "Path to the Ed25519 public key PEM. Defaults to "
            "~/.topofuzz/dev.pub.pem (the local dev key)."
        ),
    ),
) -> None:
    """Verify the DSSE attestation on a topofuzz run.

    Exit codes:

    * 0 — PASS: signature verified, JCS round-trip stable, manifest +
      run.sqlite digests match (when applicable). Banner shows ``[DEV-KEY]``.
    * 1 — FAIL: signature mismatch, decode error, JCS round-trip
      mismatch, missing / drifted on-disk files, or attestation bundle
      not discoverable.
    * 2 — usage error (missing pubkey, run_dir is not a directory).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    pubkey_path = pubkey if pubkey is not None else default_dev_pub_path()
    try:
        pub_key = load_public_key(pubkey_path)
    except KeyNotFoundError as e:
        typer.echo(f"topofuzz verify: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e
    except ValueError as e:
        typer.echo(f"topofuzz verify: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    # Serialize the public key back to PEM bytes; the verify helper
    # accepts PEM (not the cryptography object) so the SECURITY.md
    # reference verifier and `topofuzz verify` share one interface.
    from cryptography.hazmat.primitives import serialization

    pub_pem = pub_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    # --bundle and run_dir are mutually exclusive in v0.5: passing
    # both would silently skip the file-vs-envelope checks against
    # the run_dir's manifest+sqlite, which is a footgun (user thinks
    # they're verifying the run, actually only verifying the bundle).
    # The '-' sentinel for run_dir is the explicit "I know there's no
    # run_dir to bind to" opt-in. Pass-2 MAJOR-1 fix.
    resolved_run_dir: Path | None
    if bundle is not None:
        if run_dir != "-":
            typer.echo(
                "topofuzz verify: --bundle is incompatible with a concrete "
                "run_dir (file-vs-envelope checks would be silently "
                "skipped). Pass '-' as run_dir, or drop --bundle.",
                err=True,
            )
            raise typer.Exit(code=_BAD_USAGE_EXIT)
        resolved_run_dir = None
    else:
        if run_dir == "-":
            typer.echo(
                "topofuzz verify: run_dir='-' requires --bundle <path>",
                err=True,
            )
            raise typer.Exit(code=_BAD_USAGE_EXIT)
        resolved_run_dir = Path(run_dir)
        if not resolved_run_dir.is_dir():
            typer.echo(
                f"topofuzz verify: run_dir is not a directory: {run_dir}",
                err=True,
            )
            raise typer.Exit(code=_BAD_USAGE_EXIT)

    outcome = load_and_verify_attestation(
        run_dir=resolved_run_dir,
        bundle=bundle,
        public_key_pem=pub_pem,
    )

    # Render the checklist. PASS/FAIL is the headline; each check's
    # detail line follows for diagnosis.
    for name, ok, detail in outcome.checks:
        prefix = "  ok " if ok else "FAIL"
        typer.echo(f"{prefix} {name}: {detail}")
    if outcome.ok:
        typer.echo(f"PASS {outcome.watermark}")
        return
    typer.echo(f"FAIL {outcome.watermark}", err=True)
    raise typer.Exit(code=_VERIFY_FAIL_EXIT)


@app.command(
    context_settings={
        # Reserved-flag rejection (§16 #2) is a universal policy across
        # every subcommand — keygen opts in for the same reason scan /
        # report / run / verify do: a user typing `topofuzz keygen --kms`
        # should see the v1.0-reserved message, not a generic "no such
        # option" from Click.
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def keygen(
    ctx: typer.Context,
    priv_out: Path | None = typer.Option(
        None,
        "--priv-out",
        help="Private key output path (default: ~/.topofuzz/dev.pem).",
    ),
    pub_out: Path | None = typer.Option(
        None,
        "--pub-out",
        help="Public key output path (default: ~/.topofuzz/dev.pub.pem).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Overwrite existing keys. WARNING: rotating the dev key "
            "invalidates every prior `topofuzz verify` against runs "
            "signed with the old key."
        ),
    ),
) -> None:
    """Generate a fresh Ed25519 dev keypair for signing attestations.

    Without flags, writes ``~/.topofuzz/dev.pem`` (private) and
    ``~/.topofuzz/dev.pub.pem`` (public). On POSIX the private file is
    chmod 0o600. On Windows, NTFS ACLs are inherited from the parent
    directory — see SECURITY.md for the operator checklist.
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))
    priv_path = priv_out if priv_out is not None else default_dev_priv_path()
    pub_path = pub_out if pub_out is not None else default_dev_pub_path()
    try:
        generate_dev_keys(priv_path, pub_path, force=force)
    except KeyAlreadyExistsError as e:
        typer.echo(f"topofuzz keygen: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e
    except FileExistsError as e:
        # Stale `.tmp` from a prior crashed keygen — give a clear,
        # actionable message instead of a raw traceback.
        typer.echo(
            f"topofuzz keygen: stale temp file blocks write: {e}. "
            f"Remove it and re-run (a previous keygen was interrupted "
            f"mid-write).",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e
    except PermissionError as e:
        # Windows raises PermissionError (WinError 5) on os.replace
        # when the destination file is held open by another process.
        # Pass-2 MAJOR-2 fix: surface a clean, actionable message.
        typer.echo(
            f"topofuzz keygen: cannot replace {priv_path} ({e}). "
            f"Close any process holding the file open and re-run.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e
    typer.echo(f"Wrote private key: {priv_path}")
    typer.echo(f"Wrote public key:  {pub_path}")
    typer.echo(
        "[DEV-KEY] this is a developer-grade key (unencrypted PEM, no KMS). "
        "See SECURITY.md for the threat model."
    )


def main() -> None:
    """Module entry-point used by ``python -m topofuzz``."""
    logging.basicConfig(level=logging.WARNING)
    app()


if __name__ == "__main__":
    main()
