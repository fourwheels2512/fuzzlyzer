# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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
import sys
from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from topofuzz.key_inspector import (
    at_least_one_present,
    format_table,
    inspect_keys,
)
from topofuzz.license_check import (
    LICENSE_REFUSED_EXIT_CODE,
    HomeUnresolvableError,
    prompt_and_record,
)

if TYPE_CHECKING:
    # Annotation-only — `from __future__ import annotations` keeps these
    # out of the runtime import graph, so the verify-only install stays
    # langgraph-free (see the heavy-import note below).
    from topofuzz.ir import Edge, Topology

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


def _variant_violates_edges(
    variant: "Topology", edges: "tuple[Edge, ...]"
) -> bool:
    """True if ``variant``'s node order violates the graph's data deps.

    The §7 runner executes ``variant.nodes`` in tuple order. As of
    v0.5.1, ``order_swap`` is edge-aware — it only emits permutations
    that are valid linear extensions of the DAG — so this predicate
    should always return False for ``order_swap`` output. It is kept
    as defence-in-depth (regression guard against a future generator
    bug) and to surface the *original* topology's own edge-validity
    when a fixture is declared out of dependency order.

    The predicate: for every edge ``(from_id, to_id)``, the position of
    ``from_id`` in the variant's node order must come strictly before the
    position of ``to_id``. If any edge runs backwards, the variant is
    edge-invalid.

    Applies to ANY topology, including the original (index 0): the
    adapter builds ``nodes`` from LangGraph's ``get_graph()`` emission
    order — i.e. ``add_node`` *declaration* order — and does NOT
    topologically sort (see ``adapter.py`` and the
    ``test_nodes_in_declared_order`` adapter test). So the original is
    edge-valid only by the *fixture-authoring convention* of declaring
    nodes in dependency order, NOT by an adapter-enforced invariant. A
    fixture declared out of dependency order — or a cyclic topology, for
    which no linearisation is valid — makes the original itself
    edge-invalid; callers must not assume index 0 is always clean.

    A self-loop edge (``from_id == to_id``) is treated as valid — equal
    positions are not "backwards". ``edges`` whose endpoints are all in
    ``variant.nodes`` is the contract: the Topology ``@model_validator``
    rejects *dangling* edges at construction, and all three v0.5 variant
    generators preserve the node-id set, so every current caller honours
    it. A future caller that adds/removes nodes would violate it — hence
    the explicit ``ValueError`` rather than a bare ``KeyError``.
    """
    position = {node.id: idx for idx, node in enumerate(variant.nodes)}
    for edge in edges:
        try:
            from_pos, to_pos = position[edge.from_id], position[edge.to_id]
        except KeyError as exc:
            raise ValueError(
                f"edge {edge.from_id!r}->{edge.to_id!r} references a node "
                f"absent from variant.nodes {tuple(position)!r}"
            ) from exc
        if from_pos > to_pos:
            return True
    return False


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
    help=(
        "topofuzz™ — Topology Safety Fuzzer for multi-agent LLM "
        "systems (TSF v5+)."
    ),
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
    # Common mistake: ``--accept-license`` is a callback flag, so it
    # must come BEFORE the subcommand (`topofuzz --accept-license run
    # ...`), not after. Click otherwise reports a generic "unknown
    # option" which is hard to debug. Surface a placement hint —
    # together with any OTHER unknown flags, so the user sees the
    # whole picture (pass-2 RT M1: hint must not eclipse co-present
    # typos).
    bare_tokens = [tok.split("=", 1)[0] for tok in argv]
    if "--accept-license" in bare_tokens:
        others = [tok for tok in bare_tokens if tok != "--accept-license"]
        msg = (
            "topofuzz: --accept-license must come BEFORE the subcommand "
            "(e.g. `topofuzz --accept-license run ...`, not "
            "`topofuzz run ... --accept-license`)."
        )
        if others:
            msg += f" Also unknown option(s): {' '.join(others)}"
        typer.echo(msg, err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT)
    typer.echo(
        f"topofuzz: unknown option(s): {' '.join(argv)}",
        err=True,
    )
    raise typer.Exit(code=_BAD_USAGE_EXIT)


# Subcommands that skip the first-run license click-through.
#
# ``verify`` is the buyer-side audit tool — re-verifying a topofuzz
# attestation bundle is not "using" topofuzz commercially, so requiring
# acceptance there would block third-party auditors. BACKLOG entry
# "CLI click-through" boundary condition: "Skip the prompt_and_record
# call ONLY for verify." Every other subcommand (scan / run / report /
# repro / inspect / publish / confirm / ablate / keygen /
# langgraph-import / ui) is gated.
_LICENSE_EXEMPT_COMMANDS: frozenset[str] = frozenset({"verify"})


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Print version and exit."
    ),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help=(
            "Accept the BSL 1.1 license terms non-interactively. "
            "Equivalent to running once interactively, answering y at the "
            "prompt, and re-running. Persists to "
            "~/.topofuzz/license-accepted (chmod 600 on POSIX)."
        ),
    ),
) -> None:
    if version:
        typer.echo(f"topofuzz {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    # First-run license click-through. ``verify`` (and any other
    # auditor-facing command) is exempt — see _LICENSE_EXEMPT_COMMANDS.
    # Detection of non-interactive context uses sys.stdin.isatty(); CI
    # / piped invocations MUST pass --accept-license or the command
    # refuses with exit code 2 and an actionable message.
    if ctx.invoked_subcommand in _LICENSE_EXEMPT_COMMANDS:
        return
    non_interactive = not sys.stdin.isatty()
    try:
        accepted = prompt_and_record(
            force=accept_license,
            non_interactive=non_interactive,
        )
    except HomeUnresolvableError as e:
        typer.echo(
            f"topofuzz: cannot resolve user home directory ({e}). "
            "Set HOME (POSIX) or USERPROFILE (Windows) and re-run.",
            err=True,
        )
        raise typer.Exit(code=LICENSE_REFUSED_EXIT_CODE) from e
    if not accepted:
        typer.echo(
            "topofuzz: license acceptance required. Re-run interactively "
            "and answer y at the prompt, or pass --accept-license.",
            err=True,
        )
        raise typer.Exit(code=LICENSE_REFUSED_EXIT_CODE)


def _not_yet(command: str, lands: str) -> None:
    typer.echo(
        f"topofuzz {command}: not yet implemented (lands {lands}).",
        err=True,
    )
    raise typer.Exit(code=_NOT_IMPLEMENTED_EXIT)


# F3 jury size. Mirrors :data:`fuzzeval.judge._JURY_SIZE` (kept in two
# places to avoid the CLI module importing the heavy fuzzeval.judge
# stack at top-level — that would break the verify-only install guard).
_JURY_SIZE = 3


def _parse_jury_models(
    raw: str,
) -> tuple[dict[str, str], bool]:
    """Parse ``--jury-models`` into ``(vendor_models, allow_duplicate_models)``.

    Splits on commas, strips whitespace, rejects empty entries, requires
    exactly :data:`_JURY_SIZE` entries. Vendor names are auto-derived
    from the LiteLLM provider prefix (everything before the first ``/``)
    lowercased; duplicate prefixes get ``_2`` / ``_3`` suffixes so the
    JuryJudge's name-uniqueness invariant holds. Duplicate MODEL ids
    flip the ``allow_duplicate_models`` flag — the JuryJudge needs that
    permission to accept the trio.

    Entries are sorted (case-folded) before vendor-name derivation per
    Fuzzer M3 audit fix: a user re-typing the trio in a different order
    (e.g. ``flash,pro,deepseek`` vs ``pro,flash,deepseek``) MUST produce
    the same vendor → model map so the persisted ``votes_json`` key
    order is stable across superficially-equivalent re-typings.

    RT B1 audit fix: the auto-suffix walk now tracks every produced
    vendor name in a set, and if ``{prefix}_{count}`` is already
    present (because the user smuggled the suffix in as an earlier
    raw input), we keep incrementing ``count`` until we land on an
    unused name. Without this, ``"google_2/x,google/y,google/z"`` would
    silently clobber entry 1 with the auto-generated ``google_2`` for
    entry 3.

    Raises:
        typer.BadParameter: wrong number of entries; empty entry; model
            id missing a ``/`` separator (LiteLLM ids require one);
            empty provider prefix or empty model name within an entry.
    """
    entries = [tok.strip() for tok in raw.split(",")]
    if any(not e for e in entries):
        raise typer.BadParameter(
            f"--jury-models has an empty entry: {raw!r}. "
            f"Pass exactly {_JURY_SIZE} comma-separated LiteLLM model ids."
        )
    if len(entries) != _JURY_SIZE:
        raise typer.BadParameter(
            f"--jury-models expects exactly {_JURY_SIZE} comma-separated "
            f"LiteLLM model ids; got {len(entries)}: {entries!r}."
        )
    for e in entries:
        if "/" not in e:
            raise typer.BadParameter(
                f"--jury-models entry {e!r} is not a LiteLLM model id "
                f"(expected '<provider>/<model>', e.g. 'google/gemini-2.5-flash')."
            )
        # RT m2 audit fix: a trailing slash ("google/") slips past the
        # "/" check but leaves the model name empty — LiteLLM downstream
        # would surface as a cryptic auth/model error in the JuryJudge
        # thread, recorded as judge_failure. Reject at parse time.
        provider, _, model_part = e.partition("/")
        if not provider.strip() or not model_part.strip():
            raise typer.BadParameter(
                f"--jury-models entry {e!r} has an empty provider or "
                "model name. Expected '<provider>/<model>'."
            )
    # Canonicalize order: sort entries case-folded BEFORE deriving vendor
    # names. The persisted vendor-name map is therefore identical for
    # any permutation of the same trio. Fuzzer M3 audit fix.
    sorted_entries = sorted(entries, key=str.lower)
    # Derive vendor names: provider prefix lowercased + numeric suffix
    # on collisions. e.g. sorted input ['deepseek/...', 'google/flash',
    # 'google/pro'] -> {'deepseek': '...', 'google': 'google/flash',
    # 'google_2': 'google/pro'}.
    name_seen: dict[str, int] = {}
    produced: set[str] = set()
    vendor_models: dict[str, str] = {}
    for model_id in sorted_entries:
        prefix = model_id.split("/", 1)[0].strip().lower()
        # Already validated above, but assert as defense-in-depth.
        assert prefix, "empty prefix passed the front-gate check"
        count = name_seen.get(prefix, 0) + 1
        name_seen[prefix] = count
        # First occurrence stays bare; collisions get _2/_3/... If the
        # user smuggled in `google_2/x` as one entry AND submitted two
        # `google/...` entries, the auto-derived `google_2` for the
        # second would collide with the user's literal. Keep
        # incrementing until we land on a free name.
        candidate = prefix if count == 1 else f"{prefix}_{count}"
        while candidate in produced:
            count += 1
            name_seen[prefix] = count
            candidate = f"{prefix}_{count}"
        produced.add(candidate)
        vendor_models[candidate] = model_id
    allow_duplicate_models = len(set(sorted_entries)) != _JURY_SIZE
    return vendor_models, allow_duplicate_models


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
        help="Opt-in jury-of-3 judging. Default trio: Anthropic + OpenAI + Gemini. Override via --jury-models. v0.5 default judge is single Gemini.",
    ),
    jury_models: str | None = typer.Option(
        None,
        "--jury-models",
        help=(
            "Override the --jury-of-3 trio. Comma-separated LiteLLM "
            "model ids (e.g. 'google/gemini-2.5-flash,google/gemini-2.5-pro,"
            "deepseek/deepseek-chat'). Exactly 3 entries required. "
            "Duplicate ids are accepted (single-vendor plumbing-validation "
            "mode); vendor names auto-derive from the LiteLLM prefix with "
            "_1/_2/_3 suffixes for duplicates. Ignored when --jury-of-3 is off."
        ),
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
    rekor: bool = typer.Option(
        False,
        "--rekor",
        help=(
            "Opt in to Sigstore Rekor: after the run signs locally, POST "
            "the DSSE envelope to the Sigstore Public Good Instance and "
            "embed the LogEntry in <run_dir>/report.json.dsse.rekor.json. "
            "Tries 3 times with bounded backoff (~30s ceiling). On "
            "timeout, writes a [REKOR-DEFERRED] marker — re-run with "
            "--rekor later to retry. Requires a dev key "
            "(run `topofuzz keygen` first)."
        ),
    ),
) -> None:
    """Run topology fuzzing against an eval set.

    The cost estimator runs first (§12 contract). Confirm to proceed.
    """
    # Pre-launch disclaimer banner (stderr so it never pollutes the
    # stdout JSON / file:// URI other subcommands emit). Lands the
    # legal-surface ask from the 2026-05-13 disclaimers/hardening audit
    # without spamming users on every `verify` / `keygen` invocation —
    # only `run` (the only subcommand that calls third-party LLMs and
    # spends money) surfaces it.
    typer.echo(
        "topofuzz: research tool, no warranty. Measures topology "
        "fragility — not safety or fitness for purpose. See LICENSE.",
        err=True,
    )

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

    # F3 fail-fast: --jury-models is only meaningful with --jury-of-3,
    # and a malformed --jury-models value should error BEFORE the
    # fixture/eval load so the user sees the parameter error first
    # rather than a fixture-not-found exit obscuring it.
    if jury_models is not None:
        if not jury_of_3:
            typer.echo(
                "topofuzz: --jury-models is only valid with --jury-of-3; "
                "ignoring.",
                err=True,
            )
            # Don't raise — silently drop into single-judge mode so a
            # user retrying a multi-judge invocation without --jury-of-3
            # still gets a working single-judge run.
            jury_models = None
        else:
            # Run the parser once now (its result is discarded — we
            # re-call it at the JuryJudge construction site). The parse
            # raises typer.BadParameter on bad input, which Typer
            # surfaces as a usage error before any work happens.
            _parse_jury_models(jury_models)

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
        if jury_models is None:
            typer.echo(
                "topofuzz: --jury-of-3 invokes Anthropic + OpenAI + Gemini "
                "judges per BLUEPRINT §6. Missing env keys "
                "(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY) will "
                "surface as judge_failure cells, not a kickoff error.",
                err=True,
            )
        else:
            typer.echo(
                f"topofuzz: --jury-of-3 with --jury-models {jury_models!r}. "
                "Missing provider env keys will surface as judge_failure "
                "cells, not a kickoff error.",
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

    # Edge-validity warning on the *original* topology. As of v0.5.1
    # ``order_swap`` is edge-aware (BLUEPRINT §6 / BACKLOG Option B), so
    # permuted variants are guaranteed to be valid linear extensions of
    # the DAG — no order_swap-blindness warning fires. But the adapter
    # preserves ``add_node`` declaration order and does NOT topologically
    # sort (see ``adapter.py`` + the ``test_nodes_in_declared_order``
    # test), so a fixture declared out of dependency order — or a cyclic
    # topology, for which no linearisation is valid — makes the original
    # itself edge-invalid. That is a fixture/topology issue independent
    # of order_swap; surface it so the user sees their baseline is
    # running on a scrambled information flow. Fires in every run mode
    # including ``--dry-run`` because it is a structural property of the
    # loaded topology.
    #
    # We additionally assert defensively that no permuted variant
    # violates the DAG. This is belt-and-suspenders against a future
    # regression in ``order_swap``: if it ever happens we want a loud
    # message rather than silently shipping the v0.5 misdiagnosis.
    if _variant_violates_edges(original, topology.edges):
        typer.echo(
            "topofuzz: the original topology's own node order violates its "
            "data dependencies — the runner will execute the baseline on a "
            "scrambled information flow. This is a fixture/topology issue "
            "(nodes declared out of dependency order, or a cycle), not an "
            "order_swap artifact.",
            err=True,
        )
    # The regression-guard only applies to variants whose NODE-ID
    # SEQUENCE actually differs from the original — i.e. order_swap
    # output. ``role_swap`` and ``decoding_param_swap`` preserve node
    # positions in place (they ``model_copy`` the role / llm_config on
    # the same slot), so if the original is edge-invalid every role /
    # decoding variant inherits the same invalid sequence. Counting
    # those here would re-blame an order_swap bug for what is a fixture-
    # authoring issue already diagnosed by the separate warning above
    # (audit F2 RT MAJOR-1). Filter by id-sequence change so only true
    # order_swap regressions trip this branch.
    original_ids = tuple(n.id for n in original.nodes)
    reordered = [
        v for v in variant_list[1:]
        if tuple(n.id for n in v.nodes) != original_ids
    ]
    edge_invalid_count = sum(
        1 for v in reordered if _variant_violates_edges(v, topology.edges)
    )
    if edge_invalid_count:
        # Defence-in-depth: order_swap is edge-aware as of v0.5.1, so
        # this branch is a regression signal. Surface it loudly and keep
        # running (the warning is diagnostic — exit code is unchanged).
        typer.echo(
            f"topofuzz: WARNING — {edge_invalid_count} of {len(reordered)} "
            "reordered variants execute agents in an order that violates "
            "the graph's data dependencies. order_swap is edge-aware as "
            "of v0.5.1, so this should not happen; please file an issue "
            "with the fixture and rng_seed.",
            err=True,
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
    # F3 audit fix (RT M1): the orphan-`--jury-models` warning lives at
    # the top of the function (right after `_reject_unknown_extras`) and
    # zeros `jury_models` there, so by the time we reach the JuryJudge
    # branch jury_models is non-None ONLY when jury_of_3 is also True.
    # No duplicate guard here.
    resolved_vendor_models: dict[str, str] | None = None
    try:
        if jury_of_3:
            jury_kwargs: dict[str, Any] = {
                "rubric_path": effective_rubric_path,
                "dry_run": dry_run,
            }
            if jury_models is not None:
                # _parse_jury_models already exited on bad input via
                # typer.BadParameter; if we get here the trio is shape-
                # valid (3 ids, all <provider>/<model>).
                vendor_models, allow_dup = _parse_jury_models(jury_models)
                jury_kwargs["vendor_models"] = vendor_models
                jury_kwargs["allow_duplicate_models"] = allow_dup
                resolved_vendor_models = vendor_models
            judge = JuryJudge(**jury_kwargs)
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
        # Verbatim --jury-models string (None when unset / no jury).
        # Preserved so a repro run can rebuild the exact vendor trio
        # without re-deriving it from the synthetic "jury_of_3:..."
        # effective_judge_model id.
        "jury_models": jury_models,
        # Post-parse provenance (RT M2 audit fix). The verbatim string
        # above carries any whitespace / capitalization the user typed;
        # a repro tool re-parsing the manifest would normalize and
        # produce different bytes. Persisting the resolved vendor →
        # model dict alongside the raw string lets readers replay the
        # canonical trio without re-parsing. Null when no jury.
        "jury_models_resolved": resolved_vendor_models,
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

    # Step 18 (§11 line 306): Rekor opt-in. Fires AFTER local sign + AFTER
    # the result banner so the user always sees the run summary even if
    # the Rekor submission hangs out the full 30s window. Catches all
    # exceptions so a Rekor bug never tanks an otherwise-successful run
    # — the BLUEPRINT contract is "Rekor is additive metadata."
    if rekor:
        _maybe_submit_to_rekor(result.run_dir)

    # §A error-taxonomy exit codes (BLUEPRINT lines 634-643).
    # Cap-tripped takes precedence (exit 3) so CI can distinguish a
    # budget-tripped run from success. Otherwise use the §A status
    # mapping from the metrics layer (0 = succeeded / degraded-low,
    # 1 = degraded-high, 2 = failed, 130 = interrupted).
    if result.cap_tripped:
        raise typer.Exit(code=_COST_CAP_EXIT)
    if result.metrics_exit_code != 0:
        raise typer.Exit(code=result.metrics_exit_code)


def _maybe_submit_to_rekor(run_dir: Path) -> None:
    """Best-effort Rekor submission for ``run_dir/attestation.json``.

    Loads the dev key, derives the public PEM, calls
    :func:`topofuzz.rekor.submit_for_run_dir`. Prints a single line of
    operator-readable outcome. Never raises — the deferred marker on
    disk is the load-bearing signal; this function's prints are
    additive UX.

    Imports are inside the function so verify-only installs that never
    pass --rekor don't pay the rekor-module import cost at CLI startup.
    """
    from cryptography.hazmat.primitives import serialization

    from topofuzz.rekor import (
        REKOR_DEFERRED_MARKER,
        looks_like_rekor_logged,
        submit_for_run_dir,
    )

    log = logging.getLogger(__name__)

    # Audit Pass-1 RT MAJOR-1 / common-path 409 avoidance: if a prior
    # invocation already logged this run, skip the submission entirely
    # — re-POSTing the identical envelope to Rekor would return 409
    # Conflict (handled but ugly). Cheaper for the user, kinder to
    # Rekor's rate limits.
    if looks_like_rekor_logged(run_dir):
        typer.echo(
            f"[REKOR] already logged for {run_dir.name}; skipping "
            "re-submission. Delete report.json.dsse.rekor.json to force "
            "a fresh submit."
        )
        return

    pub_path = default_dev_pub_path()
    try:
        # Use the public-key bytes already on disk (written by
        # ``topofuzz keygen``). Loading the PRIVATE key + deriving its
        # public half would also work, but the public file is the
        # canonical witness — and reading it doesn't require touching
        # the encrypted private key material.
        public_key = load_public_key(pub_path)
    except KeyNotFoundError:
        # Audit Pass-1 RT MAJOR-3 / CR MINOR-1 fix: name the file the
        # operator actually needs to inspect (the public key path), not
        # the private key path the prior message named.
        typer.echo(
            f"[REKOR] no dev public key at {pub_path}; run "
            "`topofuzz keygen` first. Skipping submission.",
            err=True,
        )
        return
    except (ValueError, OSError) as e:
        typer.echo(
            f"[REKOR] dev key load failed: {e}. Skipping submission.",
            err=True,
        )
        return

    public_pem_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    try:
        outcome = submit_for_run_dir(run_dir, public_pem_bytes)
    except (OSError, ValueError) as e:
        # Audit Pass-1 CR MAJOR-1 fix: widen catch to include
        # ValueError (envelope-not-UTF-8 path from
        # ``_build_proposed_entry`` reraise). Without this, a future
        # signer that produced non-UTF-8 envelope bytes would crash
        # the CLI AFTER an otherwise-successful run, costing the
        # operator the run summary they were waiting for. We do NOT
        # catch the broader ``Exception`` so a real bug (e.g.
        # AttributeError from a refactor) still surfaces to tests —
        # same discipline as ``orchestrator._maybe_emit_attestation``.
        log.warning("rekor: persistence failed at %s: %s", run_dir, e)
        typer.echo(
            f"[REKOR] persistence failed: {e}. See logs for details.",
            err=True,
        )
        return

    if outcome.ok and outcome.log_entry is not None:
        # Rekor wraps the LogEntry in ``{"<uuid>": {...}}`` for fresh
        # entries; the 409-already-logged stub uses keys
        # ``_rekor_409_already_logged`` + ``location_url``. Both
        # variants are surfaced in a single readable line. The CLI
        # never assumes a specific shape beyond "JSON dict" — extra
        # keys are passed through transparently.
        if outcome.log_entry.get("_rekor_409_already_logged"):
            location = outcome.log_entry.get("location_url") or "(unknown)"
            typer.echo(
                f"[REKOR] already logged (server-side); "
                f"location={location}"
            )
        else:
            uuid_str: str | None = None
            log_index: Any = None
            first_key = next(iter(outcome.log_entry))
            uuid_str = first_key
            inner = outcome.log_entry[first_key]
            if isinstance(inner, dict):
                log_index = inner.get("logIndex")
            parts = ["[REKOR] logged"]
            if uuid_str:
                parts.append(f"uuid={uuid_str}")
            if log_index is not None:
                parts.append(f"logIndex={log_index}")
            typer.echo(" ".join(parts))
    else:
        typer.echo(
            f"[REKOR-DEFERRED] {outcome.error or 'submission failed'}. "
            f"Marker: {run_dir / REKOR_DEFERRED_MARKER}. "
            "Re-run with --rekor to retry.",
            err=True,
        )


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
    # ``default_resolver`` is the vendor-dispatched canonical→served
    # mapper (gemini/openai/anthropic prefix-strip + Anthropic snapshot
    # map). Without it the §7 cache keys on canonical IDs while the
    # provider responds with bare/served IDs, mismatch-warning per call
    # and 5-10× cache miss amplification on warm runs.
    # (closes BACKLOG: model_served_version_resolver)
    from topofuzz.orchestrator import execute_run
    from topofuzz.runner import default_resolver

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
                resolver=default_resolver,
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
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def repro(
    ctx: typer.Context,
    run_dir: str = typer.Argument(
        ...,
        help="Run directory (contains manifest.json + run.sqlite).",
    ),
    eval_id: str | None = typer.Option(
        None,
        "--eval",
        help=(
            "Specific eval id to reproduce. Default: lexically-first eval "
            "id whose variants flip across the verdict threshold."
        ),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Destination for the generated script. Defaults to "
            "<run_dir>/repro_<eval_id>.py."
        ),
    ),
) -> None:
    """Emit a standalone reproducer script for a verdict flip.

    BLUEPRINT_v5plus.md step 19b. Imports the user's graph module,
    replays the two losing orderings against the failing input, prints
    both verdicts side-by-side, and ``assert``s the mismatch.

    Exit codes:

    * 0 — script written; path printed to stdout.
    * 1 — no verdict flip found in the run (or for ``--eval``).
    * 2 — usage error (run_dir is not a directory, missing files).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    run_dir_path = Path(run_dir)
    if not run_dir_path.is_dir():
        typer.echo(
            f"topofuzz repro: run_dir is not a directory: {run_dir}",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    from topofuzz.repro import ReproError, write_repro_script

    try:
        target = write_repro_script(
            run_dir_path, eval_id=eval_id, out_path=out
        )
    except ReproError as e:
        typer.echo(f"topofuzz repro: {e}", err=True)
        raise typer.Exit(code=_VERIFY_FAIL_EXIT) from e
    except FileNotFoundError as e:
        typer.echo(f"topofuzz repro: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    typer.echo(str(target))


@app.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def inspect(
    ctx: typer.Context,
    run_dir: str = typer.Argument(
        ...,
        help="Run directory (contains manifest.json + run.sqlite).",
    ),
    eval_id: str | None = typer.Option(
        None,
        "--eval",
        help=(
            "Specific eval id to inspect. Default: lexically-first eval "
            "id whose original variant disagrees with another variant."
        ),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Destination for the generated HTML. Defaults to "
            "<run_dir>/inspect_<eval_id>.html."
        ),
    ),
) -> None:
    """Render a side-by-side HTML inspector for one verdict flip.

    Shows the original ordering on the left and the flipping ordering
    on the right, stacked agent-by-agent so the reader can see exactly
    where the reasoning diverged. Key risk phrases ("deny", "approve",
    "concerning", ...) are colour-coded.

    Exit codes:

    * 0 — HTML written; path printed to stdout.
    * 1 — no verdict flip found in the run (or for ``--eval``).
    * 2 — usage error (run_dir is not a directory, missing files).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    run_dir_path = Path(run_dir)
    if not run_dir_path.is_dir():
        typer.echo(
            f"topofuzz inspect: run_dir is not a directory: {run_dir}",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    from topofuzz.inspect import InspectError, write_inspect_html

    try:
        target = write_inspect_html(
            run_dir_path, eval_id=eval_id, out_path=out
        )
    except InspectError as e:
        typer.echo(f"topofuzz inspect: {e}", err=True)
        raise typer.Exit(code=_VERIFY_FAIL_EXIT) from e

    typer.echo(str(target))


@app.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def publish(
    ctx: typer.Context,
    run_dir: str = typer.Argument(
        ...,
        help="Run directory (contains manifest.json + run.sqlite).",
    ),
    pdf: Path | None = typer.Option(
        None,
        "--pdf",
        help=(
            "Destination PDF path for the full buyer-pitch PDF "
            "(report.html + top flipping inspect pages). Overwritten "
            "atomically if it exists. At least one of --pdf or "
            "--ceo-card is required."
        ),
    ),
    ceo_card: Path | None = typer.Option(
        None,
        "--ceo-card",
        help=(
            "Destination PDF path for the single-page CEO-card "
            "derivative — cold-email-shaped, 1-2 pages, headline "
            "metric + Wilson CI + one flipped verdict + attestation "
            "footer. At least one of --pdf or --ceo-card is required."
        ),
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        help=(
            "Optional company name to substitute into the CEO-card's "
            "'what this means for $TARGET' paragraph. Only meaningful "
            "with --ceo-card. Max 80 chars."
        ),
    ),
    max_flips: int = typer.Option(
        # Hardcoded 5 here mirrors topofuzz.publish.DEFAULT_MAX_FLIPS;
        # the constant isn't reachable at decorator-eval time without
        # pulling the module into cli.py's import surface, which would
        # break the lazy-import contract for verify-only installs.
        # ``test_cli_publish_default_max_flips_matches_module_constant``
        # pins them together.
        5,
        "--max-flips",
        min=0,
        max=50,
        help=(
            "Include up to this many inspect_<eid>.html pages — the "
            "evals with the most variant-disagreements. Default 5; "
            "hard limit 50. Only meaningful with --pdf."
        ),
    ),
) -> None:
    """Bundle report.html + top flipping inspect pages into one PDF.

    Produces a single share-by-email artifact for buyer pitches. The
    HTML files in ``run_dir`` remain canonical — the PDF loses
    interactivity (hover rationales, expand panels) and exists so a CTO
    reader can re-attest the bundle by copy-pasting the footer's
    ``run_id`` into ``topofuzz verify``.

    With ``--ceo-card OUT.pdf`` the command also (or instead) emits a
    single-page CEO-card derivative — designed for the opening cold-
    email attachment. The full F5 PDF is the follow-up audit link.

    Side effects: ``report.html`` and ``inspect_<eid>.html`` are
    rendered into ``run_dir`` if absent (only when ``--pdf`` is set).

    Exit codes:

    * 0 — PDF(s) written; paths printed to stdout (one per line).
    * 1 — publish failed (missing optional deps, missing files, render
      failure).
    * 2 — usage error (run_dir is not a directory, or neither --pdf nor
      --ceo-card supplied).

    Requires the ``[pdf]`` install extra::

        pip install 'topofuzz[pdf]'
        playwright install chromium
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    if pdf is None and ceo_card is None:
        typer.echo(
            "topofuzz publish: at least one of --pdf or --ceo-card "
            "must be supplied.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)
    # F6 pass-1 RT M2: ``--target ""`` slips past `is not None`. Treat
    # any truthy target as "supplied" so the --pdf-only path doesn't
    # silently ignore an empty string the operator intended as a value.
    if target and ceo_card is None:
        typer.echo(
            "topofuzz publish: --target is only meaningful with "
            "--ceo-card.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    run_dir_path = Path(run_dir)
    if not run_dir_path.is_dir():
        typer.echo(
            f"topofuzz publish: run_dir is not a directory: {run_dir}",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    from topofuzz.publish import (
        PublishError,
        build_ceo_card_pdf,
        build_publish_pdf,
    )

    try:
        if pdf is not None:
            full_target = build_publish_pdf(
                run_dir_path, out_path=pdf, max_flips=max_flips
            )
            typer.echo(str(full_target))
        if ceo_card is not None:
            card_target = build_ceo_card_pdf(
                run_dir_path,
                out_path=ceo_card,
                target_company=target,
            )
            typer.echo(str(card_target))
    except PublishError as e:
        typer.echo(f"topofuzz publish: {e}", err=True)
        raise typer.Exit(code=_VERIFY_FAIL_EXIT) from e


@app.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def confirm(
    ctx: typer.Context,
    run_dir: str = typer.Argument(
        ...,
        help="Run directory (contains manifest.json + run.sqlite).",
    ),
    repeats: int = typer.Option(
        3,
        "--repeats",
        min=1,
        help=(
            "K — re-runs per screened flip candidate AND per reference "
            "cell. Majority-vote over K separates the topology effect "
            "from LLM noise."
        ),
    ),
    max_cost_usd: float | None = typer.Option(
        None,
        "--max-cost-usd",
        min=0.0,
        help=(
            "Soft-warn at 80%%, hard-stop at 100%%. None = uncapped. "
            "Must be positive — a cap of 0 is a usage error (it would "
            "hard-stop before the first repeat)."
        ),
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge",
        help=(
            "Judge model id. Default: read from the run's manifest "
            "(cli_args.judge)."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt (the confirm pass bills real LLM calls).",
    ),
) -> None:
    """Re-run screened verdict flips K times to separate topology from noise.

    The pipeline is non-deterministic — re-running the SAME ordering
    flips the verdict some fraction of the time. ``topofuzz confirm``
    re-runs every screened flip candidate K times AND re-runs the
    original variant K times per relevant eval, persisting every repeat
    into the ``flip_confirmations`` table. The noise-corrected metrics
    (``verdict_stability``, ``confirmed_verdict_flip_rate``,
    ``voo_noise_floor``) then majority-vote over the K repeats.

    Exit codes:

    * 0 — confirm pass completed (possibly with partial results if the
      cost cap tripped — that is surfaced in the summary, not the code).
    * 2 — usage error (run_dir not a directory, missing / malformed
      manifest, fixture can't be loaded).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    run_dir_path = Path(run_dir)
    if not run_dir_path.is_dir():
        typer.echo(
            f"topofuzz confirm: run_dir is not a directory: {run_dir}",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    # N-RT-2: ``--max-cost-usd 0`` is a usage error, not "free run". The
    # hard-stop predicate is ``total_cost >= max_cost_usd``, so a cap of
    # 0 trips on ``0 >= 0`` BEFORE the first repeat — an instantly
    # degenerate partial pass that re-runs nothing. Reject it with a
    # clear message rather than letting the operator get an empty result.
    if max_cost_usd is not None and max_cost_usd <= 0.0:
        typer.echo(
            "topofuzz confirm: --max-cost-usd must be a positive amount "
            "(a cap of 0 would hard-stop before the first repeat runs). "
            "Omit the flag for an uncapped pass.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    typer.echo(
        "topofuzz: research tool, no warranty. The confirm pass re-runs "
        "the pipeline and bills real LLM calls. See LICENSE.",
        err=True,
    )

    # Lazy import: the confirm pass pulls the runtime stack (loaders →
    # langgraph, runner → litellm, the variant generators). Keeping it
    # out of module scope preserves the §11a verify-path-safe contract.
    from topofuzz.confirm import ConfirmError, run_confirm_pass

    if not yes:
        if not typer.confirm(
            f"Re-run screened flip candidates {repeats}× each "
            f"(bills real LLM calls)?",
            default=False,
        ):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    try:
        summary = run_confirm_pass(
            run_dir_path,
            repeats=repeats,
            max_cost_usd=max_cost_usd,
            judge_model=judge_model,
        )
    except ConfirmError as e:
        typer.echo(f"topofuzz confirm: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    typer.echo("")
    typer.echo(
        f"Candidates found: {summary.n_candidates} "
        f"({summary.n_candidate_variants} distinct variant(s))"
    )
    typer.echo(
        f"Repeats run: {summary.reference_repeats_run} reference + "
        f"{summary.candidate_repeats_run} candidate "
        f"(K={summary.repeats})"
    )
    typer.echo(f"Cost (priced cells only): ${summary.total_cost_usd:.4f}")
    if summary.partial:
        typer.echo(
            "Cost cap tripped; results are PARTIAL — re-run "
            "`topofuzz confirm` to complete the remaining repeats."
        )


@app.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def ablate(
    ctx: typer.Context,
    run_dir: str = typer.Argument(
        ...,
        help="Run directory (contains manifest.json + run.sqlite).",
    ),
    max_cost_usd: float | None = typer.Option(
        None,
        "--max-cost-usd",
        min=0.0,
        help=(
            "Soft-warn at 80%%, hard-stop at 100%%. None = uncapped. "
            "A cap of 0 is rejected — it would trip before the first "
            "ablation point ran."
        ),
    ),
    fixture: str | None = typer.Option(
        None,
        "--fixture",
        help=(
            "Override the manifest's recorded fixture spec (use when the "
            "fixture file has moved on disk since the source run)."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Short-circuit every terminal re-execution AND every jury "
            "re-judgment into deterministic stubs. Exercises the full "
            "pass with $0 LLM spend. Useful for plumbing-validation in "
            "CI; the attribution numbers it produces are not meaningful."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt (the ablate pass bills real LLM calls).",
    ),
) -> None:
    """Run counterfactual leave-one-out attribution on a source run's flips.

    For each flipping ``(canonical_variant, flipping_variant, eval)``
    triple in the source run, walk every node in the flipping
    variant's topology and ablate it: substitute the node's response
    with the canonical variant's, re-execute the terminal node with
    that hybrid prior, and re-judge under the source run's jury-of-3.
    Whichever node's swap most often *closes the flip* is the
    load-bearing agent — the buyer-actionable "which agent caused
    this?" answer.

    The source run MUST have used ``--jury-of-3``; SingleJudge runs are
    rejected at the gate so the leave-one-out re-judgments don't inherit
    the source's ~26 % per-judgment verdict noise (see BLUEPRINT §F4).

    Writes ``<run_dir>/ablation.json``. ``topofuzz inspect`` consumes
    the sidecar to render a per-flip attribution panel next to the
    existing jury-vote panel.

    Exit codes:

    * 0 — pass completed (possibly with partial results when the cost
      cap tripped; partial state is surfaced in the summary, not the
      exit code).
    * 2 — usage error (run_dir not a directory; missing / malformed
      manifest; non-jury-of-3 source; fixture can't be loaded).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    run_dir_path = Path(run_dir)
    if not run_dir_path.is_dir():
        typer.echo(
            f"topofuzz ablate: run_dir is not a directory: {run_dir}",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    if max_cost_usd is not None and max_cost_usd <= 0.0:
        typer.echo(
            "topofuzz ablate: --max-cost-usd must be a positive amount. "
            "Omit the flag for an uncapped pass.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    if not dry_run:
        typer.echo(
            "topofuzz: research tool, no warranty. The ablate pass re-"
            "executes the terminal node and re-judges under the jury — "
            "this bills real LLM calls. See LICENSE.",
            err=True,
        )

    from topofuzz.ablation import AblationError, run_ablation_pass

    if not yes and not dry_run:
        if not typer.confirm(
            "Run counterfactual ablation on every flipping scenario "
            "(bills real LLM calls)?",
            default=False,
        ):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    try:
        summary = run_ablation_pass(
            run_dir_path,
            max_cost_usd=max_cost_usd,
            fixture_override=fixture,
            dry_run=dry_run,
        )
    except AblationError as e:
        typer.echo(f"topofuzz ablate: {e}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from e

    typer.echo("")
    typer.echo(
        f"Flipping triples: {summary.n_flipping_pairs} | "
        f"Ablation points: {summary.n_ablation_points} "
        f"({summary.n_ablation_points_meaningful} meaningful)"
    )
    typer.echo(f"Cost (priced cells only): ${summary.total_cost_usd:.4f}")
    # Pass-1 Fuzzer MAJOR M5 fix: print partial banner BEFORE the
    # attribution table, not after. A buyer copy-pasting the table can
    # easily miss a footer line; a header banner is harder to miss.
    if summary.partial:
        typer.echo(
            "WARNING: cost cap tripped mid-pass; results are PARTIAL. "
            "Per-node rows below may have an incomplete denominator "
            "(some flipping triples weren't reached). Re-run "
            "`topofuzz ablate` to complete the remaining ablation points."
        )
    # Pass-1 Fuzzer BLOCKER B3 fix: dry-run produces stub jury scores
    # (every vendor votes 0.5 → majority APPROVE → score 1.0 — see
    # ablation.py dry_run docstring). Banner says so unmistakably.
    if dry_run:
        typer.echo(
            "DRY-RUN: every jury vote was stubbed to APPROVE. The "
            "percentages below exercise the pass plumbing only; do not "
            "interpret them as attribution. Re-run without --dry-run "
            "for measured percentages."
        )
    # Pass-1 Fuzzer BLOCKER B1 fix: if every upstream agent in V's
    # execution order ran AFTER the terminal node, by_node will contain
    # ONLY the terminal-baseline entry at 100%. A buyer reading the
    # one-line table would conclude "the verdict agent caused 100% of
    # flips" — circular by construction. Surface the structural cause.
    by_node_non_terminal = {
        nid: slot for nid, slot in summary.by_node.items()
        if not slot.get("is_terminal")
    }
    if (
        summary.n_flipping_pairs > 0
        and not by_node_non_terminal
        and summary.by_node
    ):
        typer.echo("")
        typer.echo(
            "WARNING: every upstream agent in V's execution order ran "
            "AFTER the terminal node; no causal attribution is possible "
            "for this flip-set. The 100% row below is the terminal-"
            "substitution baseline (control column), not an attribution "
            "signal. Try a different flipping triple in this run, or a "
            "topology that puts the terminal last in every permutation."
        )
    if summary.by_node:
        typer.echo("")
        typer.echo(
            "Per-node attribution "
            f"(aggregate across all {summary.n_flipping_pairs} flipping "
            "triple(s); inspect view shows per-flip breakdown):"
        )
        # Sort: non-terminal first (the load-bearing signal), then
        # the terminal baseline last. Within each group sort by
        # attribution_pct descending so the strongest signal lands at
        # the top.
        rows = sorted(
            summary.by_node.items(),
            key=lambda kv: (
                kv[1].get("is_terminal", False),
                -kv[1].get("attribution_pct", 0.0),
            ),
        )
        for node_id, slot in rows:
            label = "(terminal — baseline)" if slot.get("is_terminal") else ""
            # Pass-1 Fuzzer MAJOR M5 fix: when partial, annotate per-
            # node samples / n_flipping_pairs so a buyer knows whether
            # the row was fully measured or only partially.
            sample_annot = ""
            if summary.partial and summary.n_flipping_pairs > 0:
                sample_annot = (
                    f"  [{slot['total_meaningful']}/"
                    f"{summary.n_flipping_pairs} triples sampled]"
                )
            typer.echo(
                f"  {node_id:<32s} "
                f"{slot['flips_closed']}/{slot['total_meaningful']}  "
                f"{slot['attribution_pct']:6.2f}%  {label}{sample_annot}"
            )
    typer.echo("")
    typer.echo(
        f"Wrote sidecar: {run_dir_path / 'ablation.json'!s}"
    )


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


@app.command(
    name="langgraph-import",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
)
def langgraph_import_cmd(
    ctx: typer.Context,
    spec: str = typer.Argument(
        ...,
        help=(
            "Fixture spec in 'path/to/file.py:attr' form. ``attr`` is "
            "either a compiled LangGraph or a no-arg callable that returns "
            "one (e.g. ``build_graph``). Same shape as ``topofuzz run``."
        ),
    ),
    label: str = typer.Option(
        ...,
        "--label",
        help="User-facing workflow name shown on the UI tile (2-60 chars).",
    ),
    rubric: str | None = typer.Option(
        None,
        "--rubric",
        help=(
            "Rubric paragraph the judge uses to extract the verdict from "
            "the final agent's output (50-2000 chars). For longer rubrics "
            "use --rubric-file. Exactly one of --rubric / --rubric-file "
            "is required."
        ),
    ),
    rubric_file: Path | None = typer.Option(
        None,
        "--rubric-file",
        help=(
            "Path to a UTF-8 text file containing the rubric paragraph. "
            "PowerShell-friendly alternative to --rubric for long rubrics. "
            "Exactly one of --rubric / --rubric-file is required."
        ),
    ),
    domain_noun: str = typer.Option(
        "case",
        "--domain-noun",
        help="What the pipeline decides on (e.g. 'loan application').",
    ),
    decision_verb: str = typer.Option(
        "approve / deny",
        "--decision-verb",
        help="What the verdict is phrased as (e.g. 'approve / deny').",
    ),
    fallback_model: str = typer.Option(
        "google/gemini-2.5-flash",
        "--fallback-model",
        help=(
            "Model used for any node not covered by --llm-config or by "
            "the module's LLM_CONFIGS dict. Must be in the BYOP allow-"
            "list. Currently google/gemini-2.5-flash or google/gemini-2.5-pro."
        ),
    ),
    llm_config: list[str] = typer.Option(
        [],
        "--llm-config",
        help=(
            "Per-node model override in NODE=MODEL form, repeatable: "
            "--llm-config reviewer=google/gemini-2.5-pro "
            "--llm-config decider=google/gemini-2.5-pro. Useful when the "
            "user's LangGraph module does not export an LLM_CONFIGS dict. "
            "CLI overrides take precedence over the module's LLM_CONFIGS."
        ),
    ),
) -> None:
    """Import a user's existing LangGraph as a topofuzz custom workflow tile.

    Walks the compiled graph at SPEC, asserts it's a linear chain, and
    writes a CustomWorkflow under ``~/.topofuzz/custom-workflows/`` —
    surfacing the graph as a clickable tile in ``topofuzz ui``. If the
    module also exports ``LLM_CONFIGS: dict[str, LLMConfig]`` (the
    gallery convention), per-node models are picked up automatically.

    Branching graphs, cycles, and multi-START topologies are refused with
    a clear message — BYOP supports linear chains only. Rename or flatten
    in your LangGraph source, then re-import.

    Exit codes:

    * 0 — workflow imported and saved.
    * 2 — usage error (bad spec, branching graph, validation failure).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    # Resolve rubric source: exactly one of --rubric / --rubric-file. Tested
    # via test_cli_langgraph_import.
    if rubric is not None and rubric_file is not None:
        typer.echo(
            "topofuzz langgraph-import: pass --rubric OR --rubric-file, "
            "not both.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)
    if rubric is None and rubric_file is None:
        typer.echo(
            "topofuzz langgraph-import: one of --rubric / --rubric-file "
            "is required.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)
    if rubric_file is not None:
        try:
            rubric = rubric_file.read_text(encoding="utf-8")
        except OSError as exc:
            typer.echo(
                f"topofuzz langgraph-import: could not read --rubric-file "
                f"{rubric_file!s}: {exc}",
                err=True,
            )
            raise typer.Exit(code=_BAD_USAGE_EXIT) from exc

    # Lazy imports: pull the BYOP and LangGraph stacks only when the
    # subcommand actually runs, preserving the §11a verify-path-safe
    # contract used elsewhere in this CLI.
    from topofuzz.ir import LLMConfig
    from topofuzz.ui.custom_storage import save_custom_workflow
    from topofuzz.ui.custom_validation import CustomFormError
    from topofuzz.ui.langgraph_import import (
        LangGraphImportError,
        custom_workflow_from_module_spec,
    )

    # Parse --llm-config NODE=MODEL pairs into LLMConfigs. Empty list is fine
    # — the function-side default detects the module's LLM_CONFIGS.
    override_configs: dict[str, LLMConfig] = {}
    for raw in llm_config:
        if "=" not in raw:
            typer.echo(
                f"topofuzz langgraph-import: --llm-config must be NODE=MODEL "
                f"(got {raw!r}).",
                err=True,
            )
            raise typer.Exit(code=_BAD_USAGE_EXIT)
        node, _, model = raw.partition("=")
        node = node.strip()
        model = model.strip()
        if not node or not model:
            typer.echo(
                f"topofuzz langgraph-import: --llm-config NODE=MODEL needs "
                f"both sides (got {raw!r}).",
                err=True,
            )
            raise typer.Exit(code=_BAD_USAGE_EXIT)
        if node in override_configs:
            typer.echo(
                f"topofuzz langgraph-import: --llm-config {node!r} "
                f"specified more than once.",
                err=True,
            )
            raise typer.Exit(code=_BAD_USAGE_EXIT)
        override_configs[node] = LLMConfig(model=model)

    try:
        cw = custom_workflow_from_module_spec(
            spec,
            label=label,
            domain_noun=domain_noun,
            decision_verb=decision_verb,
            rubric_paragraph=rubric,
            fallback_model=fallback_model,
            llm_config_overrides=override_configs or None,
        )
    except LangGraphImportError as exc:
        typer.echo(f"topofuzz langgraph-import: {exc}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from exc
    except CustomFormError as exc:
        # Per-field errors — render them as a short list so the user can
        # fix the offending node id / rubric / etc.
        typer.echo("topofuzz langgraph-import: validation failed:", err=True)
        for field, msg in exc.errors.items():
            typer.echo(f"  {field}: {msg}", err=True)
        raise typer.Exit(code=_BAD_USAGE_EXIT) from exc

    workflow = save_custom_workflow(cw)

    typer.echo(f"Imported {label!r} as custom workflow {cw.key}.")
    ordered = " -> ".join(a.agent_id for a in cw.agents)
    typer.echo(f"{len(cw.agents)} agents in order: {ordered}")
    typer.echo(f"Saved to {workflow.fixture_path.parent}")
    typer.echo(
        "Start the UI with `topofuzz ui` -- the workflow appears as a tile."
    )


@app.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def ui(
    ctx: typer.Context,
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help=(
            "Bind address. Locked to loopback (127.0.0.1, localhost, ::1) "
            "because the UI has no auth and shouldn't reach the LAN. "
            "Override only via --i-understand-no-auth."
        ),
    ),
    port: int = typer.Option(
        8765,
        "--port",
        help=(
            "Preferred port. If taken, the launcher walks up to find a "
            "free one and prints the actual URL."
        ),
    ),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the default browser to the UI on startup.",
    ),
    allow_remote: bool = typer.Option(
        False,
        "--i-understand-no-auth",
        help=(
            "Explicit acknowledgement that exposing the UI beyond loopback "
            "lets ANYONE on your network start runs (spending your Gemini "
            "key) and read every past run's report. Required to use a non-"
            "loopback --host. Do NOT pass this on a coffee-shop network."
        ),
    ),
) -> None:
    """Launch the topofuzz web UI on http://localhost:<port>.

    Plug-and-play: one command, browser opens, click and go. No CSV
    editing, no Python knowledge, no config files. See
    ``topofuzz.ui.app:build_app`` for the route map.

    Loopback-only by default — there is no auth on this UI. RT-audit
    blocker B1: a `--host 0.0.0.0` invocation on a public Wi-Fi network
    would let anyone on the LAN POST /setup and spend the user's API
    budget. ``--i-understand-no-auth`` is the explicit override.
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    # Loopback enforcement — RT-audit B1.
    _LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    # Note: "0.0.0.0" is technically loopback-binding-equivalent on some
    # systems but exposes every interface on others; treat it as remote.
    is_loopback = host in {"127.0.0.1", "localhost", "::1"}
    if not is_loopback and not allow_remote:
        typer.echo(
            f"topofuzz ui: refusing to bind {host!r}. The UI has no "
            f"authentication. Pass --i-understand-no-auth to override. "
            f"See `topofuzz ui --help` for details.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    # Lazy import — FastAPI / uvicorn aren't on the verify-path install.
    try:
        import socket
        import threading
        import time
        import webbrowser

        import uvicorn

        from topofuzz.ui.app import build_app
    except ImportError as exc:
        typer.echo(
            f"topofuzz ui: required dependency missing ({exc}). "
            "Install with `pip install topofuzz[ui]`.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT) from exc

    # Walk to find a free port if the preferred one is taken. RT axis 9.
    actual_port = port
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, candidate))
                actual_port = candidate
                break
            except OSError:
                continue
    else:
        typer.echo(
            f"topofuzz ui: no free port found between {port} and {port + 19}.",
            err=True,
        )
        raise typer.Exit(code=_BAD_USAGE_EXIT)

    url = f"http://{host}:{actual_port}/"
    typer.echo(f"topofuzz UI running at {url}")
    typer.echo("Press Ctrl+C to stop.")

    fastapi_app = build_app()

    if open_browser:
        # Open browser shortly after uvicorn starts. Run in a thread so
        # we don't block on a hung browser launch.
        def _open_later() -> None:
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except OSError:
                pass

        threading.Thread(target=_open_later, daemon=True).start()

    config = uvicorn.Config(
        fastapi_app,
        host=host,
        port=actual_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


@app.command(
    context_settings={
        # Reserved-flag rejection (§16 #2) is universal — `keys` opts
        # in the same way every other subcommand does so a user who
        # types `topofuzz keys --kms-region` gets the v1.0-reserved
        # message instead of a generic "no such option".
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
def keys(ctx: typer.Context) -> None:
    """Pre-flight: list which vendor API keys are visible in the env.

    Prints one row per known vendor (google, openai, anthropic,
    deepseek) showing the env var name, status, and a masked preview
    of the value. The full key is NEVER printed — only the first 4 and
    last 4 chars, with an ASCII ``..`` between. Keys shorter than 12
    chars are flagged ``too_short`` and printed with no value at all
    (the entire short string could BE the key). Whitespace-only
    values are treated as ``missing``.

    Status taxonomy:

    * ``present`` — variable set, stripped value length ≥ 12.
    * ``missing`` — variable unset, empty, or whitespace-only.
    * ``too_short`` — variable set, stripped value length < 12
      (likely a placeholder or typo).

    Exit codes:

    * 0 — at least one key is present.
    * 1 — no keys present at all; the operator hasn't configured
      anything yet. Re-export at least one of ``GEMINI_API_KEY``,
      ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``DEEPSEEK_API_KEY``
      and re-run.
    * 2 — usage error (e.g. reserved flag passed).
    """
    _reject_reserved_flags(list(ctx.args))
    _reject_unknown_extras(list(ctx.args))

    rows = inspect_keys()
    typer.echo(format_table(rows))
    if not at_least_one_present(rows):
        raise typer.Exit(code=1)


def main() -> None:
    """Module entry-point used by ``python -m topofuzz``."""
    logging.basicConfig(level=logging.WARNING)
    app()


if __name__ == "__main__":
    main()
