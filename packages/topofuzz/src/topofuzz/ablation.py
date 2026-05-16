"""``topofuzz ablate``: counterfactual leave-one-out per-agent attribution.

After F2 (edge-aware order_swap) and F3 (jury-of-3) shipped, an evals
buyer still asks one more question: *which agent caused the flip?*
``topofuzz ablate`` answers it.

Design contract (locked)
------------------------

For each flipping ``(canonical_variant, flipping_variant, eval)`` triple
in a source run, run an ablation experiment over every node in the
variant's topology:

* **Non-terminal node N (re-execute terminal).** Build a hybrid context
  for the terminal: the flipping variant's per-node outputs in the
  *flipping variant's execution order*, but with node N's response_text
  swapped for the canonical variant's NodeOutput for the same node_id.
  Re-execute the terminal node ONCE with that hybrid prior (one LLM call
  to the terminal's vendor). Re-judge under the source run's jury-of-3.
  If the new jury verdict matches the canonical variant's verdict, then
  N's V-output was load-bearing for the flip.
* **Terminal node N == terminal (substitute terminal).** Skip the
  re-execution: substitute the canonical variant's terminal response_text
  directly and re-judge under the source run's jury-of-3. By
  construction the verdict matches the canonical's — this is a deliberate
  baseline-check column, surfaced (and labeled) in the report so the
  buyer can see the experimental control alongside the upstream
  attribution numbers.
* **Node ran AFTER the terminal in V's execution order.** Permuted
  variants may run the verdict-node early; nodes that ran later
  contributed nothing to the verdict the judge scored. We record these
  as ``kind="skip_downstream"`` rather than running a no-op re-execution
  whose result would be statistically indistinguishable from the
  original.
* **Canonical produced no NodeOutput for N (judge_failure / error
  cell).** Record as ``kind="skip_no_canonical_output"``; cannot ablate.

The source run MUST have been a ``--jury-of-3`` run (manifest
``cli_args.jury_of_3 == True``). The leave-one-out re-judgments live
on the same noise substrate the source run paid for: a SingleJudge
source run is rejected up front so the attribution numbers don't
inherit ~26 % per-judgment verdict noise (see
[[project_pipeline_nondeterminism_finding]]).

Outputs
-------

* ``<run_dir>/ablation.json`` — a sidecar mirroring the
  ``confirm.json`` pattern; readable without opening ``run.sqlite``.
  The inspector consumes it (when present) to render the new
  attribution panel next to the existing jury-vote panel for the
  same flip.
* No new SQLite schema. Per-ablation rows live in the sidecar's
  ``results`` array. Adversarial-relay-size runs produce ~50 KB of
  sidecar JSON; lending-size runs typically produce an empty
  ``results: []`` (no flips → nothing to ablate).

The pass is sequential: one variant at a time, one ablation point at
a time. ``--max-cost-usd`` honours the same soft-warn (80 %) / hard-stop
(100 %) pattern as ``topofuzz run`` / ``topofuzz confirm``; on hard-stop
whatever completed is persisted and ``partial: true`` lands in the
sidecar.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

ABLATION_SIDECAR_FILENAME = "ablation.json"
_ABLATION_SIDECAR_SCHEMA_VERSION = 1

# Outcome kinds for one ablation point. Persisted as a string so the
# sidecar reader doesn't depend on the dataclass shape.
_KIND_REEXECUTE_TERMINAL = "reexecute_terminal"
_KIND_SUBSTITUTE_TERMINAL = "substitute_terminal"
_KIND_SKIP_DOWNSTREAM = "skip_downstream"
_KIND_SKIP_NO_CANONICAL_OUTPUT = "skip_no_canonical_output"
_KIND_SKIP_NO_V_OUTPUT = "skip_no_v_output"
_KIND_FAILED = "failed"

# Verdict-binarisation threshold for "did the flip close". Pinned to
# 0.5 to match :func:`fuzzeval.judge.binary_from_continuous`'s default
# and the headline verdict pill — so the panel's "flip_closed" count
# agrees with what the inspector's verdict pill shows.
_VERDICT_THRESHOLD = 0.5

# Abort the pass after N consecutive failures (terminal re-exec OR
# jury re-judge). A genuine vendor outage produces sporadic failures
# interleaved with successes; a programmer bug (signature drift,
# manifest mismatch, response-shape change) fails every cell in a row.
# Aborting at N keeps the second case loud while letting the first
# degrade gracefully. Pass-1 RT MAJOR M2 (and Fuzzer parallel).
_MAX_CONSECUTIVE_FAILURES = 3

# Fallback non-zero floor for ``_estimate_jury_cost`` when reflection
# fails (M1 fix). Approximate cost of one jury re-judge across three
# vendors at the safe-default rate of $0.005/1K with a ~2KB envelope:
# 3 × $0.005 × 2 = $0.03. The actual real-LLM jury cost is typically
# $0.02–$0.05 per re-judge, so this floor is order-of-magnitude
# correct and the cap accounting can never silently see $0 because
# reflection broke.
_JURY_COST_FALLBACK_FLOOR_USD = 0.03


class AblationError(ValueError):
    """Raised when an ablation pass cannot run.

    Includes: missing run.sqlite / manifest, source run was not
    jury-of-3, fixture spec can't be reloaded, ``repeats`` non-positive.
    """


@dataclass(frozen=True)
class AblationResult:
    """One leave-one-out experiment's outcome.

    Persisted into ``ablation.json`` under ``results``. Field names are
    explicit-prefixed (``v_`` for the flipping variant, ``canonical_``)
    so the sidecar reads as plain English without a legend.
    """

    canonical_variant_id: str
    flipping_variant_id: str
    eval_id: str
    ablated_node_id: str
    kind: str
    # Verdict scores (after jury majority-vote binarisation) in [0, 1].
    # None when the ablation point was skipped before re-judging.
    original_v_score: float | None
    canonical_score: float | None
    new_v_score: float | None
    flip_closed: bool | None
    cost_usd: float
    # New terminal response_text after re-execution / substitution. None
    # for skipped points. Truncated to _RESPONSE_TEXT_CAP bytes before
    # persistence so the sidecar doesn't grow unbounded for chatty
    # vendors.
    new_terminal_response_text: str | None
    # Per-vendor jury vote breakdown for the re-judgment. Empty tuple
    # when skipped or when the source run was somehow single-judge
    # (refused at the gate, but defended here).
    jury_votes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AblationSummary:
    """``topofuzz ablate`` summary, returned by :func:`run_ablation_pass`.

    Mirrors the :class:`topofuzz.confirm.ConfirmSummary` pattern: the
    fields that land in the sidecar plus a few diagnostics for the CLI
    output. Field shapes are stable across v0.5 — extending requires a
    sidecar schema bump.

    Attributes:
        n_flipping_pairs: distinct ``(canonical, V, eval)`` triples.
        n_ablation_points: total experiments attempted (incl. skipped).
        n_ablation_points_meaningful: experiments that produced a
            re-judged score (kind == reexecute_terminal or
            substitute_terminal).
        total_cost_usd: real-LLM spend across the pass.
        partial: True when ``--max-cost-usd`` hard-stopped before the
            full sweep finished.
        by_node: ``{node_id: {flips_closed, total_meaningful,
            attribution_pct, is_terminal}}``.
        results: per-experiment rows. Lives both here (for the
            programmatic caller) and in the sidecar.
    """

    n_flipping_pairs: int
    n_ablation_points: int
    n_ablation_points_meaningful: int
    total_cost_usd: float
    partial: bool
    by_node: dict[str, dict[str, Any]]
    results: tuple[AblationResult, ...]


# ---------------------------------------------------------------------------
# Public read-side: sidecar I/O + flipping-pair enumeration
# ---------------------------------------------------------------------------


def read_ablation_sidecar(run_dir: Path | str) -> dict | None:
    """Return ``<run_dir>/ablation.json`` parsed, or ``None`` when absent.

    Mirrors :func:`topofuzz.confirm.read_confirm_sidecar`. A corrupt
    sidecar raises :class:`AblationError` rather than masquerading as
    "no pass run".
    """
    path = Path(run_dir) / ABLATION_SIDECAR_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AblationError(
            f"{ABLATION_SIDECAR_FILENAME} in {Path(run_dir).name!r} is "
            f"present but unreadable / not valid JSON: {e}"
        ) from e
    if not isinstance(data, dict):
        raise AblationError(
            f"{ABLATION_SIDECAR_FILENAME} in {Path(run_dir).name!r} is "
            "not a JSON object."
        )
    return data


def _write_ablation_sidecar(run_dir: Path, summary: AblationSummary) -> None:
    payload = {
        "schema_version": _ABLATION_SIDECAR_SCHEMA_VERSION,
        "n_flipping_pairs": summary.n_flipping_pairs,
        "n_ablation_points": summary.n_ablation_points,
        "n_ablation_points_meaningful": summary.n_ablation_points_meaningful,
        "total_cost_usd": summary.total_cost_usd,
        "partial": summary.partial,
        "by_node": summary.by_node,
        "results": [
            {
                "canonical_variant_id": r.canonical_variant_id,
                "flipping_variant_id": r.flipping_variant_id,
                "eval_id": r.eval_id,
                "ablated_node_id": r.ablated_node_id,
                "kind": r.kind,
                "original_v_score": r.original_v_score,
                "canonical_score": r.canonical_score,
                "new_v_score": r.new_v_score,
                "flip_closed": r.flip_closed,
                "cost_usd": r.cost_usd,
                "new_terminal_response_text": r.new_terminal_response_text,
                "jury_votes": list(r.jury_votes),
            }
            for r in summary.results
        ],
    }
    _atomic_write_text(
        run_dir / ABLATION_SIDECAR_FILENAME,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def _atomic_write_text(target: Path, text: str) -> None:
    """Atomically replace ``target`` with ``text`` via temp + ``os.replace``.

    The naïve ``Path.write_text`` opens, truncates to zero, writes, closes
    — a ``SIGKILL`` / OOM / OS reboot anywhere in the middle destroys the
    PRIOR good sidecar with no recovery (Pass-1 RT BLOCKER B1). Two
    concurrent ``topofuzz ablate`` invocations on the same run dir would
    likewise interleave their writes into a syntactically-valid but
    shape-wrong JSON.

    ``os.replace`` is atomic on every OS we support (Windows included,
    unlike ``rename``): the reader sees either the prior complete file
    or the new complete one, never a torn write. Mirrors
    :func:`topofuzz.inspect._atomic_write_text` (kept inline rather than
    factoring to a shared module per minimalist scope — when a third
    sidecar writer lands the right move is to lift both).
    """
    target = Path(target)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        dir=str(target.parent),
        suffix="",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        try:
            os.replace(tmp_path, target)
        except PermissionError as e:
            raise AblationError(
                f"Could not write {target.name}: {e}. On Windows this "
                "usually means another process (antivirus, file indexer) "
                "has the file open. Retry, or pass a different run_dir."
            ) from e
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class FlippingTriple:
    """One ``(canonical_variant, flipping_variant, eval)`` candidate."""

    canonical_variant_id: str
    flipping_variant_id: str
    eval_id: str


def find_all_flipping_triples(
    run_dir: Path | str,
    *,
    threshold: float = _VERDICT_THRESHOLD,
) -> list[FlippingTriple]:
    """Enumerate every flipping triple in a source run.

    Reuses :func:`topofuzz.repro.find_all_flip_candidates` (the same
    enumerator the confirm pass uses) and pairs each candidate with
    ``manifest.variant_canonical_hashes[0]`` as the canonical. Order is
    deterministic: sorted by ``(eval_id, flipping_variant_id)``.

    Returns an empty list when the run has no flips (non-fragile run
    — legitimate; the caller decides whether that's a no-op or an
    error).
    """
    from topofuzz.repro import ReproError, find_all_flip_candidates

    run_dir = Path(run_dir)
    try:
        candidates = find_all_flip_candidates(run_dir, threshold=threshold)
    except ReproError as e:
        raise AblationError(str(e)) from e
    manifest = _load_manifest(run_dir)
    hashes = manifest.get("variant_canonical_hashes") or []
    if not hashes:
        raise AblationError(
            f"manifest.variant_canonical_hashes is empty in run "
            f"{run_dir.name} — cannot identify the canonical variant."
        )
    canonical_id = hashes[0]
    return [
        FlippingTriple(
            canonical_variant_id=canonical_id,
            flipping_variant_id=vid,
            eval_id=eid,
        )
        for eid, vid in candidates
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_manifest(run_dir: Path) -> dict:
    try:
        return json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as e:
        raise AblationError(
            f"manifest.json missing in run_dir {run_dir.name!r}"
        ) from e
    except json.JSONDecodeError as e:
        raise AblationError(
            f"manifest.json in {run_dir.name!r} is not valid JSON: {e}"
        ) from e


def _load_cell_outputs(
    run_dir: Path, variant_id: str, eval_id: str
) -> list[dict[str, Any]] | None:
    """Read ``cells.node_outputs_json`` for one (variant, eval) pair.

    Returns the parsed list of NodeOutput-shaped dicts, or ``None`` if
    no cell row exists (e.g. cancelled before this variant ran). Raises
    :class:`AblationError` on malformed JSON.
    """
    db_path = run_dir / "run.sqlite"
    if not db_path.exists():
        raise AblationError(
            f"run.sqlite not found under run_dir={run_dir!r}"
        )
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT node_outputs_json FROM cells "
            "WHERE variant_id=? AND eval_id=?",
            (variant_id, eval_id),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    try:
        outputs = json.loads(row[0])
    except (TypeError, json.JSONDecodeError) as e:
        raise AblationError(
            f"node_outputs_json malformed for variant={variant_id[:8]} "
            f"eval={eval_id}: {e}"
        ) from e
    if not isinstance(outputs, list):
        raise AblationError(
            f"node_outputs_json for variant={variant_id[:8]} eval="
            f"{eval_id} is not a JSON array."
        )
    for i, item in enumerate(outputs):
        if not isinstance(item, dict):
            raise AblationError(
                f"node_outputs_json[{i}] for variant={variant_id[:8]} "
                f"eval={eval_id} is {type(item).__name__}, expected dict."
            )
    return outputs


def _load_judge_score(
    run_dir: Path, variant_id: str, eval_id: str
) -> float | None:
    """Read the cell's judge score for ablation's flip-closed comparison.

    Priority is ``score_value`` (continuous) FIRST, then ``passed``
    (binary) as fallback — RT Pass-1 BLOCKER B2 fix. The prior
    ``passed``-first ordering meant a binary-judge cell with
    ``passed=0, score_value=0.72`` returned ``0.0``, silently lying
    about the continuous score in :attr:`AblationResult.canonical_score`
    (which the persisted sidecar exposes to compliance readers). The F3
    gate ensures source-side ``passed=NULL`` for jury runs in v0.5, so
    in practice this matters only for forensic readers of mixed-schema
    or future-runner cells — but the field-name contract is real and
    the fix is cheap.

    Returns ``None`` when the cell has no judge_results row, when both
    ``score_value`` and ``passed`` are NULL, or when the judge failed.
    """
    db_path = run_dir / "run.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT score_value, passed FROM judge_results "
            "WHERE variant_id=? AND eval_id=?",
            (variant_id, eval_id),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    score_value, passed = row
    if score_value is not None:
        return float(score_value)
    if passed is not None:
        return float(passed)
    return None


# Persisted-response-text cap. Same rationale as
# ``fuzzeval.judge._RATIONALE_MAX_CHARS`` and the inspector's
# ``_MAX_RESPONSE_BYTES`` — bound the sidecar against pathological
# vendor responses while keeping enough context for forensic reading.
_RESPONSE_TEXT_CAP = 8192


def _truncate_response(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text.encode("utf-8")) <= _RESPONSE_TEXT_CAP:
        return text
    # Codepoint-aware truncation so the sidecar is always valid UTF-8.
    encoded = text.encode("utf-8")
    cut = _RESPONSE_TEXT_CAP
    while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
        cut -= 1
    return encoded[:cut].decode("utf-8", errors="ignore") + "…[truncated]"


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_ablation_pass(
    run_dir: Path | str,
    *,
    max_cost_usd: float | None = None,
    llm_callable: Any = None,
    fixture_override: str | None = None,
    dry_run: bool = False,
) -> AblationSummary:
    """Run the counterfactual ablation pass on a source run.

    See the module docstring for the locked design contract. Persists
    ``<run_dir>/ablation.json`` as a sidecar mirroring the confirm-pass
    sidecar pattern.

    Args:
        run_dir: An existing ``topofuzz run`` directory whose manifest
            says ``cli_args.jury_of_3 == True``.
        max_cost_usd: Soft-warn at 80 %, hard-stop at 100 %. ``None`` is
            uncapped. On hard-stop the sidecar lands with
            ``partial: true``.
        llm_callable: Test injection point — async, services the
            terminal-node re-execution only. The jury vendors are NOT
            auto-wired with this callable (their SingleJudge wrappers
            invoke their callable synchronously, so an async callable
            explodes at ``judge.score()`` time). For controlled jury
            verdicts in tests, construct a custom judge and call
            :func:`_ablate_async` directly, or pass ``dry_run=True``
            below.
        fixture_override: Path that wins over ``cli_args.fixture`` when
            the fixture has moved on disk since the source run. Same
            escape hatch the confirm pass exposes.
        dry_run: When True, every terminal LLM call AND every judge LLM
            call short-circuits. The terminal returns ``[DRY-RUN]``
            text; each vendor's SingleJudge returns ``ContinuousScore
            (0.5)`` — and crucially JuryJudge's F3 majority-vote
            aggregator binarises 0.5 ``>=`` 0.5 to approve, so all
            three vendors approve, ``approve_count * 2 > total`` →
            jury verdict is ``ContinuousScore(value=1.0)``, NOT 0.5
            (Pass-1 Fuzzer BLOCKER B3 fix — the prior docstring
            claimed 0.5 here, which the F3 aggregator contract
            falsifies). Therefore under ``--dry-run`` every
            ``new_v_score`` equals 1.0 (APPROVE) and ``flip_closed`` is
            purely a function of canonical's binarised verdict.
            *Attribution percentages under dry-run are not meaningful
            and the CLI prints a banner saying so.*

    Raises:
        AblationError: manifest / run.sqlite missing or malformed; the
            source run was not jury-of-3; the fixture can't be loaded;
            the manifest's canonical hash isn't in the regenerated
            variant set.
    """
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise AblationError("manifest.run_id missing or not a string.")
    cli_args = manifest.get("cli_args") or {}
    if not isinstance(cli_args, dict):
        raise AblationError("manifest.cli_args missing or wrong shape.")
    if not cli_args.get("jury_of_3"):
        raise AblationError(
            "ablation requires a jury-of-3 source run "
            "(manifest.cli_args.jury_of_3 != True). The leave-one-out "
            "re-judgments inherit the source run's judge-noise floor; "
            "without jury voting that floor is ~26 % per-judgment, which "
            "drowns the attribution signal. Re-run the source with "
            "`--jury-of-3` first."
        )

    # Pass-1 Fuzzer MAJOR M4 fix: if the source run used a custom
    # ``prompt_builder``, the terminal re-execution under ablation would
    # use the runner's ``_default_prompt`` and silently produce
    # incomparable output. v0.5 CLI exposes no such knob, but a
    # forward-compat manifest from a custom orchestrator caller could
    # carry one. Refuse rather than measure something different than
    # the source measured.
    _PROMPT_BUILDER_KEYS = (
        "prompt_builder", "prompt_builder_module", "prompt_builder_path"
    )
    custom_builder = [k for k in _PROMPT_BUILDER_KEYS if cli_args.get(k)]
    if custom_builder:
        raise AblationError(
            "ablation refuses to run on a source manifest that carries a "
            "custom prompt_builder field "
            f"({', '.join(custom_builder)}): F4 re-executes the terminal "
            "node with the runner's default prompt builder, which would "
            "produce an incomparable prompt and contaminate the "
            "attribution measurement. v0.5.1 BACKLOG: plumb the builder "
            "through the manifest the same way jury_models_resolved is."
        )

    # Pass-1 Fuzzer MAJOR M3 fix: if the source run was constructed with
    # custom jury kwargs (temperature / max_tokens / seed / per-vendor
    # llm_callable), the rebuilt JuryJudge here would run on the
    # BLUEPRINT defaults instead, silently shifting the noise substrate
    # F4 measures against. v0.5 CLI exposes none of these knobs, but a
    # future-version manifest could carry them under the conventional
    # name. Refuse rather than silently contaminate.
    _RECONSTRUCT_INCOMPATIBLE_KEYS = (
        "jury_temperature", "jury_max_tokens", "jury_seed",
        "jury_kwargs", "judge_kwargs",
    )
    incompatible = [k for k in _RECONSTRUCT_INCOMPATIBLE_KEYS if k in cli_args]
    if incompatible:
        raise AblationError(
            "ablation refuses to run on a source manifest that carries "
            f"unrecognised jury-construction fields ({', '.join(incompatible)}): "
            "the rebuilt jury would not match the source's. v0.5.1 BACKLOG: "
            "plumb these knobs through _build_jury before lifting the gate."
        )

    triples = find_all_flipping_triples(run_dir)
    if not triples:
        # No flips → nothing to ablate. Write a sidecar with empty
        # results so a later `topofuzz ablate` re-run on the same dir
        # is idempotent and the inspector knows an ablation pass has
        # actually been run (vs "no sidecar = not yet attempted").
        empty = AblationSummary(
            n_flipping_pairs=0,
            n_ablation_points=0,
            n_ablation_points_meaningful=0,
            total_cost_usd=0.0,
            partial=False,
            by_node={},
            results=(),
        )
        _write_ablation_sidecar(run_dir, empty)
        return empty

    # Reconstruct the variant Topology objects (canonical + each
    # flipping variant) by canonical_hash. confirm._reconstruct_variants
    # is the right tool; reuse to keep the regeneration path single-
    # sourced.
    from topofuzz.confirm import _reconstruct_variants

    by_hash = _reconstruct_variants(manifest, fixture_override=fixture_override)
    needed = {t.canonical_variant_id for t in triples} | {
        t.flipping_variant_id for t in triples
    }
    missing = sorted(h for h in needed if h not in by_hash)
    if missing:
        raise AblationError(
            "ablation: regenerated variant set is missing "
            f"{len(missing)} variant(s) referenced by the flipping "
            f"triples: {', '.join(h[:12] + '…' for h in missing)}. "
            "Fixture or rng_seed has drifted from the recorded run."
        )

    # Eval rows.
    from topofuzz.confirm import _load_eval_rows_by_id

    eval_rows_by_id = _load_eval_rows_by_id(cli_args, run_dir)

    # Construct the JuryJudge mirroring the source run's vendor map.
    judge = _build_jury(cli_args, llm_callable=llm_callable, dry_run=dry_run)

    # Drive the pass in asyncio.run so we can await _call_llm directly.
    summary = asyncio.run(
        _ablate_async(
            run_dir=run_dir,
            run_id=run_id,
            triples=triples,
            by_hash=by_hash,
            eval_rows_by_id=eval_rows_by_id,
            judge=judge,
            max_cost_usd=max_cost_usd,
            llm_callable=llm_callable,
            dry_run=dry_run,
        )
    )

    _write_ablation_sidecar(run_dir, summary)
    return summary


# ---------------------------------------------------------------------------
# Internal: jury reconstruction
# ---------------------------------------------------------------------------


def _build_jury(
    cli_args: dict,
    *,
    llm_callable: Any,
    dry_run: bool,
) -> Any:
    """Re-construct the JuryJudge from manifest cli_args.

    Mirrors :func:`topofuzz.cli.run` 's construction path: prefers the
    post-parse ``jury_models_resolved`` dict, falls back to re-parsing
    the verbatim ``jury_models`` string. The rubric_path comes from
    ``cli_args.rubric``; missing path falls through to the packaged
    default (matching cli.py's behavior).

    When ``llm_callable`` is supplied, it's applied to EVERY jury vendor.
    Tests inject a callable that returns a parseable judge response for
    deterministic re-judgments; production passes ``None`` so each
    vendor uses its real LiteLLM client.
    """
    from fuzzeval.judge import JuryJudge

    rubric_path_str = cli_args.get("rubric")
    if rubric_path_str:
        rubric_path = Path(rubric_path_str)
    else:
        from topofuzz.cli import _packaged_default_rubric_path

        rubric_path = _packaged_default_rubric_path()

    kwargs: dict[str, Any] = {
        "rubric_path": rubric_path,
        "dry_run": dry_run,
    }

    resolved = cli_args.get("jury_models_resolved")
    verbatim = cli_args.get("jury_models")
    vendor_models: dict[str, str] | None = None
    allow_dup = False
    if isinstance(resolved, dict) and resolved:
        # Trust the post-parse map; the original CLI run already
        # validated entry shapes, so we don't re-run _parse_jury_models
        # here. Recompute allow_duplicate_models from the values so
        # the JuryJudge invariant still holds without leaning on a
        # field that's not in the manifest.
        vendor_models = {str(k): str(v) for k, v in resolved.items()}
        allow_dup = len({m for m in vendor_models.values()}) != len(
            vendor_models
        )
        # Pass-1 RT MAJOR M4 fix: when BOTH fields are present, verify
        # they agree. Manifest tamper or older-version replay can leave
        # ``jury_models`` carrying one trio while ``jury_models_resolved``
        # carries another — silently preferring the dict would re-judge
        # under a DIFFERENT jury than the source ran. Refuse the pass
        # rather than produce contaminated attribution numbers.
        if isinstance(verbatim, str) and verbatim.strip():
            from topofuzz.cli import _parse_jury_models

            try:
                reparsed_models, reparsed_allow_dup = _parse_jury_models(
                    verbatim
                )
            except Exception as e:  # noqa: BLE001
                raise AblationError(
                    "manifest.cli_args.jury_models is unparseable while "
                    "jury_models_resolved is present; the two fields "
                    f"cannot be cross-checked: {e}"
                ) from e
            if (
                dict(reparsed_models) != dict(vendor_models)
                or reparsed_allow_dup != allow_dup
            ):
                raise AblationError(
                    "manifest.cli_args.jury_models and "
                    "jury_models_resolved disagree about the jury "
                    "vendor map. The manifest has been tampered with or "
                    "written by an incompatible version. Resolve "
                    "manually before re-running ablate."
                )
    else:
        # Older manifest that pre-dates jury_models_resolved (or one
        # written without a custom trio). Fall back to the verbatim
        # string when present; otherwise let JuryJudge default to the
        # BLUEPRINT trio.
        if isinstance(verbatim, str) and verbatim.strip():
            from topofuzz.cli import _parse_jury_models

            vendor_models, allow_dup = _parse_jury_models(verbatim)

    if vendor_models is not None:
        kwargs["vendor_models"] = vendor_models
        kwargs["allow_duplicate_models"] = allow_dup

    # NOTE: ``llm_callable`` (async, for the runner's ``_call_llm``) is
    # intentionally NOT plumbed into JuryJudge's per-vendor
    # ``llm_callables`` slot — JuryJudge's SingleJudge wrappers call the
    # callable synchronously, so an async callable would explode at
    # judge.score() time. Tests that need controlled jury verdicts
    # construct a custom judge and invoke :func:`_ablate_async` directly,
    # OR pass ``dry_run=True`` to short-circuit both the terminal
    # re-exec AND the jury into deterministic stubs.

    try:
        return JuryJudge(**kwargs)
    except (FileNotFoundError, ValueError) as e:
        raise AblationError(
            f"failed to construct JuryJudge from manifest cli_args: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Internal: async driver
# ---------------------------------------------------------------------------


async def _ablate_async(
    *,
    run_dir: Path,
    run_id: str,
    triples: list[FlippingTriple],
    by_hash: dict[str, Any],
    eval_rows_by_id: dict[str, dict],
    judge: Any,
    max_cost_usd: float | None,
    llm_callable: Any,
    dry_run: bool,
) -> AblationSummary:
    from fuzzeval.costest import cost_for_call

    from topofuzz.runner import (
        _call_llm,
        _default_prompt,
        _extract_text_and_served,
        _extract_usage,
    )

    results: list[AblationResult] = []
    total_cost = 0.0
    partial = False
    soft_warn_emitted = False
    consecutive_failures = 0

    def _cap_hit() -> bool:
        return max_cost_usd is not None and total_cost >= max_cost_usd

    def _maybe_soft_warn() -> None:
        nonlocal soft_warn_emitted
        if (
            max_cost_usd is not None
            and not soft_warn_emitted
            and total_cost >= 0.80 * max_cost_usd
        ):
            _log.warning(
                "ablate: cost soft-warn — $%.4f / $%.4f cap (80%% threshold).",
                total_cost,
                max_cost_usd,
            )
            soft_warn_emitted = True

    for triple in triples:
        if _cap_hit():
            partial = True
            break
        canonical_id = triple.canonical_variant_id
        v_id = triple.flipping_variant_id
        eval_id = triple.eval_id

        canonical_outputs = _load_cell_outputs(
            run_dir, canonical_id, eval_id
        )
        v_outputs = _load_cell_outputs(run_dir, v_id, eval_id)
        eval_row = eval_rows_by_id.get(eval_id)
        canonical_score = _load_judge_score(run_dir, canonical_id, eval_id)
        v_score = _load_judge_score(run_dir, v_id, eval_id)

        if canonical_outputs is None or v_outputs is None:
            _log.warning(
                "ablate: cell missing for triple "
                "(canonical=%s, v=%s, eval=%s); skipping.",
                canonical_id[:8], v_id[:8], eval_id,
            )
            continue
        if eval_row is None:
            _log.warning(
                "ablate: eval_id %r not in eval file; skipping triple "
                "(canonical=%s, v=%s).",
                eval_id, canonical_id[:8], v_id[:8],
            )
            continue
        if v_score is None or canonical_score is None:
            _log.warning(
                "ablate: triple (canonical=%s, v=%s, eval=%s) lacks "
                "scored verdicts on both sides; skipping.",
                canonical_id[:8], v_id[:8], eval_id,
            )
            continue

        v_topology = by_hash[v_id]
        terminal_ids = tuple(sorted(v_topology.terminal))
        if not terminal_ids:
            _log.warning(
                "ablate: variant %s has empty Topology.terminal; skipping.",
                v_id[:8],
            )
            continue
        # Use the same lex-first reduction the runner uses for the
        # judge target (runner._run_judge sorts terminal and takes the
        # first present output).
        terminal_id = terminal_ids[0]
        # Find terminal position in V's execution order.
        v_node_ids = [n.id for n in v_topology.nodes]
        try:
            terminal_pos = v_node_ids.index(terminal_id)
        except ValueError:
            _log.warning(
                "ablate: terminal %r not in v_topology.nodes for variant "
                "%s; skipping.", terminal_id, v_id[:8],
            )
            continue
        terminal_node = v_topology.nodes[terminal_pos]
        # The prior outputs the terminal SAW in the original V run were
        # everything that ran before it in V's execution order.
        v_prior_dicts = list(v_outputs[:terminal_pos])

        # Pass-1 RT MAJOR M3 fix: defend against duplicate node_id in
        # the persisted node_outputs_json (illegal under the IR, but a
        # future loop / retry / langgraph-import edge case could
        # surface them). The dict comprehensions below silently keep
        # the LAST occurrence per node_id; if a duplicate exists,
        # ablation would attribute against the wrong snapshot. Refuse
        # to ablate those node_ids cleanly.
        v_duplicates = _duplicate_node_ids(v_outputs)
        canonical_duplicates = _duplicate_node_ids(canonical_outputs)
        if v_duplicates:
            _log.warning(
                "ablate: variant %s eval %s has duplicate node_outputs "
                "entries for %s; ablating those nodes would silently "
                "use the last snapshot only. Recording as skip.",
                v_id[:8], eval_id, sorted(v_duplicates),
            )
        if canonical_duplicates:
            _log.warning(
                "ablate: canonical %s eval %s has duplicate node_outputs "
                "entries for %s; ablating those nodes would silently "
                "use the last snapshot only. Recording as skip.",
                canonical_id[:8], eval_id, sorted(canonical_duplicates),
            )
        ambiguous_node_ids = v_duplicates | canonical_duplicates

        canonical_by_id = {
            o.get("node_id"): o for o in canonical_outputs if isinstance(o, dict)
        }
        v_by_id = {
            o.get("node_id"): o for o in v_outputs if isinstance(o, dict)
        }

        # Ablate each node in the topology, deterministic order.
        # Use V's topology.nodes for the iteration so a buyer reading
        # the panel sees the agents in V's execution order — the
        # "story" of the flip.
        for node in v_topology.nodes:
            if _cap_hit():
                partial = True
                break

            node_id = node.id
            is_terminal = node_id == terminal_id

            # RT M3: refuse ablation on nodes with duplicate snapshots
            # in EITHER side's persisted outputs. Routes through the
            # existing skip-no-canonical-output kind so the buyer-
            # visible CLI / panel surface treats it consistently with
            # other "no measurement" cases (paired with the WARNING log
            # emitted above so an operator investigating the panel
            # finds a trail).
            if node_id in ambiguous_node_ids:
                results.append(
                    AblationResult(
                        canonical_variant_id=canonical_id,
                        flipping_variant_id=v_id,
                        eval_id=eval_id,
                        ablated_node_id=node_id,
                        kind=_KIND_SKIP_NO_CANONICAL_OUTPUT,
                        original_v_score=v_score,
                        canonical_score=canonical_score,
                        new_v_score=None,
                        flip_closed=None,
                        cost_usd=0.0,
                        new_terminal_response_text=None,
                        jury_votes=(),
                    )
                )
                continue

            canonical_node_dict = canonical_by_id.get(node_id)
            v_node_dict = v_by_id.get(node_id)

            if canonical_node_dict is None:
                results.append(
                    AblationResult(
                        canonical_variant_id=canonical_id,
                        flipping_variant_id=v_id,
                        eval_id=eval_id,
                        ablated_node_id=node_id,
                        kind=_KIND_SKIP_NO_CANONICAL_OUTPUT,
                        original_v_score=v_score,
                        canonical_score=canonical_score,
                        new_v_score=None,
                        flip_closed=None,
                        cost_usd=0.0,
                        new_terminal_response_text=None,
                        jury_votes=(),
                    )
                )
                continue
            if v_node_dict is None:
                results.append(
                    AblationResult(
                        canonical_variant_id=canonical_id,
                        flipping_variant_id=v_id,
                        eval_id=eval_id,
                        ablated_node_id=node_id,
                        kind=_KIND_SKIP_NO_V_OUTPUT,
                        original_v_score=v_score,
                        canonical_score=canonical_score,
                        new_v_score=None,
                        flip_closed=None,
                        cost_usd=0.0,
                        new_terminal_response_text=None,
                        jury_votes=(),
                    )
                )
                continue

            # Decide the experiment shape.
            if is_terminal:
                kind = _KIND_SUBSTITUTE_TERMINAL
                new_terminal_text = str(
                    canonical_node_dict.get("response_text", "")
                )
                exec_cost = 0.0
            else:
                # Find this node in V's prior slice. If it's not there,
                # the node ran AFTER the terminal in V's order: ablating
                # it can't change what the terminal saw. Record and skip.
                in_prior = any(
                    p.get("node_id") == node_id for p in v_prior_dicts
                )
                if not in_prior:
                    results.append(
                        AblationResult(
                            canonical_variant_id=canonical_id,
                            flipping_variant_id=v_id,
                            eval_id=eval_id,
                            ablated_node_id=node_id,
                            kind=_KIND_SKIP_DOWNSTREAM,
                            original_v_score=v_score,
                            canonical_score=canonical_score,
                            new_v_score=None,
                            flip_closed=None,
                            cost_usd=0.0,
                            new_terminal_response_text=None,
                            jury_votes=(),
                        )
                    )
                    continue
                kind = _KIND_REEXECUTE_TERMINAL
                # Build the hybrid prior: V's prior outputs with the
                # ablated node's response_text swapped for canonical's.
                hybrid_prior_dicts: list[dict[str, Any]] = []
                for p in v_prior_dicts:
                    if p.get("node_id") == node_id:
                        # Use canonical's response_text but keep V's
                        # role/model/etc. metadata so the prompt
                        # builder's role-formatted line still reads as
                        # this agent's role. response_text is what the
                        # terminal's prompt template actually consumes.
                        swapped = dict(p)
                        swapped["response_text"] = canonical_node_dict.get(
                            "response_text", ""
                        )
                        hybrid_prior_dicts.append(swapped)
                    else:
                        hybrid_prior_dicts.append(p)
                # Build prior tuple in the shape _default_prompt expects.
                hybrid_prior = _dicts_to_node_outputs(hybrid_prior_dicts)
                prompt = _default_prompt(
                    terminal_node, eval_row, tuple(hybrid_prior)
                )
                # Re-execute the terminal.
                try:
                    if dry_run:
                        new_terminal_text = (
                            "[DRY-RUN] terminal re-executed under ablation"
                        )
                        exec_cost = 0.0
                    else:
                        started = time.monotonic()
                        response = await _call_llm(
                            terminal_node, prompt, llm_callable
                        )
                        _latency_ms = int(
                            (time.monotonic() - started) * 1000
                        )
                        new_terminal_text, _served = _extract_text_and_served(
                            response
                        )
                        inp, out = _extract_usage(response)
                        exec_cost = (
                            cost_for_call(terminal_node.llm_config.model, inp, out)
                            or 0.0
                        )
                except (
                    KeyboardInterrupt,
                    SystemExit,
                    asyncio.CancelledError,
                ):
                    raise
                # Pass-1 RT MAJOR M2 fix: keep the broad catch (real
                # litellm vendor errors are custom subclasses of
                # Exception; shape-drift can surface as TypeError /
                # KeyError; either should demote one cell, not crash
                # the pass) BUT add a consecutive-failure counter that
                # aborts the whole pass after N back-to-back failures.
                # A real vendor outage typically produces a sporadic
                # mix; a programmer bug (signature drift) fails EVERY
                # cell. The counter distinguishes the two and surfaces
                # the bug loudly instead of silently emitting N
                # kind=failed rows the buyer can't tell apart.
                except Exception as e:  # noqa: BLE001 — boundary: vendor SDK
                    _log.warning(
                        "ablate: terminal re-execution failed for "
                        "(canonical=%s, v=%s, eval=%s, node=%s): %s: %s",
                        canonical_id[:8], v_id[:8], eval_id, node_id,
                        type(e).__name__, e,
                    )
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        raise AblationError(
                            f"ablate: {_MAX_CONSECUTIVE_FAILURES} consecutive "
                            "terminal re-execution failures — aborting the "
                            "pass. This usually indicates a programmer bug "
                            "(signature drift, response-shape change), not a "
                            "vendor hiccup. Last error: "
                            f"{type(e).__name__}: {e}"
                        ) from e
                    results.append(
                        AblationResult(
                            canonical_variant_id=canonical_id,
                            flipping_variant_id=v_id,
                            eval_id=eval_id,
                            ablated_node_id=node_id,
                            kind=_KIND_FAILED,
                            original_v_score=v_score,
                            canonical_score=canonical_score,
                            new_v_score=None,
                            flip_closed=None,
                            cost_usd=0.0,
                            new_terminal_response_text=None,
                            jury_votes=(),
                        )
                    )
                    continue

            # Re-judge under the jury.
            payload = {
                "variant_id": v_id,
                "eval_row": eval_row,
                "agent_output": new_terminal_text,
            }
            try:
                judge_result = await asyncio.to_thread(judge.score, payload)
            except (
                KeyboardInterrupt,
                SystemExit,
                asyncio.CancelledError,
            ):
                raise
            except Exception as e:  # noqa: BLE001 — boundary: judge SDK
                _log.warning(
                    "ablate: jury re-judge failed for (canonical=%s, "
                    "v=%s, eval=%s, node=%s): %s: %s",
                    canonical_id[:8], v_id[:8], eval_id, node_id,
                    type(e).__name__, e,
                )
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise AblationError(
                        f"ablate: {_MAX_CONSECUTIVE_FAILURES} consecutive "
                        "failures across terminal re-exec + jury re-judge "
                        "— aborting. This usually indicates a programmer "
                        f"bug, not a vendor hiccup. Last error: "
                        f"{type(e).__name__}: {e}"
                    ) from e
                # Cost from the terminal re-exec already counted; add a
                # best-effort jury cost estimate so the cap accounting
                # is conservative even on failure (Pass-1 RT Mi6 fix).
                failed_jury_cost = _estimate_jury_cost(judge)
                results.append(
                    AblationResult(
                        canonical_variant_id=canonical_id,
                        flipping_variant_id=v_id,
                        eval_id=eval_id,
                        ablated_node_id=node_id,
                        kind=_KIND_FAILED,
                        original_v_score=v_score,
                        canonical_score=canonical_score,
                        new_v_score=None,
                        flip_closed=None,
                        cost_usd=exec_cost + failed_jury_cost,
                        new_terminal_response_text=_truncate_response(
                            new_terminal_text
                        ),
                        jury_votes=(),
                    )
                )
                total_cost += exec_cost + failed_jury_cost
                _maybe_soft_warn()
                continue
            # Reset the consecutive-failure counter on every successful
            # re-judge — a sporadic vendor blip should not accumulate.
            consecutive_failures = 0

            new_v_score = float(judge_result.value)
            # Flip closed iff V's new verdict matches canonical's
            # binarised verdict. Use the same threshold as the headline
            # pill so the panel and the pill agree.
            canonical_binary = canonical_score >= _VERDICT_THRESHOLD
            new_binary = new_v_score >= _VERDICT_THRESHOLD
            flip_closed = (canonical_binary == new_binary)
            jury_votes_tup = tuple(
                {
                    "vendor": str(v),
                    "score": float(s),
                    "rationale": str(r),
                }
                for v, s, r in getattr(judge_result, "votes", ())
            )
            # Jury cost: we don't have per-vendor token counts back from
            # JuryJudge in v0.5 — the SingleJudge inside doesn't surface
            # them. Use the cost estimator's static rate × an assumed
            # 2 KB-per-call envelope as a rough surcharge so the cap
            # accounting isn't blind to jury spend. The cli's preflight
            # uses the same shape via cost_for_call. This is an estimate;
            # the manifest's authoritative cost lives in the runner's
            # cells.cell_cost_usd column (which ablation doesn't touch).
            jury_cost = _estimate_jury_cost(judge)

            point_cost = exec_cost + jury_cost
            total_cost += point_cost
            _maybe_soft_warn()

            results.append(
                AblationResult(
                    canonical_variant_id=canonical_id,
                    flipping_variant_id=v_id,
                    eval_id=eval_id,
                    ablated_node_id=node_id,
                    kind=kind,
                    original_v_score=v_score,
                    canonical_score=canonical_score,
                    new_v_score=new_v_score,
                    flip_closed=flip_closed,
                    cost_usd=point_cost,
                    new_terminal_response_text=_truncate_response(
                        new_terminal_text
                    ),
                    jury_votes=jury_votes_tup,
                )
            )

        if partial:
            break

    # Aggregate by node_id over MEANINGFUL ablation points (kind in
    # {reexecute_terminal, substitute_terminal}). The "terminal swap
    # always closes" sanity check shows up as is_terminal=True with
    # 100 % attribution — surfaced (not hidden) so the buyer sees the
    # baseline against which upstream attribution percentages are
    # measured.
    by_node: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.kind not in (_KIND_REEXECUTE_TERMINAL, _KIND_SUBSTITUTE_TERMINAL):
            continue
        slot = by_node.setdefault(
            r.ablated_node_id,
            {
                "flips_closed": 0,
                "total_meaningful": 0,
                "attribution_pct": 0.0,
                "is_terminal": r.kind == _KIND_SUBSTITUTE_TERMINAL,
            },
        )
        slot["total_meaningful"] += 1
        if r.flip_closed:
            slot["flips_closed"] += 1
    for node_id, slot in by_node.items():
        if slot["total_meaningful"] > 0:
            slot["attribution_pct"] = round(
                100.0 * slot["flips_closed"] / slot["total_meaningful"], 2
            )

    n_meaningful = sum(
        1
        for r in results
        if r.kind in (_KIND_REEXECUTE_TERMINAL, _KIND_SUBSTITUTE_TERMINAL)
    )

    return AblationSummary(
        n_flipping_pairs=len(triples),
        n_ablation_points=len(results),
        n_ablation_points_meaningful=n_meaningful,
        total_cost_usd=total_cost,
        partial=partial,
        by_node=by_node,
        results=tuple(results),
    )


# ---------------------------------------------------------------------------
# Internal: glue
# ---------------------------------------------------------------------------


def _dicts_to_node_outputs(dicts: list[dict[str, Any]]) -> list[Any]:
    """Hydrate a list of NodeOutput-shaped dicts into NodeOutput dataclasses.

    The default :func:`topofuzz.runner._default_prompt` reads ``.role``
    and ``.response_text`` off each entry; we lift the saved-back-out
    JSON into the original dataclass so the prompt builder sees the
    same shape it does during the live run.
    """
    from topofuzz.runner import NodeOutput

    out: list[NodeOutput] = []
    for d in dicts:
        out.append(
            NodeOutput(
                node_id=str(d.get("node_id", "")),
                role=str(d.get("role", "")),
                response_text=str(d.get("response_text", "")),
                model_served_version=str(d.get("model_served_version", "")),
                cache_hit=bool(d.get("cache_hit", False)),
                latency_ms=int(d.get("latency_ms", 0) or 0),
                input_tokens=d.get("input_tokens"),
                output_tokens=d.get("output_tokens"),
                cost_usd=d.get("cost_usd"),
            )
        )
    return out


def _duplicate_node_ids(outputs: list[dict[str, Any]]) -> set[str]:
    """Return the set of ``node_id`` values that appear more than once.

    Defensive against malformed ``node_outputs_json`` blobs from custom
    runners / langgraph-import edge cases (Pass-1 RT MAJOR M3). The IR
    contract is "node_id unique per topology", but ``run.sqlite`` is
    plain JSON and can carry duplicates that the dict-comprehension at
    the call site would silently keep only the LAST snapshot of.
    """
    seen: set[str] = set()
    dupes: set[str] = set()
    for o in outputs:
        if not isinstance(o, dict):
            continue
        nid = o.get("node_id")
        if not isinstance(nid, str):
            continue
        if nid in seen:
            dupes.add(nid)
        else:
            seen.add(nid)
    return dupes


# ENVELOPE_TOKENS is the per-vendor static estimate envelope for the
# cost-cap accounting only (NOT for billing). Promoted to a module
# constant — same rationale as ``_VERDICT_THRESHOLD`` upstream.
_JURY_ESTIMATE_ENVELOPE_TOKENS = 2_000


def _estimate_jury_cost(judge: Any) -> float:
    """Best-effort static-rate estimate for one jury re-judgment.

    JuryJudge doesn't surface per-vendor token counts to the caller, so
    we can't bill the actual draw. We approximate as
    ``sum(rate_per_1k × _JURY_ESTIMATE_ENVELOPE_TOKENS) over jury
    vendors`` — same shape as :func:`topofuzz.cli._preflight_cost`'s
    judge-call estimate.

    The important property for the cap is that this is a NON-ZERO
    floor. Pass-1 RT MAJOR M1 / Fuzzer MAJOR M2 fix: a bare ``except
    Exception → return 0.0`` defeated that property — a future
    JuryJudge refactor (private `_judges` → `_judge_map`, or `vendors`
    promoted to a method that errors on a half-constructed instance)
    would silently return $0 per jury call, letting an unbounded pass
    sneak past ``--max-cost-usd``. The fixed version:

    1. Narrows the catch to ``(AttributeError, ImportError,
       TypeError, ValueError, KeyError)`` — the shapes that reflection
       failures actually produce. Other exception classes propagate so
       a real bug surfaces.
    2. WARNS rather than silently swallows.
    3. Falls back to :data:`_JURY_COST_FALLBACK_FLOOR_USD` (~$0.03)
       instead of 0.0 — order-of-magnitude correct for a typical
       jury-of-3 call.
    """
    try:
        from fuzzeval.costest import _RATE_USD_PER_1K

        vendors = getattr(judge, "vendors", ())
        cost = 0.0
        for v in vendors:
            # vendors are normalized names; the rate table is keyed by
            # the litellm model id. Walk the judge's per-vendor
            # SingleJudges to map vendor → model.
            model = _vendor_to_model(judge, v)
            rate = _RATE_USD_PER_1K.get(model, 0.005)  # safe default
            cost += rate * (_JURY_ESTIMATE_ENVELOPE_TOKENS / 1000.0)
        # If reflection succeeded but produced $0 (e.g. vendors == ()),
        # still return the floor so the cap can never silently see zero.
        return max(cost, _JURY_COST_FALLBACK_FLOOR_USD)
    except (
        AttributeError, ImportError, TypeError, ValueError, KeyError,
    ) as e:
        _log.warning(
            "ablate: jury-cost reflection failed (%s: %s); using "
            "fallback floor $%.4f. Cap accounting may under-count by "
            "a small amount.",
            type(e).__name__, e, _JURY_COST_FALLBACK_FLOOR_USD,
        )
        return _JURY_COST_FALLBACK_FLOOR_USD
    except Exception as e:  # noqa: BLE001 — accounting must never crash the pass
        # Defense-in-depth: a truly unexpected exception class (e.g. a
        # vendor SDK exception we didn't anticipate) still returns the
        # non-zero floor, so the cap accounting never silently sees $0.
        _log.warning(
            "ablate: jury-cost reflection raised unexpected %s: %s; "
            "using fallback floor $%.4f.",
            type(e).__name__, e, _JURY_COST_FALLBACK_FLOOR_USD,
        )
        return _JURY_COST_FALLBACK_FLOOR_USD


def _vendor_to_model(judge: Any, vendor: str) -> str:
    """Reflect into JuryJudge to recover the LiteLLM model id for a vendor.

    Defensive: walks the private ``_judges`` map only when present, and
    returns the first plausible string we can find. Used only for the
    soft cost estimate above — a wrong model id costs us a slightly
    off cap estimate, not correctness.
    """
    judges = getattr(judge, "_judges", None)
    if isinstance(judges, dict):
        sj = judges.get(vendor)
        if sj is not None and hasattr(sj, "model"):
            return str(sj.model)
    return ""
