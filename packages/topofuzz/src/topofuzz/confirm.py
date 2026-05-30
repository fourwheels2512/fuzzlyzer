# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""``topofuzz confirm``: noise-floor / verdict-confirmation pass.

The finding (2026-05-14): the pipeline (Gemini Flash, temp=0) is
non-deterministic — re-running the SAME variant flips the verdict ~26%
of the time. So the shipped ``verdict_flip_rate`` conflates the topology
effect with LLM noise.

The fix: re-run every screened flip candidate K times AND re-run the
original variant K times per relevant eval, persist every repeat into
the separate ``flip_confirmations`` table, and let the noise-corrected
metrics (:mod:`topofuzz.metrics.verdict_stability`,
:mod:`topofuzz.metrics.confirmed_verdict_flip`,
:mod:`topofuzz.metrics.voo_noise_floor`) majority-vote over the K repeats.

This module owns the orchestration of that pass. Heavy imports
(``litellm``, ``langgraph`` via ``load_topology``, the variant
generators) are lazy so a ``topofuzz[verify]`` install stays importable
— same discipline as :mod:`topofuzz.cli` and :mod:`topofuzz.repro`.

Design contract (locked — do NOT redesign):

* Candidates are enumerated from ``run.sqlite`` by
  :func:`topofuzz.repro.find_all_flip_candidates` (binarised verdict vs
  the original variant, manifest index 0).
* The variant set is regenerated deterministically from the manifest's
  ``cli_args`` (fixture spec, ``rng_seed``, ``variants`` count) — the
  same ``order_swap`` / ``role_swap`` / ``decoding_param_swap`` +
  ``_round_robin_select`` path ``topofuzz run`` uses — then matched by
  ``canonical_hash()`` to the candidate variant_ids and the original.
* Repeats run with the LLM cache DISABLED (``llm_cache=None``) — we WANT
  fresh draws; caching would return the same response K times and the
  noise floor would read a spurious 0.
* ``--max-cost-usd`` is honoured with the orchestrator's soft-warn /
  hard-stop pattern; on hard-stop whatever completed is persisted and the
  caller is told the results are partial.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``Topology`` is light (no langgraph dependency); hoist to module top
# so the L910 re-construction path inside ``_reconstruct_variants`` does
# not pay the per-call local-import cost on every confirm pass.
from topofuzz.ir import Topology

_log = logging.getLogger(__name__)

# Default repeat count for the confirm pass. K=3 gives a strict majority
# over an odd number of draws (a tie is impossible for a single
# candidate's own repeats; ties only happen at the reference-baseline
# level when two evals' references split) while keeping the re-run bill
# at 3× the candidate count. Overridable via ``--repeats``.
DEFAULT_REPEATS = 3

# Sidecar JSON the confirm pass writes into the run dir — same artifact
# pattern as manifest.json / report.json / attestation.json. The attest
# path reads it to decide whether the noise-floor headline metrics can
# be populated (B-RT-1 spec-owner decision: NO new SQLite table).
CONFIRM_SIDECAR_FILENAME = "confirm.json"


class ConfirmError(ValueError):
    """Raised when the confirm pass cannot run (missing manifest, etc.)."""


@dataclass(frozen=True)
class ConfirmSummary:
    """What :func:`run_confirm_pass` returns for the CLI summary line.

    ``partial`` is True when ``--max-cost-usd`` hard-stopped the pass
    before every planned repeat ran — whatever completed is still
    persisted in ``flip_confirmations``, but the noise-floor headline
    metrics MUST NOT be populated from a partial pass (the majority vote
    over a ragged repeat block is not honest — see B-RT-1).

    ``n_unmatched_candidates`` counts screened candidate variant_ids that
    could not be matched to a regenerated topology (fixture / rng_seed
    drift). They contribute zero repeats; a non-zero count means the
    confirm data is incomplete and — like ``partial`` — blocks the
    headline (audit M-RT-2).

    ``skipped_eval_ids`` lists candidate eval_ids whose row was absent
    from the run's eval file (so their repeats could not run). Surfaced
    rather than only ``_log.warning``-ed (audit N5).

    The sidecar-relevant fields (``repeats``, ``partial``,
    ``n_candidates``, ``n_candidate_variants``, ``reference_repeats_run``,
    ``candidate_repeats_run``, ``n_unmatched_candidates``,
    ``total_cost_usd``) are written verbatim to ``confirm.json``.
    """

    n_candidates: int
    n_candidate_variants: int
    repeats: int
    reference_repeats_run: int
    candidate_repeats_run: int
    total_cost_usd: float
    partial: bool
    n_unmatched_candidates: int = 0
    skipped_eval_ids: tuple[str, ...] = ()


def read_confirm_sidecar(run_dir: Path | str) -> dict | None:
    """Read ``<run_dir>/confirm.json`` if a confirm pass has been run.

    Returns the parsed sidecar dict, or ``None`` when the file is absent
    (no confirm pass has run). Raises :class:`ConfirmError` only when the
    file exists but is unreadable / not valid JSON — a corrupt sidecar is
    a real signal, not a "no confirm pass" condition.
    """
    path = Path(run_dir) / CONFIRM_SIDECAR_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfirmError(
            f"{CONFIRM_SIDECAR_FILENAME} in {Path(run_dir).name!r} is "
            f"present but unreadable / not valid JSON: {e}"
        ) from e
    if not isinstance(data, dict):
        raise ConfirmError(
            f"{CONFIRM_SIDECAR_FILENAME} in {Path(run_dir).name!r} is not "
            f"a JSON object."
        )
    return data


def _write_confirm_sidecar(run_dir: Path, summary: "ConfirmSummary") -> None:
    """Write the confirm-pass sidecar JSON into the run dir.

    Mirrors the run-dir artifact pattern (manifest.json / report.json /
    attestation.json). The attest path consumes this — if it is missing
    OR ``partial`` is true OR ``n_unmatched_candidates`` is non-zero, the
    noise-floor headline metrics stay ``null`` (B-RT-1 / M-RT-2).
    """
    payload = {
        "repeats": summary.repeats,
        "partial": summary.partial,
        "n_candidates": summary.n_candidates,
        "n_candidate_variants": summary.n_candidate_variants,
        "reference_repeats_run": summary.reference_repeats_run,
        "candidate_repeats_run": summary.candidate_repeats_run,
        "n_unmatched_candidates": summary.n_unmatched_candidates,
        "skipped_eval_ids": list(summary.skipped_eval_ids),
        "total_cost_usd": summary.total_cost_usd,
    }
    (run_dir / CONFIRM_SIDECAR_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_flip_confirmations(
    run_dir: Path | str,
    *,
    run_id: str | None = None,
    role: str | None = None,
) -> list[dict]:
    """Read ``flip_confirmations`` rows from a run's ``run.sqlite``.

    Returns a list of dicts (column name → value) ordered by
    ``(eval_id, variant_id, repeat_idx)`` for determinism. The
    noise-corrected metrics consume these; the attest path is the
    primary caller.

    Args:
        run_dir: directory containing ``run.sqlite``.
        run_id: restrict to one run_id; ``None`` reads every row.
        role: restrict to ``'reference'`` or ``'candidate'``; ``None``
            reads both.

    Returns an empty list when the table exists but has no rows (no
    ``confirm`` pass run yet). Raises :class:`ConfirmError` if
    ``run.sqlite`` is missing or lacks the ``flip_confirmations`` schema.
    """
    import sqlite3

    db_path = Path(run_dir) / "run.sqlite"
    if not db_path.exists():
        raise ConfirmError(
            f"run.sqlite not found under run_dir={run_dir!r} "
            f"(expected at {db_path!s})"
        )
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        # v7 added a nullable ``verdict_class``. Select it only when the
        # column exists so a legacy (pre-v7) run.sqlite still reads — the
        # 3-state ``verdict_stability`` compare falls back to the score
        # threshold when the key is absent.
        _flip_cols = {
            r[1]
            for r in con.execute(
                "PRAGMA table_info(flip_confirmations)"
            ).fetchall()
        }
        _vclass_sel = (
            ", verdict_class" if "verdict_class" in _flip_cols else ""
        )
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if role is not None:
            clauses.append("role = ?")
            params.append(role)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            rows = con.execute(
                "SELECT run_id, variant_id, eval_id, repeat_idx, role, "
                "node_outputs_json, judge_kind, score_value, rationale, "
                "failure_reason, error, cell_cost_usd" + _vclass_sel + " "
                "FROM flip_confirmations" + where + " "
                "ORDER BY eval_id, variant_id, repeat_idx",
                params,
            ).fetchall()
        except sqlite3.DatabaseError as e:
            raise ConfirmError(
                f"run.sqlite in {Path(run_dir).name} is missing the "
                f"flip_confirmations schema or is corrupt: {e}"
            ) from e
    finally:
        con.close()
    return [dict(r) for r in rows]


def _load_manifest(run_dir: Path) -> dict:
    try:
        return json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as e:
        raise ConfirmError(
            f"manifest.json missing in run_dir {run_dir.name!r}"
        ) from e
    except json.JSONDecodeError as e:
        raise ConfirmError(
            f"manifest.json in {run_dir.name!r} is not valid JSON: {e}"
        ) from e


def _reconstruct_variants(
    manifest: dict,
    *,
    fixture_override: str | None = None,
) -> dict[str, Any]:
    """Regenerate the run's variant set, keyed by ``canonical_hash()``.

    Mirrors :func:`topofuzz.cli.run`'s variant-generation path exactly:
    ``load_topology`` → ``order_swap`` / ``role_swap`` /
    ``decoding_param_swap`` with the manifest's ``rng_seed``, then
    ``_round_robin_select`` capped at the manifest's ``variants`` count.
    The result maps each variant's canonical hash to its Topology so the
    confirm pass can resolve a candidate variant_id back to an executable
    object.

    Heavy imports (loaders pull langgraph, the generators) are lazy.

    Raises:
        ConfirmError: the fixture can't be loaded, or the manifest is
            missing the ``cli_args`` the regeneration needs.
    """
    from topofuzz.cli import _round_robin_select
    from topofuzz.loaders import load_topology
    from topofuzz.variants.generators import (
        decoding_param_swap,
        order_swap,
        role_swap,
    )

    cli_args = manifest.get("cli_args")
    if not isinstance(cli_args, dict):
        raise ConfirmError(
            "manifest.cli_args missing — cannot regenerate the variant set "
            "for the confirm pass."
        )
    fixture = fixture_override or cli_args.get("fixture")
    if not fixture:
        raise ConfirmError(
            "manifest.cli_args.fixture is empty — cannot load the topology."
        )
    # Faithful regeneration is required — guessing ``variants`` /
    # ``rng_seed`` would silently produce a DIFFERENT variant set than
    # the recorded run, and every candidate hash would then fail to
    # match. Refuse rather than guess (audit N6).
    if "rng_seed" not in cli_args:
        raise ConfirmError(
            "manifest.cli_args.rng_seed missing — cannot faithfully "
            "regenerate the variant set for the confirm pass."
        )
    if "variants" not in cli_args:
        raise ConfirmError(
            "manifest.cli_args.variants missing — cannot faithfully "
            "regenerate the variant set for the confirm pass."
        )
    rng_seed = cli_args["rng_seed"]
    variants_count = cli_args["variants"]

    try:
        topology = load_topology(fixture)
    except (
        FileNotFoundError,
        ValueError,
        AttributeError,
        TypeError,
        ImportError,
        NotImplementedError,
    ) as e:
        raise ConfirmError(
            f"failed to load fixture {fixture!r} for the confirm pass: "
            f"{type(e).__name__}: {e}"
        ) from e

    # Step 20a/F6 follow-up (L910): re-apply manifest-persisted v0.6
    # fields (verdict_node + verdict_label_true/false) BEFORE variant
    # generation. The orchestrator persists these whenever they differ
    # from the IR defaults (orchestrator.py:310-330, gated independently
    # for verdict_node and pair-level for the labels).
    #
    # ``load_topology`` only forwards these via fixture-file dunders
    # (``__topofuzz_verdict_node__``, ``__topofuzz_verdict_label_*__``).
    # A Python-API consumer who passed these via ``from_langgraph(...)``
    # has them in the manifest but NOT in the fixture file, so the
    # regenerated topology comes back as v0.5-shape, the strip rules in
    # canonical_hash fire asymmetrically, and every candidate hash
    # diverges from the persisted run. Re-apply here so confirm-pass
    # buyer-side "re-run my bundle" works on non-lending domains (the
    # exact buyer-visible artifact the F6 CEO card targets).
    #
    # **Precedence policy: manifest wins.** This is the buyer-attestation
    # contract — the signed manifest is the source of truth a
    # ``topofuzz verify`` consumer is re-running. A tampered fixture
    # file cannot rebind verdict identity away from what the producer
    # signed. If a fixture-file dunder disagrees with the manifest, we
    # log a WARNING so a forensic operator has evidence of the override.
    #
    # **Partial-pair defense.** The orchestrator emits the label pair as
    # a unit (both or neither). A hand-edited/tampered manifest with
    # only ONE label key is NOT producer-emitted state — accept it
    # would silently fall back to the topology default for the absent
    # side, rendering e.g. "ESCALATE/DENY" instead of "ESCALATE/DEFER"
    # on the CEO PDF. Reject partial pairs at the confirm boundary.
    m_label_true = manifest.get("verdict_label_true")
    m_label_false = manifest.get("verdict_label_false")
    if (m_label_true is None) != (m_label_false is None):
        raise ConfirmError(
            f"manifest verdict_label_* pair half-written — both keys must "
            f"be present or both absent (the producer-side pair-level "
            f"gate emits them as a unit). Got verdict_label_true="
            f"{m_label_true!r}, verdict_label_false={m_label_false!r}. "
            f"run_id={manifest.get('run_id')!r}"
        )
    m_verdict_node = manifest.get("verdict_node")
    needs_v06_rebind = (
        m_label_true is not None
        or m_label_false is not None
        or m_verdict_node is not None
    )
    if needs_v06_rebind:
        # Log when manifest values OVERRIDE fixture-file dunders that
        # set a different value — a forensic operator needs an audit
        # trail when the attestation contract supersedes the source file.
        if (
            m_verdict_node is not None
            and topology.verdict_node is not None
            and m_verdict_node != topology.verdict_node
        ):
            _log.warning(
                "confirm: manifest.verdict_node=%r overrides fixture-file "
                "__topofuzz_verdict_node__=%r (manifest wins — attestation "
                "contract). run_id=%r",
                m_verdict_node,
                topology.verdict_node,
                manifest.get("run_id"),
            )
        # Label override warning: ONLY fire when the loaded topology
        # ALREADY carries a non-default label value (i.e. the fixture
        # file explicitly set ``__topofuzz_verdict_label_*__``) AND
        # that value disagrees with the manifest. The IR defaults to
        # APPROVE/DENY for un-set labels, so a Python-API consumer
        # whose fixture has no dunders comes back as APPROVE/DENY
        # post-load — comparing those defaults to a manifest's
        # ESCALATE/DEFER would always trigger a "manifest overrides"
        # warning on the legitimate L910 path. Train-the-operator-to-
        # ignore-this-log is the failure mode pass-2 caught (CR M1 /
        # RT N1): the audit-trail value of the override-log only
        # exists when the warning fires SELDOM and means something.
        #
        # Cheap correct gate: the fixture-file dunder produced a
        # non-default value iff the loaded topology is non-default.
        fixture_has_custom_labels = (
            topology.verdict_label_true != "APPROVE"
            or topology.verdict_label_false != "DENY"
        )
        if (
            fixture_has_custom_labels
            and m_label_true is not None
            and m_label_true != topology.verdict_label_true
        ):
            _log.warning(
                "confirm: manifest.verdict_label_true=%r overrides "
                "fixture-file value %r (manifest wins). run_id=%r",
                m_label_true, topology.verdict_label_true,
                manifest.get("run_id"),
            )
        if (
            fixture_has_custom_labels
            and m_label_false is not None
            and m_label_false != topology.verdict_label_false
        ):
            _log.warning(
                "confirm: manifest.verdict_label_false=%r overrides "
                "fixture-file value %r (manifest wins). run_id=%r",
                m_label_false, topology.verdict_label_false,
                manifest.get("run_id"),
            )
        # Re-construct through ``Topology(...)`` rather than
        # ``model_copy(update=...)``: ``model_copy`` does NOT re-run
        # the field-validators or model-validators in pydantic v2, which
        # would let a tampered NFKC-collision / RTL-override / 13-char
        # label slip past the IR's hardening. Going through the
        # constructor fires every validator (label normalize + pair
        # collision check + structural integrity + tds_version derive).
        try:
            topology = Topology(
                name=topology.name,
                nodes=topology.nodes,
                edges=topology.edges,
                entrypoint=topology.entrypoint,
                terminal=topology.terminal,
                verdict_node=(
                    m_verdict_node
                    if m_verdict_node is not None
                    else topology.verdict_node
                ),
                verdict_label_true=(
                    m_label_true
                    if m_label_true is not None
                    else topology.verdict_label_true
                ),
                verdict_label_false=(
                    m_label_false
                    if m_label_false is not None
                    else topology.verdict_label_false
                ),
            )
        except (ValueError, TypeError) as e:
            # Pydantic raises ValidationError (a ValueError subclass) on
            # bad labels / unknown verdict_node; TypeError covers the
            # str-subclass guard. Wrap so the operator sees "the
            # manifest's persisted fields were rejected" with the
            # run_id pinned, rather than a bare pydantic stacktrace.
            raise ConfirmError(
                f"manifest v0.6 fields rejected by IR validation "
                f"(verdict_node={m_verdict_node!r}, "
                f"verdict_label_true={m_label_true!r}, "
                f"verdict_label_false={m_label_false!r}, "
                f"run_id={manifest.get('run_id')!r}): {e}"
            ) from e

    ord_list = list(order_swap(topology, rng_seed=rng_seed))
    role_list = list(role_swap(topology))
    dec_list = list(decoding_param_swap(topology))
    original = ord_list[0]
    ord_perms = ord_list[1:]
    variant_list = _round_robin_select(
        original=original,
        family_pools=(ord_perms, role_list, dec_list),
        total=variants_count,
    )
    by_hash: dict[str, Any] = {}
    for v in variant_list:
        by_hash[v.canonical_hash()] = v
    return by_hash


def run_confirm_pass(
    run_dir: Path | str,
    *,
    repeats: int = DEFAULT_REPEATS,
    max_cost_usd: float | None = None,
    judge_model: str | None = None,
    llm_callable: Any = None,
    fixture_override: str | None = None,
) -> ConfirmSummary:
    """Re-run screened flip candidates + the original variant K times each.

    See the module docstring for the locked design contract. Returns a
    :class:`ConfirmSummary`; every repeat is persisted into the run's
    ``flip_confirmations`` table via
    :meth:`topofuzz.runner.RunWriter.write_flip_confirmation`.

    A re-run REPLACES the prior confirm pass's rows — it does not merge
    (the pass calls
    :meth:`topofuzz.runner.RunWriter.delete_flip_confirmations` before the
    re-run loop so a changing ``K`` stays idempotent — audit M-RT-3).
    There is no transactional rollback: if the pass crashes mid-loop the
    prior data is already gone, and a ``partial: true`` sidecar is written
    on the way out so the half-rewritten table is *marked* incomplete
    rather than silently invisible to the attest path (audit N-RT-1).

    Args:
        run_dir: the existing ``topofuzz run`` directory.
        repeats: K — re-runs per candidate and per reference cell.
        max_cost_usd: soft-warn at 80%, hard-stop at 100%. ``None`` =
            uncapped. On hard-stop the pass persists what completed and
            returns ``ConfirmSummary.partial=True``.
        judge_model: judge model id. ``None`` reads the run's manifest
            (``cli_args.judge``).
        llm_callable: test injection point — an async LLM callable
            (same contract as :func:`topofuzz.runner.run_variant`'s
            ``llm_callable``). ``None`` uses the real LiteLLM path.
        fixture_override: test / forensic escape hatch to point at a
            relocated fixture file; ``None`` uses the manifest's spec.

    Raises:
        ConfirmError: manifest / run.sqlite missing or malformed, or the
            fixture can't be loaded.
        ValueError: ``repeats`` is not positive.
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be positive; got {repeats!r}")

    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ConfirmError("manifest.run_id missing or not a string.")
    cli_args = manifest.get("cli_args", {})

    # Lazy heavy imports — kept here so importing this module stays light.
    from fuzzeval.judge import SingleJudge

    from topofuzz.repro import find_all_flip_candidates
    from topofuzz.runner import RunWriter, default_resolver, run_variant

    # 1. Enumerate candidate flips from run.sqlite.
    candidates = find_all_flip_candidates(run_dir)
    candidate_variant_ids = sorted({vid for _eid, vid in candidates})
    candidate_eval_ids = sorted({eid for eid, _vid in candidates})

    # 2. Reconstruct the variant Topology objects, matched by hash.
    by_hash = _reconstruct_variants(
        manifest, fixture_override=fixture_override
    )
    variant_hashes = manifest.get("variant_canonical_hashes") or []
    if not variant_hashes:
        raise ConfirmError(
            "manifest.variant_canonical_hashes is empty — cannot identify "
            "the original variant for the reference repeats."
        )
    original_hash = variant_hashes[0]
    if original_hash not in by_hash:
        raise ConfirmError(
            f"regenerated variant set does not contain the manifest's "
            f"original variant {original_hash[:12]}… — the fixture or "
            "rng_seed has drifted from the recorded run."
        )

    # Eval rows: load the run's eval file the same way `topofuzz run` did.
    eval_rows_by_id = _load_eval_rows_by_id(cli_args, run_dir)

    # Judge: same rubric as the run; rubric_hash recomputed.
    effective_judge_model = (
        judge_model
        if judge_model is not None
        else cli_args.get("judge", "google/gemini-2.5-pro")
    )
    rubric_path = cli_args.get("rubric")
    judge = _build_judge(
        SingleJudge, rubric_path, effective_judge_model
    )
    # F3 Fuzzer m1 audit fix: a jury-of-3 source run gets downgraded to
    # a SingleJudge here on purpose (the confirm pass is a per-judgment
    # K-repeat to bound judge noise, not a per-vendor repeat — that's
    # what flip_confirmations dropping votes_json formalises). Surface
    # that decision so a user expecting K × 3 vendor calls doesn't get
    # surprised by the (cheaper) single-judge confirm.
    if cli_args.get("jury_of_3"):
        try:
            import typer
            typer.echo(
                f"topofuzz confirm: source run used --jury-of-3, but "
                f"confirm K-repeats with a SingleJudge ({effective_judge_model!r}) "
                "by design — the noise-floor estimator measures per-judgment "
                "variance, not per-vendor.",
                err=True,
            )
        except ImportError:
            # confirm is callable as a library too; if typer isn't
            # available swallow the hint rather than failing the run.
            pass

    # Candidate variant_ids that couldn't be matched to a regenerated
    # topology (fixture / rng_seed drift). They get zero repeats — and
    # per audit M-RT-2 a non-zero count is a measurement-integrity signal
    # that must reach the sidecar (and thereby block the headline), not
    # just a log line.
    unmatched = [
        vid for vid in candidate_variant_ids if vid not in by_hash
    ]
    if unmatched:
        _log.warning(
            "confirm: %d candidate variant(s) had no match in the "
            "regenerated variant set and were skipped: %s",
            len(unmatched),
            ", ".join(v[:12] + "…" for v in unmatched),
        )

    # 3 + 4 + 5. Re-run + persist.
    #
    # N-RT-1: ``_confirm_async`` runs ``delete_flip_confirmations`` BEFORE
    # the re-run loop, so a crash mid-loop has already destroyed the prior
    # confirm pass's rows. If we let the exception propagate with no
    # sidecar written, the attest path treats the run as "no confirm pass"
    # (sidecar absent) — safe, but the operator gets no signal that a pass
    # was attempted and left the table half-populated. So on ANY failure
    # we write a ``partial: true`` sidecar: whatever rows survived are at
    # least *marked* incomplete (the attest path then refuses the headline
    # with a partial-pass note) rather than silently invisible. We do NOT
    # attempt transactional rollback — a re-run REPLACES prior confirm
    # data (delete-then-write), and the finally-writes-partial marker is
    # sufficient (see ``delete_flip_confirmations`` / this function's
    # caveat note).
    try:
        summary = asyncio.run(
            _confirm_async(
                run_dir=run_dir,
                run_id=run_id,
                topology=by_hash[original_hash],
                original_variant=by_hash[original_hash],
                candidate_variants={
                    vid: by_hash[vid]
                    for vid in candidate_variant_ids
                    if vid in by_hash
                },
                candidates=candidates,
                candidate_eval_ids=candidate_eval_ids,
                eval_rows_by_id=eval_rows_by_id,
                repeats=repeats,
                max_cost_usd=max_cost_usd,
                judge=judge,
                judge_model=effective_judge_model,
                resolver=default_resolver,
                llm_callable=llm_callable,
                run_variant=run_variant,
                RunWriter=RunWriter,
                n_unmatched_candidates=len(unmatched),
            )
        )
    except BaseException:
        # A crash mid-pass left the table partially rewritten. Mark it
        # partial so the attest path refuses the headline rather than
        # trusting a ragged block — then re-raise (the caller still sees
        # the failure).
        _log.warning(
            "confirm: the confirm pass crashed mid-run; writing a "
            "partial sidecar so the attest path treats the "
            "flip_confirmations table as incomplete.",
        )
        try:
            _write_confirm_sidecar(
                run_dir,
                ConfirmSummary(
                    n_candidates=len(candidates),
                    n_candidate_variants=sum(
                        1 for vid in candidate_variant_ids if vid in by_hash
                    ),
                    repeats=repeats,
                    reference_repeats_run=0,
                    candidate_repeats_run=0,
                    total_cost_usd=0.0,
                    partial=True,
                    n_unmatched_candidates=len(unmatched),
                ),
            )
        except OSError:
            # Best-effort — if we cannot even write the marker, the
            # absent sidecar is still the safe ("no confirm pass") state.
            _log.warning("confirm: could not write the partial sidecar.")
        raise
    # B-RT-1 / M-RT-2: persist the sidecar so the attest path can decide
    # whether the noise-floor headline is populatable.
    _write_confirm_sidecar(run_dir, summary)
    return summary


def _resolve_eval_path(cli_args: dict, run_dir: Path) -> Path:
    """Locate the run's eval file from ``manifest.cli_args.eval``.

    Resolution order, first existing match wins:

    * an **absolute** ``eval`` path is tried as-is and ONLY as-is —
      ``cwd / abs`` and ``run_dir.parent / abs`` collapse back to the
      same absolute path on every OS, so trying them is dead weight that
      also makes the not-found error message print the same path 4×
      (audit N3).
    * a **relative** ``eval`` path is resolved run-dir-relative FIRST,
      then run-dir-parent-relative, then cwd-relative, then literal.
      Run-dir-relative is preferred over cwd-relative (audit m-RT-4):
      ``confirm`` runs over the operator's own run directory and a
      same-named eval file in ``cwd`` must not shadow the one that
      actually belongs to the run.

    Unlike :func:`topofuzz.repro._resolve_eval_path`, this does NOT
    refuse absolute paths: ``confirm`` is a first-party post-run pass
    over the operator's OWN run directory (it already loads and executes
    the fixture), so the untrusted-manifest threat model that hardening
    addresses — leaking arbitrary file contents into a *generated
    standalone script* — does not apply here.
    """
    raw = cli_args.get("eval", "")
    if not raw:
        raise ConfirmError(
            "manifest.cli_args.eval is empty — cannot locate the eval file."
        )
    p = Path(str(raw))
    if p.is_absolute():
        # An absolute path resolves to itself under every prefix; only
        # the literal path is meaningful (audit N3 — no 4× duplicate).
        tried = [p]
    else:
        # Run-dir-relative before cwd-relative so a same-named file in
        # cwd cannot shadow the run's own eval file (audit m-RT-4).
        tried = [run_dir / p, run_dir.parent / p, Path.cwd() / p, p]
    for c in tried:
        if c.is_file():
            return c.resolve()
    raise ConfirmError(
        f"eval file not found from manifest.cli_args.eval={raw!r}; tried "
        + ", ".join(str(c) for c in tried)
    )


def _load_eval_rows_by_id(cli_args: dict, run_dir: Path) -> dict[str, dict]:
    """Load the run's eval file → ``{eval_id: row}``.

    Resolves the path via :func:`_resolve_eval_path`, then parses with
    :func:`topofuzz.loaders.load_eval_rows`.
    """
    from topofuzz.loaders import load_eval_rows

    eval_path = _resolve_eval_path(cli_args, run_dir)
    try:
        rows = load_eval_rows(eval_path)
    except (FileNotFoundError, ValueError) as e:
        raise ConfirmError(
            f"failed to load eval rows for the confirm pass: {e}"
        ) from e
    return {r["id"]: r for r in rows}


def _build_judge(SingleJudge: Any, rubric_path: Any, judge_model: str) -> Any:
    """Construct the SingleJudge for the confirm pass (rubric_hash recompute).

    ``rubric_path`` comes from the manifest's ``cli_args.rubric``; when
    absent (older manifest) we fall back to the packaged default the same
    way :func:`topofuzz.cli.run` does.
    """
    if rubric_path:
        path = Path(rubric_path)
    else:
        from topofuzz.cli import _packaged_default_rubric_path

        path = _packaged_default_rubric_path()
    try:
        return SingleJudge(model=judge_model, rubric_path=path)
    except (FileNotFoundError, ValueError) as e:
        raise ConfirmError(
            f"failed to construct judge for the confirm pass "
            f"(rubric={path!s}): {e}"
        ) from e


async def _confirm_async(
    *,
    run_dir: Path,
    run_id: str,
    topology: Any,
    original_variant: Any,
    candidate_variants: dict[str, Any],
    candidates: list[tuple[str, str]],
    candidate_eval_ids: list[str],
    eval_rows_by_id: dict[str, dict],
    repeats: int,
    max_cost_usd: float | None,
    judge: Any,
    judge_model: str,
    resolver: Any,
    llm_callable: Any,
    run_variant: Any,
    RunWriter: Any,
    n_unmatched_candidates: int = 0,
) -> ConfirmSummary:
    """Inner async driver: re-run repeats sequentially, persist each.

    Sequential by design — the confirm pass is a small, bounded re-run
    (candidates are a fraction of the original sweep) and a serial loop
    keeps the ``--max-cost-usd`` accounting trivially correct without a
    shared-semaphore drain. Cache is disabled (``llm_cache=None``) so
    every repeat is a fresh LLM draw.
    """
    rubric_hash = getattr(judge, "rubric_hash", "")
    judge_renderer = getattr(judge, "render_prompt", None)

    total_cost = 0.0
    reference_repeats_run = 0
    candidate_repeats_run = 0
    partial = False
    soft_warn_emitted = False
    skipped_eval_ids: list[str] = []

    def _cap_hit() -> bool:
        return max_cost_usd is not None and total_cost >= max_cost_usd

    with RunWriter(run_dir, run_id, topology, rubric_hash=rubric_hash) as writer:
        # M-RT-3: delete-then-write. INSERT OR REPLACE alone would leave
        # stale higher-index rows behind when this pass uses a smaller K
        # than a prior one — re-running confirm must be idempotent across
        # changing K. Delete every prior row for this run_id first.
        writer.delete_flip_confirmations()

        # ----- reference repeats: original variant × K × candidate evals
        for eid in candidate_eval_ids:
            eval_row = eval_rows_by_id.get(eid)
            if eval_row is None:
                _log.warning(
                    "confirm: eval_id %r not found in the run's eval file; "
                    "skipping its reference repeats.", eid,
                )
                skipped_eval_ids.append(eid)
                continue
            for ridx in range(repeats):
                if _cap_hit():
                    partial = True
                    break
                cell = await run_variant(
                    original_variant,
                    eval_row,
                    llm_cache=None,
                    judge=judge,
                    judge_cache=None,
                    rubric_hash=rubric_hash,
                    judge_model_version=judge_model,
                    judge_prompt_renderer=judge_renderer,
                    model_served_version_resolver=resolver,
                    llm_callable=llm_callable,
                )
                writer.write_flip_confirmation(
                    cell, repeat_idx=ridx, role="reference"
                )
                reference_repeats_run += 1
                total_cost += _sum_cell_cost(cell)
                soft_warn_emitted = _maybe_soft_warn(
                    total_cost, max_cost_usd, soft_warn_emitted
                )
            if partial:
                break

        # ----- candidate repeats: each (variant, eval) × K
        if not partial:
            for eid, vid in candidates:
                variant = candidate_variants.get(vid)
                if variant is None:
                    continue  # unmatched — warned by the caller
                eval_row = eval_rows_by_id.get(eid)
                if eval_row is None:
                    _log.warning(
                        "confirm: eval_id %r not found in the run's eval "
                        "file; skipping candidate %s.", eid, vid[:12],
                    )
                    if eid not in skipped_eval_ids:
                        skipped_eval_ids.append(eid)
                    continue
                for ridx in range(repeats):
                    if _cap_hit():
                        partial = True
                        break
                    cell = await run_variant(
                        variant,
                        eval_row,
                        llm_cache=None,
                        judge=judge,
                        judge_cache=None,
                        rubric_hash=rubric_hash,
                        judge_model_version=judge_model,
                        judge_prompt_renderer=judge_renderer,
                        model_served_version_resolver=resolver,
                        llm_callable=llm_callable,
                    )
                    writer.write_flip_confirmation(
                        cell, repeat_idx=ridx, role="candidate"
                    )
                    candidate_repeats_run += 1
                    total_cost += _sum_cell_cost(cell)
                    soft_warn_emitted = _maybe_soft_warn(
                        total_cost, max_cost_usd, soft_warn_emitted
                    )
                if partial:
                    break

    if partial:
        _log.warning(
            "confirm: --max-cost-usd hard-stop at $%.4f / $%.4f cap. "
            "Persisted %d reference + %d candidate repeats before stopping; "
            "results are PARTIAL.",
            total_cost, max_cost_usd,
            reference_repeats_run, candidate_repeats_run,
        )

    return ConfirmSummary(
        n_candidates=len(candidates),
        n_candidate_variants=len(candidate_variants),
        repeats=repeats,
        reference_repeats_run=reference_repeats_run,
        candidate_repeats_run=candidate_repeats_run,
        total_cost_usd=total_cost,
        partial=partial,
        n_unmatched_candidates=n_unmatched_candidates,
        skipped_eval_ids=tuple(skipped_eval_ids),
    )


def _sum_cell_cost(cell: Any) -> float:
    """Sum a Cell's NodeOutput costs, treating ``None`` as 0.0.

    Mirrors :func:`topofuzz.orchestrator._sum_cost` — the confirm pass
    wants a running float total even for un-priced (mocked) cells.
    """
    return sum((o.cost_usd or 0.0) for o in cell.node_outputs)


def _maybe_soft_warn(
    total_cost: float, max_cost_usd: float | None, already_warned: bool
) -> bool:
    """Emit the 80%-of-cap soft-warn once. Returns the updated flag."""
    if (
        max_cost_usd is not None
        and not already_warned
        and total_cost >= 0.80 * max_cost_usd
    ):
        _log.warning(
            "confirm: cost soft-warn — $%.4f / $%.4f cap (80%% threshold).",
            total_cost, max_cost_usd,
        )
        return True
    return already_warned
