# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""``topofuzz repro``: emit a standalone reproducer script.

BLUEPRINT_v5plus.md step 19b (line 441). Answers the first skeptical
HN comment ("how do I know this isn't noise?") in 30 seconds. The
generated script imports the user's graph module, replays the two
losing orderings against the failing input, prints both verdicts
side-by-side, and ``assert``s the mismatch.

Design contract:

* Script is **standalone** — no ``import topofuzz`` anywhere in
  the generated file. Skeptics paste into a gist + run.
* Script imports ``LLM_CONFIGS`` (and the user's ``build_graph``
  factory) from ``manifest.cli_args.fixture`` when the path can be
  dotted; otherwise emits a ``# TODO`` placeholder + warning.
* Output is **deterministic** given a fixed ``(run_dir, eval_id)`` —
  the byte-identical contract is covered by
  ``test_build_repro_script_byte_identical_on_double_render``.
* Generation never imports the user's fixture (untrusted-fixture
  concern; see BACKLOG ``--allow-untrusted-fixture``). All inputs
  come from ``manifest.json``, ``run.sqlite``, and the eval file.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

# jinja2 is in the ``topofuzz[full]`` extra only — keep the import lazy
# so a base install (``pip install topofuzz`` for verify-only callers)
# stays importable and we surface a friendly error at the first ``repro``
# invocation instead of a bare ``ModuleNotFoundError`` traceback. Matches
# ``topofuzz.report``'s lazy-jinja2 pattern (audit CR pass-2 MAJOR-1).

_log = logging.getLogger(__name__)

# Sourced from :mod:`topofuzz.metrics.verdict_flip` so the repro CLI's
# binarisation matches the §14 headline metric (closes BACKLOG: lift
# _FLIP_THRESHOLD).
from topofuzz.metrics.verdict_flip import (
    DEFAULT_FLIP_THRESHOLD as DEFAULT_VERDICT_THRESHOLD,
)

# Shared 3-state class rule (2026-06-11): an explicit v7 ``verdict_class``
# must outrank the score threshold HERE exactly as it does in the metrics
# layer — otherwise ``topofuzz repro`` could emit a reproducer for a
# "flip" whose conditional side the §14 headline metric excluded as a
# non-clear pair, contradicting the published number.
from topofuzz.metrics._verdict import verdict_of as _verdict_of

# eval_id is interpolated into the default output filename. Constrain
# to characters that are filesystem-safe on POSIX *and* Windows so a
# hostile CSV ``id`` column cannot drive ``write_repro_script`` to
# clobber files outside the run_dir (audit RT-pass-1 BLOCKER).
_SAFE_EID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ReproError(ValueError):
    """Raised when a reproducer cannot be built (no flip, missing file, etc)."""


def find_flip_candidate(
    run_dir: Path,
    *,
    eval_id: str | None = None,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> tuple[str, int, int]:
    """Return ``(eval_id, variant_idx_neg, variant_idx_pos)`` for one flip.

    Variant indices are positions in ``manifest.variant_canonical_hashes``.
    ``variant_idx_neg`` is the variant whose binarised verdict is below
    threshold ("DENY"); ``variant_idx_pos`` is above ("APPROVE").

    Selection is deterministic:

    * If ``eval_id`` is given, use that eval (or raise if it doesn't flip).
    * Otherwise scan eval_ids in lexicographic order and pick the first
      one that has at least one variant on each side of ``threshold``.
    * Within the chosen eval, pick the **smallest-index** variant on each
      side. Ties are broken by variant-index, which is itself stable in
      the manifest (insertion order of canonical_hashes).

    Cells with ``passed IS NULL AND score_value IS NULL`` (judge failures)
    are skipped — they contribute neither side of the flip. Cells whose
    v7 ``verdict_class`` is explicitly ``"conditional"`` are skipped for
    the same reason: the §14 headline metric counts clear APPROVE↔DENY
    pairs only, and the reproducer must agree with it.

    Raises:
        ReproError: no flip found, or ``eval_id`` doesn't exist in the run.
    """
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    hashes = manifest["variant_canonical_hashes"]
    hash_to_idx = {h: i for i, h in enumerate(hashes)}

    con = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        try:
            # v7 ``verdict_class`` probe — same conditional-read shape as
            # :func:`topofuzz.metrics.score_matrix.read_score_matrix`, so
            # a pre-v7 DB still loads (its rows fall back to the legacy
            # passed/threshold rule below).
            judge_cols = {
                row[1]
                for row in con.execute(
                    "PRAGMA table_info(judge_results)"
                ).fetchall()
            }
            has_verdict_class = "verdict_class" in judge_cols
            select_cols = "variant_id, eval_id, score_value, passed"
            if has_verdict_class:
                select_cols += ", verdict_class"
            rows = list(
                con.execute(
                    f"SELECT {select_cols} "
                    "FROM judge_results "
                    "ORDER BY variant_id, eval_id"
                )
            )
        except sqlite3.DatabaseError as e:
            # Audit RT-pass-1 M-RT-1: surface schema/corruption errors
            # as ReproError so the CLI returns a friendly exit-1 with a
            # message instead of an unfriendly traceback.
            raise ReproError(
                f"run.sqlite in {run_dir.name} is missing the expected "
                f"judge_results schema or is corrupt: {e}"
            ) from e
    finally:
        con.close()

    by_eval: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row in rows:
        vid, eid, score, passed = row[0], row[1], row[2], row[3]
        vclass = row[4] if has_verdict_class else None
        idx = hash_to_idx.get(vid)
        if idx is None:
            continue
        # 3-state rule, shared with the metrics layer via ``verdict_of``:
        # an explicit ``verdict_class`` outranks both ``passed`` and the
        # score threshold, and an explicitly CONDITIONAL cell contributes
        # NEITHER side of a flip (the §14 metric counts clear
        # APPROVE↔DENY pairs only — a reproducer for a pair the headline
        # rate excluded would contradict the published number). Rows
        # without an explicit class keep the legacy passed → threshold
        # precedence byte-identically.
        explicit = _verdict_of({"verdict_class": vclass}, threshold)
        if explicit == "conditional":
            continue
        if explicit is not None:
            verdict = explicit == "approve"
        elif passed is not None:
            verdict = bool(passed)
        elif score is not None:
            verdict = bool(score >= threshold)
        else:
            continue
        by_eval[eid].append((idx, verdict))

    if eval_id is not None:
        if eval_id not in by_eval:
            raise ReproError(
                f"eval_id={eval_id!r} has no scored judge results in run "
                f"{run_dir.name} (rows may be judge-failures or "
                f"explicitly CONDITIONAL)"
            )
        candidates: list[str] = [eval_id]
    else:
        candidates = sorted(by_eval.keys())

    for eid in candidates:
        pos = sorted(i for i, v in by_eval[eid] if v)
        neg = sorted(i for i, v in by_eval[eid] if not v)
        if pos and neg:
            return eid, neg[0], pos[0]

    suffix = f" for eval {eval_id!r}" if eval_id else ""
    raise ReproError(
        f"No verdict flip found in run {run_dir.name}{suffix}. "
        "Topofuzz repro requires at least one eval row where variants "
        "disagree on the binarised verdict."
    )


def find_all_flip_candidates(
    run_dir: Path,
    *,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> list[tuple[str, str]]:
    """Return every ``(eval_id, variant_id)`` whose verdict flips vs the original.

    A *candidate* is a cell whose binarised verdict (``score >= threshold``,
    or ``passed`` when present) differs from the **original variant's**
    verdict on the same ``eval_id``. The original variant is
    ``manifest.variant_canonical_hashes[0]`` — the §6 kickoff-insertion-
    order contract, same as :func:`find_flip_candidate`.

    Unlike :func:`find_flip_candidate` (which returns ONE flip for the
    standalone reproducer), this returns the FULL candidate set — the
    ``topofuzz confirm`` pass re-runs every one of them K times to
    separate the topology effect from LLM noise.

    Selection rules, all deterministic:

    * Judge-failure cells (``passed IS NULL AND score_value IS NULL``)
      are skipped — they contribute no verdict to compare.
    * An eval_id whose original-variant cell is a judge failure (or
      absent) is skipped entirely: with no reference verdict there is
      nothing to flip against.
    * The original variant is never its own candidate.
    * Output is sorted lexicographically by ``(eval_id, variant_id)``
      so the re-run order is stable across invocations.

    Returns an empty list when no flip exists (this is NOT an error —
    a non-fragile run legitimately has zero candidates; the caller
    decides whether to treat that as a no-op).

    Raises:
        ReproError: ``manifest.json`` missing / malformed, or
            ``run.sqlite`` is missing the ``judge_results`` schema.
    """
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    hashes = manifest["variant_canonical_hashes"]
    if not hashes:
        raise ReproError(
            f"manifest.variant_canonical_hashes is empty in run "
            f"{run_dir.name} — cannot identify the original variant."
        )
    original_hash = hashes[0]
    known_hashes = set(hashes)

    con = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        try:
            # v7 ``verdict_class`` probe — identical shape to
            # :func:`find_flip_candidate` above so every flip enumerator
            # in this module applies the same 3-state rule. This one
            # matters MOST: ``confirm`` (K-repeat spend), F4 ``ablate``
            # (attribution triples), and ``attest`` (consistency card)
            # all consume this enumerator, and before 2026-06-11 an
            # explicitly CONDITIONAL cell (continuous score 0.5 →
            # legacy-binarised APPROVE) could be screened, re-run, and
            # attributed as a "flip" the §14 headline metric excluded.
            judge_cols = {
                row[1]
                for row in con.execute(
                    "PRAGMA table_info(judge_results)"
                ).fetchall()
            }
            has_verdict_class = "verdict_class" in judge_cols
            select_cols = "variant_id, eval_id, score_value, passed"
            if has_verdict_class:
                select_cols += ", verdict_class"
            rows = list(
                con.execute(
                    f"SELECT {select_cols} "
                    "FROM judge_results "
                    "ORDER BY eval_id, variant_id"
                )
            )
        except sqlite3.DatabaseError as e:
            raise ReproError(
                f"run.sqlite in {run_dir.name} is missing the expected "
                f"judge_results schema or is corrupt: {e}"
            ) from e
    finally:
        con.close()

    # Binarise once; only cells whose variant_id is a known manifest
    # hash count (a stray cell under an unknown hash is forensic noise).
    by_eval: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        vid, eid, score, passed = row[0], row[1], row[2], row[3]
        vclass = row[4] if has_verdict_class else None
        if vid not in known_hashes:
            continue
        # Same 3-state precedence as :func:`find_flip_candidate`.
        explicit = _verdict_of({"verdict_class": vclass}, threshold)
        if explicit == "conditional":
            continue
        if explicit is not None:
            verdict = explicit == "approve"
        elif passed is not None:
            verdict = bool(passed)
        elif score is not None:
            verdict = bool(score >= threshold)
        else:
            continue  # judge failure — no verdict
        by_eval[eid][vid] = verdict

    candidates: list[tuple[str, str]] = []
    for eid in sorted(by_eval):
        verdicts = by_eval[eid]
        orig_verdict = verdicts.get(original_hash)
        if orig_verdict is None:
            # Original variant has no scored verdict for this eval — no
            # reference frame, skip the whole eval.
            continue
        for vid in sorted(verdicts):
            if vid == original_hash:
                continue
            if verdicts[vid] != orig_verdict:
                candidates.append((eid, vid))
    return candidates


def resolve_graph_import(fixture_spec: str) -> tuple[str | None, str]:
    """Parse ``<path>:<factory>`` into ``(dotted_module, factory_name)``.

    Returns ``(None, factory_name)`` when the path can't be safely dotted:
    absolute paths, parent traversal, non-``.py`` files, or names that
    aren't valid Python identifiers. Caller emits a ``# TODO`` placeholder
    in those cases.

    Examples (cross-platform — manifest may store forward or back slashes):

    * ``examples/gallery/lending/fixture.py:build_graph`` →
      ``("examples.gallery.lending.fixture", "build_graph")``
    * ``examples\\gallery\\lending\\fixture.py:build_graph`` → same.
    * ``/abs/path/fixture.py:build_graph`` → ``(None, "build_graph")``.
    * ``../foo/fixture.py:build_graph`` → ``(None, "build_graph")``.
    """
    if ":" not in fixture_spec:
        return None, "build_graph"
    # rpartition (not partition) so a Windows drive-letter colon in the
    # path doesn't get consumed as the path/factory separator
    # (audit CR pass-1 M-CR-2 + the regression test it added).
    path_str, _, factory = fixture_spec.rpartition(":")
    factory = factory or "build_graph"
    if not factory.isidentifier():
        return None, "build_graph"
    # Normalise Windows separators so this works regardless of which OS
    # produced the manifest.
    raw = path_str.replace("\\", "/")
    # Audit pass-1 (CR M2 + RT MN3): reject any path that's absolute under
    # POSIX *or* Windows semantics, including POSIX-style absolutes seen
    # from a manifest produced on the other OS. ``Path.is_absolute()``
    # on Windows returns ``False`` for ``/etc/passwd``, so a single check
    # leaks the cross-platform case.
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
        return None, factory
    # Drive-letter check: ``Path("C:/abs/fixture.py").parts`` on POSIX
    # yields ``("C:", "abs", "fixture.py")`` and ``"C:".isidentifier()``
    # is ``False`` only because of the colon. Belt-and-braces: explicit
    # reject of any leading drive-letter form.
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        return None, factory
    p = Path(raw)
    if p.suffix != ".py":
        return None, factory
    parts = list(p.with_suffix("").parts)
    if not parts or any(part in ("", "..", ".") for part in parts):
        return None, factory
    if not all(part.isidentifier() for part in parts):
        return None, factory
    return ".".join(parts), factory


def read_eval_input(eval_path: Path, eval_id: str) -> str:
    """Return the ``input`` column for ``eval_id`` from a CSV or JSONL eval file.

    The returned value is always a ``str`` — JSONL rows with a non-string
    ``input`` field (e.g. nested dict) raise ``ReproError`` rather than
    silently propagating into the generated script where they'd break the
    f-string at runtime (audit RT-pass-1 M-RT-3).

    Raises:
        ReproError: eval_id missing, non-string ``input``, or unsupported
            file extension.
    """
    suffix = eval_path.suffix.lower()
    if suffix == ".csv":
        with eval_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") == eval_id:
                    val = row.get("input", "")  # csv.DictReader yields str
                    if not val:
                        raise ReproError(
                            f"eval_id={eval_id!r} in {eval_path}: ``input`` "
                            "field is empty or missing. A meaningless prompt "
                            "would produce a meaningless reproducer "
                            "(audit RT pass-2 M-RT2-1)."
                        )
                    return val
    elif suffix in (".jsonl", ".json"):
        with eval_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("id") == eval_id:
                    val = d.get("input", "")
                    if not isinstance(val, str):
                        raise ReproError(
                            f"eval_id={eval_id!r} in {eval_path}: ``input`` "
                            f"field is {type(val).__name__}, expected str"
                        )
                    if not val:
                        raise ReproError(
                            f"eval_id={eval_id!r} in {eval_path}: ``input`` "
                            "field is empty (audit RT pass-2 M-RT2-1)."
                        )
                    return val
    else:
        raise ReproError(
            f"Unsupported eval-file format {eval_path.suffix!r}; "
            "topofuzz repro reads .csv or .jsonl."
        )
    raise ReproError(f"eval_id={eval_id!r} not found in {eval_path}")


def _resolve_verdict_node(manifest: dict) -> str | None:
    """Pick the node id whose output the generated repro should compare.

    Precedence mirrors :func:`topofuzz.runner._run_judge` and
    :func:`topofuzz.publish._excerpt_terminal_output` (Step 20a / F6 BLOCKER):

    1. ``manifest.verdict_node`` (v0.6 first-class field) — used as-is if a
       non-empty trimmed ``str``. A present-but-malformed value (empty
       string, non-string type) is rejected with a one-line warning so an
       operator can grep the log; resolution falls through to (2).
    2. The lex-first entry of a SINGLE-element ``manifest.terminal_nodes``
       list (v0.5.5 transitional shape). A list with two or more entries
       is REFUSED on multi-terminal manifests without an explicit
       ``verdict_node``: see L1028 below.
    3. ``None`` — for v0.5 manifests with neither field. The rendered script
       falls back to ``history[-1][1]`` (legacy behaviour preserved bit-for-
       bit for existing gallery runs).

    Returns the resolved node id or ``None`` for the v0.5 fallback path.
    Raises :class:`ReproError` on a multi-terminal manifest where
    ``verdict_node`` was not resolved in step (1).

    **L1028 multi-terminal refusal (closed 2026-05-16).** On a
    multi-terminal manifest without ``verdict_node``, this function
    used to return ``sorted(terminal_nodes)[0]`` unconditionally — but
    :func:`publish._excerpt_terminal_output` picks the first sorted
    node that is *also present in the cell's* ``by_node`` (the
    output-level survival filter). repro has no per-cell access at
    render time, so it cannot replicate that filter. Returning
    lex-first regardless meant the repro script could name a different
    node than the CEO PDF for any cell where the lex-first terminal
    was cancelled / silently dropped. We refuse multi-terminal here so
    the buyer hits a clear ``ReproError`` with remediation guidance,
    not a silent disagreement between two buyer-facing artifacts on
    the same run.
    """
    raw_verdict = manifest.get("verdict_node")
    if raw_verdict is not None:
        if not isinstance(raw_verdict, str):
            _log.warning(
                "topofuzz repro: manifest.verdict_node is a %s, not a string; "
                "ignoring (a malformed manifest is the underlying bug — fix "
                "the orchestrator, don't lean on this fallback)",
                type(raw_verdict).__name__,
            )
        elif not raw_verdict.strip():
            _log.warning(
                "topofuzz repro: manifest.verdict_node is empty / whitespace; "
                "falling through to terminal_nodes (mirrors publish)."
            )
        else:
            return raw_verdict

    raw_terminals = manifest.get("terminal_nodes")
    if raw_terminals is None:
        return None
    if not isinstance(raw_terminals, list):
        _log.warning(
            "topofuzz repro: manifest.terminal_nodes is a %s, not a list; "
            "ignoring.",
            type(raw_terminals).__name__,
        )
        return None
    if not all(isinstance(t, str) and t.strip() for t in raw_terminals):
        _log.warning(
            "topofuzz repro: manifest.terminal_nodes contains blank or "
            "non-string entries (%r); the whole list is rejected to surface "
            "the malformed manifest rather than silently pick a survivor.",
            raw_terminals,
        )
        return None
    # Length dispatch after content validation:
    #   len == 0 → return None (line below; v0.5 fallback)
    #   len == 1 → return raw_terminals[0] (single-terminal v0.5.5 happy path)
    #   len >= 2 → raise ReproError (L1028 multi-terminal refusal)
    if not raw_terminals:
        return None
    if len(raw_terminals) >= 2:
        raise ReproError(
            "topofuzz repro cannot disambiguate a multi-terminal manifest "
            f"({raw_terminals!r}) without an explicit ``verdict_node``. "
            "On multi-terminal topologies, ``publish``'s CEO card picks "
            "the first sorted terminal that ALSO produced output for each "
            "cell (the output-level survival filter on ``by_node``); "
            "repro has no per-cell access at render time and would have "
            "to fall back to lex-first unconditionally — silently naming "
            "a different node than the CEO PDF whenever the lex-first "
            "terminal was cancelled. Set ``Topology(..., verdict_node=...)``"
            " on the fixture and re-run ``topofuzz run`` to regenerate "
            "the manifest, then re-run ``topofuzz repro``. "
            "(L1028 closed 2026-05-16, option (c).)"
        )
    return raw_terminals[0]


def _load_manifest(run_dir: Path) -> dict:
    """Read and parse ``manifest.json`` once. Centralises error message."""
    try:
        return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ReproError(
            f"manifest.json missing in run_dir {run_dir.name!r}"
        ) from e
    except json.JSONDecodeError as e:
        raise ReproError(
            f"manifest.json in {run_dir.name!r} is not valid JSON: {e}"
        ) from e


def _resolve_eval_path(cli_args: dict, run_dir: Path) -> Path:
    """Locate the eval file referenced by the manifest.

    Tries the literal path first, then cwd-relative, run_dir-parent-relative,
    and run_dir-relative. Returns the first existing match.

    Audit RT-pass-1 M-RT-2: parent-traversal in the manifest's eval path
    is rejected. A hostile run_dir with ``cli_args.eval = "../secrets.csv"``
    would otherwise leak file contents into the generated script's
    ``EVAL_INPUT`` literal.
    """
    raw = cli_args.get("eval", "")
    if not raw:
        raise ReproError(
            "manifest.cli_args.eval is empty — cannot locate eval file"
        )
    raw_norm = raw.replace("\\", "/")
    if ".." in PurePosixPath(raw_norm).parts:
        raise ReproError(
            f"manifest.cli_args.eval={raw!r} contains parent-traversal; "
            "refusing to read paths outside the run_dir tree."
        )
    # Audit pass-2 (CR MAJOR-2 + RT MN-RT2-1): reject ANY absolute path —
    # POSIX-absolute (``/etc/passwd``) and Windows-absolute (``C:\…``) —
    # so a hostile manifest can't direct ``read_eval_input`` at arbitrary
    # system files. Eval files always live inside the project tree.
    if (
        PurePosixPath(raw_norm).is_absolute()
        or PureWindowsPath(raw_norm).is_absolute()
        or (len(raw_norm) >= 2 and raw_norm[1] == ":" and raw_norm[0].isalpha())
    ):
        raise ReproError(
            f"manifest.cli_args.eval={raw!r} is an absolute path; "
            "topofuzz repro only resolves eval files relative to the "
            "project tree."
        )
    p = Path(raw_norm)
    tried: list[Path] = [p, Path.cwd() / p, run_dir.parent / p, run_dir / p]
    for c in tried:
        if c.is_file():
            return c.resolve()
    raise ReproError(
        f"eval file not found from manifest.cli_args.eval={raw!r}; tried "
        + ", ".join(str(c) for c in tried)
    )


def build_repro_script(
    run_dir: Path,
    *,
    eval_id: str | None = None,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> tuple[str, str]:
    """Return ``(script_text, chosen_eval_id)``.

    Pure function: byte-identical output for identical ``(run_dir, eval_id)``.
    Never imports or executes the user's fixture. All data comes from
    ``manifest.json``, ``run.sqlite``, and the eval file referenced by
    the manifest.
    """
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    cli_args = manifest.get("cli_args", {})
    fixture_spec = cli_args.get("fixture", "")

    eid, idx_neg, idx_pos = find_flip_candidate(
        run_dir, eval_id=eval_id, threshold=threshold
    )

    orderings = manifest["variant_orderings"]
    ordering_neg = list(orderings[idx_neg])
    ordering_pos = list(orderings[idx_pos])
    hashes = manifest["variant_canonical_hashes"]
    hash_neg = hashes[idx_neg]
    hash_pos = hashes[idx_pos]

    # Resolve the judged node the same way ``runner._run_judge`` and
    # ``publish._excerpt_terminal_output`` do (Step 20a / F6 BLOCKER).
    # Without this, the generated repro returns whatever node a permutation
    # walked last, which on multi-terminal / supervisor topologies is not
    # the node the judge actually scored — the same lexical-luck fragility
    # ``verdict_node`` was introduced to kill.
    verdict_node = _resolve_verdict_node(manifest)
    # Audit RT-pass-1 M1: if the orchestrator declared ``verdict_node``
    # but neither variant ordering actually walked it, the rendered script
    # would AssertionError mid-run after the buyer has already paid for the
    # first Gemini turn. Catch that here — surface as ReproError with both
    # orderings so the operator can diagnose without running the script.
    if verdict_node is not None and (
        verdict_node not in ordering_neg or verdict_node not in ordering_pos
    ):
        raise ReproError(
            f"manifest.verdict_node={verdict_node!r} is not present in "
            f"both flip orderings (NEG={ordering_neg!r}, POS={ordering_pos!r}). "
            "Topology drifted since the run was captured — re-run "
            "``topofuzz run`` to refresh the manifest."
        )

    eval_path = _resolve_eval_path(cli_args, run_dir)
    eval_input = read_eval_input(eval_path, eid)

    module_dotted, factory = resolve_graph_import(fixture_spec)
    if module_dotted is None:
        import_line = (
            f"# TODO: import your graph factory + LLM_CONFIGS here.\n"
            f"# manifest.fixture={fixture_spec!r} could not be auto-resolved\n"
            f"# (absolute path, parent-traversal, or non-python file).\n"
            f"# from your_module import {factory}, LLM_CONFIGS"
        )
        _log.warning(
            "topofuzz repro: could not auto-resolve graph_module from "
            "fixture spec %r; emitting a TODO placeholder. Edit the "
            "generated script's import_line before running it.",
            fixture_spec,
        )
    else:
        import_line = f"from {module_dotted} import {factory}, LLM_CONFIGS"

    # PackageLoader works under wheel install + zipped imports, where
    # ``FileSystemLoader(str(resources.files(...)))`` silently fails
    # because the path is virtual (audit pass-1 MN-CR-1 / MN-RT-2).
    # jinja2 lives in the ``[full]`` extra; surface a friendly error
    # for base-install callers (audit CR pass-2 MAJOR-1).
    try:
        from jinja2 import Environment, PackageLoader, StrictUndefined
    except ImportError as e:
        raise ReproError(
            "topofuzz repro requires the 'full' install extra "
            "(jinja2 not found). Install with `pip install topofuzz[full]`."
        ) from e
    env = Environment(
        loader=PackageLoader("topofuzz", "templates"),
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tpl = env.get_template("repro.py.j2")
    text = tpl.render(
        run_id=manifest["run_id"],
        eval_id=eid,
        import_line=import_line,
        eval_input=eval_input,
        ordering_neg=ordering_neg,
        ordering_pos=ordering_pos,
        hash_neg_short=hash_neg[:8],
        hash_pos_short=hash_pos[:8],
        threshold=threshold,
        verdict_node=verdict_node,
    )
    return text, eid


def write_repro_script(
    run_dir: Path,
    *,
    eval_id: str | None = None,
    out_path: Path | None = None,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
) -> Path:
    """Build + write the script. Returns the path written.

    Default destination: ``<run_dir>/repro_<eval_id>.py``. When the caller
    doesn't override ``out_path``, ``eval_id`` is validated against
    ``_SAFE_EID_RE`` to refuse path-traversal/symlink-bait substrings —
    audit RT-pass-1 BLOCKER. Callers supplying their own ``out_path``
    own that path; we honour it as-is.
    """
    run_dir = Path(run_dir)
    text, eid = build_repro_script(run_dir, eval_id=eval_id, threshold=threshold)
    if out_path is None:
        if not _SAFE_EID_RE.match(eid):
            raise ReproError(
                f"eval_id {eid!r} contains characters not permitted in the "
                f"default output filename (allowed: A-Z, a-z, 0-9, '.', '_', '-'). "
                "Pass --out <path> to override."
            )
        target = run_dir / f"repro_{eid}.py"
    else:
        target = Path(out_path)
    target.write_text(text, encoding="utf-8")
    return target
