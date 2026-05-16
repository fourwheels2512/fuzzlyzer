# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 6 CLI loaders: fixture-spec + eval-row I/O.

Two small functions:

* :func:`load_topology` — parse ``"path/to/fixture.py:attr"``, import the
  module, resolve ``attr`` (callable → call, else use directly), then
  convert via :func:`topofuzz.adapter.from_langgraph`. If the module also
  exports a top-level ``LLM_CONFIGS: dict[str, LLMConfig]`` it is
  threaded into the adapter so §6 ``role_swap`` /
  ``decoding_param_swap`` generators fire.

* :func:`load_eval_rows` — read ``.jsonl`` or ``.csv`` into the list-of-
  dicts shape that :func:`run_variant` expects (``{"id": str, "input":
  any, ...}``). Empty / malformed files raise with file + line context.

Out of scope for step 6: HuggingFace datasets, parquet, YAML, glob
patterns. Add when a user asks.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from topofuzz.adapter import from_langgraph
from topofuzz.ir import LLMConfig, Topology

_log = logging.getLogger(__name__)

# Tracks which fixture paths we've already warned about in this process,
# so the "executes arbitrary Python" warning fires once per path rather
# than once per call. Cleared between processes (no need to persist).
_WARNED_FIXTURE_PATHS: set[str] = set()


def _import_module_from_path(path: Path) -> Any:
    """Import a Python file by absolute path under a private module name.

    ``topofuzz._fixture_<sha1>`` is the module name. The cache key
    includes the file's ``mtime_ns`` so a user editing the fixture
    between two CLI invocations in the same process (or two pytest
    tests sharing a process) sees the *current* content — the previous
    cache-by-path-only behavior returned stale modules (audit M-6).

    Emits a one-time stderr warning on the first load of each fixture
    path because the call performs ``exec_module`` on user-supplied
    code (audit M-5). The user is the one running the CLI, but a stray
    "topofuzz run evil.py:g" via a copy-paste from the internet could
    still own the user; the warning makes the boundary visible.
    """
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Fixture file not found: {path!s} (resolved to {resolved!s})"
        )

    # One-shot per-path warning. Skip when the caller has already loaded
    # this path — pytest reruns share a process, so we don't want every
    # test invocation to spam stderr.
    resolved_str = str(resolved)
    if resolved_str not in _WARNED_FIXTURE_PATHS:
        _WARNED_FIXTURE_PATHS.add(resolved_str)
        print(  # noqa: T201 — intentional stderr surface
            f"topofuzz: loading fixture {resolved!s} — this executes "
            f"arbitrary Python. Only run trusted fixtures.",
            file=sys.stderr,
        )

    mtime_ns = resolved.stat().st_mtime_ns
    digest = hashlib.sha1(
        f"{resolved_str}|{mtime_ns}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    mod_name = f"topofuzz._fixture_{digest}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load fixture module from {resolved!s}: "
            "importlib.util.spec_from_file_location returned None."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Roll back so the next attempt re-executes cleanly.
        sys.modules.pop(mod_name, None)
        raise
    return module


def _split_spec(spec: str) -> tuple[str, str]:
    """Split a fixture spec ``path:attr`` while tolerating Windows drive letters.

    A Windows path ``C:\\Users\\fx.py:graph`` contains TWO colons; the
    first is the drive letter and the second is the attr separator.
    Strip the drive prefix before splitting on the LAST colon (which
    is always the attr separator on either platform).
    """
    drive_prefix = ""
    rest = spec
    if len(spec) >= 2 and spec[1] == ":" and spec[0].isalpha():
        drive_prefix = spec[:2]
        rest = spec[2:]
    if ":" not in rest:
        raise ValueError(
            f"Fixture spec {spec!r} must be of the form 'path/to/file.py:attr'; "
            "missing ':'."
        )
    path_tail, _, attr = rest.rpartition(":")
    path_str = drive_prefix + path_tail
    if not path_str or not attr:
        raise ValueError(
            f"Fixture spec {spec!r} must have both a path and an attribute; "
            f"got path={path_str!r} attr={attr!r}."
        )
    return path_str, attr


def load_topology(spec: str) -> Topology:
    """Parse ``"path:attr"`` and return a Topology.

    ``attr`` may be a compiled LangGraph, or a no-arg callable that
    returns one (e.g. the canonical ``build_graph()`` from
    ``lending_4role.py``). If the module also exports ``LLM_CONFIGS:
    dict[str, LLMConfig]`` it is passed through to ``from_langgraph``.

    Raises:
        ValueError: ``spec`` is missing ``:`` or the attribute is empty.
        FileNotFoundError: the file does not exist.
        AttributeError: the attribute is not on the module.
        TypeError: ``LLM_CONFIGS`` is not a dict of {str: LLMConfig}.
    """
    path_str, attr = _split_spec(spec)
    module = _import_module_from_path(Path(path_str))
    if not hasattr(module, attr):
        raise AttributeError(
            f"Module {path_str!r} has no attribute {attr!r}. "
            f"Available exports: {sorted(n for n in dir(module) if not n.startswith('_'))[:10]}..."
        )
    target = getattr(module, attr)
    compiled = target() if callable(target) else target

    llm_configs_raw = getattr(module, "LLM_CONFIGS", None)
    llm_configs: dict[str, LLMConfig] | None = None
    if llm_configs_raw is not None:
        if not isinstance(llm_configs_raw, dict):
            raise TypeError(
                f"{path_str}:LLM_CONFIGS must be dict[str, LLMConfig]; "
                f"got {type(llm_configs_raw).__name__}."
            )
        for k, v in llm_configs_raw.items():
            if not isinstance(k, str) or not isinstance(v, LLMConfig):
                raise TypeError(
                    f"{path_str}:LLM_CONFIGS contains a bad entry "
                    f"{k!r} -> {type(v).__name__}; both keys and values "
                    "must be str and LLMConfig respectively."
                )
        llm_configs = llm_configs_raw

    name = getattr(module, "__topofuzz_name__", Path(path_str).stem)
    # Step 20a: forward optional ``__topofuzz_verdict_node__`` so
    # ``confirm._reconstruct_variants`` (which regenerates the IR via
    # this loader) round-trips a v0.6 fixture with the same canonical
    # hash as the kickoff run. Dropping the field would silently
    # rebucket every variant_id and break confirm-pass resolution.
    verdict_node = getattr(module, "__topofuzz_verdict_node__", None)
    if verdict_node is not None and not isinstance(verdict_node, str):
        raise TypeError(
            f"{path_str}:__topofuzz_verdict_node__ must be str; got "
            f"{type(verdict_node).__name__}."
        )
    # Step 20a/F6 follow-up: forward optional
    # ``__topofuzz_verdict_label_{true,false}__`` so producer-side
    # custom labels survive the confirm-pass IR regeneration. Same
    # rationale as verdict_node — a dropped label would change the
    # variant canonical_hash and confirm wouldn't find the cells.
    # Type-check at the boundary; defer normalisation + length rules
    # to the IR field-validator so the recipe lives in one place.
    # Red-team pass-1 M5: lock to canonical ``str`` (not subclass) and
    # M6: WARN log if attr is explicitly None so a fixture author who
    # wrote ``__topofuzz_verdict_label_true__ = None`` thinking they
    # were being explicit about "use default" gets a visible signal
    # rather than silent fallback.
    # Local variable named ``dunder_attr`` to avoid shadowing the outer
    # ``attr`` from ``_split_spec`` (red-team pass-2 MAJ-2).
    _label_attr_pairs = [
        ("verdict_label_true", "__topofuzz_verdict_label_true__"),
        ("verdict_label_false", "__topofuzz_verdict_label_false__"),
    ]
    _label_forwards: dict[str, str | None] = {}
    _SENTINEL = object()
    for kwarg, dunder_attr in _label_attr_pairs:
        raw = getattr(module, dunder_attr, _SENTINEL)
        if raw is _SENTINEL:
            _label_forwards[kwarg] = None
            continue
        if raw is None:
            _log.warning(
                "%s:%s = None — using default (APPROVE/DENY). Remove the "
                "attribute entirely to silence this warning, or set it "
                "to a non-empty str (e.g. 'ESCALATE') to opt in.",
                path_str, dunder_attr,
            )
            _label_forwards[kwarg] = None
            continue
        if type(raw) is not str:
            raise TypeError(
                f"{path_str}:{dunder_attr} must be str (not subclass); got "
                f"{type(raw).__name__}."
            )
        _label_forwards[kwarg] = raw
    verdict_label_true = _label_forwards["verdict_label_true"]
    verdict_label_false = _label_forwards["verdict_label_false"]
    return from_langgraph(
        compiled,
        name=name,
        llm_configs=llm_configs,
        verdict_node=verdict_node,
        verdict_label_true=verdict_label_true,
        verdict_label_false=verdict_label_false,
    )


def load_eval_rows(path: str | Path) -> list[dict]:
    """Read ``.jsonl`` or ``.csv`` into a list of dicts.

    Every row must have a non-empty string ``id`` and an ``input`` key.
    Missing-id rows raise with the 1-based line number so a user with a
    1000-row CSV can grep to the bad line.

    JSONL: one JSON object per line; blank lines are skipped.
    CSV: first row is the header; remaining rows are coerced to strings
    (CSV doesn't carry types).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {path!s}")
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows = _load_jsonl(p)
    elif suffix == ".csv":
        rows = _load_csv(p)
    else:
        raise ValueError(
            f"Unsupported eval format {suffix!r} (path={path!s}). "
            "Supported: .jsonl, .csv."
        )
    if not rows:
        raise ValueError(f"Eval file {path!s} is empty (no rows after parse).")
    _validate_eval_rows(rows, path=p)
    return rows


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path!s}:{lineno}: invalid JSON: {e.msg}"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path!s}:{lineno}: expected JSON object, got "
                    f"{type(obj).__name__}."
                )
            rows.append(obj)
    return rows


def _load_csv(path: Path) -> list[dict]:
    # ``utf-8-sig`` transparently strips a leading BOM. Excel exports
    # UTF-8 CSVs with a BOM by default; without ``-sig`` the first
    # header becomes ``"﻿id"`` and every row fails the id check
    # with the confusing "got None" message (audit M-7).
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        # Strip whitespace on header keys so a CSV like ``id, input``
        # (one space after the comma — extremely common from spread-
        # sheet exports) doesn't produce a `" input"` key that the
        # validator then reports as missing (audit M4).
        rows: list[dict] = []
        for raw in reader:
            rows.append(
                {
                    (k.strip() if isinstance(k, str) else k): v
                    for k, v in raw.items()
                }
            )
        return rows


def _validate_eval_rows(rows: list[dict], *, path: Path) -> None:
    seen_ids: set[str] = set()
    for i, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"{path!s} row {i}: 'id' must be a non-empty str; got {row_id!r}."
            )
        if "input" not in row:
            raise ValueError(
                f"{path!s} row {i} (id={row_id!r}): missing required 'input' field."
            )
        if row_id in seen_ids:
            raise ValueError(
                f"{path!s} row {i}: duplicate id {row_id!r}. Eval ids must "
                "be unique — the runner keys cache and persistence on (variant, id)."
            )
        seen_ids.add(row_id)
