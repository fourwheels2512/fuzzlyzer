# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""On-disk storage for user-defined ("BYOP — bring your own pipeline") workflows.

Each custom workflow lives under ``~/.topofuzz/custom-workflows/{workflow_id}/``
as three files:

* ``meta.json``  — the :class:`CustomWorkflow` serialized.
* ``fixture.py`` — generated from the template (linear-chain LangGraph stubs).
* ``rubric.txt`` — user's rubric paragraph wrapped in the gallery format.

Design notes:
  * The on-disk dir is the source of truth — ``meta.json`` is rebuilt on every
    save. Reload via :func:`load_custom_workflow` reconstructs the dataclass.
  * The directory layout mirrors the gallery, so the existing
    ``topofuzz run <fixture>:build_graph --rubric <rubric>`` CLI works
    without any new flags.
  * All writes are atomic — the ``.tmp`` → ``os.replace`` dance prevents a
    half-written ``fixture.py`` from being loaded by a later ``list``.
  * Path containment is enforced on every read. A ``custom:foo/../bar``
    attempt resolves outside the storage root and is refused.
  * Symlinks are refused on read — both the workflow dir itself and the
    files inside it. This stops a confused-deputy where a user pastes a
    custom workflow that secretly points fixture.py at /etc/passwd.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from topofuzz.ui.presets import (
    ALLOWED_CUSTOM_MODEL_KEYS,
    CustomAgent,
    CustomWorkflow,
    StarterScenario,
    Workflow,
)


_log = logging.getLogger(__name__)

# Directory name under ~/.topofuzz/ where custom workflows live.
_CUSTOM_DIR_NAME = "custom-workflows"

# Per-workflow file names. Match the gallery layout so the CLI works as-is.
_META_FILENAME = "meta.json"
_FIXTURE_FILENAME = "fixture.py"
_RUBRIC_FILENAME = "rubric.txt"

# workflow_id format: YYYYMMDDThhmmss-<8hex>. Same shape as run_id so the
# format is one fact in the codebase, not two.
_WORKFLOW_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$")


_FORBIDDEN_HOME_PREFIXES = (
    # System dirs we refuse to use as TOPOFUZZ_HOME even if the env var
    # points there. Prevents an attacker-influenced env from causing the
    # writer to drop fixture.py into a system-write location.
    Path("/etc"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/boot"),
    Path("/sys"),
    Path("/proc"),
    Path("C:/Windows"),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
    Path("C:/ProgramData"),
)


def _validate_topofuzz_home(override: str) -> Path | None:
    """Return a sanitized Path for TOPOFUZZ_HOME, or None to fall back to ~.

    Refuses paths that resolve under a known system dir. Auto-creation is
    fine — same as ``~/.topofuzz/`` — so we don't require the dir to
    pre-exist; the system-path block is the load-bearing security check.
    """
    candidate = Path(override).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        _log.warning("TOPOFUZZ_HOME=%r could not be resolved; ignoring", override)
        return None
    for forbidden in _FORBIDDEN_HOME_PREFIXES:
        try:
            forbidden_real = forbidden.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        try:
            resolved.relative_to(forbidden_real)
        except ValueError:
            continue
        _log.warning(
            "TOPOFUZZ_HOME=%r resolves under system dir %s; ignoring",
            override,
            forbidden,
        )
        return None
    return resolved


def custom_workflows_root() -> Path:
    """Directory holding every saved custom workflow.

    Honors ``$TOPOFUZZ_HOME`` if set AND it sanitizes to a real, non-system
    path; otherwise falls back to ``~/.topofuzz/``. Creates the
    custom-workflows subdir on first call so callers never have to
    ``mkdir(exist_ok=True)``.
    """
    override = os.environ.get("TOPOFUZZ_HOME", "").strip()
    base: Path | None = None
    if override:
        base = _validate_topofuzz_home(override)
    if base is None:
        home = Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
        base = home / ".topofuzz"
    root = base / _CUSTOM_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_workflow_id() -> str:
    """Generate a fresh workflow id. Server-side only — never accept from form."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}-{suffix}"


def is_valid_workflow_id(workflow_id: str) -> bool:
    """Strict regex check on workflow_id format. Used as a path-injection guard."""
    return bool(_WORKFLOW_ID_RE.match(workflow_id or ""))


def workflow_dir(workflow_id: str) -> Path:
    """Resolve a workflow_id to its directory, with path-containment guard.

    Raises :class:`ValueError` if the id has an invalid format OR if its
    resolved path escapes ``custom_workflows_root()`` (defense in depth
    against a malformed id slipping past the regex).
    """
    if not is_valid_workflow_id(workflow_id):
        raise ValueError(f"invalid workflow_id: {workflow_id!r}")
    root = custom_workflows_root()
    candidate = root / workflow_id
    # Path containment: even though the regex forbids ``/`` and ``..``,
    # belt-and-suspenders. A symlinked root could otherwise bypass.
    try:
        candidate_real = candidate.resolve(strict=False)
        root_real = root.resolve(strict=False)
        candidate_real.relative_to(root_real)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"workflow_id {workflow_id!r} resolves outside custom workflows root"
        ) from exc
    return candidate


def _atomic_write(path: Path, content: str) -> None:
    """Write text to ``path`` via a sibling ``.tmp`` + ``os.replace``.

    Either the file is written entirely, or the previous version (if any)
    survives untouched. Crash mid-write leaves no half-baked fixture.py.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _safe_to_load(dir_path: Path) -> bool:
    """Reject obviously unsafe workflow dirs before reading them.

    Specifically:
      * The dir itself must not be a symlink (could escape root).
      * fixture.py and rubric.txt, if present, must not be symlinks
        (could secretly redirect the runner to system files).
    """
    if dir_path.is_symlink():
        return False
    for name in (_FIXTURE_FILENAME, _RUBRIC_FILENAME, _META_FILENAME):
        candidate = dir_path / name
        # is_symlink() is fine on a missing path — returns False.
        if candidate.is_symlink():
            return False
    return True


def save_custom_workflow(cw: CustomWorkflow) -> Workflow:
    """Persist a CustomWorkflow to disk and return its :class:`Workflow`.

    Writes ``meta.json`` + ``fixture.py`` + ``rubric.txt`` atomically. The
    returned Workflow's ``key`` is ``custom:{workflow_id}`` so it can never
    collide with a built-in.

    Caller is responsible for validating the inputs (agent_id format,
    model allow-list, agent count, rubric length). See
    :func:`topofuzz.ui.custom_validation.validate_custom_workflow_form`.
    """
    if not is_valid_workflow_id(cw.workflow_id):
        raise ValueError(f"invalid workflow_id: {cw.workflow_id!r}")
    target = workflow_dir(cw.workflow_id)
    target.mkdir(parents=True, exist_ok=True)
    # Explicit write — to_workflow() is now pure (no I/O), so save must
    # call write_workflow_files itself. Previously the side effect lived
    # inside to_workflow() and fired on every get_workflow lookup.
    write_workflow_files(cw, target)
    return cw.to_workflow(target)


def list_custom_workflows() -> list[CustomWorkflow]:
    """Return every saved custom workflow, newest ``created_at`` first.

    Robust to corruption: a dir with a missing or unparseable ``meta.json``,
    or one that fails the symlink check, is logged at WARNING and skipped.
    The return value is always a (possibly empty) list — never raises.
    """
    root = custom_workflows_root()
    out: list[CustomWorkflow] = []
    try:
        candidates = sorted(root.iterdir())
    except OSError as exc:
        # Permission denied / disk error / mount broken — surface as a
        # warning. Previously swallowed silently; the user saw an empty
        # list with no hint that the dir was unreadable.
        _log.warning(
            "could not enumerate custom workflows root %s: %s", root, exc
        )
        return out
    for entry in candidates:
        if not entry.is_dir():
            continue
        if not is_valid_workflow_id(entry.name):
            # An unrelated dir snuck in — ignore quietly.
            continue
        if not _safe_to_load(entry):
            _log.warning(
                "skipping custom workflow %s: failed symlink/safety check",
                entry.name,
            )
            continue
        cw = _load_meta(entry)
        if cw is None:
            continue
        out.append(cw)
    out.sort(key=lambda c: c.created_at, reverse=True)
    return out


def load_custom_workflow(workflow_id: str) -> CustomWorkflow | None:
    """Load a single custom workflow by id, or None if missing/corrupt."""
    if not is_valid_workflow_id(workflow_id):
        return None
    try:
        target = workflow_dir(workflow_id)
    except ValueError:
        return None
    if not target.exists() or not target.is_dir():
        return None
    if not _safe_to_load(target):
        _log.warning(
            "load_custom_workflow %s: failed symlink/safety check", workflow_id
        )
        return None
    return _load_meta(target)


def delete_custom_workflow(workflow_id: str) -> bool:
    """Remove the workflow's dir. Returns True if it existed and was removed.

    Refuses to delete if the dir would resolve outside the storage root
    (path-containment guard) or contains a symlink (we never follow it).
    Walks the directory manually rather than using ``shutil.rmtree`` —
    rmtree on Windows historically follows directory junctions.
    """
    if not is_valid_workflow_id(workflow_id):
        return False
    try:
        target = workflow_dir(workflow_id)
    except ValueError:
        return False
    if not target.exists():
        return False
    if target.is_symlink():
        # Top-level dir IS a symlink — refuse rather than follow. The
        # underlying target stays untouched; the user can clean it up by
        # hand if it really is a stale link they want gone.
        _log.warning(
            "delete_custom_workflow %s: dir is a symlink; refusing to follow",
            workflow_id,
        )
        return False
    if not _safe_to_load(target):
        # Some inner file is a symlink — do not unlink it (could escape
        # the storage root). Surface the failure.
        _log.warning(
            "delete_custom_workflow %s: contains a symlinked file; refusing",
            workflow_id,
        )
        return False
    try:
        for entry in target.iterdir():
            if entry.is_symlink():
                # Belt-and-suspenders against a symlink racing in between
                # _safe_to_load and now (TOCTOU window).
                _log.warning(
                    "delete_custom_workflow %s: skipping symlink %s",
                    workflow_id,
                    entry.name,
                )
                continue
            if entry.is_file():
                entry.unlink()
            elif entry.is_dir():
                # Custom workflow dirs are flat — no nested subdirs in the
                # spec. If one exists (corrupt state), refuse rather than
                # recurse blindly.
                _log.warning(
                    "delete_custom_workflow %s: unexpected nested dir %s; "
                    "refusing to delete recursively",
                    workflow_id,
                    entry.name,
                )
                return False
        target.rmdir()
        return True
    except OSError as exc:
        _log.warning("delete_custom_workflow %s: %s", workflow_id, exc)
        return False


def _load_meta(dir_path: Path) -> CustomWorkflow | None:
    """Parse ``meta.json`` into a :class:`CustomWorkflow` or return None."""
    meta_path = dir_path / _META_FILENAME
    if not meta_path.exists():
        _log.warning("custom workflow %s: missing meta.json", dir_path.name)
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning(
            "custom workflow %s: meta.json unreadable: %s", dir_path.name, exc
        )
        return None
    try:
        agents_raw = raw.get("agents") or []
        agents = tuple(
            CustomAgent(agent_id=a["agent_id"], model=a["model"]) for a in agents_raw
        )
        # Defensive: skip if any agent lists a model outside the allow-list.
        # The data could have been hand-edited on disk.
        for a in agents:
            if a.model not in ALLOWED_CUSTOM_MODEL_KEYS:
                _log.warning(
                    "custom workflow %s: agent %s uses disallowed model %s",
                    dir_path.name,
                    a.agent_id,
                    a.model,
                )
                return None
        starters_raw = raw.get("starter_scenarios") or []
        starters = tuple(
            StarterScenario(
                case=s.get("case", ""),
                why_borderline=s.get("why_borderline", ""),
                expected=s.get("expected", "deny"),
            )
            for s in starters_raw
        )
        cw = CustomWorkflow(
            workflow_id=raw["workflow_id"],
            label=raw["label"],
            domain_noun=raw["domain_noun"],
            decision_verb=raw["decision_verb"],
            agents=agents,
            verdict_agent_id=raw["verdict_agent_id"],
            rubric_paragraph=raw["rubric_paragraph"],
            created_at=float(raw["created_at"]),
            starter_scenarios=starters,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _log.warning(
            "custom workflow %s: meta.json malformed: %s", dir_path.name, exc
        )
        return None
    if not is_valid_workflow_id(cw.workflow_id):
        # Should never happen if save_custom_workflow was used.
        return None
    return cw


def serialize_meta(cw: CustomWorkflow) -> str:
    """Return the JSON text written to meta.json. Pure — no I/O."""
    payload = {
        "workflow_id": cw.workflow_id,
        "label": cw.label,
        "domain_noun": cw.domain_noun,
        "decision_verb": cw.decision_verb,
        "agents": [asdict(a) for a in cw.agents],
        "verdict_agent_id": cw.verdict_agent_id,
        "rubric_paragraph": cw.rubric_paragraph,
        "created_at": cw.created_at,
        "starter_scenarios": [asdict(s) for s in cw.starter_scenarios],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_workflow_files(cw: CustomWorkflow, target: Path) -> None:
    """Atomically write meta.json + fixture.py + rubric.txt under ``target``.

    Used by :meth:`CustomWorkflow.to_workflow` — kept here so the "what
    files live in a custom workflow dir" knowledge is centralized.
    """
    from topofuzz.ui.custom_templates import (
        render_fixture_py,
        render_rubric_txt,
    )

    target.mkdir(parents=True, exist_ok=True)
    _atomic_write(target / _META_FILENAME, serialize_meta(cw))
    _atomic_write(target / _FIXTURE_FILENAME, render_fixture_py(cw))
    _atomic_write(target / _RUBRIC_FILENAME, render_rubric_txt(cw))
