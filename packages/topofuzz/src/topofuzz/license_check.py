# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""First-run license click-through acknowledgment for the topofuzz CLI.

BACKLOG entry "MAJOR — First-run CLI license click-through acknowledgment"
(2026-05-16). On first invocation of any topofuzz CLI command (except
``topofuzz verify``, which is the buyer-side audit tool), the CLI prints
a short BSL summary and prompts for acknowledgment. The acceptance is
persisted to ``~/.topofuzz/license-accepted`` as JSON; subsequent runs
do not re-prompt. A ``--accept-license`` flag bypasses the prompt for
CI / unattended use. Version-mismatch (a later topofuzz minor that
might re-license under modified BSL) re-prompts.

This module never sends acceptance data to a server. The
``~/.topofuzz/license-accepted`` file is local-only — see PRIVACY.md
"License-Acceptance Telemetry" (= "None").
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from topofuzz import __version__

# License-summary text printed to the operator on first run. Kept short
# (~10 lines): the full BSL body is in the LICENSE file at the repo
# root. The contact email is the same placeholder used in SECURITY.md /
# PRIVACY.md; user substitutes before launch.
LICENSE_SUMMARY: str = (
    "Topofuzz is licensed under the Business Source License 1.1 (BSL).\n"
    "Non-production use, internal development, academic research, and\n"
    "small-scale use (annual revenue under USD $100,000 from any use of\n"
    "Topofuzz) are permitted without charge. Production use above this\n"
    "threshold, or offering Topofuzz as a managed service, requires a\n"
    "commercial license — contact kiran@modelbrew.ai.\n"
    "Each version converts to the Apache License 2.0 four years after\n"
    "its release date. Full license text in the LICENSE file in your\n"
    "installation directory.\n"
)

PROMPT: str = "Acknowledge these terms? [y/N]: "

# §A error-taxonomy alignment: ``cli.py`` defines ``_BAD_USAGE_EXIT = 2``
# and the BACKLOG spec ties non-interactive refusal to "exit code 2 ...
# matches existing CLI exit-code contract." Re-stated as a module-level
# constant so tests can assert against the same value the wiring uses.
# Note: this code intentionally collides with _BAD_USAGE_EXIT,
# _RESERVED_FLAG_EXIT, _NOT_IMPLEMENTED_EXIT (all also 2). The collision
# is per spec; CI must discriminate on the stderr message, not the code.
LICENSE_REFUSED_EXIT_CODE: int = 2

# Public so the SECURITY.md / PRIVACY.md regression tests and any
# operator tooling can import the tag rather than re-typing the literal
# string. Stored verbatim in the acceptance JSON file.
#
# Stability: this value MAY change on a future re-license (BUSL-1.1 →
# custom EULA, etc.). External callers should treat it as a runtime
# constant for comparison against the acceptance file's `license` key
# — DO NOT pin against the literal "BUSL-1.1" in third-party code.
LICENSE_TAG: str = "BUSL-1.1"


class HomeUnresolvableError(RuntimeError):
    """``Path.home()`` could not be resolved.

    Raised when neither ``HOME`` (POSIX) nor ``USERPROFILE`` (Windows)
    is set — e.g., a container entrypoint that strips env, ``su`` without
    ``-l``, or a misconfigured CI runner. The CLI catches this and emits
    a clean actionable message instead of a Python traceback.
    """


def acceptance_path() -> Path:
    """Return the per-user acceptance file path.

    ``Path.home()`` honours ``USERPROFILE`` on Windows and ``HOME`` on
    POSIX — both are the OS-native "user home" sources. We deliberately
    do NOT read ``os.environ["HOME"]`` directly so Windows users get a
    valid path without setting HOME by hand.

    Raises:
        HomeUnresolvableError: neither ``HOME`` nor ``USERPROFILE`` is
            set (Path.home() raises RuntimeError).
    """
    try:
        home = Path.home()
    except RuntimeError as e:
        raise HomeUnresolvableError(str(e)) from e
    return home / ".topofuzz" / "license-accepted"


def _read_acceptance(path: Path) -> dict | None:
    """Read and JSON-decode the acceptance file. Return ``None`` on any
    failure (missing / corrupt / non-dict)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _is_accepted(data: dict | None) -> bool:
    """A stored acceptance counts only when the version matches the
    currently-installed topofuzz version AND the license tag matches.

    Rationale: if v0.7 ships under a modified BSL (different threshold,
    different conversion timeline, etc.), the prior acceptance is for a
    different contract. Re-prompt rather than silently carry over.
    """
    if data is None:
        return False
    if data.get("license") != LICENSE_TAG:
        return False
    if data.get("version") != __version__:
        return False
    return True


def _write_acceptance(path: Path) -> None:
    """Persist an acceptance record. Atomic write + chmod 0o600 on POSIX.

    Atomic write: data is staged to a sibling ``.tmp`` in the same
    directory, ``os.replace``-d into place, so a crash mid-write never
    leaves a half-written acceptance file that subsequent runs might
    decode as ``None`` and silently re-prompt.

    Best-effort race minimization (NOT elimination): just before the
    rename, re-check whether the destination already contains a valid
    acceptance for the current version+tag (a parallel CLI process may
    have raced us). If so, skip the rename — typically preserving the
    FIRST acceptance's ``accepted_at`` timestamp. A narrow TOCTOU
    window remains between the recheck and ``os.replace``; we accept
    the residual race because (a) the payload is identical except for
    timestamp, (b) BSL evidence requirements only need ONE acceptance
    record per machine, not "the first". For atomic
    first-acceptance-wins semantics a future change could open dest
    with ``O_CREAT | O_EXCL`` and retry; out of scope for v0.6.

    chmod 0o600: only meaningful on POSIX. On Windows ``os.chmod`` is a
    no-op (NTFS ACL is inherited from the parent ``%USERPROFILE%``
    directory, which is already user-only). We still call it for parity
    with the POSIX path.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": __version__,
        # Microsecond precision per ISO 8601. "+00:00" → "Z" suffix is
        # canonical UTC notation; some legal evidence trails expect Z.
        "accepted_at": (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        ),
        "license": LICENSE_TAG,
    }
    # Pretty-print so the file is human-readable if a security
    # reviewer opens it. Trailing newline so editors that strip-on-save
    # don't show a no-newline-at-eof warning.
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    # tempfile.mkstemp creates with mode 0o600 atomically (no race
    # window) on POSIX. ``dir=parent`` forces same filesystem so
    # ``os.replace`` is atomic on POSIX and on Windows when source +
    # dest are on the same volume.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".license-accepted.", suffix=".tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(serialized)
        # chmod the tmp file first, then rename — keeps the destination
        # never visible with looser perms even for a microsecond.
        # (mkstemp already sets 0o600, but be explicit in case future
        # changes to umask or platform semantics change the default.)
        try:
            os.chmod(tmp_name, 0o600)
        except (OSError, NotImplementedError):
            # Windows: chmod cannot enforce POSIX bits on NTFS. The
            # file inherits the user-only ACL from the parent dir.
            pass
        # Last-second race re-check. If another process already wrote
        # a valid acceptance for this version+tag, leave it alone —
        # their ``accepted_at`` is the canonical first-consent moment.
        if _is_accepted(_read_acceptance(path)):
            return
        os.replace(tmp_name, path)
    finally:
        # Cleanup the tmp on every exit path:
        # * Success: os.replace consumed it → unlink raises OSError, swallow.
        # * Race-loss (early return): tmp still exists, remove it.
        # * Exception (including KeyboardInterrupt): tmp may be partial,
        #   remove it before propagating.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def is_accepted() -> bool:
    """Read-only check: does a valid acceptance record exist right now?

    Never prompts and never writes — the sibling of :func:`prompt_and_record`
    for callers that must not touch stdin (e.g. the UI server process
    deciding whether to show the in-browser accept gate before a run
    can be launched; see ``topofuzz.ui.app``'s ``/accept-license`` route).

    ``HomeUnresolvableError`` (no HOME/USERPROFILE) is treated as "not
    accepted" rather than propagated — a server-side caller should
    degrade to showing the accept gate, never crash.
    """
    try:
        path = acceptance_path()
    except HomeUnresolvableError:
        return False
    return _is_accepted(_read_acceptance(path))


def record_acceptance() -> None:
    """Persist an acceptance record unconditionally, without prompting.

    Equivalent to answering ``y`` at the interactive prompt or passing
    ``--accept-license``, minus the CLI flag/stdin plumbing — the entry
    point for the UI's in-browser "I accept" control.
    """
    _write_acceptance(acceptance_path())


def prompt_and_record(
    *,
    force: bool = False,
    non_interactive: bool = False,
) -> bool:
    """Check (or establish) license acceptance.

    Returns True if the contract is accepted, False otherwise.

    Branch ordering is load-bearing:

    1. Already-accepted check first → silent success on the steady-state
       path. This is the common case after first invocation; the rest of
       the function is for first run and version-rollover scenarios.
    2. ``force`` (i.e. ``--accept-license``) BEFORE ``non_interactive``.
       The CI / unattended path is "no TTY but I want to accept" — the
       flag must override the non-interactive refusal, else CI is
       perma-broken even with the documented bypass.
    3. ``non_interactive`` without force → refuse silently. Caller emits
       the actionable "rerun with --accept-license or interactively" hint
       and exits 2.
    4. Interactive prompt last. EOF (stdin closed mid-prompt) is a
       refusal with a clear message so the operator does not blame an
       unrelated bug.
    """
    path = acceptance_path()
    if _is_accepted(_read_acceptance(path)):
        return True
    # Order matters: force wins over non_interactive (see docstring).
    if force:
        _write_acceptance(path)
        return True
    if non_interactive:
        return False
    # Interactive prompt. Write to stderr so callers can pipe stdout
    # cleanly (consistent with how scan/run print banners to stderr).
    sys.stderr.write(LICENSE_SUMMARY)
    sys.stderr.write("\n")
    sys.stderr.write(PROMPT)
    sys.stderr.flush()
    try:
        answer = input()
    except EOFError:
        # stdin closed mid-prompt (uncommon but possible — e.g., piping
        # an empty file, or running under a redirected-stdin harness
        # that did not pass --accept-license). Treat as refusal AND
        # surface a clear diagnostic so the operator does not blame an
        # unrelated bug.
        sys.stderr.write(
            "\ntopofuzz: stdin closed before y/N answer — treating as refusal.\n"
        )
        return False
    if answer.strip().lower() in ("y", "yes"):
        _write_acceptance(path)
        return True
    return False
