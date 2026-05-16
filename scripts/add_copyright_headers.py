"""Idempotent copyright-header inserter for the BSL-licensed packages.

Closes ``packages/topofuzz/BACKLOG.md`` entry "MINOR -- Copyright + ``(C)``
notice headers in source files". Walks the four watched directories and
prepends a 4-line header (3-line copyright/license/LICENSE pointer +
1-line patent-pending placeholder) to every ``.py`` file whose canonical
header block is not already present. Preserves shebangs and PEP 263
encoding declarations by inserting after them. Module docstrings stay
as the first executable construct after the header.

The patent-pending marker is the literal placeholder

    # <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

A future agent replaces this line in-place once a provisional patent
application number is assigned (post-funding). The marker is deliberately
recognizable so a single grep finds every line that needs updating.

Run modes:

* ``python scripts/add_copyright_headers.py`` -- apply mode (default).
  Inserts headers, prints a summary, exits 0 if everything is consistent
  afterwards.
* ``python scripts/add_copyright_headers.py --check`` -- CI mode. Touches
  no files; exits 0 if every watched ``.py`` file has the FULL canonical
  header block, exits 1 otherwise (listing the offenders on stderr).

Year handling (BACKLOG-acknowledged tradeoff): the header uses literal
2026, which is correct for the v0.6 publication year. Files added in a
later calendar year should declare year-of-first-publication per BACKLOG
spec; future contributors will need to bump ``COPYRIGHT_HEADER_LINES``
or parameterize the year before v0.7 ships. The single-source COPY in
this script keeps the bump to one edit when that happens.

The header lines are exported as ``COPYRIGHT_HEADER_LINES`` for the
pytest suite at ``packages/topofuzz/tests/test_copyright_headers.py``,
which exercises every branch (shebang vs no-shebang, PEP 263 encoding
declaration, idempotent re-run, empty file, docstring preservation,
canonical-block enforcement, CRLF refusal) on synthetic ``tmp_path``
files without mutating the repo.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WATCHED_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "packages" / "topofuzz" / "src" / "topofuzz",
    REPO_ROOT / "packages" / "topofuzz" / "tests",
    REPO_ROOT / "packages" / "fuzzeval" / "src" / "fuzzeval",
    REPO_ROOT / "packages" / "fuzzeval" / "tests",
)

COPYRIGHT_HEADER_LINES: tuple[str, ...] = (
    "# Copyright (c) 2026 Kiran Nayudu. All rights reserved.",
    "# Licensed under the Business Source License 1.1.",
    "# See LICENSE file in the project root for full terms.",
    "# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->",
)

# Fast-path sentinel used to short-circuit a no-op apply. Detection of
# spec compliance for `--check` uses `header_is_canonical` below, which
# verifies the FULL block byte-for-byte. The two-tier design lets apply
# stay cheap while CI catches stale / wrong-year / paraphrased headers
# that a sentinel-only check would silently accept (pass-1 RT MAJOR-1).
HEADER_SENTINEL = "# Copyright (c)"

# PEP 263 encoding declarations must live in the first or second line.
# When present, the header inserts AFTER the encoding line so the
# directive remains effective; otherwise Python would silently fall
# back to UTF-8 and editors that honor the declaration would misread
# the file. The same idea applies to ``#!`` shebangs.
# Per PEP 263: must be a comment line (leading ``#``) with the
# ``coding`` directive. The leading ``^[ \t\v]*#`` anchor prevents
# false positives on prose comments like
# ``# This is the coding: utf-8 explanation`` that happen to contain
# the literal text. (Pass-2 RT MAJOR-1.)
_PEP263_RE = re.compile(rb"^[ \t\v]*#.*?coding[=:][ \t]*([-_.a-zA-Z0-9]+)")


def iter_python_files(roots: tuple[Path, ...] = WATCHED_DIRS) -> list[Path]:
    """Return sorted list of `.py` files under the watched roots.

    Skips `__pycache__` directories. Sort yields stable ordering so the
    CI failure list is reproducible across machines.
    """
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def _preamble_end(lines: list[str]) -> int:
    """Index of the first line that is NOT a shebang or PEP 263 encoding line.

    PEP 263 allows the encoding declaration on line 1 or line 2; when a
    shebang occupies line 1 the encoding line MAY be on line 2. We
    accept both layouts and treat them as preamble that the header
    inserts after.
    """
    if not lines:
        return 0
    idx = 0
    if lines[idx].startswith("#!"):
        idx += 1
    if idx < len(lines) and _PEP263_RE.search(lines[idx].encode("utf-8")):
        idx += 1
    return idx


def has_header(text: str) -> bool:
    """Cheap sentinel check used by apply-mode to short-circuit no-ops.

    Returns True when the first line after any shebang / PEP 263
    encoding declaration begins with ``# Copyright (c)``. Does NOT
    verify the rest of the block matches the canonical lines -- use
    ``header_is_canonical`` for that strict check.
    """
    lines = text.splitlines()
    idx = _preamble_end(lines)
    if idx >= len(lines):
        return False
    return lines[idx].startswith(HEADER_SENTINEL)


def header_is_canonical(text: str) -> bool:
    """Strict check: the full 4-line block matches ``COPYRIGHT_HEADER_LINES``.

    Used by ``--check`` so a file with a stale year, paraphrased license
    name, or wrong copyright holder is flagged as an offender instead of
    silently passing. (Pass-1 RT MAJOR-1: the sentinel-only check would
    accept ``# Copyright (c) 1999 SomeoneElse`` as headered, defeating
    the BSL hardening intent.)
    """
    lines = text.splitlines()
    idx = _preamble_end(lines)
    if idx + len(COPYRIGHT_HEADER_LINES) > len(lines):
        return False
    for offset, expected in enumerate(COPYRIGHT_HEADER_LINES):
        if lines[idx + offset] != expected:
            return False
    return True


def insert_header(text: str) -> str:
    """Return ``text`` with the 4-line header inserted at the canonical slot.

    Idempotent: if ``has_header(text)`` is already True, returns ``text``
    unchanged byte-for-byte.

    Slot rules (per BACKLOG entry 2):
      * Empty file -> just the header (with trailing newline).
      * File starts with ``#!shebang`` -> header inserts after it.
      * File starts with a PEP 263 encoding declaration (or has one on
        line 2 after a shebang) -> header inserts after the encoding
        line so the directive remains effective.
      * Otherwise header goes on line 1..4.
      * One blank line always separates header from existing content,
        so module docstrings remain the first executable construct.
    """
    if has_header(text):
        return text

    header_block = "\n".join(COPYRIGHT_HEADER_LINES) + "\n"

    if not text:
        return header_block

    lines = text.split("\n")
    idx = _preamble_end(lines)
    if idx == 0:
        return header_block + "\n" + text
    preamble = "\n".join(lines[:idx]) + "\n"
    rest = "\n".join(lines[idx:])
    return preamble + header_block + "\n" + rest


class CRLFBodyError(RuntimeError):
    """Raised when a file body contains CRLF line endings.

    The repo's ``.gitattributes`` pins ``*.py`` to ``eol=lf``. Writing a
    file with a mix of LF (new header) and CRLF (existing body) would
    silently produce a mixed-ending file that confuses ``git blame``
    and lint. We refuse to write and point the contributor at the
    canonical fix instead of papering over it (pass-1 RT MAJOR-2).
    """


class StaleHeaderError(RuntimeError):
    """Raised when a file already has a ``# Copyright (c)`` line that is
    NOT the canonical block.

    The sentinel matches but the full block does not -- typical causes
    are a stale year (``# Copyright (c) 2025 ...``), a vendored snippet
    from a third party (``# Copyright (c) 2018 SomeOther ...``), or a
    paraphrased license name. Silently rewriting would either lose the
    original author's notice (legally risky) or papered-over the spec
    violation. Refuse to write and require the contributor to resolve
    by hand (pass-2 RT BLOCKER-1).
    """


def has_crlf_body(path: Path) -> bool:
    """True if ``path`` has any ``\\r\\n`` byte sequence on disk.

    Used by ``--check`` so a CRLF-bodied file is surfaced as an
    offender even when ``read_text`` universal-newline normalization
    would otherwise hide the issue (pass-2 RT MINOR-1).
    """
    return b"\r\n" in path.read_bytes()


def apply_to_file(path: Path) -> bool:
    """Insert the header into ``path`` if missing. Returns True if file changed.

    Reads and writes bytes with explicit LF line endings so the file
    matches the repo-wide ``.gitattributes eol=lf`` policy regardless
    of the host OS.

    Raises:
      ``CRLFBodyError`` -- body has CRLF endings; rewriting would
      mix line endings. Contributor must fix ``core.autocrlf`` first.
      ``StaleHeaderError`` -- a non-canonical ``# Copyright (c)`` line
      is already present. Contributor must reconcile by hand
      (replace the stale line with ``COPYRIGHT_HEADER_LINES[0]`` or
      delete the foreign notice if rights were re-cleared).
    """
    raw = path.read_bytes()
    if b"\r\n" in raw:
        raise CRLFBodyError(
            f"{path}: file body uses CRLF line endings; .gitattributes "
            f"requires LF for *.py. Run `git rm --cached -- <file>` and "
            f"re-add after setting `core.autocrlf=input`, or convert with "
            f"`dos2unix`. Refusing to write to avoid creating a mixed-ending file."
        )
    text = raw.decode("utf-8")
    if header_is_canonical(text):
        return False  # nothing to do
    if has_header(text):
        raise StaleHeaderError(
            f"{path}: file already has a `# Copyright (c)` line that does "
            f"not match the canonical block. Edit by hand to reconcile "
            f"(or delete the foreign notice if rights were re-cleared), "
            f"then re-run this script. Refusing to overwrite a third-party "
            f"copyright notice automatically."
        )
    updated = insert_header(text)
    path.write_bytes(updated.encode("utf-8"))
    return True


def missing_header_files(roots: tuple[Path, ...] = WATCHED_DIRS) -> list[Path]:
    """Return watched `.py` files that lack the canonical header block
    or have a non-LF body.

    Uses the strict ``header_is_canonical`` check so a stale or
    paraphrased header counts as missing -- the whole point of the BSL
    hardening pass is that the notice is correct, not just present in
    some form. Also surfaces CRLF-bodied files because apply-mode
    refuses to write them: the two modes must agree on what counts as
    "clean" (pass-2 RT MINOR-1).
    """
    offenders: list[Path] = []
    for p in iter_python_files(roots):
        if has_crlf_body(p):
            offenders.append(p)
            continue
        if not header_is_canonical(p.read_text(encoding="utf-8")):
            offenders.append(p)
    return offenders


def _display(p: Path) -> str:
    """Render ``p`` relative to repo root when possible; else as-is.

    Forward-slashes regardless of host OS so CI output (Linux) and
    local Windows output match grep-wise. Falls back to ``str(p)``
    when the path is outside ``REPO_ROOT`` (test fixtures using
    ``tmp_path``).
    """
    try:
        rel = str(p.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(p)
    return rel.replace("\\", "/")


_EPILOG = (
    "examples:\n"
    "  python scripts/add_copyright_headers.py            # apply mode (fix)\n"
    "  python scripts/add_copyright_headers.py --check    # CI mode\n"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] if __doc__ else "",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not modify files; exit 1 if any watched `.py` lacks the "
            "canonical header. Run without --check to fix."
        ),
    )
    args = parser.parse_args(argv)

    # Look up WATCHED_DIRS at call time so test-side monkeypatching of
    # the module attribute reaches the helpers. Default-arg binding
    # captures WATCHED_DIRS at `def`-time, which a monkeypatch on the
    # module global would not affect.
    roots = WATCHED_DIRS

    if args.check:
        offenders = missing_header_files(roots)
        if offenders:
            print(
                f"copyright-header check FAILED: {len(offenders)} file(s) missing or non-canonical header:",
                file=sys.stderr,
            )
            for p in offenders:
                print(f"  {_display(p)}", file=sys.stderr)
            print(
                "\nRun `python scripts/add_copyright_headers.py` to fix.",
                file=sys.stderr,
            )
            return 1
        print(f"copyright-header check passed: {len(iter_python_files(roots))} .py file(s) OK")
        return 0

    files = iter_python_files(roots)
    changed: list[Path] = []
    crlf_offenders: list[Path] = []
    stale_offenders: list[Path] = []
    for p in files:
        try:
            if apply_to_file(p):
                changed.append(p)
        except CRLFBodyError as exc:
            crlf_offenders.append(p)
            print(f"  ! {exc}", file=sys.stderr)
        except StaleHeaderError as exc:
            stale_offenders.append(p)
            print(f"  ! {exc}", file=sys.stderr)

    skipped = len(crlf_offenders) + len(stale_offenders)
    untouched = len(files) - len(changed) - skipped
    print(
        f"copyright-headers: {len(changed)} updated, {untouched} already had header, "
        f"{len(crlf_offenders)} skipped (CRLF), {len(stale_offenders)} skipped (stale header)"
    )
    for p in changed:
        print(f"  + {_display(p)}")

    if crlf_offenders or stale_offenders:
        return 3

    leftover = missing_header_files(roots)
    if leftover:
        print(
            f"\nINTERNAL ERROR: {len(leftover)} file(s) still missing canonical header after apply:",
            file=sys.stderr,
        )
        for p in leftover:
            print(f"  {_display(p)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
