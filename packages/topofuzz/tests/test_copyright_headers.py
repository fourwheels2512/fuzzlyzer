# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``scripts/add_copyright_headers.py``.

Two responsibilities:

1. **Unit tests** against ``tmp_path`` exercise the header-insertion
   logic in isolation: idempotency, shebang preservation, docstring
   preservation, empty-file handling, CLI ``--check`` exit codes.

2. **Repo invariant** — every watched ``.py`` file already has the
   header at commit time. This is the same check ``--check`` runs in
   CI; duplicating it in pytest gives every contributor's local
   ``pytest`` an early warning before they push.

The script is imported as a normal module via ``importlib`` because it
lives at ``scripts/add_copyright_headers.py``, outside any installed
package. ``REPO_ROOT`` is resolved from this test file's location.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "add_copyright_headers.py"


def _load_script_module():
    """Import scripts/add_copyright_headers.py as a fresh module object."""
    spec = importlib.util.spec_from_file_location(
        "add_copyright_headers_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script_module()


def test_header_constant_matches_backlog_spec(script):
    # BACKLOG entry 2 ("Copyright + (C) notice headers in source files")
    # locks the exact three lines plus the patent-pending marker. If
    # this test fails, either the spec changed (update both) or someone
    # edited the constant by accident.
    assert script.COPYRIGHT_HEADER_LINES == (
        "# Copyright (c) 2026 Kiran Nayudu. All rights reserved.",
        "# Licensed under the Business Source License 1.1.",
        "# See LICENSE file in the project root for full terms.",
        "# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->",
    )
    assert script.HEADER_SENTINEL == "# Copyright (c)"


def test_insert_into_empty_file(script):
    result = script.insert_header("")
    assert result == "\n".join(script.COPYRIGHT_HEADER_LINES) + "\n"


def test_insert_preserves_docstring_first_class(script):
    src = '"""Module docstring."""\n\nimport os\n'
    result = script.insert_header(src)
    lines = result.split("\n")
    # Header occupies lines 0..3; blank separator on line 4; docstring on line 5.
    for i, header_line in enumerate(script.COPYRIGHT_HEADER_LINES):
        assert lines[i] == header_line
    assert lines[4] == ""
    assert lines[5] == '"""Module docstring."""'


def test_insert_preserves_shebang(script):
    src = "#!/usr/bin/env python\n\"\"\"With shebang.\"\"\"\n"
    result = script.insert_header(src)
    lines = result.split("\n")
    assert lines[0] == "#!/usr/bin/env python"
    for i, header_line in enumerate(script.COPYRIGHT_HEADER_LINES):
        assert lines[1 + i] == header_line
    assert lines[5] == ""
    assert lines[6] == '"""With shebang."""'


def test_idempotent_no_op_when_header_present(script):
    src = (
        "\n".join(script.COPYRIGHT_HEADER_LINES)
        + "\n\n"
        + '"""Module docstring."""\n'
    )
    once = script.insert_header(src)
    twice = script.insert_header(once)
    assert once == src, "first apply on already-headered file must be a no-op"
    assert twice == src, "second apply must remain byte-identical"


def test_idempotent_with_shebang(script):
    src = (
        "#!/usr/bin/env python\n"
        + "\n".join(script.COPYRIGHT_HEADER_LINES)
        + "\n\n"
        + '"""Shebanged."""\n'
    )
    assert script.insert_header(src) == src


def test_has_header_detects_shebang_offset(script):
    headered = (
        "#!/usr/bin/env python\n"
        + "\n".join(script.COPYRIGHT_HEADER_LINES)
        + "\nimport os\n"
    )
    assert script.has_header(headered) is True


def test_has_header_false_when_only_shebang(script):
    assert script.has_header("#!/usr/bin/env python\n") is False


def test_has_header_false_on_empty(script):
    assert script.has_header("") is False


def test_apply_to_file_changes_then_no_change(script, tmp_path):
    f = tmp_path / "sample.py"
    # write_bytes (not write_text) so Windows text mode does NOT
    # convert "\n" to "\r\n" -- the CRLFBodyError guard would reject.
    f.write_bytes(b'"""Sample."""\nimport os\n')
    assert script.apply_to_file(f) is True
    # Second run must be a no-op and bit-identical.
    before = f.read_bytes()
    assert script.apply_to_file(f) is False
    assert f.read_bytes() == before


def test_apply_writes_lf_line_endings(script, tmp_path):
    # .gitattributes pins .py to eol=lf. The script must not emit CRLF
    # even on Windows hosts.
    f = tmp_path / "lf.py"
    f.write_bytes(b'"""LF only."""\nimport os\n')
    script.apply_to_file(f)
    assert b"\r\n" not in f.read_bytes()


def test_apply_to_file_handles_empty_file(script, tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    assert script.apply_to_file(f) is True
    assert script.has_header(f.read_text(encoding="utf-8")) is True


def test_iter_python_files_skips_pycache(script, tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "kept.py").write_text("", encoding="utf-8")
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "dropped.py").write_text("", encoding="utf-8")
    files = script.iter_python_files((root,))
    names = [p.name for p in files]
    assert "kept.py" in names
    assert "dropped.py" not in names


def test_iter_python_files_returns_sorted(script, tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    for name in ("zebra.py", "alpha.py", "mango.py"):
        (root / name).write_text("", encoding="utf-8")
    files = script.iter_python_files((root,))
    assert files == sorted(files)


def test_repo_invariant_every_watched_py_has_header(script):
    # Every .py under the four watched directories must carry the
    # canonical header at commit time. This is the same assertion the
    # CI ``--check`` job runs; duplicating it here gives ``pytest`` an
    # early warning before push. ASCII-only failure text so cp1252
    # consoles render it cleanly (pass-1 POLISH-1).
    offenders = script.missing_header_files()
    assert offenders == [], (
        "These .py files are missing the canonical copyright header -- run "
        "`python scripts/add_copyright_headers.py` to fix:\n"
        + "\n".join(str(p.relative_to(REPO_ROOT)) for p in offenders)
    )


def test_header_is_canonical_rejects_wrong_year(script):
    # Pass-1 RT MAJOR-1: sentinel-only check used to accept any
    # `# Copyright (c)` line. Canonical check must reject anything
    # that isn't the exact spec block.
    wrong_year = (
        "# Copyright (c) 1999 Someone Else. All wrongs reserved.\n"
        + "\n".join(script.COPYRIGHT_HEADER_LINES[1:])
        + "\nimport os\n"
    )
    assert script.has_header(wrong_year) is True  # cheap sentinel still matches
    assert script.header_is_canonical(wrong_year) is False
    # And the file would appear as an offender in --check.


def test_header_is_canonical_accepts_real_header(script):
    src = "\n".join(script.COPYRIGHT_HEADER_LINES) + "\n\nimport os\n"
    assert script.header_is_canonical(src) is True


def test_header_is_canonical_with_shebang_and_encoding(script):
    src = (
        "#!/usr/bin/env python\n"
        "# -*- coding: utf-8 -*-\n"
        + "\n".join(script.COPYRIGHT_HEADER_LINES)
        + "\n\n"
        '"""doc."""\n'
    )
    assert script.header_is_canonical(src) is True


def test_pep263_encoding_line_preserved(script):
    # MINOR-1: header must NOT shift a PEP 263 encoding declaration
    # off line 1/2 -- doing so silently invalidates the directive.
    src = '# -*- coding: utf-8 -*-\n"""doc."""\nimport os\n'
    result = script.insert_header(src)
    lines = result.split("\n")
    assert lines[0] == "# -*- coding: utf-8 -*-"
    for i, header_line in enumerate(script.COPYRIGHT_HEADER_LINES):
        assert lines[1 + i] == header_line
    # Idempotent on re-apply.
    assert script.insert_header(result) == result


def test_pep263_with_shebang_preserved(script):
    src = (
        "#!/usr/bin/env python\n"
        "# coding=utf-8\n"
        '"""doc."""\nimport os\n'
    )
    result = script.insert_header(src)
    lines = result.split("\n")
    assert lines[0] == "#!/usr/bin/env python"
    assert lines[1] == "# coding=utf-8"
    for i, header_line in enumerate(script.COPYRIGHT_HEADER_LINES):
        assert lines[2 + i] == header_line


def test_apply_to_file_refuses_crlf_body(script, tmp_path):
    # Pass-1 RT MAJOR-2: silently mixing LF header into a CRLF body
    # creates mixed line endings. Refuse instead of papering over it.
    f = tmp_path / "crlf.py"
    f.write_bytes(b'"""crlf body."""\r\nimport os\r\n')
    with pytest.raises(script.CRLFBodyError) as excinfo:
        script.apply_to_file(f)
    assert "CRLF" in str(excinfo.value)
    assert ".gitattributes" in str(excinfo.value)
    # File on disk must be untouched (refuse-to-write contract).
    assert f.read_bytes() == b'"""crlf body."""\r\nimport os\r\n'


def test_cli_apply_reports_crlf_skip(script, tmp_path, monkeypatch, capsys):
    fake_root = tmp_path / "fakepkg"
    fake_root.mkdir()
    (fake_root / "crlf.py").write_bytes(b'import os\r\n')
    monkeypatch.setattr(script, "WATCHED_DIRS", (fake_root,))
    rc = script.main([])
    assert rc == 3
    out = capsys.readouterr()
    assert "CRLF" in (out.err + out.out)


def test_display_uses_forward_slashes(script):
    # MINOR-4: CI is Linux, contributors on Windows. Output must be
    # path-separator-consistent so grep / diff / dedup across logs work.
    p = REPO_ROOT / "packages" / "topofuzz" / "src" / "topofuzz" / "cli.py"
    rendered = script._display(p)
    assert "\\" not in rendered
    assert rendered.startswith("packages/topofuzz/")


def test_cli_check_exits_zero_when_repo_clean():
    # Subprocess so we exercise the actual main() + argparse path
    # under the same Python interpreter that runs pytest.
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"--check should pass on a clean repo, got rc={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "passed" in result.stdout


def test_cli_check_detects_missing_header(script, tmp_path, monkeypatch):
    # Point WATCHED_DIRS at a tmp tree that contains one un-headered
    # file, then call main(["--check"]) directly. We can't subprocess
    # this case without mutating the real repo.
    fake_root = tmp_path / "fakepkg"
    fake_root.mkdir()
    # write_bytes so Windows text mode does not convert "\n" to "\r\n"
    # (would also trip the CRLF guard; pass-2 RT POLISH-1).
    (fake_root / "unhead.py").write_bytes(b"import os\n")
    monkeypatch.setattr(script, "WATCHED_DIRS", (fake_root,))
    rc = script.main(["--check"])
    assert rc == 1


# --- pass-2 audit folds ---------------------------------------------------


def test_apply_refuses_stale_header_does_not_rewrite(script, tmp_path):
    # Pass-2 RT BLOCKER-1: a file with `# Copyright (c)` line that is
    # NOT the canonical block must be refused (not silently overwritten,
    # not papered over with rc=2 INTERNAL ERROR).
    f = tmp_path / "stale.py"
    stale_body = (
        b"# Copyright (c) 1999 Someone Else. All wrongs reserved.\n"
        b"# Licensed under MIT.\n"
        b"# See LICENSE.\n"
        b"# placeholder.\n"
        b"\n"
        b"import os\n"
    )
    f.write_bytes(stale_body)
    with pytest.raises(script.StaleHeaderError) as excinfo:
        script.apply_to_file(f)
    assert "stale" in str(excinfo.value).lower() or "canonical" in str(excinfo.value).lower()
    # File must be byte-identical -- refusing to overwrite a foreign
    # copyright notice is a legal-safety contract.
    assert f.read_bytes() == stale_body


def test_cli_apply_reports_stale_skip(script, tmp_path, monkeypatch, capsys):
    fake_root = tmp_path / "fakepkg"
    fake_root.mkdir()
    (fake_root / "stale.py").write_bytes(
        b"# Copyright (c) 2024 Foo.\n"
        b"# Licensed under Apache 2.0.\n"
        b"# Notice.\n"
        b"# placeholder.\n"
        b"\nimport os\n"
    )
    monkeypatch.setattr(script, "WATCHED_DIRS", (fake_root,))
    rc = script.main([])
    assert rc == 3
    captured = capsys.readouterr()
    assert "stale" in (captured.err + captured.out).lower()


def test_check_surfaces_crlf_body_as_offender(script, tmp_path, monkeypatch):
    # Pass-2 RT MINOR-1: a CRLF-bodied file with header silently
    # passed --check before this fix (read_text universal-newlines
    # normalizes), but apply-mode would refuse to write. The two
    # modes must agree.
    fake_root = tmp_path / "fakepkg"
    fake_root.mkdir()
    headered_crlf = (
        b"\r\n".join(line.encode("utf-8") for line in script.COPYRIGHT_HEADER_LINES)
        + b"\r\n\r\nimport os\r\n"
    )
    (fake_root / "crlf.py").write_bytes(headered_crlf)
    monkeypatch.setattr(script, "WATCHED_DIRS", (fake_root,))
    rc = script.main(["--check"])
    assert rc == 1, "CRLF-bodied file must fail --check even when header text matches"


def test_pep263_regex_matches_cpython_tokenizer(script):
    # Pass-2 RT MAJOR-1 closed: the `^[ \t\v]*#.*?coding[=:]` form
    # is intentionally identical to CPython's tokenize.py cookie_re.
    # Pure prose (no `coding` keyword + value) must NOT be treated as
    # preamble; any line CPython itself would treat as PEP 263 IS
    # preserved as preamble (safer than risking a misclassification
    # in the other direction).
    pure_prose = '# Just a regular comment\n"""doc."""\nimport os\n'
    result = script.insert_header(pure_prose)
    lines = result.split("\n")
    # Header on lines 0..3; prose preserved as body on line 5.
    for i, header_line in enumerate(script.COPYRIGHT_HEADER_LINES):
        assert lines[i] == header_line
    assert lines[5] == "# Just a regular comment"


def test_pep263_regex_does_not_match_line_3(script):
    # PEP 263 declarations are valid ONLY on line 1 or 2. A `coding:`
    # comment on line 3 must NOT be treated as preamble -- it's body.
    src = (
        '"""doc."""\n'  # line 0: docstring (not a comment)
        "import os\n"  # line 1
        "# coding: utf-8\n"  # line 2 (would be treated as PEP 263 by CPython too -- but only if on lines 1-2)
        "x = 1\n"
    )
    result = script.insert_header(src)
    lines = result.split("\n")
    # Header on 0..3; original content starts on line 5; the `coding`
    # line stays at its original relative position (now line 7).
    for i, header_line in enumerate(script.COPYRIGHT_HEADER_LINES):
        assert lines[i] == header_line
    assert lines[5] == '"""doc."""'
    assert "# coding: utf-8" in lines[7]


def test_pep263_regex_accepts_real_declarations(script):
    # Sanity: the anchored regex still matches every PEP 263-legal form.
    for line in (
        "# -*- coding: utf-8 -*-",
        "# coding=utf-8",
        "# coding: latin-1",
        "#  coding: utf-8",  # leading space inside comment
        "\t# coding: utf-8",  # tab + comment per PEP 263 `[ \t\v]*`
    ):
        result = script.insert_header(line + '\n"""doc."""\n')
        first_line = result.split("\n")[0]
        assert first_line == line, f"PEP 263 line not preserved as preamble: {line!r}"


def test_header_is_canonical_rejects_partial_match(script):
    # Pass-2 code-reviewer POLISH: lock in "all 4 lines must match",
    # not just line 1.
    almost = (
        script.COPYRIGHT_HEADER_LINES[0] + "\n"
        + script.COPYRIGHT_HEADER_LINES[1] + "\n"
        + script.COPYRIGHT_HEADER_LINES[2] + "\n"
        + "# WRONG LAST LINE\n"
        + "\nimport os\n"
    )
    assert script.header_is_canonical(almost) is False


def test_header_is_canonical_rejects_out_of_order(script):
    swapped = (
        script.COPYRIGHT_HEADER_LINES[1] + "\n"  # license line first
        + script.COPYRIGHT_HEADER_LINES[0] + "\n"  # copyright second
        + script.COPYRIGHT_HEADER_LINES[2] + "\n"
        + script.COPYRIGHT_HEADER_LINES[3] + "\n"
        + "\nimport os\n"
    )
    assert script.header_is_canonical(swapped) is False


def test_cli_apply_mode_writes_then_passes_check(script, tmp_path, monkeypatch):
    fake_root = tmp_path / "fakepkg"
    fake_root.mkdir()
    # write_bytes so the CRLFBodyError guard does not trigger on
    # Windows (where text mode converts "\n" to "\r\n").
    (fake_root / "a.py").write_bytes(b'"""A."""\n')
    (fake_root / "b.py").write_bytes(b"import os\n")
    monkeypatch.setattr(script, "WATCHED_DIRS", (fake_root,))
    assert script.main([]) == 0
    # After apply, every file must carry the header and a re-check passes.
    assert script.main(["--check"]) == 0
