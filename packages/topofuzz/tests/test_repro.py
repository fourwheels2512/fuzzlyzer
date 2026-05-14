"""Tests for ``topofuzz repro`` — step 19b reproducer-script emitter."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topofuzz.cli import app
from topofuzz.repro import (
    DEFAULT_VERDICT_THRESHOLD,
    ReproError,
    build_repro_script,
    find_flip_candidate,
    read_eval_input,
    resolve_graph_import,
    write_repro_script,
)


GALLERY_RUN = Path(
    "examples/gallery/langgraph-lending-4role/runs/"
    "2026-05-14-01KRJ2WADFPMF6W1MJSF777EW0"
)


def _make_synthetic_run(tmp_path: Path, *, with_flip: bool) -> Path:
    """Construct a minimal valid run_dir for unit tests.

    Schema mirrors topofuzz.runner ``judge_results`` + the real manifest
    keys ``build_repro_script`` consults. Avoids depending on the gallery
    run for unit-level tests.
    """
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()

    manifest = {
        "run_id": "synth-run-0001",
        "topology_canonical_hash": "h" * 64,
        "topology_name": "synth",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": 2,
        "n_eval_rows": 1,
        "variant_canonical_hashes": ["a" * 64, "b" * 64],
        "variant_orderings": [["n1", "n2"], ["n2", "n1"]],
        "eval_ids": ["eval_001"],
        "eval_ground_truth": [1],
        "agent_vendors": ["google"],
        "judge_model": "google/gemini-2.5-pro",
        "rubric_hash": "r" * 64,
        "semaphore_size": 8,
        "max_cost_usd": 1.0,
        "cli_args": {
            "fixture": "tests/data/fake_fixture.py:build_graph",
            "eval": "eval.csv",
            "rng_seed": 42,
        },
        "kickoff_ts": 0,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # eval file alongside run_dir parent so _resolve_eval_path finds it.
    (tmp_path / "eval.csv").write_text(
        "id,input\neval_001,Will this loan be approved?\n", encoding="utf-8"
    )

    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, failure_reason TEXT)"
    )
    if with_flip:
        rows = [
            ("synth-run-0001", "a" * 64, "eval_001", "continuous", 0.9, None, None, None),
            ("synth-run-0001", "b" * 64, "eval_001", "continuous", 0.1, None, None, None),
        ]
    else:
        rows = [
            ("synth-run-0001", "a" * 64, "eval_001", "continuous", 0.9, None, None, None),
            ("synth-run-0001", "b" * 64, "eval_001", "continuous", 0.8, None, None, None),
        ]
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    db.commit()
    db.close()
    return run_dir


# ---------------------------------------------------------------------------
# find_flip_candidate
# ---------------------------------------------------------------------------


def test_find_flip_candidate_returns_deterministic_indices(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    eid, neg, pos = find_flip_candidate(run_dir)
    assert eid == "eval_001"
    # Variant b (idx=1) scored 0.1 → neg; variant a (idx=0) scored 0.9 → pos
    assert neg == 1
    assert pos == 0


def test_find_flip_candidate_no_flip_raises(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=False)
    with pytest.raises(ReproError, match="No verdict flip"):
        find_flip_candidate(run_dir)


def test_find_flip_candidate_skips_judge_failures(tmp_path):
    """NULL-score + NULL-passed cells must not be counted as either side."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("synth-run-0001", "c" * 64, "eval_001", "continuous", None, None, None, "no parse"),
    )
    db.commit()
    db.close()
    # Should still find the original a/b flip, ignoring the NULL row.
    eid, neg, pos = find_flip_candidate(run_dir)
    assert (eid, neg, pos) == ("eval_001", 1, 0)


def test_find_flip_candidate_eval_id_with_no_flip_raises(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=False)
    with pytest.raises(ReproError, match="No verdict flip"):
        find_flip_candidate(run_dir, eval_id="eval_001")


# ---------------------------------------------------------------------------
# resolve_graph_import
# ---------------------------------------------------------------------------


def test_resolve_graph_import_dottable_path():
    assert resolve_graph_import("pkg/sub/mod.py:make") == ("pkg.sub.mod", "make")


def test_resolve_graph_import_windows_separators_normalised():
    assert resolve_graph_import("pkg\\sub\\mod.py:make") == ("pkg.sub.mod", "make")


def test_resolve_graph_import_emits_none_for_hyphenated_dir():
    """Hyphens are not valid Python identifiers → fail soft to placeholder."""
    module, factory = resolve_graph_import(
        "examples/gallery/langgraph-lending-4role/fixture.py:build_graph"
    )
    assert module is None
    assert factory == "build_graph"


def test_resolve_graph_import_emits_none_for_absolute_path():
    # Use a path that's absolute on both Windows and POSIX semantics.
    module, _ = resolve_graph_import("/abs/path/fixture.py:f")
    assert module is None


def test_resolve_graph_import_emits_none_for_parent_traversal():
    module, _ = resolve_graph_import("../foo/fixture.py:f")
    assert module is None


def test_resolve_graph_import_emits_none_for_non_python_file():
    module, _ = resolve_graph_import("pkg/sub/mod.txt:f")
    assert module is None


def test_resolve_graph_import_missing_colon():
    assert resolve_graph_import("pkg/sub/mod.py") == (None, "build_graph")


# ---------------------------------------------------------------------------
# read_eval_input
# ---------------------------------------------------------------------------


def test_read_eval_input_csv(tmp_path):
    p = tmp_path / "evals.csv"
    p.write_text(
        "id,input\neval_001,first prompt\neval_002,second prompt\n",
        encoding="utf-8",
    )
    assert read_eval_input(p, "eval_002") == "second prompt"


def test_read_eval_input_missing_id_raises(tmp_path):
    p = tmp_path / "evals.csv"
    p.write_text("id,input\neval_001,only\n", encoding="utf-8")
    with pytest.raises(ReproError, match="not found"):
        read_eval_input(p, "missing_id")


def test_read_eval_input_unsupported_extension(tmp_path):
    p = tmp_path / "evals.parquet"
    p.write_text("not real parquet", encoding="utf-8")
    with pytest.raises(ReproError, match="Unsupported eval-file format"):
        read_eval_input(p, "eval_001")


# ---------------------------------------------------------------------------
# build_repro_script — the load-bearing contracts
# ---------------------------------------------------------------------------


def test_build_repro_script_byte_identical_on_double_render(tmp_path):
    """BLUEPRINT line 444: 'Output is deterministic given the same --rng-seed.'

    The generator is itself pure given (run_dir, eval_id) — no rng involved
    at this layer. Render twice and assert byte equality.
    """
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    text1, eid1 = build_repro_script(run_dir)
    text2, eid2 = build_repro_script(run_dir)
    assert text1 == text2
    assert eid1 == eid2


def test_build_repro_script_never_contains_topofuzz_import(tmp_path):
    """BLUEPRINT line 443: 'NO topofuzz import in the generated script.'

    Use AST so prose mentioning the word in docstrings doesn't trip us.
    """
    import ast

    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    text, _ = build_repro_script(run_dir)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("topofuzz"), (
                    f"generated script imports {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("topofuzz"), (
                f"generated script does `from {module} import ...`"
            )


def test_build_repro_script_contains_required_tokens(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    text, eid = build_repro_script(run_dir)
    assert eid == "eval_001"
    assert "EVAL_INPUT" in text
    assert "ORDERING_NEG" in text
    assert "ORDERING_POS" in text
    # The flipping variant b had ordering ["n2", "n1"] (NEG side)
    # The reference variant a had ordering ["n1", "n2"] (POS side)
    assert '["n2", "n1"]' in text
    assert '["n1", "n2"]' in text
    assert "Will this loan be approved?" in text
    assert "assert out_neg != out_pos" in text


def test_build_repro_script_emits_todo_for_unresolvable_fixture(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    # Patch the manifest to use a hyphenated (unresolvable) path.
    m = json.loads((run_dir / "manifest.json").read_text())
    m["cli_args"]["fixture"] = (
        "examples/gallery/langgraph-lending-4role/fixture.py:build_graph"
    )
    (run_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    text, _ = build_repro_script(run_dir)
    assert "# TODO: import your graph factory" in text


# ---------------------------------------------------------------------------
# write_repro_script — file-write integration
# ---------------------------------------------------------------------------


def test_write_repro_script_default_path(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    target = write_repro_script(run_dir)
    assert target == run_dir / "repro_eval_001.py"
    assert target.is_file()
    assert "assert out_neg != out_pos" in target.read_text(encoding="utf-8")


def test_write_repro_script_respects_out_option(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    out = tmp_path / "custom_name.py"
    target = write_repro_script(run_dir, out_path=out)
    assert target == out
    assert out.is_file()


# ---------------------------------------------------------------------------
# CLI integration — `topofuzz repro <run_dir>`
# ---------------------------------------------------------------------------


def test_cli_repro_writes_script_to_default_path(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    result = CliRunner().invoke(app, ["repro", str(run_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    expected = run_dir / "repro_eval_001.py"
    assert expected.is_file()
    assert str(expected) in result.stdout


def test_cli_repro_no_flip_exits_1(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=False)
    result = CliRunner(mix_stderr=False).invoke(app, ["repro", str(run_dir)])
    assert result.exit_code == 1
    assert "No verdict flip" in result.stderr


def test_cli_repro_bad_run_dir_exits_2(tmp_path):
    nonexistent = tmp_path / "no-such-dir"
    result = CliRunner(mix_stderr=False).invoke(app, ["repro", str(nonexistent)])
    assert result.exit_code == 2
    assert "not a directory" in result.stderr


def test_cli_repro_reserved_flag_rejected(tmp_path):
    """Reserved-flag policy (§16 #2) applies to repro like every subcommand."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    result = CliRunner(mix_stderr=False).invoke(
        app, ["repro", str(run_dir), "--sandbox", "foo"]
    )
    assert result.exit_code == 2
    assert "reserved" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Gallery-fixture integration — end-to-end against real committed run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not GALLERY_RUN.is_dir(),
    reason="Lending gallery run not present (skipped outside repo checkout)",
)
def test_build_repro_script_on_real_lending_gallery_run():
    """End-to-end smoke: real run dir produces a parseable Python script."""
    import ast

    text, eid = build_repro_script(GALLERY_RUN)
    assert eid.startswith("lending_")
    # Generated script must be syntactically valid Python.
    tree = ast.parse(text, filename=f"repro_{eid}.py")
    # BLUEPRINT line 443: never imports topofuzz.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("topofuzz")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("topofuzz")


# Sanity-check the module's public constant — flagged by review-pattern
# tests that look for accidental mutation of the default.
def test_default_threshold_is_half():
    assert DEFAULT_VERDICT_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Audit pass-1 hardening regression tests
# ---------------------------------------------------------------------------


def test_resolve_graph_import_rejects_windows_drive_letter():
    """A POSIX manifest with Windows-style drive letters must not silently
    produce ``from c.abs.fixture import ...`` (audit CR pass-1 M-CR-2)."""
    assert resolve_graph_import("C:/abs/fixture.py:f") == (None, "f")
    assert resolve_graph_import("C:\\abs\\fixture.py:f") == (None, "f")


def test_resolve_graph_import_rejects_posix_abs_path_on_windows():
    """``/etc/passwd`` is absolute on POSIX even when this runs on Windows
    (audit RT pass-1 MN-RT-3)."""
    assert resolve_graph_import("/etc/passwd.py:f") == (None, "f")


def test_write_repro_script_refuses_unsafe_eval_id(tmp_path):
    """Audit RT pass-1 BLOCKER — hostile eval ``id`` column with traversal
    must not drive ``write_repro_script`` to clobber files outside run_dir."""
    run_dir = tmp_path / "evil-run"
    run_dir.mkdir()
    manifest = json.loads(
        (_make_synthetic_run(tmp_path, with_flip=True) / "manifest.json").read_text()
    )
    # Replace eval row id with a traversal pattern; manifest references that id.
    manifest["eval_ids"] = ["../../tmp/pwn"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Rebuild a sqlite + eval CSV that uses the unsafe id.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, failure_reason TEXT)"
    )
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("synth-run-0001", "a" * 64, "../../tmp/pwn", "continuous", 0.9, None, None, None),
            ("synth-run-0001", "b" * 64, "../../tmp/pwn", "continuous", 0.1, None, None, None),
        ],
    )
    db.commit()
    db.close()
    (tmp_path / "eval.csv").write_text(
        'id,input\n"../../tmp/pwn","oops"\n', encoding="utf-8"
    )

    with pytest.raises(ReproError, match="characters not permitted"):
        write_repro_script(run_dir)


def test_write_repro_script_unsafe_eid_allowed_with_explicit_out(tmp_path):
    """Caller-supplied --out path bypasses the eid safety check (they own it)."""
    # Use the same evil-run fixture: write succeeds when out_path is given.
    run_dir = tmp_path / "evil-run-2"
    run_dir.mkdir()
    src = _make_synthetic_run(tmp_path, with_flip=True)
    manifest = json.loads((src / "manifest.json").read_text())
    manifest["eval_ids"] = ["bad/eid"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, failure_reason TEXT)"
    )
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("synth-run-0001", "a" * 64, "bad/eid", "continuous", 0.9, None, None, None),
            ("synth-run-0001", "b" * 64, "bad/eid", "continuous", 0.1, None, None, None),
        ],
    )
    db.commit()
    db.close()
    (tmp_path / "eval.csv").write_text(
        'id,input\n"bad/eid","prompt"\n', encoding="utf-8"
    )

    out = tmp_path / "explicit_safe_name.py"
    target = write_repro_script(run_dir, out_path=out)
    assert target == out
    assert out.is_file()


def test_resolve_eval_path_rejects_parent_traversal(tmp_path):
    """Audit RT pass-1 M-RT-2 — hostile manifest can't read /etc/secrets via
    ``eval`` field."""
    from topofuzz.repro import _resolve_eval_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ReproError, match="parent-traversal"):
        _resolve_eval_path({"eval": "../../etc/passwd"}, run_dir)


def test_find_flip_candidate_corrupt_sqlite_raises_reproerror(tmp_path):
    """Audit RT pass-1 M-RT-1 — missing schema must surface as ReproError,
    not bare ``sqlite3.OperationalError`` traceback."""
    run_dir = tmp_path / "corrupt-run"
    run_dir.mkdir()
    # Valid manifest but empty/wrong sqlite file.
    minimal_manifest = {
        "run_id": "x",
        "variant_canonical_hashes": ["a" * 64, "b" * 64],
        "variant_orderings": [["n1"], ["n2"]],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(minimal_manifest), encoding="utf-8"
    )
    # Write a sqlite file without the judge_results table.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute("CREATE TABLE not_judge_results (x INTEGER)")
    db.commit()
    db.close()

    with pytest.raises(ReproError, match="judge_results schema"):
        find_flip_candidate(run_dir)


def test_read_eval_input_rejects_non_string_jsonl_input(tmp_path):
    """Audit RT pass-1 M-RT-3 — JSONL row with dict-shaped ``input`` must
    raise rather than silently produce a non-string EVAL_INPUT."""
    p = tmp_path / "evals.jsonl"
    p.write_text(
        '{"id": "x", "input": {"nested": "obj"}}\n', encoding="utf-8"
    )
    with pytest.raises(ReproError, match="expected str"):
        read_eval_input(p, "x")


# ---------------------------------------------------------------------------
# Audit pass-2 hardening regression tests
# ---------------------------------------------------------------------------


def test_read_eval_input_empty_csv_input_raises(tmp_path):
    """Audit RT pass-2 M-RT2-1 — empty ``input`` column produces a
    meaningless reproducer; must raise."""
    p = tmp_path / "evals.csv"
    p.write_text("id,input\nempty_001,\n", encoding="utf-8")
    with pytest.raises(ReproError, match="empty"):
        read_eval_input(p, "empty_001")


def test_read_eval_input_empty_jsonl_input_raises(tmp_path):
    p = tmp_path / "evals.jsonl"
    p.write_text('{"id": "empty_002", "input": ""}\n', encoding="utf-8")
    with pytest.raises(ReproError, match="empty"):
        read_eval_input(p, "empty_002")


def test_resolve_eval_path_rejects_absolute_posix(tmp_path):
    """Audit pass-2 (CR MAJOR-2 + RT MN-RT2-1) — absolute POSIX paths
    cannot be used to read system files into the generated script."""
    from topofuzz.repro import _resolve_eval_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ReproError, match="absolute path"):
        _resolve_eval_path({"eval": "/etc/passwd"}, run_dir)


def test_resolve_eval_path_rejects_absolute_windows(tmp_path):
    from topofuzz.repro import _resolve_eval_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ReproError, match="absolute path"):
        _resolve_eval_path({"eval": "C:/Windows/win.ini"}, run_dir)
    with pytest.raises(ReproError, match="absolute path"):
        _resolve_eval_path({"eval": "C:\\Windows\\win.ini"}, run_dir)


def test_resolve_eval_path_legitimate_relative_with_dashes(tmp_path):
    """A relative path containing ``..`` only inside a segment name
    (e.g. ``runs/2026-05-14-foo``) must still be accepted."""
    from topofuzz.repro import _resolve_eval_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "eval-dash..suffix.csv").write_text(
        "id,input\nx,y\n", encoding="utf-8"
    )
    out = _resolve_eval_path({"eval": "eval-dash..suffix.csv"}, run_dir)
    assert out.name == "eval-dash..suffix.csv"


def test_build_repro_script_jinja2_missing_raises_friendly(tmp_path, monkeypatch):
    """Audit CR pass-2 MAJOR-1 — missing jinja2 ([full] extra) surfaces
    as ReproError with a friendly install hint, not a bare ImportError."""
    import builtins

    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "jinja2" or name.startswith("jinja2."):
            raise ImportError("No module named 'jinja2'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ReproError, match=r"\[full\]"):
        build_repro_script(run_dir)
