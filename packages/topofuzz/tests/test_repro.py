# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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


def _runner() -> CliRunner:
    """Mirror the ``test_cli_scan._runner`` Click 8.1/8.2 compat shim.

    The CLI tests below assert diagnostics land on STDERR specifically,
    which needs split streams: ``mix_stderr=False`` on Click 8.1 (whose
    default mixed runner raises on ``result.stderr``); on Click 8.2+ the
    kwarg is removed and separation is the default.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


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
    result = _runner().invoke(app, ["repro", str(run_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    expected = run_dir / "repro_eval_001.py"
    assert expected.is_file()
    assert str(expected) in result.stdout


def test_cli_repro_no_flip_exits_1(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, with_flip=False)
    result = _runner().invoke(app, ["repro", str(run_dir)])
    assert result.exit_code == 1
    assert "No verdict flip" in result.stderr


def test_cli_repro_bad_run_dir_exits_2(tmp_path):
    nonexistent = tmp_path / "no-such-dir"
    result = _runner().invoke(app, ["repro", str(nonexistent)])
    assert result.exit_code == 2
    assert "not a directory" in result.stderr


def test_cli_repro_reserved_flag_rejected(tmp_path):
    """Reserved-flag policy (§16 #2) applies to repro like every subcommand."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    result = _runner().invoke(
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


# ---------------------------------------------------------------------------
# Step 20a / F6 BLOCKER — repro must dispatch on manifest.verdict_node
# ---------------------------------------------------------------------------


def _patch_manifest(run_dir: Path, **fields) -> None:
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    m.update(fields)
    (run_dir / "manifest.json").write_text(
        json.dumps(m), encoding="utf-8"
    )


def test_repro_emits_explicit_verdict_node_when_manifest_sets_it(tmp_path):
    """v0.6 manifest path — generated script must carry the named node.

    Without the Step 20a fix the template hardcoded ``history[-1][1]`` which
    silently judged whichever node a permutation walked last — fabricating
    flips on multi-terminal / supervisor topologies. The reproducer must
    carry the orchestrator's declared verdict_node so the comparison
    matches what ``runner._run_judge`` actually scored.
    """
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    # ``n2`` is NOT the last node in either ordering (NEG is ["n2", "n1"],
    # POS is ["n1", "n2"]); picking it surfaces any test that latently
    # depends on the last-walked-node fallback.
    _patch_manifest(run_dir, verdict_node="n2", terminal_nodes=["n2"])
    text, _ = build_repro_script(run_dir)
    assert 'VERDICT_NODE = "n2"' in text
    assert "run_chain(ORDERING_NEG, LLM_CONFIGS, VERDICT_NODE)" in text


def test_repro_uses_single_terminal_node_when_verdict_node_unset(tmp_path):
    """v0.5.5 transitional path — only a SINGLE-element ``terminal_nodes``
    list set (no ``verdict_node``). Must use that node so the script and
    the runner score the same node.

    Multi-terminal manifests without ``verdict_node`` are refused by
    L1028 — see ``test_repro_refuses_multi_terminal_without_verdict_node``
    for the refusal contract. This test pins the single-terminal happy
    path that survives the L1028 fix.
    """
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(run_dir, terminal_nodes=["n1"])
    text, _ = build_repro_script(run_dir)
    assert 'VERDICT_NODE = "n1"' in text


def test_repro_refuses_multi_terminal_without_verdict_node(tmp_path):
    """L1028 — multi-terminal manifest without ``verdict_node`` must
    raise ``ReproError`` at render time with remediation guidance.

    Why this is the right shape:
      * On multi-terminal topologies, ``publish._excerpt_terminal_output``
        picks the first sorted terminal that ALSO produced output for
        each cell (the output-level survival filter on ``by_node``).
      * ``repro`` has no per-cell access at render time — it operates
        only on the manifest's field-level state.
      * If both functions returned ``sorted(terminal_nodes)[0]``
        unconditionally, they would silently name DIFFERENT nodes
        whenever the lex-first terminal was cancelled in some cells.
        That divergence breaks the "re-run the bundle yourself" claim
        on the CEO card.
      * Refusing in repro forces the operator to declare
        ``verdict_node`` explicitly, eliminating the ambiguity.

    Single-terminal manifests are unaffected — see
    ``test_repro_uses_single_terminal_node_when_verdict_node_unset``.
    """
    from topofuzz.repro import ReproError

    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(run_dir, terminal_nodes=["n2", "n1"])
    with pytest.raises(ReproError) as exc_info:
        build_repro_script(run_dir)
    msg = str(exc_info.value)
    assert "multi-terminal" in msg.lower()
    assert "verdict_node" in msg
    # Operator must see what was rejected and how to fix.
    assert "n2" in msg or "n1" in msg
    assert "topofuzz run" in msg
    assert "L1028" in msg


def test_repro_accepts_multi_terminal_with_explicit_verdict_node(tmp_path):
    """L1028 contract — multi-terminal manifest WITH ``verdict_node`` set
    works unchanged. The refusal applies only when ``verdict_node`` is
    unresolved and the manifest carries 2+ terminals."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(
        run_dir,
        verdict_node="n2",
        terminal_nodes=["n1", "n2"],
    )
    text, _ = build_repro_script(run_dir)
    assert 'VERDICT_NODE = "n2"' in text


def test_repro_no_verdict_fields_emits_none(tmp_path):
    """v0.5 manifest — neither ``verdict_node`` nor ``terminal_nodes``.

    Backwards-compat: the script must compile, ``VERDICT_NODE`` is ``None``,
    and the fallback path returns ``history[-1][1]`` so legacy gallery runs
    keep reproducing without re-running the orchestrator. Jinja's ``tojson``
    filter emits JSON ``null`` for Python ``None``, which is NOT a valid
    Python literal — a regression there would NameError the generated
    script at load. This test catches that via AST parse.
    """
    import ast

    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    text, _ = build_repro_script(run_dir)
    tree = ast.parse(text)
    found_verdict = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "VERDICT_NODE"
        ):
            found_verdict = True
            assert isinstance(node.value, ast.Constant)
            assert node.value.value is None, (
                "v0.5 manifest must render VERDICT_NODE = None (not 'null')"
            )
    assert found_verdict, "VERDICT_NODE assignment missing from rendered script"


def test_repro_verdict_node_preferred_over_terminal_nodes(tmp_path):
    """When both fields are present, ``verdict_node`` wins — matches the
    runner's precedence so a manifest that pinned the verdict node never
    silently gets re-interpreted by repro. ``n1`` is in both synthetic
    orderings, so the render-time presence check (RT M1) accepts it.
    """
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(
        run_dir,
        verdict_node="n1",
        terminal_nodes=["n2"],
    )
    text, _ = build_repro_script(run_dir)
    assert 'VERDICT_NODE = "n1"' in text


def test_repro_empty_verdict_node_falls_through(tmp_path):
    """Mirror publish's empty/whitespace rejection — an empty string is
    not a valid node id; resolution must fall through to terminal_nodes
    (or None when neither yields a valid id)."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(
        run_dir,
        verdict_node="   ",
        terminal_nodes=["n2"],
    )
    text, _ = build_repro_script(run_dir)
    assert 'VERDICT_NODE = "n2"' in text


def test_repro_terminal_nodes_with_blank_entry_rejected(tmp_path):
    """A malformed terminal_nodes list (e.g. an empty string) must be
    treated as absent; the renderer must NOT pick the blank as lex-first
    and emit ``VERDICT_NODE = ""`` (which the runtime would then fail to
    match against any history entry)."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(run_dir, terminal_nodes=["", "alpha"])
    text, _ = build_repro_script(run_dir)
    # Falls all the way through to None — neither "" nor "alpha" is picked
    # because the whole list is rejected when any entry fails the validator.
    import ast
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "VERDICT_NODE"
        ):
            assert node.value.value is None


def test_repro_byte_identical_with_verdict_node_set(tmp_path):
    """The byte-identical contract (covered for v0.5 manifests by
    ``test_build_repro_script_byte_identical_on_double_render``) must also
    hold for v0.6 manifests with ``verdict_node`` set."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(run_dir, verdict_node="n2", terminal_nodes=["n2"])
    text1, _ = build_repro_script(run_dir)
    text2, _ = build_repro_script(run_dir)
    assert text1 == text2


def test_repro_script_remains_syntactically_valid_with_verdict_node(tmp_path):
    """Sanity: with verdict_node set the generated script still parses as
    Python — the new top-level constant + ``run_chain`` signature change
    must not produce invalid syntax."""
    import ast

    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(run_dir, verdict_node="n2", terminal_nodes=["n2"])
    text, eid = build_repro_script(run_dir)
    ast.parse(text, filename=f"repro_{eid}.py")  # raises on bad syntax


# ---------------------------------------------------------------------------
# Pass-1 audit folds — RT M1 / M2 / M3, CR M1
# ---------------------------------------------------------------------------


def test_repro_render_time_check_rejects_verdict_node_not_in_orderings(tmp_path):
    """RT pass-1 M1 — if the orchestrator declared a ``verdict_node`` that
    isn't actually walked by either flip ordering, the render must fail
    with ``ReproError`` (not silently emit a script that AssertionErrors
    mid-run after the buyer has paid for the first Gemini turn)."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    # The synthetic orderings are ["n1", "n2"] / ["n2", "n1"]; "ghost" is
    # NOT in either, so the rendered script would never match it at runtime.
    _patch_manifest(run_dir, verdict_node="ghost", terminal_nodes=["ghost"])
    with pytest.raises(ReproError, match="ghost.*not present in both flip orderings"):
        build_repro_script(run_dir)


def test_repro_render_time_check_passes_when_verdict_node_walked(tmp_path):
    """The render-time check must NOT false-positive when verdict_node IS
    in both orderings (regardless of position)."""
    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    # "n1" appears in both ORDERING_NEG and ORDERING_POS even though it
    # isn't last in NEG.
    _patch_manifest(run_dir, verdict_node="n1")
    text, _ = build_repro_script(run_dir)
    assert 'VERDICT_NODE = "n1"' in text


def test_resolve_verdict_node_logs_warning_on_non_string_type(tmp_path, caplog):
    """RT pass-1 M3 — a non-string verdict_node (list/dict/int/bool) is
    silently demoted to None elsewhere; here we surface it as a warning
    so an operator can grep the log for a malformed manifest."""
    import logging
    from topofuzz.repro import _resolve_verdict_node

    with caplog.at_level(logging.WARNING, logger="topofuzz.repro"):
        out = _resolve_verdict_node({"verdict_node": ["risk", "policy"]})
    assert out is None
    assert any("not a string" in r.message for r in caplog.records)


def test_resolve_verdict_node_logs_warning_on_empty_string(tmp_path, caplog):
    """RT pass-1 M3 mirror — empty/whitespace verdict_node logs and falls
    through (matches publish's behaviour for log parity)."""
    import logging
    from topofuzz.repro import _resolve_verdict_node

    with caplog.at_level(logging.WARNING, logger="topofuzz.repro"):
        out = _resolve_verdict_node({"verdict_node": "   "})
    assert out is None
    assert any("empty" in r.message.lower() for r in caplog.records)


def test_resolve_verdict_node_logs_warning_on_non_list_terminal_nodes(caplog):
    """RT pass-1 M3 — same for malformed terminal_nodes."""
    import logging
    from topofuzz.repro import _resolve_verdict_node

    with caplog.at_level(logging.WARNING, logger="topofuzz.repro"):
        out = _resolve_verdict_node({"terminal_nodes": "decision"})
    assert out is None
    assert any("not a list" in r.message for r in caplog.records)


def test_resolve_verdict_node_handles_empty_terminal_list_silently():
    """Empty list (orchestrator wrote ``terminal_nodes: []``) → None,
    but quietly: it's not malformed, just empty (e.g. cancelled run).
    """
    import logging
    from topofuzz.repro import _resolve_verdict_node

    out = _resolve_verdict_node({"terminal_nodes": []})
    assert out is None


def _publish_pick_verdict_node_field(manifest: dict) -> str | None:
    """Faithful inline copy of publish.py:927-942's field-resolution.

    The actual code lives inside ``_build_ceo_card_context`` (publish.py
    around line 920), where it sets ``verdict_node_id`` and
    ``terminal_nodes`` before passing to ``_excerpt_terminal_output``.
    If this canary copy ever drifts from publish.py's inline block, this
    test will fail — that's the point. **If you update publish.py's
    resolver, update this copy AND topofuzz.repro._resolve_verdict_node
    in lockstep.**
    """
    verdict_node_id = manifest.get("verdict_node")
    if isinstance(verdict_node_id, str) and not verdict_node_id.strip():
        verdict_node_id = None
    if not isinstance(verdict_node_id, str):
        verdict_node_id = None
    terminal_nodes_raw = manifest.get("terminal_nodes")
    if isinstance(terminal_nodes_raw, list) and all(
        isinstance(t, str) and t.strip() for t in terminal_nodes_raw
    ):
        terminal_nodes = tuple(terminal_nodes_raw)
    else:
        terminal_nodes = None
    # Reduce to a single node id the same way _excerpt_terminal_output
    # does on the field level (sorted lex-first when verdict_node is
    # unset). _excerpt_terminal_output also has a "first PRESENT in
    # by_node" survival rule, but that operates on outputs, not on the
    # manifest fields; the canary is field-level only.
    if verdict_node_id is not None:
        return verdict_node_id
    if terminal_nodes:
        return sorted(terminal_nodes)[0]
    return None


def test_resolve_verdict_node_field_level_parity_with_publish():
    """RT pass-1 M2 — field-level resolver canary (runner-vs-repro contract).

    repro and publish each re-implement the same field-level precedence
    (verdict_node string → lex-first terminal_nodes → None). A future PR
    that loosens one without the other re-introduces the silent
    wrong-node-scored bug on the buyer-facing artifact (the demo PDF and
    the standalone repro script would name different nodes, and a
    skeptical compliance reviewer declares the product broken).

    **Scope:** field-level only. ``publish._excerpt_terminal_output``
    additionally applies a "first present in by_node" survival rule on the
    actual node-output set, which repro cannot replicate at render time
    (no access to per-cell ``node_outputs_json``). The output-level
    divergence on multi-terminal with cancelled outputs is BACKLOG, not
    in this canary's scope.
    """
    from topofuzz.repro import _resolve_verdict_node

    # Cases where repro and publish must agree byte-for-byte.
    # L1028 closed 2026-05-16: multi-terminal-without-verdict_node is
    # NOT in this list — repro now raises ReproError there while
    # publish picks lex-first (and then filters by_node at the cell
    # level). See ``test_resolve_verdict_node_diverges_from_publish_on_multi_terminal``
    # for the deliberate-divergence contract.
    cases = [
        ("explicit verdict_node wins",
         {"verdict_node": "explicit", "terminal_nodes": ["other"]}),
        ("empty-string verdict_node falls through",
         {"verdict_node": "", "terminal_nodes": ["fallback"]}),
        ("whitespace verdict_node falls through",
         {"verdict_node": "   ", "terminal_nodes": ["only-one"]}),
        ("solo verdict_node, no terminal_nodes",
         {"verdict_node": "winner"}),
        ("solo terminal_nodes, no verdict_node",
         {"terminal_nodes": ["solo"]}),
        ("v0.5 manifest with neither field", {}),
        ("verdict_node with empty terminal_nodes list",
         {"verdict_node": "single", "terminal_nodes": []}),
        ("list-typed verdict_node ignored, terminal_nodes used",
         {"verdict_node": ["a", "b"], "terminal_nodes": ["fallback"]}),
        ("int-typed terminal_nodes entry rejects whole list",
         {"terminal_nodes": ["valid", 0]}),
        # RT pass-2 MIN-2: type widening — these all must demote to None.
        ("bool-typed verdict_node", {"verdict_node": False}),
        ("int-typed verdict_node", {"verdict_node": 0}),
        ("float-typed verdict_node", {"verdict_node": 0.0}),
        ("dict-typed verdict_node", {"verdict_node": {"a": "b"}}),
        ("None-valued verdict_node", {"verdict_node": None}),
        ("dict-typed terminal_nodes ignored",
         {"terminal_nodes": {"a": "b"}}),
    ]
    for label, fields in cases:
        repro_pick = _resolve_verdict_node(fields)
        publish_pick = _publish_pick_verdict_node_field(fields)
        assert repro_pick == publish_pick, (
            f"resolver drift on case '{label}' ({fields}): "
            f"repro picked {repro_pick!r}, publish picked {publish_pick!r}"
        )


def test_resolve_verdict_node_diverges_from_publish_on_multi_terminal():
    """L1028 contract — repro and publish DELIBERATELY diverge on
    multi-terminal-without-verdict_node, by design.

    publish's field-level pick is ``sorted(terminal_nodes)[0]`` (then
    filtered by the cell's ``by_node`` at the output level). repro has
    no per-cell access at render time, so it cannot replicate the
    output-level filter — and returning lex-first regardless would
    silently name a different node than the CEO card on any cell where
    the lex-first terminal was cancelled. Refusing here is the only
    safe option: ``publish`` keeps working (its by_node filter is
    intact), ``repro`` raises with remediation guidance, and the buyer
    is forced to set ``verdict_node`` explicitly — which removes the
    ambiguity at the source.

    This test pins the divergence so a future PR that "fixes" the
    parity canary by relaxing repro's refusal trips this test first.
    """
    from topofuzz.repro import _resolve_verdict_node, ReproError

    multi_terminal_no_verdict = {"terminal_nodes": ["zebra", "alpha", "mango"]}

    # publish (field-level canary) picks lex-first.
    publish_pick = _publish_pick_verdict_node_field(multi_terminal_no_verdict)
    assert publish_pick == "alpha"

    # repro REFUSES — output-level by_node filter not replicable at
    # render time.
    with pytest.raises(ReproError) as exc_info:
        _resolve_verdict_node(multi_terminal_no_verdict)
    msg = str(exc_info.value)
    assert "multi-terminal" in msg.lower()
    assert "verdict_node" in msg
    assert "L1028" in msg


def test_repro_runtime_failure_uses_banner_exit_not_assertion(tmp_path):
    """RT pass-1 m3 — the generated script's verdict_node-missing failure
    path should be a clean ``sys.exit(...)`` with a stderr banner, not a
    raw ``AssertionError`` traceback (compliance reviewers misread
    tracebacks as 'API problems' and dismiss the result).

    RT pass-2 MAJ-2: exit code must NOT be 2 (collides with cli.py's
    ``_BAD_USAGE_EXIT == 2``). We exit 1 to match the existing
    empty-LLM-response branch (both are "inconclusive reproduction").
    """
    import ast

    run_dir = _make_synthetic_run(tmp_path, with_flip=True)
    _patch_manifest(run_dir, verdict_node="n2", terminal_nodes=["n2"])
    text, _ = build_repro_script(run_dir)
    assert "REPRO INCONCLUSIVE" in text
    assert "sys.exit(2)" not in text  # exit 2 reserved for CLI bad-usage
    # And the raw AssertionError must NOT appear inside run_chain.
    tree = ast.parse(text)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "run_chain":
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call):
                    called = sub.exc.func
                    name = getattr(called, "id", None) or getattr(called, "attr", None)
                    assert name != "AssertionError", (
                        "run_chain still raises AssertionError on missing "
                        "verdict_node — must use sys.exit(...) for the "
                        "compliance-reviewer-friendly path."
                    )
