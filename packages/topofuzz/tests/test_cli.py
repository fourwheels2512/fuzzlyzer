"""Tests for the topofuzz CLI.

Three planes:

* ``scan`` / ``report`` / ``verify`` are stubs until later weekends —
  smoke tests confirm a friendly exit, not a traceback.
* ``run`` is wired in step 6. Tests cover parse-layer rejections
  (reserved flags, --jury-of-3, bad fixture / eval). E2E artifact
  shape is exercised in :mod:`test_orchestrator` against the
  orchestrator directly (no Typer indirection).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("diskcache")  # CLI run command opens caches

from typer.testing import CliRunner  # noqa: E402

from topofuzz.cli import RESERVED_FLAGS, app  # noqa: E402


def test_help_succeeds():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "topofuzz" in result.output.lower()


def test_version_prints_and_exits_cleanly():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "topofuzz" in result.output


def test_scan_stub_exits_with_friendly_message():
    result = CliRunner().invoke(app, ["scan", "some-fixture"])
    assert result.exit_code == 2
    assert "not yet implemented" in (result.output or "").lower()


def test_report_stub_exits_with_friendly_message():
    result = CliRunner().invoke(app, ["report", "some-dir"])
    assert result.exit_code == 2


# `verify` is no longer a stub — see test_cli_verify_keygen.py for the
# step-10a tests of the real command.


def test_reserved_flags_constant_matches_blueprint_section_16():
    """BLUEPRINT §16 #2 names 9 reserved flags. Keep the constant in sync."""
    expected = {
        "--conformance", "--signoff", "--judge-pool",
        "--sandbox", "--key-uri", "--kms",
        "--allow-side-effects", "--tenant-id", "--per-cell-entropy",
    }
    assert set(RESERVED_FLAGS) == expected


# ---------------------------------------------------------------------------
# Step 6 — `run` parse-layer guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", list(RESERVED_FLAGS))
def test_run_rejects_each_reserved_flag(flag, tmp_path):
    """Every reserved flag must surface a friendly v1.0 message.

    Parametrized so a future contributor adding a flag to RESERVED_FLAGS
    automatically gets coverage. Trip the guard via the ``run`` subcommand
    rather than the root callback so the test exercises the actual user
    path.
    """
    result = CliRunner().invoke(
        app, ["run", "fixture.py:graph", "--eval", str(tmp_path / "x.jsonl"), flag]
    )
    assert result.exit_code == 2
    assert flag in result.output
    assert "reserved for v1.0" in result.output.lower()


def test_run_jury_of_3_rejected_in_v05(tmp_path):
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text('{"id": "e1", "input": "x"}\n', encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "fixture.py:graph",
            "--eval", str(eval_path),
            "--jury-of-3",
        ],
    )
    assert result.exit_code == 2
    assert "jury-of-3" in result.output.lower()


def test_run_friendly_error_on_missing_fixture(tmp_path):
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text('{"id": "e1", "input": "x"}\n', encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "nonexistent/file.py:graph",
            "--eval", str(eval_path),
            "--yes",
        ],
    )
    assert result.exit_code == 2
    assert "failed to load fixture" in result.output.lower()


def test_run_friendly_error_on_missing_eval(tmp_path):
    # A minimal fixture file the loader can find — failure must come
    # from the eval, not the fixture.
    fixture_file = tmp_path / "fx.py"
    fixture_file.write_text(
        "graph = None  # never reached; eval check runs first\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture_file}:graph",
            "--eval", str(tmp_path / "missing.jsonl"),
            "--yes",
        ],
    )
    # Either the fixture loader complains (graph = None → adapter raises
    # NotImplementedError → exit 64 per §A error taxonomy) or the eval
    # loader does (missing file → exit 2). The point is a clean exit,
    # not a stack trace.
    assert result.exit_code in (2, 64)
    assert "topofuzz:" in result.output.lower()


# ---------------------------------------------------------------------------
# Preflight refuses topologies with any None llm_config (audit B-2)
# ---------------------------------------------------------------------------


def test_run_rejects_topology_with_missing_llm_config(tmp_path):
    """A fixture with partial LLM_CONFIGS leaves some nodes with
    ``llm_config=None``. The runner would crash mid-run with a
    confusing ValueError. Catch it at the CLI's preflight so the
    user gets a clear actionable message AND the cost estimator
    doesn't display numbers computed against the wrong rate."""
    pytest.importorskip("langgraph")
    fixture = tmp_path / "fx.py"
    fixture.write_text(
        "from langgraph.graph import END, START, StateGraph\n"
        "from typing import TypedDict\n"
        "from topofuzz.ir import LLMConfig\n"
        "\n"
        "class _S(TypedDict, total=False):\n"
        "    x: str\n"
        "\n"
        "def _a(s): return s\n"
        "\n"
        "def build_graph():\n"
        "    g = StateGraph(_S)\n"
        "    g.add_node('a', _a)\n"
        "    g.add_node('b', _a)\n"
        "    g.add_edge(START, 'a')\n"
        "    g.add_edge('a', 'b')\n"
        "    g.add_edge('b', END)\n"
        "    return g.compile()\n"
        "\n"
        "# Only 'a' has a config; 'b' is left as None.\n"
        "LLM_CONFIGS = {'a': LLMConfig(model='google/gemini-2.5-pro')}\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text('{"id": "e1", "input": "x"}\n', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--yes",
        ],
    )
    assert result.exit_code == 2
    assert "no llm_config" in result.output.lower()
    assert "'b'" in result.output


# ---------------------------------------------------------------------------
# --variants is a cross-family cap (audit B-2 second pass)
# ---------------------------------------------------------------------------


def test_round_robin_select_variants_total_cap():
    """Direct test of the per-family round-robin helper.

    `--variants 1` must produce exactly 1 variant (the original).
    `--variants 4` with 3 populated families yields original + one
    from each family, totaling 4.
    """
    from topofuzz.cli import _round_robin_select

    original = "ORIG"
    fam0 = ["o1", "o2", "o3"]
    fam1 = ["r1", "r2"]
    fam2 = ["d1", "d2", "d3", "d4"]

    assert _round_robin_select(
        original=original, family_pools=(fam0, fam1, fam2), total=1
    ) == ["ORIG"]
    assert _round_robin_select(
        original=original, family_pools=(fam0, fam1, fam2), total=4
    ) == ["ORIG", "o1", "r1", "d1"]
    # Larger N cycles when a pool runs out — fam1 exhausts at 2 picks.
    assert _round_robin_select(
        original=original, family_pools=(fam0, fam1, fam2), total=10
    ) == ["ORIG", "o1", "r1", "d1", "o2", "r2", "d2", "o3", "d3", "d4"]


def test_round_robin_select_zero_total():
    from topofuzz.cli import _round_robin_select
    assert _round_robin_select(
        original="X", family_pools=([1, 2], [3]), total=0
    ) == []


# ---------------------------------------------------------------------------
# True CLI end-to-end (audit M-2 second pass)
# ---------------------------------------------------------------------------


def test_cli_run_end_to_end_with_mocked_orchestrator(tmp_path, monkeypatch):
    """End-to-end: CliRunner → loaders → preflight → execute_run.

    The orchestrator's `execute_run` is monkey-patched to inject a
    fake `llm_callable` and a small `delay_s`, so the test doesn't
    touch the network. Verifies that:
      * fixture and eval rows load cleanly
      * preflight accepts `--yes`
      * the orchestrator runs to completion
      * a run directory is created at `out_dir / <run_id>` with
        manifest.json, run.sqlite, and traces.jsonl
      * `--variants 1` yields exactly 1 variant in the manifest
    """
    pytest.importorskip("langgraph")

    fixture = tmp_path / "fx.py"
    fixture.write_text(
        "from langgraph.graph import END, START, StateGraph\n"
        "from typing import TypedDict\n"
        "from topofuzz.ir import LLMConfig\n"
        "\n"
        "class _S(TypedDict, total=False):\n"
        "    x: str\n"
        "\n"
        "def _a(s): return s\n"
        "\n"
        "def build_graph():\n"
        "    g = StateGraph(_S)\n"
        "    g.add_node('a', _a)\n"
        "    g.add_node('b', _a)\n"
        "    g.add_edge(START, 'a')\n"
        "    g.add_edge('a', 'b')\n"
        "    g.add_edge('b', END)\n"
        "    return g.compile()\n"
        "\n"
        "LLM_CONFIGS = {\n"
        "    'a': LLMConfig(model='google/gemini-2.5-pro'),\n"
        "    'b': LLMConfig(model='google/gemini-2.5-pro'),\n"
        "}\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "e.jsonl"
    # ``ground_truth_decision`` is required by the packaged lending
    # rubric's ``render_prompt`` (even in --dry-run, which renders to
    # surface template bugs early). Without it every cell would raise
    # KeyError inside the judge → judge_failure → §A status='failed' →
    # exit 2. Step 9 makes that failure path observable; pre-step-9 the
    # CLI hard-coded status='succeeded' regardless of judge failures,
    # which is what this test was inadvertently relying on.
    eval_path.write_text(
        '{"id": "e1", "input": "x", "ground_truth_decision": "approve"}\n'
        '{"id": "e2", "input": "y", "ground_truth_decision": "deny"}\n',
        encoding="utf-8",
    )

    async def fake_llm(**kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    # Wrap the real execute_run so the fake llm_callable is threaded in.
    from topofuzz import cli as cli_mod
    from topofuzz import orchestrator as orch_mod

    real_execute_run = orch_mod.execute_run

    async def execute_run_with_fake(plan, **kwargs):
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    # Step 10a moved `execute_run` to a lazy import inside `cli.run`
    # (§11a verify-path safety). Patch at the source module so the
    # lazy import resolves to the fake.
    monkeypatch.setattr(orch_mod, "execute_run", execute_run_with_fake)

    out_dir = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--n", "2",
            "--variants", "1",
            "--out", str(out_dir),
            "--no-cache",
            # ``--dry-run`` short-circuits the judge to a deterministic
            # ContinuousScore(0.5). Without it the packaged
            # ``lending_judge.txt`` rubric would try to parse the
            # ``fake_llm`` response "ok" as ``Score: N`` → every cell
            # marked judge_failure → step-9 §A status='failed' → exit 2.
            # This test exercises the CLI plumbing, not judge parsing.
            "--dry-run",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output

    # Discover the run dir (CLI mints a fresh date-prefixed ULID).
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "run.sqlite").exists()
    assert (run_dir / "traces.jsonl").exists()

    # --variants 1 must result in exactly 1 variant in the manifest.
    import json
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_variants"] == 1
    assert manifest["n_eval_rows"] == 2

    # SQLite must record 2 successful cells (1 variant × 2 eval rows).
    import sqlite3
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        cells = db.execute(
            "SELECT eval_id, error, cell_cost_usd FROM cells "
            "WHERE run_id = ?", (manifest["run_id"],)
        ).fetchall()
    finally:
        db.close()
    assert len(cells) == 2
    for eid, err, cost in cells:
        assert err is None
        assert cost is not None and cost > 0.0


# ---------------------------------------------------------------------------
# Step 7 — packaged rubric + --rubric + --dry-run
# ---------------------------------------------------------------------------


def test_packaged_default_rubric_resolves_to_existing_file():
    """The CLI's default rubric must resolve to a file that ships with
    the wheel. Hatch picks up everything under src/topofuzz/; the test
    confirms the resource is actually loadable via importlib.resources.
    """
    from topofuzz.cli import _packaged_default_rubric_path

    path = _packaged_default_rubric_path()
    assert path.exists(), f"rubric missing at {path}"
    text = path.read_text(encoding="utf-8")
    assert "{variant_id}" in text
    assert "{applicant_blurb}" in text
    assert "{ground_truth_decision}" in text
    assert "{agent_output}" in text


def test_run_rejects_missing_rubric_path(tmp_path):
    """A user-supplied --rubric path that doesn't exist must surface
    a friendly error, not a stack trace.
    """
    pytest.importorskip("langgraph")
    fixture = tmp_path / "fx.py"
    fixture.write_text(
        "from langgraph.graph import END, START, StateGraph\n"
        "from typing import TypedDict\n"
        "from topofuzz.ir import LLMConfig\n"
        "\n"
        "class _S(TypedDict, total=False):\n"
        "    x: str\n"
        "\n"
        "def _a(s): return s\n"
        "\n"
        "def build_graph():\n"
        "    g = StateGraph(_S)\n"
        "    g.add_node('a', _a)\n"
        "    g.add_edge(START, 'a')\n"
        "    g.add_edge('a', END)\n"
        "    return g.compile()\n"
        "\n"
        "LLM_CONFIGS = {'a': LLMConfig(model='google/gemini-2.5-pro')}\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text('{"id": "e1", "input": "x"}\n', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--rubric", str(tmp_path / "nope.txt"),
            "--yes",
        ],
    )
    assert result.exit_code == 2
    assert "rubric" in result.output.lower()


def test_run_dry_run_flag_skips_llm_judge(tmp_path, monkeypatch):
    """End-to-end dry-run: judge never calls an LLM; cells get
    ContinuousScore(0.5); manifest cli_args records ``dry_run: true``.
    """
    pytest.importorskip("langgraph")

    fixture = tmp_path / "fx.py"
    fixture.write_text(
        "from langgraph.graph import END, START, StateGraph\n"
        "from typing import TypedDict\n"
        "from topofuzz.ir import LLMConfig\n"
        "\n"
        "class _S(TypedDict, total=False):\n"
        "    x: str\n"
        "\n"
        "def _a(s): return s\n"
        "\n"
        "def build_graph():\n"
        "    g = StateGraph(_S)\n"
        "    g.add_node('a', _a)\n"
        "    g.add_edge(START, 'a')\n"
        "    g.add_edge('a', END)\n"
        "    return g.compile()\n"
        "\n"
        "LLM_CONFIGS = {'a': LLMConfig(model='google/gemini-2.5-pro')}\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text(
        '{"id": "e1", "input": "x", "ground_truth_decision": "approve"}\n'
        '{"id": "e2", "input": "y", "ground_truth_decision": "deny"}\n',
        encoding="utf-8",
    )

    async def fake_llm(**kwargs):
        return {
            "choices": [{"message": {"content": "agent decision: ok"}}],
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    from topofuzz import cli as cli_mod
    from topofuzz import orchestrator as orch_mod

    real_execute_run = orch_mod.execute_run

    async def execute_run_with_fake(plan, **kwargs):
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    # Step 10a moved `execute_run` to a lazy import inside `cli.run`
    # (§11a verify-path safety). Patch at the source module so the
    # lazy import resolves to the fake.
    monkeypatch.setattr(orch_mod, "execute_run", execute_run_with_fake)

    out_dir = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--n", "2",
            "--variants", "1",
            "--out", str(out_dir),
            "--no-cache",
            "--dry-run",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[DRY-RUN]" in result.output

    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    import json
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cli_args"]["dry_run"] is True
    assert manifest["cli_args"]["rubric"]  # packaged path was resolved
    assert manifest["cli_args"]["rubric_hash"]  # non-empty hash

    # Every cell's judge_result must be ContinuousScore(0.5) when dry-run.
    import sqlite3
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    try:
        rows = db.execute(
            "SELECT judge_kind, score_value FROM judge_results "
            "WHERE run_id = ?", (manifest["run_id"],)
        ).fetchall()
    finally:
        db.close()
    assert len(rows) == 2
    for kind, value in rows:
        assert kind == "continuous"
        assert value == 0.5


def test_run_dry_run_threads_renderer_into_orchestrator(tmp_path, monkeypatch):
    """The CLI must pass ``judge.render_prompt`` as the orchestrator's
    judge_prompt_renderer so the §7 cache slot keys on the rendered
    prompt (not the eval_row fallback). Verify via capture.
    """
    pytest.importorskip("langgraph")

    fixture = tmp_path / "fx.py"
    fixture.write_text(
        "from langgraph.graph import END, START, StateGraph\n"
        "from typing import TypedDict\n"
        "from topofuzz.ir import LLMConfig\n"
        "\n"
        "class _S(TypedDict, total=False):\n"
        "    x: str\n"
        "\n"
        "def _a(s): return s\n"
        "\n"
        "def build_graph():\n"
        "    g = StateGraph(_S)\n"
        "    g.add_node('a', _a)\n"
        "    g.add_edge(START, 'a')\n"
        "    g.add_edge('a', END)\n"
        "    return g.compile()\n"
        "\n"
        "LLM_CONFIGS = {'a': LLMConfig(model='google/gemini-2.5-pro')}\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text(
        '{"id": "e1", "input": "x", "ground_truth_decision": "approve"}\n',
        encoding="utf-8",
    )

    async def fake_llm(**kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    captured: dict = {}
    from topofuzz import cli as cli_mod
    from topofuzz import orchestrator as orch_mod
    real_execute_run = orch_mod.execute_run

    async def capture_execute_run(plan, **kwargs):
        captured["judge"] = kwargs.get("judge")
        captured["renderer"] = kwargs.get("judge_prompt_renderer")
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    monkeypatch.setattr(orch_mod, "execute_run", capture_execute_run)

    out_dir = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--n", "1",
            "--variants", "1",
            "--out", str(out_dir),
            "--no-cache",
            "--dry-run",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["judge"] is not None
    assert callable(captured["renderer"])
    # Renderer is judge.render_prompt — invoke and confirm it returns a str.
    rendered = captured["renderer"]({
        "variant_id": "v",
        "eval_row": {"id": "e", "input": "x", "ground_truth_decision": "approve"},
        "agent_output": "out",
    })
    assert isinstance(rendered, str)
    assert "x" in rendered  # the applicant input flowed in


# ---------------------------------------------------------------------------
# Step 7 — packaged lending_eval.csv loads cleanly
# ---------------------------------------------------------------------------


def test_packaged_lending_eval_csv_loads_with_loader():
    """The bundled lending_eval.csv must satisfy the loader's id+input
    contract AND carry the ground_truth_decision column the judge needs.
    """
    from importlib import resources

    from topofuzz.loaders import load_eval_rows

    resource = resources.files("topofuzz").joinpath("fixtures/lending_eval.csv")
    with resources.as_file(resource) as csv_path:
        rows = load_eval_rows(Path(csv_path))
    assert len(rows) == 50
    for row in rows:
        assert isinstance(row.get("id"), str) and row["id"].startswith("lending_")
        assert isinstance(row.get("input"), str) and len(row["input"]) > 20
        assert row.get("ground_truth_decision") in {"approve", "deny"}


# Audit B2: dry-run must NOT pollute the JudgeCache. The CLI guarantees
# this by forcing --no-cache when --dry-run is set. Verify the manifest's
# recorded cache field is null in dry-run, even when --cache is supplied.
def test_dry_run_forces_no_cache_at_cli_layer(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")

    fixture = tmp_path / "fx.py"
    fixture.write_text(
        "from langgraph.graph import END, START, StateGraph\n"
        "from typing import TypedDict\n"
        "from topofuzz.ir import LLMConfig\n"
        "\n"
        "class _S(TypedDict, total=False):\n"
        "    x: str\n"
        "\n"
        "def _a(s): return s\n"
        "\n"
        "def build_graph():\n"
        "    g = StateGraph(_S)\n"
        "    g.add_node('a', _a)\n"
        "    g.add_edge(START, 'a')\n"
        "    g.add_edge('a', END)\n"
        "    return g.compile()\n"
        "\n"
        "LLM_CONFIGS = {'a': LLMConfig(model='google/gemini-2.5-pro')}\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text(
        '{"id": "e1", "input": "x", "ground_truth_decision": "approve"}\n',
        encoding="utf-8",
    )

    async def fake_llm(**kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

    from topofuzz import cli as cli_mod
    from topofuzz import orchestrator as orch_mod
    real_execute_run = orch_mod.execute_run

    async def execute_run_with_fake(plan, **kwargs):
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    # Step 10a moved `execute_run` to a lazy import inside `cli.run`
    # (§11a verify-path safety). Patch at the source module so the
    # lazy import resolves to the fake.
    monkeypatch.setattr(orch_mod, "execute_run", execute_run_with_fake)

    out_dir = tmp_path / "runs"
    cache_dir = tmp_path / "real_cache"
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--n", "1",
            "--variants", "1",
            "--out", str(out_dir),
            "--cache", str(cache_dir),  # user explicitly asked for cache
            "--dry-run",                # …but dry-run wins
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output

    # Manifest's cli_args["cache"] should be null AND the cache dir
    # should NOT have been created on disk.
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    import json
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cli_args"]["cache"] is None, (
        "dry-run must force cache=None at the CLI layer (audit B2)"
    )
    assert not cache_dir.exists(), (
        "dry-run must not create the cache dir even when --cache is supplied"
    )
