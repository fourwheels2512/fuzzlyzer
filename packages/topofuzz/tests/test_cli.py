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


def test_scan_bad_fixture_exits_with_friendly_message():
    """``scan`` is wired in step 14 — bad fixture surfaces as exit 2,
    not a stub message. End-to-end coverage of the badge dispatch +
    lending_4role HIGH-RISK live in test_scan.py + test_cli_scan.py."""
    result = CliRunner().invoke(app, ["scan", "some-nonexistent-fixture:graph"])
    assert result.exit_code == 2
    assert "failed to load fixture" in (result.output or "").lower()


def test_report_bad_run_dir_exits_with_friendly_message():
    """`report` is wired in step 11 — bad run_dir surfaces as exit 2,
    not a stub message. End-to-end coverage lives in test_cli_report.py."""
    result = CliRunner().invoke(app, ["report", "some-nonexistent-dir"])
    assert result.exit_code == 2
    assert "not a directory" in (result.output or "").lower()


# `verify` is no longer a stub — see test_cli_verify_keygen.py for the
# step-10a tests of the real command. `report` likewise — see
# test_cli_report.py for the step-11 tests.


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


def test_run_jury_of_3_accepted_past_parse_layer(tmp_path):
    """Step 12b: --jury-of-3 is no longer a hard CLI reject. It now flows
    past the parse layer; downstream failures (missing fixture, missing
    eval, no env keys) surface as their normal exit codes — NOT as the
    pre-12b "v0.5 has Gemini only" usage error.

    Trip an error downstream of the parse layer (missing fixture path)
    to verify --jury-of-3 isn't trapped at the gate, and confirm the
    stderr hint about env keys is present.
    """
    eval_path = tmp_path / "e.jsonl"
    eval_path.write_text('{"id": "e1", "input": "x"}\n', encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "nonexistent/path.py:graph",
            "--eval", str(eval_path),
            "--jury-of-3",
        ],
    )
    # Non-zero exit is from the missing-fixture branch, not the old
    # jury rejection. The hint about env keys is informational — it
    # routes to stderr at parse time, BEFORE the fixture load fails,
    # so it must appear regardless of the downstream exit code.
    assert result.exit_code == 2
    # Typer's default CliRunner mixes stderr into ``output``; both the
    # env-key hint and the downstream "failed to load fixture" message
    # land there. The hint is what proves --jury-of-3 made it past the
    # parse layer (it's emitted AFTER the reserved-flag gate and
    # BEFORE the fixture load).
    # Pass-1 RT-6 audit fix: loosen the substring matchers to a
    # case-insensitive regex so a future "ANTHROPIC API key" / "Anthropic
    # API key" copy-edit of the hint doesn't break the test. The load-
    # bearing claim is "the message mentions all three vendor env-key
    # names"; exact substring is too brittle.
    import re

    combined = result.output or ""
    assert re.search(r"anthropic[_ ]api[_ ]key", combined, re.IGNORECASE)
    assert re.search(r"openai[_ ]api[_ ]key", combined, re.IGNORECASE)
    assert re.search(r"gemini[_ ]api[_ ]key", combined, re.IGNORECASE)
    assert re.search(r"failed to load fixture", combined, re.IGNORECASE)
    # Negative assertion: must NOT carry the pre-12b "v0.5 has Gemini
    # only" message — that means the bad-usage exit is gone.
    assert not re.search(r"v0\.5 has gemini only", combined, re.IGNORECASE)


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


def test_cli_run_wires_default_resolver_into_execute_run(tmp_path, monkeypatch):
    """The CLI MUST pass ``default_resolver`` (the vendor-dispatched
    canonical→served mapper) to ``execute_run``. Without this wiring
    the §7 cache keys on canonical IDs while the provider responds
    with bare/snapshot IDs — every call mismatch-warns and the cache
    misses on warm runs.

    Asserts the kwarg by capturing the call from a fake ``execute_run``.
    (closes BACKLOG: model_served_version_resolver)
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
        "LLM_CONFIGS = {\n"
        "    'a': LLMConfig(model='google/gemini-2.5-flash'),\n"
        "}\n",
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
            "model": "gemini-2.5-flash",  # bare served id
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    from topofuzz import orchestrator as orch_mod
    from topofuzz.runner import default_resolver

    captured: dict = {}
    real_execute_run = orch_mod.execute_run

    async def execute_run_capturing(plan, **kwargs):
        captured["kwargs"] = kwargs
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    monkeypatch.setattr(orch_mod, "execute_run", execute_run_capturing)

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

    # Object identity — not "a callable that happens to map the same".
    # If a future refactor decomposes the resolver into a partial / lambda
    # we'd want this assertion to fail and prompt updating the contract.
    assert captured["kwargs"].get("resolver") is default_resolver, (
        "CLI must wire default_resolver into execute_run; got "
        f"{captured['kwargs'].get('resolver')!r}"
    )


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


def test_run_jury_of_3_plumbs_into_manifest_cli_args(tmp_path, monkeypatch):
    """Pass-1 RT-M5 audit fix: end-to-end CLI run with --jury-of-3 --dry-run
    must write a manifest whose ``cli_args.jury_of_3 == True`` AND whose
    ``judge_model`` reflects the synthetic jury identity. This closes the
    chain the [JURY] banner depends on (see ``topofuzz.report.render_report``).
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
            "choices": [{"message": {"content": "agent decision: ok"}}],
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    from topofuzz import orchestrator as orch_mod
    real_execute_run = orch_mod.execute_run

    async def execute_run_with_fake(plan, **kwargs):
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    monkeypatch.setattr(orch_mod, "execute_run", execute_run_with_fake)

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
            "--jury-of-3",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output

    import json
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    manifest = json.loads(
        (run_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    # Falsifiability of the [JURY] banner: report.py:165 reads
    # manifest["cli_args"]["jury_of_3"]. If this field doesn't survive
    # CLI → write_manifest plumbing, the banner is dead.
    assert manifest["cli_args"]["jury_of_3"] is True
    # The user's --judge value is still preserved in cli_args for
    # forensic completeness, but the plan's judge_model reflects the
    # synthetic jury identity per the locked design.
    assert manifest["judge_model"].startswith("jury_of_3:")
    assert "anthropic" in manifest["judge_model"]
    assert "openai" in manifest["judge_model"]
    assert "gemini" in manifest["judge_model"]
    # Post-run accounting echo must appear in CLI output (DoD #14
    # falsifiability surface). Polish-audit DRIFT-3: the label is now
    # "vendor-call failure rate" to disambiguate it from the §8
    # cell-level judge_failure_rate.
    assert "vendor-call failure rate" in result.output.lower()
    assert "vendor_overlap" in result.output.lower()


def test_run_jury_of_3_warns_when_cache_intent_dropped(tmp_path, monkeypatch):
    """Pass-1 CR-M1 / RT-M3 audit fix: --jury-of-3 + --cache <path>
    drops the cache silently in v0.5. Emit a stderr hint so the user
    isn't surprised — and so a v0.5.1 per-vendor cache slicing follow-up
    has a clear remediation pointer.
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

    from topofuzz import orchestrator as orch_mod
    real_execute_run = orch_mod.execute_run

    async def execute_run_with_fake(plan, **kwargs):
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    monkeypatch.setattr(orch_mod, "execute_run", execute_run_with_fake)

    user_cache = tmp_path / "user-supplied-cache-dir"
    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--n", "1",
            "--variants", "1",
            "--out", str(tmp_path / "runs"),
            "--cache", str(user_cache),
            "--jury-of-3",
            "--dry-run",
            "--yes",
        ],
    )
    # Run completes (dry-run, fake LLM); warning is present in mixed stderr.
    # Pass-2 RT MINOR-5 audit fix: regex match keeps the assertion robust
    # against future copy-edits of the hint message.
    import re

    assert result.exit_code == 0, result.output
    assert re.search(
        r"disables\s+the\s+judge\s*cache", result.output, re.IGNORECASE
    )
    assert "ignored" in result.output.lower()


def test_run_jury_of_3_silent_when_cache_not_supplied(tmp_path, monkeypatch):
    """Pass-2 CR-Critical / RT-MAJOR-1 audit fix: a bare ``--jury-of-3``
    with NO ``--cache`` argument must NOT trigger the "you supplied"
    warning. Before pass-2, the typer default ``Path('.cache/topofuzz')``
    made every jury run claim the user supplied a path they hadn't.
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

    from topofuzz import orchestrator as orch_mod
    real_execute_run = orch_mod.execute_run

    async def execute_run_with_fake(plan, **kwargs):
        kwargs.setdefault("llm_callable", fake_llm)
        return await real_execute_run(plan, **kwargs)

    monkeypatch.setattr(orch_mod, "execute_run", execute_run_with_fake)

    result = CliRunner().invoke(
        app,
        [
            "run",
            f"{fixture}:build_graph",
            "--eval", str(eval_path),
            "--n", "1",
            "--variants", "1",
            "--out", str(tmp_path / "runs"),
            # No --cache argument!
            "--jury-of-3",
            "--dry-run",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    # The cache-disabled hint must NOT appear — user did not supply --cache.
    import re

    assert not re.search(
        r"disables\s+the\s+judge\s*cache", result.output, re.IGNORECASE
    ), f"Cache-disable warning fired without --cache. Output:\n{result.output}"
    # But the env-key hint about jury should still appear (it's
    # unconditional on --jury-of-3).
    assert re.search(r"anthropic[_ ]api[_ ]key", result.output, re.IGNORECASE)


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
