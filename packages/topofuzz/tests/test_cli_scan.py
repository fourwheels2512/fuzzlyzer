# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Step 14: ``topofuzz scan FIXTURE_PATH`` end-to-end coverage.

Three planes:

* Happy path on the canonical ``lending_4role`` fixture — must report
  HIGH RISK + a JSON-on-stdout document that parses cleanly.
* Exit-code policy per BLUEPRINT DoD #11: exit 0 regardless of badge
  colour. Tested on HIGH (lending_4role), MEDIUM (2-agent distinct-
  models chain), and a promotion-cascade case (HIGH via base+promos).
* Reserved-flag / unknown-flag rejection routes through the same
  ``_reject_*`` helpers as the rest of the CLI — symmetry sanity test.

Stdout/stderr are tested separately via ``CliRunner(mix_stderr=False)``
so the test cannot be fooled by future ordering changes (audit fix
RT-m3, pass-1): the banner MUST live on stderr; the JSON MUST live on
stdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
from typer.testing import CliRunner

# ``topofuzz scan`` transitively imports langgraph at command time
# (via loaders.load_topology). Skip the whole module on verify-only
# installs, matching the pattern in test_cli.py.
pytest.importorskip("langgraph")

from topofuzz.cli import app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runner() -> CliRunner:
    """``CliRunner`` with stdout/stderr split so tests can assert the
    banner-to-stderr / JSON-to-stdout contract per RT-m3 (pass-1).

    Click 8.2 (Jan 2025) removed the ``mix_stderr`` constructor kwarg
    and made stream separation the default — the kwarg now raises
    ``TypeError``. Pass-2 audit fix CR-BLOCKER B-1: try the legacy
    signature first; on TypeError fall through to the modern default.
    ``topofuzz`` does not pin Click in pyproject so both versions must
    work for downstream consumers.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        # Click >= 8.2: streams are separated by default. ``result.stderr``
        # is populated and ``result.stdout`` does not include it.
        return CliRunner()


def _parse_scan_stdout(stdout: str) -> dict:
    """Parse the JSON document on stdout. With ``mix_stderr=False``
    stdout contains *only* the JSON line so a single ``json.loads`` on
    the stripped output is sufficient; the helper exists so a future
    change to stdout (e.g. adding a trailing summary line) surfaces here
    rather than in every test.
    """
    stripped = stdout.strip()
    assert stripped, f"Expected JSON on stdout; got empty. Stdout: {stdout!r}"
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scan_on_packaged_lending_fixture_reports_high_risk() -> None:
    """The canonical ``lending_4role`` fixture is the BLUEPRINT §5a
    HIGH-RISK marketing example: 4-agent linear chain, no aggregator,
    no judge node. Two model pairs (not all-share) so no model promo.
    """
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "topofuzz"
        / "fixtures"
        / "lending_4role.py"
    )
    spec = f"{fixture_path}:build_graph"
    result = _runner().invoke(app, ["scan", spec])
    assert result.exit_code == 0, (
        f"Expected exit 0 per DoD #11; got {result.exit_code}.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    # Banner MUST land on stderr (RT-m3 contract).
    assert "HIGH RISK" in result.stderr
    payload = _parse_scan_stdout(result.stdout)
    assert payload["badge"] == "HIGH"
    assert "linear_chain_3plus_no_aggregator" in payload["matched_patterns"]
    # lending_4role has 2 model pairs, NOT all-share → promo must NOT fire.
    assert "all_agents_share_model" not in payload["matched_patterns"]
    assert "rationale" in payload and isinstance(payload["rationale"], str)


def _write_two_agent_distinct_models_fixture(path: Path) -> None:
    """Helper: a 2-agent LangGraph linear chain with distinct models and
    temperature=0.0 on every agent. The §5a base rule fires
    (``linear_chain_2_no_aggregator``, MEDIUM); neither promotion does.
    Result: MEDIUM, no cap-out — the inverse of the lending_4role HIGH
    case.
    """
    path.write_text(
        dedent(
            """\
            from typing import TypedDict

            from topofuzz.ir import LLMConfig


            class _S(TypedDict, total=False):
                x: str


            LLM_CONFIGS = {
                "a0": LLMConfig(model="m_a", temperature=0.0),
                "a1": LLMConfig(model="m_b", temperature=0.0),
            }


            def _stub(state):
                return state


            def build_graph():
                from langgraph.graph import END, START, StateGraph

                g = StateGraph(_S)
                g.add_node("a0", _stub)
                g.add_node("a1", _stub)
                g.add_edge(START, "a0")
                g.add_edge("a0", "a1")
                g.add_edge("a1", END)
                return g.compile()
            """
        ),
        encoding="utf-8",
    )


def test_scan_medium_badge_still_exits_zero(tmp_path: Path) -> None:
    """CR-MAJOR-3 audit fix (pass-1): a non-HIGH badge must also exit 0.
    DoD #11 says "exit code 0 on success regardless of badge color." A
    future refactor that wires ``typer.Exit(code=...)`` to badge severity
    would silently break this; the test guards that contract from the
    CLI plane (the scanner unit tests cover badge semantics).
    """
    fixture = tmp_path / "medium_chain.py"
    _write_two_agent_distinct_models_fixture(fixture)
    result = _runner().invoke(app, ["scan", f"{fixture}:build_graph"])
    assert result.exit_code == 0, (
        f"Expected exit 0 per DoD #11; got {result.exit_code}.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    payload = _parse_scan_stdout(result.stdout)
    assert payload["badge"] == "MEDIUM"
    assert "linear_chain_2_no_aggregator" in payload["matched_patterns"]
    # No promotions on this fixture (distinct models, temp=0).
    assert "all_agents_share_model" not in payload["matched_patterns"]
    assert "temperature_gt_0" not in payload["matched_patterns"]
    assert "MEDIUM RISK" in result.stderr


def test_scan_rejects_reserved_flag() -> None:
    """``topofuzz scan --sandbox foo path:graph`` must surface the v1.0
    reserved-flag message, not a "no such option" error."""
    result = _runner().invoke(app, ["scan", "x:y", "--sandbox", "foo"])
    assert result.exit_code == 2
    assert "--sandbox" in result.stderr.lower()
    assert "reserved" in result.stderr.lower()


def test_scan_rejects_unknown_flag() -> None:
    """Unknown flags (typos) must surface as a usage error per the
    pattern in :func:`topofuzz.cli._reject_unknown_extras`."""
    result = _runner().invoke(app, ["scan", "x:y", "--no-such-flag"])
    assert result.exit_code == 2
    assert "unknown option" in result.stderr.lower()


def test_scan_bad_fixture_path_exits_2(tmp_path: Path) -> None:
    """Missing fixture file → friendly exit 2, matches ``run`` behaviour."""
    result = _runner().invoke(
        app, ["scan", f"{tmp_path / 'nonexistent.py'}:graph"]
    )
    assert result.exit_code == 2
    assert "failed to load fixture" in result.stderr.lower()


def test_scan_broken_fixture_module_exits_2_not_traceback(
    tmp_path: Path,
) -> None:
    """CR-MAJOR-1 audit fix (pass-1): a fixture file with a Python
    syntax error or import-time failure must produce the documented
    exit-2 "failed to load fixture" message, not a raw traceback.
    """
    fixture = tmp_path / "broken.py"
    # Genuine SyntaxError surfaced by ``exec_module`` — historically this
    # bubbled up untouched and produced a raw traceback. The audit fix
    # adds SyntaxError + ImportError to the caught set.
    fixture.write_text(
        "this is not valid python syntax !!\n", encoding="utf-8"
    )
    result = _runner().invoke(app, ["scan", f"{fixture}:graph"])
    assert result.exit_code == 2, (
        f"Expected exit 2; got {result.exit_code}.\n"
        f"stderr: {result.stderr!r}"
    )
    assert "failed to load fixture" in result.stderr.lower()


def test_scan_stdout_json_is_machine_parseable(tmp_path: Path) -> None:
    """The JSON line on stdout must parse as a dict with the documented
    keys and types — protects CI consumers piping ``topofuzz scan ... | jq``.

    The fixture is a real LangGraph 2-agent chain wired through the
    public ``LLM_CONFIGS`` surface (same shape as ``lending_4role``).
    Both nodes share model ``"m"`` and one has temperature > 0, so we
    cover the §5a all-share + temp>0 promotions through the CLI.
    """
    fixture = tmp_path / "tiny.py"
    fixture.write_text(
        dedent(
            """\
            from typing import TypedDict

            from topofuzz.ir import LLMConfig


            class _S(TypedDict, total=False):
                x: str


            LLM_CONFIGS = {
                "a0": LLMConfig(model="m", temperature=0.5),
                "a1": LLMConfig(model="m", temperature=0.0),
            }


            def _stub(state):
                return state


            def build_graph():
                from langgraph.graph import END, START, StateGraph

                g = StateGraph(_S)
                g.add_node("a0", _stub)
                g.add_node("a1", _stub)
                g.add_edge(START, "a0")
                g.add_edge("a0", "a1")
                g.add_edge("a1", END)
                return g.compile()
            """
        ),
        encoding="utf-8",
    )
    result = _runner().invoke(app, ["scan", f"{fixture}:build_graph"])
    assert result.exit_code == 0, (
        f"Expected exit 0 per DoD #11; got {result.exit_code}.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    payload = _parse_scan_stdout(result.stdout)
    assert payload.keys() == {"badge", "matched_patterns", "rationale"}
    assert payload["badge"] in {"LOW", "MEDIUM", "HIGH"}
    # 2-agent chain, all-same-model + temp>0 → MEDIUM base + 2 promos → HIGH.
    assert payload["badge"] == "HIGH"
    assert "linear_chain_2_no_aggregator" in payload["matched_patterns"]
    assert "all_agents_share_model" in payload["matched_patterns"]
    assert "temperature_gt_0" in payload["matched_patterns"]
    # Stdout must contain ONLY the JSON line, not the banner — protects
    # the ``| jq`` pipe contract (RT-m3 audit fix, pass-1).
    assert "RISK" not in result.stdout
    assert "Heuristic only" not in result.stdout
