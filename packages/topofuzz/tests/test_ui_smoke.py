"""topofuzz.ui smoke tests — page renders, form posts, cost math, etc.

These tests stub the subprocess at the launch boundary so no real
``topofuzz run`` ever spawns. The contract under test is the FastAPI
route surface + the form-to-CSV-to-launch wiring. Real subprocess
behavior is covered by the separate ``test_runner`` / ``test_orchestrator``
suites.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from topofuzz.ui.app import build_app  # noqa: E402
from topofuzz.ui.cost import (  # noqa: E402
    COST_DIVISOR,
    format_dollars,
    raw_cap_for_real_budget,
    real_dollars,
)
from topofuzz.ui.presets import (  # noqa: E402
    DEFAULT_THOROUGHNESS,
    DEFAULT_WORKFLOW,
    THOROUGHNESS,
    WORKFLOWS,
    get_thoroughness,
    get_workflow,
)
from topofuzz.ui.runs import (  # noqa: E402
    Run,
    RunInProgressError,
    RunRegistry,
    Scenario,
    UnknownRunError,
    write_eval_csv,
)


# --------- cost.py -------------------------------------------------------


class TestCostMath:
    def test_divisor_is_2_75(self) -> None:
        """The 2.75x topofuzz-meter inflation must be the documented value."""
        assert COST_DIVISOR == 2.75

    def test_real_dollars_inverts_raw_cap(self) -> None:
        raw = 8.0
        real = real_dollars(raw)
        # round-trip
        assert raw_cap_for_real_budget(real) == pytest.approx(raw)

    def test_real_dollars_zero(self) -> None:
        assert real_dollars(0) == 0.0

    def test_real_dollars_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            real_dollars(-1)

    def test_raw_cap_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            raw_cap_for_real_budget(-1)

    def test_format_dollars_two_decimals(self) -> None:
        assert format_dollars(1.5) == "$1.50"
        assert format_dollars(0) == "$0.00"
        assert format_dollars(1234.567) == "$1,234.57"


# --------- presets.py ----------------------------------------------------


class TestPresets:
    def test_default_workflow_exists(self) -> None:
        assert DEFAULT_WORKFLOW in WORKFLOWS

    def test_default_thoroughness_exists(self) -> None:
        assert DEFAULT_THOROUGHNESS in THOROUGHNESS

    def test_balanced_uses_pro_judge(self) -> None:
        # The balanced preset is the sellable-demo configuration —
        # promoting it away from Pro silently would tank demo quality.
        assert THOROUGHNESS["balanced"].judge_model == "google/gemini-2.5-pro"

    def test_workflow_fixtures_resolve(self) -> None:
        # Each preset must point at a real fixture and rubric on disk —
        # otherwise the user clicks Run and we crash spawning the
        # subprocess.
        for workflow in WORKFLOWS.values():
            assert workflow.fixture_path.exists(), (
                f"workflow {workflow.key!r} fixture missing: "
                f"{workflow.fixture_path}"
            )
            assert workflow.rubric_path.exists(), (
                f"workflow {workflow.key!r} rubric missing: "
                f"{workflow.rubric_path}"
            )

    def test_get_workflow_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_workflow("does-not-exist")

    def test_get_thoroughness_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_thoroughness("does-not-exist")

    def test_starters_are_balanced(self) -> None:
        # Every workflow must include at least one approve-leaning
        # starter — the RT-specialist's "balance meter" finding. An
        # all-deny seed set teaches the user to file an all-deny eval,
        # which fuzzes to zero flips (degenerate).
        for workflow in WORKFLOWS.values():
            verdicts = {s.expected for s in workflow.starters}
            assert verdicts == {"approve", "deny"}, (
                f"workflow {workflow.key!r} starters must include both "
                f"approve and deny: got {verdicts!r}"
            )


# --------- runs.py: write_eval_csv ---------------------------------------


class TestWriteEvalCSV:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.csv"
        scenarios = [
            Scenario(
                case="Loan applicant A facts.",
                why_borderline="Short tenure.",
                expected="deny",
            ),
            Scenario(
                case="Loan applicant B facts.",
                why_borderline="",
                expected="approve",
            ),
        ]
        write_eval_csv(path, scenarios)
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["id"] == "scn_001"
        assert rows[1]["id"] == "scn_002"
        assert rows[0]["expected_canonical_verdict"] == "deny"
        assert rows[1]["expected_canonical_verdict"] == "approve"
        # id must NEVER be user-supplied — guard against the RT axis 1
        # duplicate-key collision.
        assert rows[0]["ground_truth_decision"] == "deny"

    def test_normalizes_smart_quotes(self, tmp_path: Path) -> None:
        # RT axis 1: a Word smart-quote paste must not reach the CSV.
        path = tmp_path / "eval.csv"
        scenarios = [
            Scenario(
                case="He said “hello” to her.",
                why_borderline="Some ‘soft quote’ risk.",
                expected="deny",
            )
        ]
        write_eval_csv(path, scenarios)
        text = path.read_text(encoding="utf-8")
        assert "“" not in text
        assert "”" not in text
        assert "‘" not in text
        assert "’" not in text
        assert '"hello"' in text

    def test_rejects_bad_verdict(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.csv"
        scenarios = [
            Scenario(case="x", why_borderline="", expected="maybe"),
        ]
        with pytest.raises(ValueError):
            write_eval_csv(path, scenarios)


# --------- runs.py: registry concurrency ---------------------------------


class TestRegistryConcurrency:
    def test_active_run_returns_none_when_empty(self, tmp_path: Path) -> None:
        reg = RunRegistry(runs_root=tmp_path)
        assert reg.active_run() is None

    def test_get_unknown_raises(self, tmp_path: Path) -> None:
        reg = RunRegistry(runs_root=tmp_path)
        with pytest.raises(UnknownRunError):
            reg.get("not-a-real-run")

    def test_second_launch_while_running_rejected(self, tmp_path: Path) -> None:
        # Inject a fake active run into the registry — the single-run
        # invariant must reject a second launch attempt.
        reg = RunRegistry(runs_root=tmp_path)
        import time as _time

        fake = Run(
            run_id="fake-1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=tmp_path / "fake-1",
            fixture_path=tmp_path / "noop.py",
            eval_csv_path=tmp_path / "noop.csv",
            rubric_path=tmp_path / "noop.txt",
            raw_cost_cap_usd=1.0,
            started_at=_time.time(),
        )
        fake.status = "running"
        reg._runs["fake-1"] = fake  # noqa: SLF001 — test reaches in deliberately
        workflow = get_workflow(DEFAULT_WORKFLOW)
        thoroughness = get_thoroughness(DEFAULT_THOROUGHNESS)
        scenarios = [
            Scenario(case="x", why_borderline="", expected="deny"),
        ]
        with pytest.raises(RunInProgressError):
            reg.launch(
                workflow=workflow, thoroughness=thoroughness, scenarios=scenarios
            )


# --------- app.py: route surface ----------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Pretend a key is set so we don't bounce into the no_key page.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    app = build_app(runs_root=tmp_path / "ui-runs")
    c = TestClient(app)
    # POST routes enforce same-origin via Origin header (CSRF defense in
    # app.py:_csrf_guard). Mirror TestClient's default Host="testserver"
    # so legitimate POSTs in tests don't trip the guard.
    c.headers["Origin"] = "http://testserver"
    return c


@pytest.fixture()
def client_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = build_app(runs_root=tmp_path / "ui-runs")
    return TestClient(app)


class TestRoutes:
    def test_healthz(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_landing_renders(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "Test your AI pipeline" in r.text

    def test_setup_renders_with_key(self, client: TestClient) -> None:
        r = client.get("/setup")
        assert r.status_code == 200
        assert "Pick a workflow" in r.text
        # "Start the test" is the primary CTA copy. Was once "Test for
        # flips" — the RT-audit's plain-English review flagged "flips"
        # as confusing for a 5th grader (could read as gymnastics).
        assert "Start the test" in r.text
        # Cost row uses the 2.75x divisor — the page must show a SMALL
        # real-dollar number, never the inflated raw topofuzz cap.
        # Balanced preset = $8.00 raw → ~$2.91 real (10 scenarios) or
        # ~$1.45 (5 starter scenarios). Either is fine; both are <$5
        # which is the test that matters (no raw $8 leaking through).
        cost_section = r.text.split('id="cost-row"', 1)[1].split("</div>")[0]
        # The cost-row div contains the formatted dollar amount.
        # Find any $ amount inside it.
        import re as _re
        amounts = _re.findall(r"\$([0-9]+)\.[0-9]{2}", cost_section)
        assert amounts, f"no dollar amount in cost row: {cost_section!r}"
        # Never show the raw $8 cap.
        assert all(int(amount) < 5 for amount in amounts), (
            f"cost row shows raw topofuzz dollars instead of real: {amounts!r}"
        )

    def test_setup_no_key_redirects_to_setup_page(
        self, client_no_key: TestClient
    ) -> None:
        r = client_no_key.get("/setup")
        assert r.status_code == 200
        assert "Gemini API key" in r.text

    def test_runs_history_empty(self, client: TestClient) -> None:
        r = client.get("/runs")
        assert r.status_code == 200
        assert "No runs yet" in r.text

    def test_setup_rejects_too_few_scenarios(self, client: TestClient) -> None:
        r = client.post(
            "/setup",
            data={
                "workflow": "loan-approval",
                "thoroughness": "quick",
                "scenario_case": ["only one"],
                "scenario_why": [""],
                "scenario_expected": ["deny"],
            },
        )
        # Form re-renders with an error message (400). Critical: the
        # subprocess must NOT have spawned.
        assert r.status_code == 400
        assert "at least three" in r.text.lower()

    def test_cost_preview_fragment(self, client: TestClient) -> None:
        r = client.get(
            "/setup/cost-preview",
            params={
                "workflow": "loan-approval",
                "thoroughness": "balanced",
                "n_scenarios": 5,
            },
        )
        assert r.status_code == 200
        assert "Estimated real cost" in r.text

    def test_add_scenario_card(self, client: TestClient) -> None:
        r = client.post("/setup/scenario")
        assert r.status_code == 200
        assert "What's the case?" in r.text

    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        r = client.get("/runs/nope/live")
        assert r.status_code == 404

    def test_setup_post_launches_run(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The hot path — POST /setup spawns a subprocess and redirects.

        We patch ``subprocess.Popen`` so no real topofuzz process spawns;
        the test only verifies the route wiring + CSV write.
        """

        class FakeProc:
            stdout = iter(["[fake] starting\n", "[fake] done\n"])
            returncode = 0

            def wait(self) -> int:
                return 0

            def terminate(self) -> None:
                return

        with patch("topofuzz.ui.runs.subprocess.Popen", return_value=FakeProc()):
            r = client.post(
                "/setup",
                data={
                    "workflow": "loan-approval",
                    "thoroughness": "quick",
                    "scenario_case": [
                        "Applicant A facts here.",
                        "Applicant B facts here.",
                        "Applicant C facts here.",
                    ],
                    "scenario_why": ["", "", ""],
                    "scenario_expected": ["deny", "approve", "deny"],
                },
                follow_redirects=False,
            )
        assert r.status_code == 303
        assert r.headers["location"].startswith("/runs/")
        # Verify the CSV was actually written with the user's content.
        # The redirect target = /runs/{id}/live, run dir = ui-runs/{id}/.
        run_id = r.headers["location"].split("/")[2]
        eval_csv = tmp_path / "ui-runs" / run_id / "eval.csv"
        assert eval_csv.exists()
        rows = list(csv.DictReader(eval_csv.open(encoding="utf-8")))
        assert len(rows) == 3
        assert rows[0]["input"] == "Applicant A facts here."


# --------- helpers -------------------------------------------------------


class TestSnapshot:
    """Run.snapshot() must always return divided-by-2.75 real dollars."""

    def test_snapshot_dollars_are_real_not_raw(self, tmp_path: Path) -> None:
        import time as _time

        run = Run(
            run_id="t1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=tmp_path,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        run.parsed.raw_spend_usd = 5.50  # raw topofuzz units
        snap = run.snapshot()
        assert snap["real_spend_usd"] == pytest.approx(5.50 / 2.75, rel=1e-3)
        assert snap["real_cap_usd"] == pytest.approx(8.0 / 2.75, rel=1e-3)


class TestErrorClassification:
    def test_auth_failure_detected_from_log_line(self, tmp_path: Path) -> None:
        import time as _time

        run = Run(
            run_id="t1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=tmp_path,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        run.append_log("openai.AuthenticationError: 401 Unauthorized")
        assert run.parsed.error_class == "auth"

    def test_inner_run_dir_rejects_symlink(self, tmp_path: Path) -> None:
        """RT-audit B3: a symlink in out_dir must NOT become inner_run_dir."""
        import time as _time

        # Skip on Windows without symlink privilege — `os.symlink` requires
        # SeCreateSymbolicLinkPrivilege which most user accounts lack.
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        target = tmp_path / "elsewhere"
        target.mkdir()
        try:
            os.symlink(target, out_dir / "malicious-symlink", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        run = Run(
            run_id="t1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=out_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        # The only subdir is a symlink pointing outside out_dir.
        # inner_run_dir MUST refuse it.
        assert run.inner_run_dir is None

    def test_inner_run_dir_caches_on_first_resolve(self, tmp_path: Path) -> None:
        """Code-review M2 + RT-audit B3: cached after first non-None find."""
        import time as _time

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        sub = out_dir / "topofuzz-run-id"
        sub.mkdir()
        run = Run(
            run_id="t1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=out_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        first = run.inner_run_dir
        assert first == sub
        # Drop a malicious dir later — cached resolve should NOT swap to it.
        intruder = out_dir / "intruder"
        intruder.mkdir()
        # Bump intruder's mtime so an mtime-ranking scan would prefer it.
        intruder_stat = intruder.stat()
        os.utime(intruder, (intruder_stat.st_atime, intruder_stat.st_mtime + 100))
        second = run.inner_run_dir
        assert second == first  # still the original

    def test_cancel_sets_explicit_flag(self, tmp_path: Path) -> None:
        """Code-review M1: cancel sets a flag, not just terminate()."""
        import time as _time

        registry = RunRegistry(runs_root=tmp_path)
        run = Run(
            run_id="t1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=tmp_path / "t1",
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        run.status = "running"
        registry._runs["t1"] = run  # noqa: SLF001
        registry.cancel("t1")
        assert run.cancel_requested is True

    def test_rate_limit_detected_from_log_line(self, tmp_path: Path) -> None:
        import time as _time

        run = Run(
            run_id="t1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=tmp_path,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        run.append_log("HTTP 429 rate-limit, retrying in 30s")
        assert run.parsed.error_class == "rate_limit"
