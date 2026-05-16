# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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


# ---------------------------------------------------------------------------
# F1: Wilson CI surfacing in the UI headline subtitle
# ---------------------------------------------------------------------------


class TestBuildHeadlineWilson:
    """Coverage for ``ui.app._build_headline``'s F1 Wilson clause."""

    @staticmethod
    def _make_run(tmp_path: Path, status: str = "succeeded") -> "Run":
        import time as _time

        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        eval_csv = tmp_path / "e.csv"
        eval_csv.write_text(
            "eval_id,prompt\nE1,p1\nE2,p2\nE3,p3\nE4,p4\nE5,p5\n",
            encoding="utf-8",
        )
        run = Run(
            run_id="t1",
            workflow_key="loan-approval",
            thoroughness_key="balanced",
            out_dir=out_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=eval_csv,
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        run.status = status  # type: ignore[assignment]
        return run

    def test_wilson_sentence_appears_in_subtitle_when_present(
        self, tmp_path: Path,
    ) -> None:
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": 0.20,
                    "wilson_ci_upper": 0.65,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence:" in out["subtitle"]
        # Rounded to integer percent.
        assert "between 20% and 65%" in out["subtitle"]
        # Default conf level rendered (no wilson_conf_level in dict).
        assert "95% confident" in out["subtitle"]

    def test_wilson_sentence_dropped_when_keys_missing(
        self, tmp_path: Path,
    ) -> None:
        """Pre-F1 runs / degenerate metric → no Wilson sentence."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                # No wilson_ci_lower / wilson_ci_upper.
                "verdict_flip_rate": {"point": 0.4},
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence" not in out["subtitle"]

    def test_wilson_sentence_dropped_on_partial_bounds(
        self, tmp_path: Path,
    ) -> None:
        """Single bound → never render half a CI."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": 0.20,
                    "wilson_ci_upper": None,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence" not in out["subtitle"]

    def test_wilson_sentence_dropped_on_out_of_range_bounds(
        self, tmp_path: Path,
    ) -> None:
        """Bound outside [0, 1] is rejected as malformed."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": 0.20,
                    "wilson_ci_upper": 1.7,  # invalid
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence" not in out["subtitle"]

    def test_wilson_sentence_dropped_on_nan_bound(
        self, tmp_path: Path,
    ) -> None:
        """NaN slipping through json round-trip → no clause, no crash."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": float("nan"),
                    "wilson_ci_upper": 0.65,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence" not in out["subtitle"]

    def test_wilson_sentence_rejects_bool_bounds(
        self, tmp_path: Path,
    ) -> None:
        """``True`` / ``False`` are ``isinstance(_, int)`` in Python; they
        must NOT slip through the float gate as 0/1 percent CIs."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": True,
                    "wilson_ci_upper": False,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence" not in out["subtitle"]

    def test_wilson_sentence_dropped_on_lower_greater_than_upper(
        self, tmp_path: Path,
    ) -> None:
        """Corrupted dict with lower > upper → no rendered nonsense
        ``"between 50% and 30%"`` (Fuzzer-MAJOR-1)."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": 0.50,
                    "wilson_ci_upper": 0.30,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence" not in out["subtitle"]

    def test_wilson_sentence_renders_persisted_conf_level(
        self, tmp_path: Path,
    ) -> None:
        """A non-default ``wilson_conf_level`` (e.g. 0.99) is reflected
        in the rendered percent (Fuzzer-MAJOR-4)."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": 0.10,
                    "wilson_ci_upper": 0.75,
                    "wilson_conf_level": 0.99,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "99% confident" in out["subtitle"]
        assert "95% confident" not in out["subtitle"]


class TestBuildHeadlineConfirmedBranch:
    """The confirmed-branch UI path: dict-shaped confirmed_verdict_flip_rate
    + Wilson CI surfacing on the noise-corrected number."""

    @staticmethod
    def _make_run(tmp_path: Path) -> "Run":
        return TestBuildHeadlineWilson._make_run(tmp_path)

    def test_confirmed_branch_fires_for_dict_shape(
        self, tmp_path: Path,
    ) -> None:
        """Pre-F1 bug: UI tested ``isinstance(confirmed_rate, (int, float))``
        but the schema is a dict — confirmed branch never fired. Fixed
        as a prerequisite of Wilson-on-confirmed surfacing."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "confirmed_verdict_flip_rate": {
                    "point": 0.3,
                    "confirmed": 2,
                    "flaky": 4,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        # title: confirmed_count = round(0.3 * 5) = 2 (the "noise-checked count")
        assert "your AI pipeline changed its mind on 2 of them" in out["title"]
        # subtitle: the confirmed-branch caveat text
        assert "noise-checked count" in out["subtitle"]

    def test_confirmed_branch_carries_wilson_from_confirmed_dict(
        self, tmp_path: Path,
    ) -> None:
        """Wilson CI on the noise-corrected metric — user's explicit
        requirement: ``"both numbers get CIs you have full coverage; if
        only raw, that's a half-fix"``."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                # Raw rate has Wilson on a DIFFERENT bound — confirmed
                # branch must use the CONFIRMED Wilson, not the raw one.
                "verdict_flip_rate": {
                    "point": 0.4,
                    "wilson_ci_lower": 0.05,
                    "wilson_ci_upper": 0.85,
                },
                "confirmed_verdict_flip_rate": {
                    "point": 0.3,
                    "confirmed": 2,
                    "flaky": 4,
                    "wilson_ci_lower": 0.15,
                    "wilson_ci_upper": 0.55,
                    "wilson_conf_level": 0.95,
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        # Confirmed branch fired (title uses confirmed_count of 2 not raw 2).
        assert "your AI pipeline changed its mind on 2 of them" in out["title"]
        # Wilson sentence reflects the CONFIRMED bounds (15-55), NOT the
        # raw bounds (5-85).
        assert "between 15% and 55%" in out["subtitle"]
        assert "between 5% and 85%" not in out["subtitle"]

    def test_confirmed_branch_no_wilson_when_keys_absent(
        self, tmp_path: Path,
    ) -> None:
        """Confirmed dict without Wilson keys → no CI sentence; the
        confirmed-branch caveat still renders."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "confirmed_verdict_flip_rate": {
                    "point": 0.3,
                    "confirmed": 2,
                    "flaky": 4,
                    # No wilson_*.
                },
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "Statistical confidence" not in out["subtitle"]
        # Caveat still there.
        assert "noise-checked count" in out["subtitle"]

    def test_scalar_legacy_form_still_renders(
        self, tmp_path: Path,
    ) -> None:
        """Forward-compat: if a hypothetical producer emits the legacy
        scalar form for confirmed_verdict_flip_rate, the count still
        renders. No CI on this path."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "confirmed_verdict_flip_rate": 0.3,
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "your AI pipeline changed its mind on 2 of them" in out["title"]
        assert "Statistical confidence" not in out["subtitle"]


class TestBuildHeadlineFinitePointGate:
    """Audit pass-2 Fuzzer-MAJOR-1: NaN / Inf in `point` would crash
    `round()` (ValueError on NaN, OverflowError on Inf) and surface as
    HTTP 500 on the results page. Producer math is finite; this is
    defence in depth at the rendering boundary on both the raw and the
    confirmed-rate branches."""

    @staticmethod
    def _make_run(tmp_path: Path) -> "Run":
        return TestBuildHeadlineWilson._make_run(tmp_path)

    def test_raw_point_nan_does_not_crash(
        self, tmp_path: Path,
    ) -> None:
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": float("nan")},
                "judge_failure_rate": 0.0,
            }
        }
        # Must not raise.
        out = _build_headline(run=run, summary=summary, workflow=wf)
        # Falls through to the malformed-summary path because raw_count
        # stays None and confirmed_count is None.
        assert out["flip_count"] is None

    def test_raw_point_inf_does_not_crash(
        self, tmp_path: Path,
    ) -> None:
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": float("inf")},
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert out["flip_count"] is None

    def test_confirmed_point_nan_does_not_crash(
        self, tmp_path: Path,
    ) -> None:
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "confirmed_verdict_flip_rate": {"point": float("nan")},
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        # Confirmed branch skipped (NaN gated), falls to raw branch.
        assert "your AI pipeline changed its mind on about 2 of them" in out["title"]

    def test_confirmed_point_inf_does_not_crash(
        self, tmp_path: Path,
    ) -> None:
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "confirmed_verdict_flip_rate": {"point": float("inf")},
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert "your AI pipeline changed its mind on about 2 of them" in out["title"]

    def test_legacy_scalar_confirmed_inf_does_not_crash(
        self, tmp_path: Path,
    ) -> None:
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "confirmed_verdict_flip_rate": float("inf"),
                "judge_failure_rate": 0.0,
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        # Legacy scalar Inf rejected, falls to raw branch.
        assert "your AI pipeline changed its mind on about 2 of them" in out["title"]

    def test_judge_failure_rate_nan_does_not_crash(
        self, tmp_path: Path,
    ) -> None:
        """Audit pass-3 Fuzzer-MAJOR-2: same NaN/Inf round() crash class
        on the judge_failure_rate branch. ``or 0`` does NOT replace NaN
        (NaN is truthy in Python). The fix is the same finite+bool gate
        pattern used on the three point sites."""
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "judge_failure_rate": float("nan"),
            }
        }
        # Must not raise. Renders the raw branch with no judge-fail caveat.
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert out["flip_count"] == 2  # 0.4 * 5

    def test_judge_failure_rate_inf_does_not_crash(
        self, tmp_path: Path,
    ) -> None:
        from topofuzz.ui.app import _build_headline

        run = self._make_run(tmp_path)
        wf = WORKFLOWS[DEFAULT_WORKFLOW]
        summary = {
            "headline": {
                "n_evals": 5,
                "n_variants": 4,
                "verdict_flip_rate": {"point": 0.4},
                "judge_failure_rate": float("inf"),
            }
        }
        out = _build_headline(run=run, summary=summary, workflow=wf)
        assert out["flip_count"] == 2
