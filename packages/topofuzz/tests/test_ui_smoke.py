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
import subprocess
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
    _build_topofuzz_cmd,
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

    def test_every_builtin_workflow_ships_enough_starters_to_run(self) -> None:
        # POST /setup refuses fewer than three cases. A built-in tile
        # with only two starters fails the first time someone picks it
        # and clicks "Run the test" without typing anything.
        for key, workflow in WORKFLOWS.items():
            filled = [s for s in workflow.starters if s.case.strip()]
            assert len(filled) >= 3, f"{key} ships only {len(filled)} starter case(s)"

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


# --------- UX-2: the spawned child must never hang on stdin --------------
#
# Pre-fix, ``topofuzz run`` is spawned with no ``stdin`` redirect and no
# ``--accept-license``. On a fresh machine (or after a version bump —
# acceptance is version-keyed) the child hits its own interactive BSL
# prompt, blocks on stdin, gets EOF, and exits 2 — surfaced to the UI user
# only as an opaque "exit code 2".


class TestBuildCmdAndLaunchNeverHang:
    def test_build_cmd_includes_accept_license_before_run_subcommand(
        self, tmp_path: Path
    ) -> None:
        cmd = _build_topofuzz_cmd(
            fixture_path=tmp_path / "fixture.py",
            eval_csv=tmp_path / "eval.csv",
            n=5,
            variants=4,
            judge_model="google/gemini-2.5-pro",
            rubric_path=tmp_path / "rubric.txt",
            raw_cost_cap_usd=8.0,
            out_dir=tmp_path / "out",
        )
        assert "--accept-license" in cmd
        assert "run" in cmd
        # It's a root-level flag — Click/typer requires it BEFORE the
        # subcommand, not after (see cli.py's _reject_unknown_extras
        # placement-hint test for the mirror-image CLI-usage mistake).
        assert cmd.index("--accept-license") < cmd.index("run")

    def test_launch_spawns_subprocess_with_stdin_devnull(
        self, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}

        class FakeProc:
            stdout = iter(["[fake] done\n"])
            returncode = 0

            def wait(self) -> int:
                return 0

            def terminate(self) -> None:
                return

        def fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN003
            captured.update(kwargs)
            return FakeProc()

        with patch("topofuzz.ui.runs.subprocess.Popen", side_effect=fake_popen):
            reg = RunRegistry(runs_root=tmp_path / "ui-runs")
            workflow = get_workflow(DEFAULT_WORKFLOW)
            thoroughness = get_thoroughness(DEFAULT_THOROUGHNESS)
            scenarios = [
                Scenario(case="x", why_borderline="", expected="deny"),
            ]
            reg.launch(
                workflow=workflow, thoroughness=thoroughness, scenarios=scenarios
            )
        assert captured.get("stdin") == subprocess.DEVNULL


# --------- app.py: route surface ----------------------------------------


@pytest.fixture(autouse=True)
def _isolate_keystore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the in-browser keystore at a tmp dir and clear runtime cache.

    Without this, tests on a developer's machine inherit
    ``~/.topofuzz/keys/{vendor}`` (set via the browser UI) and the no-key
    paths look like they have keys. Also clear every non-Google env var
    so a developer with OPENAI_API_KEY set doesn't surprise the tests.
    """
    from topofuzz.ui import _keystore as _ks
    monkeypatch.setattr(_ks, "_KEY_DIR", tmp_path / "keys-isolated")
    monkeypatch.setattr(_ks, "_KEY_FILE", tmp_path / "keys-isolated" / "google")
    monkeypatch.setattr(_ks, "_runtime", {})
    for spec in _ks.VENDORS.values():
        if spec.vendor != "google":
            # Google is set/unset per-fixture; clear the rest unconditionally
            # so multi-vendor tests start from a clean slate.
            monkeypatch.delenv(spec.env_var, raising=False)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Pretend a key is set via env so we don't bounce into the key-entry card.
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
    c = TestClient(app)
    c.headers["Origin"] = "http://testserver"  # for POST /key
    return c


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``Path.home()`` (and therefore ``license_check.acceptance_
    path()``) at a FRESH tmp dir with no acceptance file on record.

    Overrides conftest.py's autouse ``_preaccept_license`` fixture (which
    pre-accepts the license for every other UI test so the pre-existing
    route tests don't have to think about it) — same override pattern as
    ``test_license_check.py``'s identically-named fixture.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def client_unaccepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_home: Path
) -> TestClient:
    """Same as ``client`` but with NO license acceptance on record —
    exercises the UX-1 in-browser accept gate."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    app = build_app(runs_root=tmp_path / "ui-runs")
    c = TestClient(app)
    c.headers["Origin"] = "http://testserver"
    return c


class TestRoutes:
    def test_healthz(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_landing_renders(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "Check your AI for unfair decisions" in r.text
        # Env-var Google key is present → landing shows the "saved" row,
        # not the paste-key card.
        assert "Google (Gemini)" in r.text
        assert "Remove" in r.text  # the saved-state button

    def test_landing_shows_scope_disclaimer(self, client: TestClient) -> None:
        """UX-6: a founder-dogfooding session found a real person who
        believed the loan/insurance demo would predict whether THEY
        personally would get a real loan. A short, always-visible,
        factual disclaimer must be on every page (it lives in the
        shared footer, so this also covers the results page)."""
        r = client.get("/")
        assert r.status_code == 200
        assert (
            "does not make real loan, insurance, hiring, or other "
            "decisions about any person" in r.text
        )

    def test_setup_renders_with_key(self, client: TestClient) -> None:
        r = client.get("/setup")
        assert r.status_code == 200
        assert "What kind of AI is this" in r.text
        # "Run the test" is the primary CTA copy.
        assert "Run the test" in r.text
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

    def test_setup_no_key_redirects_to_landing_with_paste_card(
        self, client_no_key: TestClient
    ) -> None:
        # /setup is gated by key presence. With no key, the server redirects
        # to / so the user can paste one in-browser. TestClient follows
        # redirects by default.
        r = client_no_key.get("/setup")
        assert r.status_code == 200
        assert "paste your Google AI key" in r.text
        assert "AIza" in r.text  # the format hint
        # No more "stop the server" instructions.
        assert "environment variable" not in r.text
        assert "Ctrl+C" not in r.text

    def test_post_google_key_valid_saves_and_redirects(
        self, client_no_key: TestClient
    ) -> None:
        valid = "AIza" + "x" * 35  # 39 chars, starts with AIza
        r = client_no_key.post(
            "/key/google",
            data={"key": valid, "remember": "1"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/"
        # Following the redirect shows the "saved" state for Google.
        r2 = client_no_key.get("/")
        assert "Google (Gemini)" in r2.text
        # Tail of the key is the last 4 chars.
        assert valid[-4:] in r2.text
        # Full key is never echoed back into HTML.
        assert valid not in r2.text

    def test_post_key_malformed_returns_400_with_error(
        self, client_no_key: TestClient
    ) -> None:
        r = client_no_key.post(
            "/key/google",
            data={"key": "not-a-real-key"},
            follow_redirects=False,
        )
        assert r.status_code == 400
        assert "starts with AIza" in r.text
        # The paste card is still shown so the user can try again.
        assert "paste your Google AI key" in r.text

    def test_post_key_empty_returns_400(
        self, client_no_key: TestClient
    ) -> None:
        r = client_no_key.post(
            "/key/google", data={"key": ""}, follow_redirects=False
        )
        assert r.status_code == 400
        assert "Paste your key" in r.text

    def test_post_key_remove_clears_runtime_and_disk(
        self, client_no_key: TestClient, tmp_path: Path
    ) -> None:
        from topofuzz.ui import _keystore as _ks
        valid = "AIza" + "y" * 35
        r = client_no_key.post(
            "/key/google",
            data={"key": valid, "remember": "1"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        disk_path = _ks._KEY_DIR / "google"
        assert disk_path.exists(), "remember=1 should write to disk"
        r2 = client_no_key.post("/key/google/remove", follow_redirects=False)
        assert r2.status_code == 303
        assert r2.headers["location"] == "/"
        assert not disk_path.exists(), "remove must delete the disk file"
        # Landing is back to the paste-key state.
        r3 = client_no_key.get("/")
        assert "paste your Google AI key" in r3.text

    def test_post_key_no_remember_does_not_write_disk(
        self, client_no_key: TestClient
    ) -> None:
        from topofuzz.ui import _keystore as _ks
        valid = "AIza" + "z" * 35
        r = client_no_key.post(
            "/key/google", data={"key": valid}, follow_redirects=False
        )
        assert r.status_code == 303
        assert not (_ks._KEY_DIR / "google").exists(), (
            "no remember checkbox should NOT persist to disk"
        )
        # But the key is still active for this session.
        r2 = client_no_key.get("/")
        assert "Google (Gemini)" in r2.text

    def test_landing_remember_checkbox_unchecked_by_default(
        self, client_no_key: TestClient
    ) -> None:
        """SECURITY FIX 1: memory-only key storage must be the default —
        the "remember this key" checkbox renders UNCHECKED on first load.
        Persisting to ``~/.topofuzz/keys/{vendor}`` is opt-in, matching
        the truthful SECURITY.md / PRIVACY.md posture."""
        r = client_no_key.get("/")
        assert r.status_code == 200
        idx = r.text.index('name="remember"')
        tag_end = r.text.index("/>", idx)
        checkbox_tag = r.text[idx:tag_end]
        assert "checked" not in checkbox_tag, (
            f"remember checkbox defaults to checked: {checkbox_tag!r}"
        )

    def test_post_key_unknown_vendor_404(
        self, client_no_key: TestClient
    ) -> None:
        valid = "AIza" + "x" * 35
        r = client_no_key.post(
            "/key/bogus", data={"key": valid}, follow_redirects=False
        )
        assert r.status_code == 404

    def test_post_openai_key_saves_then_appears_in_optional_list(
        self, client: TestClient
    ) -> None:
        valid = "sk-" + "x" * 25
        r = client.post(
            "/key/openai", data={"key": valid}, follow_redirects=False
        )
        assert r.status_code == 303
        r2 = client.get("/")
        assert "OpenAI" in r2.text
        assert valid[-4:] in r2.text

    def test_post_anthropic_wrong_prefix_rejected(
        self, client_no_key: TestClient
    ) -> None:
        # Anthropic requires sk-ant-, not bare sk-.
        r = client_no_key.post(
            "/key/anthropic",
            data={"key": "sk-" + "x" * 30},
            follow_redirects=False,
        )
        assert r.status_code == 400
        assert "sk-ant-" in r.text

    def test_jury_unlocks_when_three_vendor_keys_saved(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Start from clean — only Google env var is set via the `client`
        # fixture. /setup should NOT show the jury checkbox yet.
        r = client.get("/setup")
        assert "Use 3 referees" not in r.text
        # Save OpenAI + Anthropic via the browser.
        client.post(
            "/key/openai",
            data={"key": "sk-" + "x" * 25},
            follow_redirects=False,
        )
        client.post(
            "/key/anthropic",
            data={"key": "sk-ant-" + "x" * 30},
            follow_redirects=False,
        )
        # Now we have 3 saved → checkbox appears on /setup.
        r2 = client.get("/setup")
        assert "Use 3 referees" in r2.text
        assert "Google (Gemini)" in r2.text
        assert "OpenAI" in r2.text
        assert "Anthropic (Claude)" in r2.text

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

    def test_cost_preview_also_refreshes_run_button_note(
        self, client: TestClient
    ) -> None:
        # The note beside "Run the test" repeats the cost; the fragment
        # must update it out-of-band or the page shows two different
        # prices after the user changes thoroughness.
        r = client.get(
            "/setup/cost-preview",
            params={"workflow": "loan-approval", "thoroughness": "quick", "n_scenarios": 3},
        )
        assert 'id="run-cost-note"' in r.text
        assert 'hx-swap-oob="true"' in r.text
        assert "3 cases queued" in r.text

    def test_setup_cost_counts_only_filled_cases(self, client: TestClient) -> None:
        # The form pads to five cards, but blank cards are dropped at
        # submit — the estimate must price the three starters, not five.
        r = client.get("/setup")
        assert r.status_code == 200
        assert "3 cases queued" in r.text
        assert "5 cases queued" not in r.text

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


# --------- UX-1: in-browser license accept gate ---------------------------
#
# `ui` is now CLI-license-exempt (test_license_check.py) so the server
# always comes up, even non-interactively. Enforcement of "no run without
# acceptance" moves here: the landing page shows a one-time BSL accept
# gate, and POST /setup (the run-starting action) refuses to spawn a
# subprocess until license_check reports accepted.


class TestLicenseAcceptGate:
    def test_landing_shows_accept_gate_when_not_accepted(
        self, client_unaccepted: TestClient
    ) -> None:
        r = client_unaccepted.get("/")
        assert r.status_code == 200
        assert "I accept these terms" in r.text
        assert "Business Source License" in r.text
        # Nothing else on the page is reachable yet.
        assert "Start a test" not in r.text

    def test_landing_shows_normal_content_once_accepted(
        self, client: TestClient
    ) -> None:
        # The default `client` fixture inherits conftest's pre-accepted
        # fake home — the gate must NOT appear.
        r = client.get("/")
        assert r.status_code == 200
        assert "I accept these terms" not in r.text

    def test_setup_get_redirects_home_when_not_accepted(
        self, client_unaccepted: TestClient
    ) -> None:
        r = client_unaccepted.get("/setup", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/"

    def test_setup_post_refuses_to_launch_without_acceptance(
        self, client_unaccepted: TestClient
    ) -> None:
        """The critical run-starting gate: POSTing /setup while
        unaccepted must NOT spawn a subprocess — it must redirect back
        to the accept gate instead."""
        with patch("topofuzz.ui.runs.subprocess.Popen") as mock_popen:
            r = client_unaccepted.post(
                "/setup",
                data={
                    "workflow": "loan-approval",
                    "thoroughness": "quick",
                    "scenario_case": ["A facts.", "B facts.", "C facts."],
                    "scenario_why": ["", "", ""],
                    "scenario_expected": ["deny", "approve", "deny"],
                },
                follow_redirects=False,
            )
        mock_popen.assert_not_called()
        assert r.status_code == 303
        assert r.headers["location"] == "/"

    def test_accept_license_then_run_can_launch(
        self, client_unaccepted: TestClient, tmp_path: Path
    ) -> None:
        """Full happy path: not accepted → POST /accept-license →
        license_check reports accepted → POST /setup now launches."""
        from topofuzz.license_check import is_accepted

        assert is_accepted() is False
        r = client_unaccepted.post("/accept-license", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/"
        assert is_accepted() is True

        # The gate is gone; normal landing content is back.
        r_home = client_unaccepted.get("/")
        assert "I accept these terms" not in r_home.text

        class FakeProc:
            stdout = iter(["[fake] starting\n", "[fake] done\n"])
            returncode = 0

            def wait(self) -> int:
                return 0

            def terminate(self) -> None:
                return

        with patch(
            "topofuzz.ui.runs.subprocess.Popen", return_value=FakeProc()
        ) as mock_popen:
            r2 = client_unaccepted.post(
                "/setup",
                data={
                    "workflow": "loan-approval",
                    "thoroughness": "quick",
                    "scenario_case": ["A facts.", "B facts.", "C facts."],
                    "scenario_why": ["", "", ""],
                    "scenario_expected": ["deny", "approve", "deny"],
                },
                follow_redirects=False,
            )
        mock_popen.assert_called_once()
        assert r2.status_code == 303
        assert r2.headers["location"].startswith("/runs/")


# --------- _keystore.py: platform-correct on-disk permission -------------


class TestKeystoreDiskPermissions:
    """FIX 2: the persisted ``~/.topofuzz/keys/{vendor}`` file must get a
    real owner-only restriction on whatever platform it's written on.
    POSIX ``chmod(0o600)`` is a silent no-op on Windows (NTFS is ACL-
    gated, not mode-bit-gated), so Windows must instead call `icacls`
    to reset inheritance and grant Full Control to the current user only.
    """

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only icacls check")
    def test_windows_calls_icacls_with_inheritance_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from topofuzz.ui import _keystore as _ks

        monkeypatch.setattr(_ks, "_KEY_DIR", tmp_path / "keys")
        monkeypatch.setattr(_ks, "_runtime", {})

        captured: dict[str, Any] = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

            class _Result:
                returncode = 0

            return _Result()

        monkeypatch.setattr(subprocess, "run", fake_run)
        _ks.set_key("google", "AIza" + "x" * 35, remember=True)

        assert captured, "icacls (subprocess.run) was never invoked"
        args = captured["args"]
        assert args[0] == "icacls"
        assert str(_ks._disk_path("google")) in args
        assert "/inheritance:r" in args
        assert any(a.startswith("/grant:r") for a in args)
        # List-args, not a shell string — no shell=True hazard.
        assert captured["kwargs"].get("check") is False

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-only chmod check")
    def test_posix_calls_chmod_0600(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from topofuzz.ui import _keystore as _ks

        monkeypatch.setattr(_ks, "_KEY_DIR", tmp_path / "keys")
        monkeypatch.setattr(_ks, "_runtime", {})

        captured: dict[str, Any] = {}

        def fake_chmod(path, mode):
            captured["path"] = path
            captured["mode"] = mode

        monkeypatch.setattr(os, "chmod", fake_chmod)
        _ks.set_key("google", "AIza" + "y" * 35, remember=True)

        assert captured.get("mode") == 0o600
        assert captured.get("path") == _ks._disk_path("google")


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

    def test_post_run_cost_from_report_json_is_also_divided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """report.json's ``headline.total_cost_usd`` is the runner's raw
        meter total — the same units as the cap. The results page must
        convert it, or a "$0.24" estimate is followed by a "$0.63" bill
        for a run that really cost about $0.23."""
        import time as _time

        from topofuzz.ui.runs import _hydrate_cost_from_report_json

        run = Run(
            run_id="t2",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=tmp_path,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=2.2,
            started_at=_time.time(),
        )
        monkeypatch.setattr(
            run, "read_summary", lambda: {"headline": {"total_cost_usd": 0.62571}}
        )
        _hydrate_cost_from_report_json(run)
        snap = run.snapshot()
        assert snap["real_spend_usd"] == pytest.approx(0.62571 / 2.75, rel=1e-3)
        assert snap["real_spend_usd"] <= snap["real_cap_usd"]


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


# ---------------------------------------------------------------------------
# Bug B: §A ``degraded-high`` (exit code 1) runs must NOT be mislabelled
# as cancelled, and must NOT show "couldn't read summary file".
# ---------------------------------------------------------------------------


def _find_registry(app, registry_cls):  # type: ignore[no-untyped-def]
    """Walk an app's route-handler closures for a captured RunRegistry.

    ``build_app`` constructs one ``RunRegistry`` and closes over it in
    every endpoint. Tests that need to inject pre-fabricated ``Run``
    objects (to exercise a results page in a known state) have to find
    that registry from outside the app — there is no public accessor.
    We walk ``app.routes[*].endpoint.__closure__`` and return the first
    matching ``RunRegistry`` cell. The walk picks the first match;
    today every closed-over ``RunRegistry`` IS the same instance.
    """
    for route in app.routes:
        ep = getattr(route, "endpoint", None)
        if ep is None or ep.__closure__ is None:
            continue
        for cell in ep.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, registry_cls):
                return val
    return None


class TestDegradedRunResultsBanners:
    """Regression coverage: a completed degraded run with a valid
    report.json must show neither the "cancelled" banner nor the
    "couldn't read summary file" headline.

    Bug B root cause was twofold: (1) ``_was_cancelled(1) is True`` on
    Windows colliding with §A's degraded-high exit code, AND (2) bare
    exit-code dispatch couldn't tell §A degraded-high apart from a
    Python uncaught-exception crash (both exit 1). The fix gates on
    report.json validity (``_has_completed_report``) so degraded runs
    classify as ``succeeded`` while crashes still classify as
    ``failed``.
    """

    @staticmethod
    def _write_run_artifacts(
        run_dir: Path, *, headline_status: str = "degraded"
    ) -> Path:
        """Lay out a fake completed run on disk: out_dir/<inner>/report.json.

        Mirrors the on-disk shape produced by ``topofuzz run`` —
        :pyattr:`Run.inner_run_dir` scans one level deep for the
        topofuzz-stamped subdir.
        """
        import json as _json

        inner = run_dir / "2026-05-16-FAKEINNER"
        inner.mkdir(parents=True)
        report = {
            "headline": {
                "n_evals": 3,
                "n_variants": 2,
                "n_total_cells": 6,
                "n_successful_cells": 6,
                "verdict_flip_rate": {"point": 0.5},
                "judge_failure_rate": 0.3333,
                "status": headline_status,
            }
        }
        (inner / "report.json").write_text(
            _json.dumps(report), encoding="utf-8"
        )
        return inner

    def test_drain_thread_assigns_succeeded_on_exit_one(
        self, tmp_path: Path
    ) -> None:
        """The drain-thread branch on exit_code=1 must dispatch to
        ``succeeded`` (NOT to ``cancelled`` via the old
        ``_was_cancelled(1)`` Windows-terminate heuristic, and NOT to
        ``failed`` via the fallthrough branch).
        """
        import time as _time

        from topofuzz.ui.runs import _drain_subprocess

        out_dir = tmp_path / "ui-runs" / "fake-run-1"
        out_dir.mkdir(parents=True)
        self._write_run_artifacts(out_dir)

        class FakeProc:
            """Subprocess that emits one line, exits with code 1."""

            stdout = iter(["topofuzz: degraded, judge_failure_rate=33.3%\n"])
            returncode = 1

            def wait(self) -> int:
                return 1

        run = Run(
            run_id="fake-run-1",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=out_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
            process=FakeProc(),  # type: ignore[arg-type]
        )
        _drain_subprocess(run)
        assert run.status == "succeeded", (
            "exit_code=1 (degraded-high) with a valid report.json must "
            "classify as succeeded (the results page surfaces the warn "
            "banner from headline.status, not a cancelled/failed banner)."
        )
        assert run.cancel_requested is False

    def test_drain_thread_assigns_failed_on_exit_one_without_report(
        self, tmp_path: Path
    ) -> None:
        """B1 regression: a crash that exits 1 with NO report.json
        must NOT be relabelled "succeeded" by the exit-code-1 branch.

        Python's default uncaught-exception handler exits 1 — a Gemini
        SDK rate-limit, an OOM, or a version-mismatched ``langgraph``
        import inside ``topofuzz run`` all hit this path. The original
        Bug-B fix's bare ``elif exit_code == 1`` would have masked
        those as "Finished with warnings" with a fabricated headline.
        The ``_has_completed_report`` probe makes the fix surgical:
        only exit-1 runs that wrote a valid report.json classify as
        succeeded; the rest fall through to ``failed``.
        """
        import time as _time

        from topofuzz.ui.runs import _drain_subprocess

        out_dir = tmp_path / "ui-runs" / "fake-crash-1"
        out_dir.mkdir(parents=True)
        # Deliberately do NOT call _write_run_artifacts — this run
        # crashed mid-flight, so there is no report.json on disk.

        class FakeProc:
            """Subprocess that crashed and exited with code 1."""

            stdout = iter(["Traceback (most recent call last):\n"])
            returncode = 1

            def wait(self) -> int:
                return 1

        run = Run(
            run_id="fake-crash-1",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=out_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
            process=FakeProc(),  # type: ignore[arg-type]
        )
        _drain_subprocess(run)
        assert run.status == "failed", (
            "exit_code=1 with NO report.json must classify as failed "
            "(uncaught-exception crash) — not silently relabelled as "
            "succeeded/degraded."
        )
        assert run.failure_reason is not None and (
            "exit code 1" in run.failure_reason
            or "stopped" in run.failure_reason.lower()
        ), "failed runs must populate failure_reason for the danger banner."

    def test_has_completed_report_helper(self, tmp_path: Path) -> None:
        """``_has_completed_report`` is the load-bearing dispatch probe.
        Cover the three branches: valid report, missing report, malformed
        report.
        """
        import json as _json
        import time as _time

        from topofuzz.ui.runs import _has_completed_report

        # 1. Valid report.json → True.
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        inner = good_dir / "2026-05-16-INNER"
        inner.mkdir()
        (inner / "report.json").write_text(
            _json.dumps({"headline": {"status": "degraded"}}),
            encoding="utf-8",
        )
        good_run = Run(
            run_id="good",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=good_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        assert _has_completed_report(good_run) is True

        # 2. Missing report.json → False (uncaught-exception crash).
        missing_dir = tmp_path / "missing"
        missing_dir.mkdir()
        missing_run = Run(
            run_id="missing",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=missing_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        assert _has_completed_report(missing_run) is False

        # 3. report.json present but headline.status unknown → False
        # (defensive against future writer drift).
        weird_dir = tmp_path / "weird"
        weird_dir.mkdir()
        weird_inner = weird_dir / "2026-05-16-INNER"
        weird_inner.mkdir()
        (weird_inner / "report.json").write_text(
            _json.dumps({"headline": {"status": "wat"}}),
            encoding="utf-8",
        )
        weird_run = Run(
            run_id="weird",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=weird_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        assert _has_completed_report(weird_run) is False

    def test_was_cancelled_no_longer_swallows_exit_one(self) -> None:
        """``_was_cancelled(1)`` used to be True (Windows terminate
        heuristic). That collided with §A's degraded-high exit code
        (the SUCCESSFUL case from
        :func:`topofuzz.metrics.classify_run_status` when
        25% < cell_failure_rate <= 50%). The heuristic now only fires
        on POSIX negative signal codes."""
        from topofuzz.ui.runs import _was_cancelled

        assert _was_cancelled(1) is False
        assert _was_cancelled(2) is False  # §A failed — that's "failed" not "cancelled"
        assert _was_cancelled(-15) is True  # SIGTERM
        assert _was_cancelled(-2) is True   # SIGINT

    def test_results_page_no_cancelled_banner_on_degraded_run(
        self, tmp_path: Path
    ) -> None:
        """End-to-end via TestClient: a completed degraded run renders
        the warn banner with "Finished with warnings" — NOT the
        "cancelled" banner and NOT the "couldn't read summary file"
        headline."""
        import time as _time

        from topofuzz.ui.app import build_app
        from topofuzz.ui.runs import RunRegistry

        ui_runs_root = tmp_path / "explicit-ui-runs"
        ui_runs_root.mkdir(parents=True)
        out_dir = ui_runs_root / "fake-run-degraded"
        out_dir.mkdir()
        self._write_run_artifacts(out_dir)

        app2 = build_app(runs_root=ui_runs_root)
        c = TestClient(app2)
        c.headers["Origin"] = "http://testserver"

        target_registry = _find_registry(app2, RunRegistry)
        assert target_registry is not None, (
            "could not locate RunRegistry in app2's route closures"
        )

        run = Run(
            run_id="fake-run-degraded",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=out_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        run.status = "succeeded"  # post-Bug-B-fix classification
        run.exit_code = 1
        run.finished_at = _time.time()
        target_registry._runs["fake-run-degraded"] = run  # noqa: SLF001

        r = c.get("/runs/fake-run-degraded")
        assert r.status_code == 200, r.text

        # Bug B fix assertions: the two false banners are GONE.
        assert "This test was cancelled before it finished" not in r.text, (
            "Bug B regression: a successful-but-degraded run is being "
            "mislabelled as cancelled."
        )
        assert "we couldn't read its summary file" not in r.text, (
            "Bug B regression: a run with a valid report.json on disk "
            "is being reported as having an unreadable summary."
        )
        # And the new warn banner IS shown.
        assert "Finished with warnings" in r.text, (
            "Bug B fix: a degraded run must surface a warn banner "
            "explaining the judge-failure caveat."
        )
        # The real headline rendered. ``_build_headline`` scales the
        # verdict_flip_rate against n_evals (= 3 here), so 0.5 → 2
        # flips on the headline (matches existing _build_headline
        # tests above). The phrasing "changed its mind on about N" is
        # the load-bearing copy; the exact N is asserted separately by
        # the unit tests on _build_headline itself.
        assert "changed its mind on about" in r.text

    def test_results_page_cancelled_wins_over_degraded_summary(
        self, tmp_path: Path
    ) -> None:
        """The "cancelled" banner is still correct when the run was
        actually cancelled (user clicked Cancel), AND it must take
        precedence over the new degraded banner even when a partial
        ``report.json`` with ``headline.status == "degraded"`` is on
        disk. The cancel_requested signal is authoritative — a user who
        clicked Cancel deserves the cancelled banner regardless of how
        much of the run made it to the metrics layer.
        """
        import time as _time

        from topofuzz.ui.app import build_app
        from topofuzz.ui.runs import RunRegistry

        ui_runs_root = tmp_path / "ui-runs-cancel-case"
        ui_runs_root.mkdir(parents=True)
        out_dir = ui_runs_root / "fake-run-cancelled"
        out_dir.mkdir()
        # CR M3 precedence test: write a degraded report.json so the
        # results template's `elif summary.headline.status=="degraded"`
        # branch is reachable. The cancelled branch comes FIRST in the
        # template and must win.
        self._write_run_artifacts(out_dir)

        app2 = build_app(runs_root=ui_runs_root)
        c = TestClient(app2)
        c.headers["Origin"] = "http://testserver"

        target_registry = _find_registry(app2, RunRegistry)
        assert target_registry is not None

        run = Run(
            run_id="fake-run-cancelled",
            workflow_key="loan-approval",
            thoroughness_key="quick",
            out_dir=out_dir,
            fixture_path=tmp_path / "f.py",
            eval_csv_path=tmp_path / "e.csv",
            rubric_path=tmp_path / "r.txt",
            raw_cost_cap_usd=8.0,
            started_at=_time.time(),
        )
        run.status = "cancelled"
        run.cancel_requested = True
        run.finished_at = _time.time()
        target_registry._runs["fake-run-cancelled"] = run  # noqa: SLF001

        r = c.get("/runs/fake-run-cancelled")
        assert r.status_code == 200, r.text
        assert "This test was cancelled before it finished" in r.text, (
            "user-driven cancel must surface the cancelled banner."
        )
        assert "Finished with warnings" not in r.text, (
            "CR M3 precedence: cancelled banner must take precedence "
            "over the degraded banner — a cancelled run with a partial "
            "degraded report.json should NOT show 'Completed with "
            "warnings'."
        )


# ---------------------------------------------------------------------------
# Vendored htmx — the UI must load no third-party script
# ---------------------------------------------------------------------------


def test_ui_pages_load_htmx_from_local_static_only(client_no_key: TestClient) -> None:
    """The UI is a local-only tool (SECURITY.md network-egress list): its
    pages must not pull a script from a CDN, or the UI breaks offline and
    the browser makes a third-party request on every page load."""
    import re

    html = client_no_key.get("/").text
    srcs = re.findall(r'<script[^>]*\bsrc="([^"]+)"', html)
    assert srcs == ["/static/htmx.min.js"]


def test_vendored_htmx_matches_pinned_integrity_hash(client_no_key: TestClient) -> None:
    """The served file must be the exact upstream htmx 1.9.12 build the
    ``integrity`` attribute in base.html names — a silently edited or
    swapped copy would otherwise be refused by the browser at demo time."""
    import base64
    import hashlib
    import re

    html = client_no_key.get("/").text
    m = re.search(r'src="/static/htmx\.min\.js"\s+integrity="sha384-([^"]+)"', html)
    assert m, "base.html must keep the integrity attribute on the htmx script"
    r = client_no_key.get("/static/htmx.min.js")
    assert r.status_code == 200
    digest = base64.b64encode(hashlib.sha384(r.content).digest()).decode("ascii")
    assert digest == m.group(1)


# ---------------------------------------------------------------------------
# History survives a restart of the UI process
# ---------------------------------------------------------------------------


def _persisted_run(runs_root: Path, run_id: str, *, status: str) -> Run:
    import time as _time

    from topofuzz.ui.runs import _persist_run

    out_dir = runs_root / run_id
    out_dir.mkdir(parents=True)
    run = Run(
        run_id=run_id,
        workflow_key="loan-approval",
        thoroughness_key="quick",
        out_dir=out_dir,
        fixture_path=out_dir / "f.py",
        eval_csv_path=out_dir / "eval.csv",
        rubric_path=out_dir / "r.txt",
        raw_cost_cap_usd=2.2,
        started_at=_time.time(),
        status=status,
    )
    _persist_run(run)
    return run


def test_registry_reloads_finished_runs_from_disk(tmp_path: Path) -> None:
    """The registry was memory-only: a test run before a demo vanished
    from "Past tests" as soon as the app was closed and reopened."""
    _persisted_run(tmp_path, "20260101T000000-aaaaaaaa", status="succeeded")
    reg = RunRegistry(runs_root=tmp_path)
    runs = reg.list_runs()
    assert [r.run_id for r in runs] == ["20260101T000000-aaaaaaaa"]
    assert runs[0].status == "succeeded"
    assert runs[0].workflow_key == "loan-approval"
    assert reg.get("20260101T000000-aaaaaaaa") is runs[0]


def test_registry_does_not_resurrect_an_interrupted_run_as_running(
    tmp_path: Path,
) -> None:
    """A record left at "running" belongs to a dead process. Reloading it
    as running would block every new run (one active run at a time)."""
    _persisted_run(tmp_path, "20260101T000000-bbbbbbbb", status="running")
    reg = RunRegistry(runs_root=tmp_path)
    assert reg.active_run() is None
    run = reg.get("20260101T000000-bbbbbbbb")
    assert run.status == "failed"
    assert "closed before this test finished" in run.failure_reason


def test_registry_skips_unreadable_or_mismatched_run_records(tmp_path: Path) -> None:
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "ui_run.json").write_text("{not json", encoding="utf-8")
    _persisted_run(tmp_path, "20260101T000000-cccccccc", status="succeeded")
    # A sidecar copied into a differently named directory is ignored.
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "ui_run.json").write_text(
        (tmp_path / "20260101T000000-cccccccc" / "ui_run.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    reg = RunRegistry(runs_root=tmp_path)
    assert [r.run_id for r in reg.list_runs()] == ["20260101T000000-cccccccc"]
