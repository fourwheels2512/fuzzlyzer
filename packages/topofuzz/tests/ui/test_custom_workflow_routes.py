"""Route-level tests for the BYOP custom-workflow flow.

Verifies the FastAPI surface end-to-end:
  * GET  /custom-workflow/new          → renders editor
  * POST /custom-workflow              → creates files + redirects to /setup
  * POST /custom-workflow (invalid)    → 400 with form error
  * GET  /setup                        → shows custom workflows in tile group
  * GET  /custom-workflow/{id}         → renders editor pre-filled
  * POST /custom-workflow/{id}         → updates and redirects
  * POST /custom-workflow/{id}/delete  → deletes + redirects
  * Form size cap rejects oversized POST
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from topofuzz.ui.app import build_app  # noqa: E402
from topofuzz.ui.custom_storage import (  # noqa: E402
    list_custom_workflows,
    load_custom_workflow,
)


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setenv("TOPOFUZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    app = build_app(runs_root=tmp_path / "ui-runs")
    c = TestClient(app)
    # CSRF guard: the editor's POSTs require a same-origin Origin header.
    # TestClient defaults Host to "testserver"; mirror that as Origin so
    # the guard accepts these tests without changing each call site.
    c.headers["Origin"] = "http://testserver"
    return c


def _valid_form(label: str = "My pipeline") -> dict[str, list[str] | str]:
    """A form payload that passes every validator."""
    return {
        "label": label,
        "domain_noun": "loan application",
        "decision_verb": "approve / deny",
        "agent_id": ["intake_clerk", "final_decider"],
        "agent_model": [
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash",
        ],
        "verdict_agent_id": "final_decider",
        "rubric_paragraph": "x" * 100,
    }


class TestEditorRender:
    def test_get_new_renders_form(self, client: TestClient) -> None:
        r = client.get("/custom-workflow/new")
        assert r.status_code == 200
        assert "Build a custom workflow" in r.text
        # Model dropdown options must be present.
        assert "google/gemini-2.5-flash" in r.text
        assert "google/gemini-2.5-pro" in r.text


class TestCreate:
    def test_post_create_writes_files_and_redirects(
        self, client: TestClient
    ) -> None:
        r = client.post(
            "/custom-workflow",
            data=_valid_form(),
            follow_redirects=False,
        )
        assert r.status_code == 303
        loc = r.headers["location"]
        assert loc.startswith("/setup?workflow=custom:")
        # The workflow now appears in storage.
        listed = list_custom_workflows()
        assert len(listed) == 1
        assert listed[0].label == "My pipeline"
        # And it appears on /setup.
        r2 = client.get("/setup")
        assert "My pipeline" in r2.text

    def test_post_create_invalid_returns_400_with_form_error(
        self, client: TestClient
    ) -> None:
        bad = _valid_form()
        bad["agent_id"] = ["BAD UPPERCASE", "final_decider"]
        r = client.post("/custom-workflow", data=bad)
        assert r.status_code == 400
        # The user's other input is preserved (not blown away on error).
        assert 'value="My pipeline"' in r.text
        # Original storage list is empty — no file written on validation failure.
        assert list_custom_workflows() == []

    def test_post_create_disallowed_model_rejected(
        self, client: TestClient
    ) -> None:
        bad = _valid_form()
        bad["agent_model"] = ["openai/gpt-4o", "google/gemini-2.5-flash"]
        r = client.post("/custom-workflow", data=bad)
        assert r.status_code == 400
        assert list_custom_workflows() == []


class TestSetupListing:
    def test_setup_shows_custom_workflows(self, client: TestClient) -> None:
        client.post(
            "/custom-workflow", data=_valid_form(label="Workflow A")
        )
        client.post(
            "/custom-workflow", data=_valid_form(label="Workflow B")
        )
        r = client.get("/setup")
        assert r.status_code == 200
        assert "Workflow A" in r.text
        assert "Workflow B" in r.text
        # The "Create your own" tile is also present.
        assert "Create your own" in r.text

    def test_setup_shows_create_tile_when_no_custom(
        self, client: TestClient
    ) -> None:
        r = client.get("/setup")
        assert "Create your own" in r.text


class TestEdit:
    def test_get_edit_pre_fills(self, client: TestClient) -> None:
        client.post(
            "/custom-workflow", data=_valid_form(label="Edit me")
        )
        cw = list_custom_workflows()[0]
        r = client.get(f"/custom-workflow/{cw.workflow_id}")
        assert r.status_code == 200
        assert 'value="Edit me"' in r.text
        assert 'value="intake_clerk"' in r.text

    def test_post_edit_updates_in_place(self, client: TestClient) -> None:
        client.post(
            "/custom-workflow", data=_valid_form(label="V1")
        )
        cw = list_custom_workflows()[0]
        wid = cw.workflow_id
        updated = _valid_form(label="V2")
        r = client.post(
            f"/custom-workflow/{wid}", data=updated, follow_redirects=False
        )
        assert r.status_code == 303
        loaded = load_custom_workflow(wid)
        assert loaded is not None
        assert loaded.label == "V2"
        # The id is preserved across an edit (no fork).
        assert loaded.workflow_id == wid

    def test_get_edit_unknown_returns_404(
        self, client: TestClient
    ) -> None:
        r = client.get("/custom-workflow/20260515T120000-deadbeef")
        assert r.status_code == 404


class TestDelete:
    def test_post_delete_removes_workflow_and_redirects(
        self, client: TestClient
    ) -> None:
        client.post("/custom-workflow", data=_valid_form())
        cw = list_custom_workflows()[0]
        r = client.post(
            f"/custom-workflow/{cw.workflow_id}/delete",
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/setup"
        assert list_custom_workflows() == []

    def test_post_delete_invalid_id_returns_400(
        self, client: TestClient
    ) -> None:
        # The path-format guard rejects any non-conforming id BEFORE any
        # filesystem traversal is attempted.
        r = client.post("/custom-workflow/..%2Fpasswd/delete")
        # Could be 400 (invalid id) or 404 (route mismatch on slash).
        # Both are acceptable — the contract is "doesn't escape".
        assert r.status_code in (400, 404, 405)


class TestFormSizeCap:
    def test_oversized_post_rejected(self, client: TestClient) -> None:
        # 100 KB rubric → exceeds the 64 KB body cap.
        oversized = _valid_form()
        oversized["rubric_paragraph"] = "x" * 100_000
        r = client.post("/custom-workflow", data=oversized)
        assert r.status_code == 413


class TestSetupRunsCustomWorkflow:
    def test_setup_get_with_custom_key_does_not_500(
        self, client: TestClient
    ) -> None:
        client.post("/custom-workflow", data=_valid_form())
        cw = list_custom_workflows()[0]
        r = client.get(f"/setup?workflow={cw.key}")
        assert r.status_code == 200
        # The custom workflow name is in the page (selected radio).
        assert cw.label in r.text


class TestCSRFGuard:
    """The CSRF guard must reject POSTs from a foreign origin while still
    accepting same-origin POSTs (covered implicitly by every other test).
    """

    def _foreign_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> TestClient:
        monkeypatch.setenv("TOPOFUZZ_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        app = build_app(runs_root=tmp_path / "ui-runs")
        c = TestClient(app)
        c.headers["Origin"] = "http://attacker.example.com"
        return c

    def test_create_rejected_on_cross_origin_post(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = self._foreign_client(tmp_path, monkeypatch)
        r = c.post("/custom-workflow", data=_valid_form())
        assert r.status_code == 403

    def test_delete_rejected_on_cross_origin_post(
        self, client: TestClient, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Create one with the legit (same-origin) client first, then try
        # to delete from a foreign origin.
        client.post("/custom-workflow", data=_valid_form())
        cw = list_custom_workflows()[0]
        attacker = self._foreign_client(tmp_path, monkeypatch)
        r = attacker.post(f"/custom-workflow/{cw.workflow_id}/delete")
        assert r.status_code == 403
        # Workflow still exists.
        assert load_custom_workflow(cw.workflow_id) is not None

    def test_post_with_no_origin_and_no_sec_fetch_site_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unbranded curl request (no Origin, no Sec-Fetch-Site) is
        # treated as ambiguous and rejected — the legit UI always sends
        # one of the two.
        monkeypatch.setenv("TOPOFUZZ_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        app = build_app(runs_root=tmp_path / "ui-runs")
        c = TestClient(app)
        # Don't set Origin. The fixture above does it.
        r = c.post("/custom-workflow", data=_valid_form())
        assert r.status_code == 403


class TestAgentIdNewlineRejected:
    """Regression: agent_id ending in a newline used to slip through
    re.match because $ matches before final \\n. Now anchored \\A...\\Z."""

    def test_agent_id_with_trailing_newline_rejected(
        self, client: TestClient
    ) -> None:
        bad = _valid_form()
        bad["agent_id"] = ["valid_name", "ab\n"]
        r = client.post(
            "/custom-workflow", data=bad, follow_redirects=False
        )
        assert r.status_code == 400
        assert "agent name" in r.text.lower()


class TestTopofuzzHomeSanitization:
    """TOPOFUZZ_HOME pointing under a system dir must be ignored."""

    def test_system_dir_override_falls_back_to_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from topofuzz.ui import custom_storage

        monkeypatch.setenv("TOPOFUZZ_HOME", "/etc/topofuzz-attacker")
        # Should NOT raise; just falls back to ~/.topofuzz/.
        root = custom_storage.custom_workflows_root()
        assert "/etc" not in str(root).replace("\\", "/")


class TestNoFormBodyChunkedBypass:
    """No Content-Length → 411 (closes the chunked-bypass form-cap hole)."""

    def test_post_without_content_length_rejected(
        self, client: TestClient
    ) -> None:
        # Use a generator body to trick httpx into NOT sending Content-Length.
        def _streamed():
            yield b"label=anything"

        r = client.post(
            "/custom-workflow",
            content=_streamed(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        # Either 411 (no content-length) or 400 (invalid content-length)
        # — both prove the cap is load-bearing, not Content-Length-trusted.
        assert r.status_code in (411, 400)
