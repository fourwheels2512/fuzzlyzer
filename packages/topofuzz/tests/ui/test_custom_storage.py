"""Tests for the custom-workflow on-disk storage layer.

Covers:
  * Round-trip: save → list → load → delete.
  * Corruption tolerance (a bad meta.json must not crash list).
  * Atomic writes (no half-written files survive a mid-write crash).
  * Path-containment + symlink rejection.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from topofuzz.ui.custom_storage import (
    custom_workflows_root,
    delete_custom_workflow,
    is_valid_workflow_id,
    list_custom_workflows,
    load_custom_workflow,
    new_workflow_id,
    save_custom_workflow,
    serialize_meta,
    workflow_dir,
)
from topofuzz.ui.presets import CustomAgent, CustomWorkflow, StarterScenario


@pytest.fixture()
def topofuzz_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.topofuzz/ to a tmp dir for the duration of the test."""
    monkeypatch.setenv("TOPOFUZZ_HOME", str(tmp_path))
    return tmp_path


def _make_cw(label: str = "Demo workflow", **overrides: object) -> CustomWorkflow:
    """Build a minimal valid CustomWorkflow."""
    defaults: dict = {
        "workflow_id": new_workflow_id(),
        "label": label,
        "domain_noun": "loan application",
        "decision_verb": "approve / deny",
        "agents": (
            CustomAgent("intake_clerk", "google/gemini-2.5-flash"),
            CustomAgent("final_decider", "google/gemini-2.5-flash"),
        ),
        "verdict_agent_id": "final_decider",
        "rubric_paragraph": "A" * 100,
        "created_at": time.time(),
        "starter_scenarios": (),
    }
    defaults.update(overrides)
    return CustomWorkflow(**defaults)  # type: ignore[arg-type]


class TestIdValidation:
    def test_valid_id_round_trip(self) -> None:
        wid = new_workflow_id()
        assert is_valid_workflow_id(wid)

    def test_invalid_ids_rejected(self) -> None:
        for bad in [
            "",
            "not-an-id",
            "../../etc/passwd",
            "20260515T125726",  # missing suffix
            "20260515T125726-tooshort",
            "20260515T125726-NOTHEX12",
            "20260515T125726-deadbeef/extra",
        ]:
            assert not is_valid_workflow_id(bad), f"should reject {bad!r}"


class TestSaveListLoadDelete:
    def test_round_trip_happy_path(self, topofuzz_home: Path) -> None:
        cw = _make_cw()
        wf = save_custom_workflow(cw)
        assert wf.key == f"custom:{cw.workflow_id}"
        assert wf.fixture_path.exists()
        assert wf.rubric_path.exists()

        listed = list_custom_workflows()
        assert len(listed) == 1
        assert listed[0].workflow_id == cw.workflow_id
        assert listed[0].label == cw.label

        loaded = load_custom_workflow(cw.workflow_id)
        assert loaded is not None
        assert loaded.label == cw.label
        assert loaded.agents == cw.agents

        assert delete_custom_workflow(cw.workflow_id) is True
        assert load_custom_workflow(cw.workflow_id) is None
        assert list_custom_workflows() == []

    def test_list_sorted_newest_first(self, topofuzz_home: Path) -> None:
        a = _make_cw(label="A", workflow_id=new_workflow_id())
        # Force ordering — created_at is the sort key.
        a = CustomWorkflow(
            **{**a.__dict__, "created_at": 1000.0}
        )  # type: ignore[arg-type]
        save_custom_workflow(a)
        time.sleep(0.01)
        b = _make_cw(label="B", workflow_id=new_workflow_id())
        b = CustomWorkflow(
            **{**b.__dict__, "created_at": 2000.0}
        )  # type: ignore[arg-type]
        save_custom_workflow(b)

        listed = list_custom_workflows()
        labels = [c.label for c in listed]
        assert labels == ["B", "A"]

    def test_delete_returns_false_when_missing(
        self, topofuzz_home: Path
    ) -> None:
        # Well-formed but nonexistent.
        wid = new_workflow_id()
        assert delete_custom_workflow(wid) is False

    def test_delete_rejects_invalid_id(self, topofuzz_home: Path) -> None:
        assert delete_custom_workflow("../../etc/passwd") is False
        assert delete_custom_workflow("") is False


class TestCorruptionTolerance:
    def test_bad_meta_json_skipped_not_raised(
        self, topofuzz_home: Path
    ) -> None:
        # Plant a workflow dir with corrupt meta.json. list_custom_workflows
        # must skip it, not crash.
        bad_id = new_workflow_id()
        bad_dir = custom_workflows_root() / bad_id
        bad_dir.mkdir()
        (bad_dir / "meta.json").write_text("{ not valid json", encoding="utf-8")

        good = _make_cw(label="Good")
        save_custom_workflow(good)

        listed = list_custom_workflows()
        assert len(listed) == 1
        assert listed[0].workflow_id == good.workflow_id

    def test_missing_meta_json_skipped(self, topofuzz_home: Path) -> None:
        bad_id = new_workflow_id()
        bad_dir = custom_workflows_root() / bad_id
        bad_dir.mkdir()
        # No meta.json at all.
        listed = list_custom_workflows()
        assert listed == []
        # And single-load returns None.
        assert load_custom_workflow(bad_id) is None

    def test_wrong_dir_name_ignored(self, topofuzz_home: Path) -> None:
        # An unrelated dir (not workflow-id-shaped) under the root must
        # be ignored quietly.
        weird = custom_workflows_root() / "unrelated-stuff"
        weird.mkdir()
        listed = list_custom_workflows()
        assert listed == []

    def test_disallowed_model_in_meta_json_skipped(
        self, topofuzz_home: Path
    ) -> None:
        # A user could hand-edit meta.json to use openai/gpt-4o; the loader
        # must refuse rather than spawning a run with an off-allow-list model.
        cw = _make_cw()
        save_custom_workflow(cw)
        meta_path = workflow_dir(cw.workflow_id) / "meta.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["agents"][0]["model"] = "openai/gpt-4o"
        meta_path.write_text(json.dumps(raw), encoding="utf-8")
        assert load_custom_workflow(cw.workflow_id) is None
        assert list_custom_workflows() == []


class TestAtomicWrites:
    def test_no_tmp_files_left_after_save(self, topofuzz_home: Path) -> None:
        cw = _make_cw()
        save_custom_workflow(cw)
        for path in workflow_dir(cw.workflow_id).iterdir():
            assert not path.name.endswith(".tmp"), (
                f"leftover .tmp file: {path}"
            )

    def test_save_overwrites_existing(self, topofuzz_home: Path) -> None:
        # Save a workflow, edit its meta, save again — the new content
        # must win; no merge or duplicate.
        cw1 = _make_cw(label="V1")
        save_custom_workflow(cw1)
        cw2 = CustomWorkflow(
            **{**cw1.__dict__, "label": "V2"}
        )  # type: ignore[arg-type]
        save_custom_workflow(cw2)
        loaded = load_custom_workflow(cw1.workflow_id)
        assert loaded is not None and loaded.label == "V2"


class TestPathContainment:
    def test_workflow_dir_rejects_invalid(self, topofuzz_home: Path) -> None:
        with pytest.raises(ValueError):
            workflow_dir("../../etc/passwd")
        with pytest.raises(ValueError):
            workflow_dir("")

    def test_load_rejects_invalid_id(self, topofuzz_home: Path) -> None:
        assert load_custom_workflow("../../etc/passwd") is None


class TestSymlinkRejection:
    """RT-axis path-injection: a symlinked workflow dir must not be loaded."""

    def test_symlinked_workflow_dir_skipped(
        self, topofuzz_home: Path
    ) -> None:
        # Plant a real workflow dir, then symlink another id at the root
        # to it. The symlink must NOT register as a separate workflow.
        cw = _make_cw()
        save_custom_workflow(cw)
        target = workflow_dir(cw.workflow_id)
        evil_id = new_workflow_id()
        link = custom_workflows_root() / evil_id
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        listed = list_custom_workflows()
        listed_ids = {c.workflow_id for c in listed}
        assert evil_id not in listed_ids
        assert cw.workflow_id in listed_ids

    def test_symlinked_fixture_file_skipped(
        self, topofuzz_home: Path
    ) -> None:
        # Plant a workflow then swap fixture.py with a symlink. load
        # must refuse it — defense against a confused-deputy attack
        # where the runner is tricked into loading code from /etc/.
        cw = _make_cw()
        save_custom_workflow(cw)
        fixture = workflow_dir(cw.workflow_id) / "fixture.py"
        # Replace the real file with a symlink to itself's content via
        # a dummy outside file (avoid sys-file targeting in tests).
        decoy = topofuzz_home / "decoy-fixture.py"
        decoy.write_text("__topofuzz_name__ = 'evil'\n", encoding="utf-8")
        fixture.unlink()
        try:
            os.symlink(decoy, fixture)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        # load must return None for the now-tampered workflow.
        assert load_custom_workflow(cw.workflow_id) is None


class TestSerializeMeta:
    def test_meta_json_round_trips_through_load(
        self, topofuzz_home: Path
    ) -> None:
        cw = _make_cw(
            starter_scenarios=(
                StarterScenario(
                    case="x", why_borderline="y", expected="approve"
                ),
            )
        )
        meta = serialize_meta(cw)
        # The serialized string must be valid JSON.
        parsed = json.loads(meta)
        assert parsed["workflow_id"] == cw.workflow_id
        assert parsed["agents"][0]["agent_id"] == "intake_clerk"

        save_custom_workflow(cw)
        loaded = load_custom_workflow(cw.workflow_id)
        assert loaded is not None
        assert loaded.starter_scenarios[0].case == "x"
        assert loaded.starter_scenarios[0].expected == "approve"
