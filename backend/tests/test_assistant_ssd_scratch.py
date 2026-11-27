"""The GUI's SSD scratch folder in what the assistant plans and suggests: with a scratch root in
Settings a workflow reads its images from the project's scratch folder (the job manager fills
--ssd_scratch_folder; the plan never does) instead of loading them into RAM, and the image copy's
space is checked there; without one the RAM rule stays, with a hint to set it. Draft jobs and the
model's own suggestions follow the same choice.
"""

from __future__ import annotations

import pytest

from test_assistant_workflows import (  # noqa: F401  (fixtures)
    CREATE, _set_autonomy, _settle, ask, data, isolated, no_destruction, schema, stub)

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_actions, assistant_workflows, settings_store

needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
pytestmark = needs_assistant
REASON = "images are read from the GUI's SSD scratch folder, set in Settings"


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    root = tmp_path / "ssd"
    root.mkdir()
    monkeypatch.setattr(settings_store, "ssd_scratch_root", lambda: root)
    return root


def _check(out):
    return next(c for c in out["plan"]["checks"] if c["name"] == "SSD scratch")


def test_workflow_reads_its_images_from_the_scratch_folder(client, run, project, data, stub, scratch, no_destruction):
    out = assistant_workflows.preview(ask(data), project["id"])
    for node in out["graph"]["nodes"]:
        assert node["args"]["load_images_to_ram"] is False and "ssd_scratch_folder" not in node["args"]
    assert out["steps"][0]["reasons"]["load_images_to_ram"] == REASON and out["ok"]
    assert _check(out)["status"] == "pass" and str(scratch) in _check(out)["message"]
    _set_autonomy("ask")
    r = run(client.post(CREATE, json={"request": ask(data), "project_id": project["id"], "run": True}))
    assert r.status_code == 200, r.text
    w = _settle(client, run, r.json()["workflow_id"])
    assert w["status"] == "succeeded", w["run_state"]
    for s in w["run_state"].values():
        job = run(client.get(f"/api/jobs/{s['job_id']}")).json()
        assert job["args"]["ssd_scratch_folder"] == str(scratch / project["id"])     # filled by the job manager
        assert "--load_images_to_ram" not in job["command"]
        assert job["command"][job["command"].index("--ssd_scratch_folder") + 1] == str(scratch / project["id"])


def test_without_a_scratch_folder_the_ram_rule_and_a_hint(project, data, monkeypatch):
    monkeypatch.setattr(settings_store, "ssd_scratch_root", lambda: None)
    out = assistant_workflows.preview(ask(data), project["id"])
    assert out["graph"]["nodes"][0]["args"]["load_images_to_ram"] is True and out["ok"]   # the small stack fits
    assert _check(out)["status"] == "warn" and "set one" in _check(out)["message"] and "Settings" in _check(out)["message"]


def test_a_full_scratch_disk_fails_the_plan(client, run, project, data, scratch, monkeypatch):
    import shutil
    from pathlib import Path
    real = shutil.disk_usage

    def full(path):
        u = real(path)
        return type(u)(u.total, u.used, 0) if Path(path) == scratch else u
    monkeypatch.setattr("hax_assistant.workflows.checks.shutil.disk_usage", full)
    out = assistant_workflows.preview(ask(data), project["id"])
    assert not out["ok"] and _check(out)["status"] == "fail" and "another scratch folder" in _check(out)["message"]
    _set_autonomy("ask")
    r = run(client.post(CREATE, json={"request": ask(data), "project_id": project["id"], "run": True}))
    assert r.status_code == 422 and any(c["name"] == "SSD scratch" for c in r.json()["detail"]["failed_checks"])


def test_draft_job_leaves_ram_loading_off(data, scratch):
    spec = assistant_actions.build_draft_spec("hetsiren", str(data["md"]))
    assert not spec["args"].get("load_images_to_ram") and "ssd_scratch_folder" not in spec["args"]
    assert "ssd_scratch_folder" in spec["auto_filled"]          # the job manager fills it at create time


@pytest.mark.parametrize("configured", [True, False])
def test_the_model_is_told_about_the_scratch_folder(schema, scratch, monkeypatch, configured):
    from hax_assistant.config import Settings
    from hax_assistant.session import AssistantSession
    seen = []
    s = object.__new__(AssistantSession)
    s.settings = Settings(backend="ollama", tier="guided")
    s.backend = type("B", (), {"supports_tools": False, "chat": lambda self, msgs, **k: seen.extend(msgs) or iter(["ok"])})()
    s.schema, s.index, s.embedder = schema, None, None
    monkeypatch.setattr(assistant_bridge, "_get", lambda: s)
    if not configured:
        monkeypatch.setattr(settings_store, "ssd_scratch_root", lambda: None)
    list(assistant_bridge.stream_events([{"role": "user", "content": "which flags for hetsiren?"}], grounded=False))
    note = next(m.content for m in seen if m.role == "system" and "ABOUT THIS FRONT END" in m.content)
    if configured:
        assert "--ssd_scratch_folder automatically" in note and "never ask the user for a scratch path" in note.lower()
    else:
        assert "No SSD scratch folder is set" in note and "Settings" in note
