"""The chat's stop card: which job a "stop the run" means, the checked stop route per autonomy and
ownership (through the job manager, audited, nothing deleted), and the socket offering the card in
the guided tier only.

Jobs are the conftest stub held running by the GATE file; the assistant's settings are per test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import GATE, wait_for_status
from test_assistant_workflows import isolated, no_destruction, schema  # noqa: F401  (fixtures)

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_tools, projects
from hax_gui.app import app
from hax_gui.jobs import manager

STOP = "/api/assistant/actions/stop-job"
needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
pytestmark = needs_assistant


@pytest.fixture
def gate():
    GATE.write_text("")                   # the stub job now waits until this is removed
    yield
    GATE.unlink(missing_ok=True)


@pytest.fixture
def ready(project):
    assert assistant_bridge.ensure_workspace() is None           # as the GUI registers its projects root
    return project


@pytest.fixture
def cancels(monkeypatch):
    calls, real = [], manager.cancel

    async def spy(job_id):
        calls.append(job_id)
        return await real(job_id)
    monkeypatch.setattr(manager, "cancel", spy)
    return calls


def _job(client, run, project_id, md_file, wait=("running",)):
    body = {"program": "filter_latents", "args": {"latents": str(md_file)}, "launch": True}
    r = run(client.post(f"/api/projects/{project_id}/jobs", json=body))
    assert r.status_code == 200, r.text
    return wait_for_status(client, run, r.json()["id"], set(wait))


def _audit(job):
    from hax_assistant.monitor.controller import AUDIT_FILE
    path = Path(job["workdir"]) / AUDIT_FILE
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _level(level):
    from hax_assistant import autonomy
    autonomy.set_autonomy(level)


# ── the route ────────────────────────────────────────────────────────────────
def test_stop_route_per_autonomy(client, run, ready, md_file, gate, cancels, no_destruction):
    job = _job(client, run, ready["id"], md_file)
    r = run(client.post(STOP, json={"job_id": job["id"]}))                    # advisory by default
    assert r.status_code == 403 and "advisory" in r.text and cancels == []
    assert _audit(job)[-1]["decision"] == "blocked"
    assert run(client.post(STOP, json={"job_id": job["id"], "force": True})).status_code == 422
    _level("ask")
    r = run(client.post(STOP, json={"job_id": job["id"], "auto": True}))       # not the assistant's job
    assert r.status_code == 403 and "with a click" in r.text and cancels == []
    assert manager.get(job["id"])["status"] == "running"
    r = run(client.post(STOP, json={"job_id": job["id"]}))
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("cancelling", "cancelled") and cancels == [job["id"]]
    rec = _audit(job)[-1]
    assert (rec["action"], rec["decision"], rec["decided_by"], rec["autonomy"]) == ("stop", "stopped", "user", "ask")
    assert wait_for_status(client, run, job["id"], {"cancelled", "failed"})["status"] == "cancelled"
    r = run(client.post(STOP, json={"job_id": job["id"]}))                    # a second click
    assert r.status_code == 409 and "already ended (cancelled)" in r.text and cancels == [job["id"]]
    assert Path(job["workdir"]).is_dir()                                       # the run folder is kept
    assert run(client.post(STOP, json={"job_id": "nope"})).status_code == 404


def test_auto_stops_only_the_assistants_job(client, run, ready, md_file, gate, cancels, no_destruction):
    own = _job(client, run, ready["id"], md_file)
    assistant_bridge.register_job(own["workdir"], own["pid"], own.get("command") or [])
    other = _job(client, run, ready["id"], md_file, wait=("queued", "running"))
    _level("auto")
    r = run(client.post(STOP, json={"job_id": other["id"], "auto": True}))
    assert r.status_code == 403 and cancels == []
    r = run(client.post(STOP, json={"job_id": own["id"], "auto": True}))
    assert r.status_code == 200 and r.json()["auto"] is True and cancels == [own["id"]]
    assert _audit(own)[-1]["decided_by"] == "auto"
    r = run(client.post(STOP, json={"job_id": other["id"]}))                  # a click stops it too
    assert r.status_code == 200 and cancels == [own["id"], other["id"]]


def test_stop_refused_outside_the_workspaces(client, run, project, md_file, gate, cancels, tmp_path, monkeypatch):
    elsewhere = tmp_path / "other_root"
    elsewhere.mkdir()
    job = _job(client, run, project["id"], md_file)
    monkeypatch.setattr(assistant_bridge.settings_store, "projects_root", lambda: elsewhere)
    _level("ask")
    r = run(client.post(STOP, json={"job_id": job["id"]}))
    assert r.status_code == 403 and "outside" in r.text and cancels == []
    assert manager.get(job["id"])["status"] == "running"
    monkeypatch.undo()
    assert run(client.post(STOP, json={"job_id": job["id"]})).status_code == 403     # still advisory here
    GATE.unlink()
    wait_for_status(client, run, job["id"], {"succeeded"})


# ── which job ────────────────────────────────────────────────────────────────
def test_which_job_a_stop_means(client, run, ready, md_file, gate):
    p, empty = projects.create("stop_card_jobs"), projects.create("stop_card_empty")
    a = _job(client, run, p["id"], md_file)
    ids = lambda out: [j["id"] for j in out["jobs"]]                          # noqa: E731
    out = assistant_tools.stop_target("stop the run", a["id"], None)          # attached (or the page's)
    assert ids(out) == [a["id"]] and out["jobs"][0]["decision"] == "propose" and "advisory" in out["jobs"][0]["reason"]
    placed = assistant_tools.place({"kind": "stop_job", "job_id": None}, "stop the run", None, empty["id"], a["id"])
    assert ids(placed) == [a["id"]]                                           # the page's job wins over the project
    assert ids(assistant_tools.stop_target("stop the run", None, p["id"])) == [a["id"]]      # the one running
    assert assistant_tools.stop_target("stop the run", None, empty["id"]) == {"jobs": []}   # nothing running
    b = _job(client, run, p["id"], md_file, wait=("queued", "running"))
    assert set(ids(assistant_tools.stop_target("stop the run", None, p["id"]))) == {a["id"], b["id"]}   # pick one
    assert {a["id"], b["id"]} <= set(ids(assistant_tools.stop_target("stop it", None, None)))  # no project: all
    named = assistant_tools.stop_target(f"stop the hetsiren training in {b['workdir']}/", a["id"], p["id"])
    assert ids(named) == [b["id"]]                                            # the folder it names
    assert assistant_tools.stop_target("stop the run in /data/nowhere/x", a["id"])["unknown"] == "/data/nowhere/x"
    GATE.unlink()
    done = wait_for_status(client, run, a["id"], {"succeeded"})
    wait_for_status(client, run, b["id"], {"succeeded"})
    ended = assistant_tools.stop_target("stop the run", a["id"])["jobs"][0]
    assert ended["status"] == done["status"] and "decision" not in ended


# ── the socket ───────────────────────────────────────────────────────────────
class Backend:
    model, supports_tools, supports_vision = "fake", True, False

    def __init__(self):
        self.calls = 0

    def parameter_count(self):
        return None

    def chat(self, messages, tools=None, stream=False, **kw):
        from hax_assistant.llm.base import Reply
        self.calls += 1
        return iter(["streamed"]) if stream else Reply("ok")


@pytest.mark.parametrize("tier", ["guided", "free"])
def test_socket_offers_the_stop_card_in_the_guided_tier_only(client, run, ready, md_file, gate, schema,
                                                             monkeypatch, tier):
    from hax_assistant.config import Settings
    from hax_assistant.session import STOP_REPLY, AssistantSession
    from hax_gui import assistant_actions
    s = object.__new__(AssistantSession)
    s.settings = Settings(backend="openai", tier=tier)
    s.backend, s.schema, s.index, s.embedder = Backend(), schema, None, None
    monkeypatch.setattr(assistant_bridge, "_get", lambda: s)
    monkeypatch.setattr(assistant_actions.settings, "assistant_actions", True)
    job = _job(client, run, ready["id"], md_file)
    with TestClient(app).websocket_connect("ws://127.0.0.1/ws/assistant/chat") as ws:
        ws.send_json({"messages": [{"role": "user", "content": "please stop the run"}], "turn": 1,
                      "project_id": ready["id"], "route_job_id": job["id"]})
        frames = []
        while not frames or "done" not in frames[-1]:
            frames.append(ws.receive_json())
    actions = [f["action"] for f in frames if "action" in f]
    if tier == "guided":
        assert len(actions) == 1 and actions[0]["kind"] == "stop_job" and actions[0]["job_id"] is None
        assert [j["id"] for j in actions[0]["jobs"]] == [job["id"]] and actions[0]["jobs"][0]["decision"]
        assert {"token": STOP_REPLY} in frames and s.backend.calls == 0
    else:
        assert actions == [] and {"token": "ok"} in frames and s.backend.calls == 1
    assert manager.get(job["id"])["status"] == "running"                      # the card stops it, not the chat
    GATE.unlink()
    wait_for_status(client, run, job["id"], {"succeeded"})
