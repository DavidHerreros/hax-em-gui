"""Assistant wiring: autonomy routes, workspace registration, the prepare-motion
refusal, the chat ask-and-wait round trip, watch with the assistant's rules, the
apply route, and a source guard against any deleting call on assistant-driven paths.

The assistant's own settings (workspaces, autonomy, job registry) go to a per-test
HAX_ASSISTANT_CONFIG, so no real user settings are read or written.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

import pytest

from conftest import wait_for_status

from hax_gui import assistant as assistant_bridge
from hax_gui import watch

BACKEND = Path(__file__).resolve().parents[1] / "hax_gui"
WS = "ws://127.0.0.1"          # a loopback Host, or the DNS-rebinding guard closes the socket
needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HAX_ASSISTANT_CONFIG", str(tmp_path / "assistant_cfg" / "settings.json"))
    monkeypatch.delenv("HAX_ASSISTANT_AUTONOMY", raising=False)
    monkeypatch.delenv("HAX_ASSISTANT_WORKSPACE", raising=False)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A projects root the bridge sees, without touching the GUI's settings file."""
    d = tmp_path / "projects"
    d.mkdir()
    monkeypatch.setattr(assistant_bridge.settings_store, "projects_root", lambda: d)
    return d


class FakeSession:
    schema = None

    def status(self):
        from hax_assistant import autonomy
        return {"autonomy": autonomy.get_autonomy(), "model": "fake"}


# ── autonomy ─────────────────────────────────────────────────────────────────
@needs_assistant
def test_autonomy_routes(client, run):
    from hax_assistant import autonomy
    r = run(client.get("/api/assistant/autonomy"))
    assert r.status_code == 200 and r.json()["level"] == "advisory"
    assert set(r.json()["levels"]) == {"advisory", "ask", "auto"}
    r = run(client.put("/api/assistant/autonomy", json={"level": "ask-first"}))
    assert r.status_code == 200 and r.json()["level"] == "ask"
    assert run(client.get("/api/assistant/autonomy")).json()["level"] == "ask"
    assert autonomy.load_user_config()["autonomy"] == "ask"
    r = run(client.put("/api/assistant/autonomy", json={"level": "yolo"}))
    assert r.status_code == 422 and "unknown autonomy level" in r.text
    assert autonomy.get_autonomy() == "ask"


def test_autonomy_routes_without_assistant(client, run, monkeypatch):
    monkeypatch.setattr(assistant_bridge, "_INSTALLED", False)
    assert run(client.get("/api/assistant/autonomy")).status_code == 400
    assert run(client.put("/api/assistant/autonomy", json={"level": "ask"})).status_code == 400
    assert assistant_bridge.ensure_workspace() == "the assistant is not installed"
    assert assistant_bridge.data_facts(__file__) == {}


# ── workspaces ───────────────────────────────────────────────────────────────
@needs_assistant
def test_workspace_auto_registration(client, run, root, tmp_path, monkeypatch):
    from hax_assistant import safety
    monkeypatch.setattr(assistant_bridge, "_get", lambda: FakeSession())
    assert safety.workspaces() == []
    st = run(client.get("/api/assistant/status")).json()
    assert st["available"] and st["workspace_error"] is None and st["autonomy"] == "advisory"
    assert safety.workspaces() == [root.resolve()]
    assert safety.inside_workspace(root / "proj" / "hetsiren_x")

    monkeypatch.setattr(assistant_bridge.settings_store, "projects_root", lambda: Path.home())
    reason = assistant_bridge.ensure_workspace()
    assert "too broad" in reason
    assert run(client.get("/api/assistant/status")).json()["workspace_error"] == reason
    monkeypatch.setattr(assistant_bridge.settings_store, "projects_root", lambda: tmp_path / "gone")
    assert "not an existing folder" in assistant_bridge.ensure_workspace()
    monkeypatch.setattr(assistant_bridge.settings_store, "projects_root", lambda: None)
    assert "no projects folder" in assistant_bridge.ensure_workspace()
    assert safety.workspaces() == [root.resolve()]       # refusals registered nothing


@needs_assistant
def test_prepare_motion_refused_outside_workspace(root, tmp_path, monkeypatch):
    import hax_assistant.actions as actions
    calls = []
    monkeypatch.setattr(actions, "prepare_motion", lambda *a, **k: calls.append(a) or "deformation")
    outside = tmp_path / "elsewhere" / "run"
    outside.mkdir(parents=True)
    with pytest.raises(assistant_bridge.Refused, match="outside"):
        assistant_bridge.run_prepare_motion(str(outside), 0)
    assert calls == [] and list(outside.iterdir()) == []
    inside = root / "proj" / "hetsiren_x"
    inside.mkdir(parents=True)
    assert assistant_bridge.run_prepare_motion(str(inside), 0) == "deformation"
    assert calls == [(str(inside), 0)]


@needs_assistant
def test_prepare_motion_ws_reports_refusal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from hax_gui.app import app
    from hax_gui.routes import assistant as routes
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(assistant_bridge, "ensure_workspace", lambda: "no projects folder is set")
    monkeypatch.setattr(routes, "manager", type("M", (), {"get": lambda self, j: {"id": j}})())
    monkeypatch.setattr(routes, "_resolve_run_artifacts", lambda job: {"run_dir": str(run_dir)})
    with TestClient(app).websocket_connect(WS + "/ws/assistant/prepare-motion") as ws:
        ws.send_json({"job_id": "j1", "gpu": 0})
        msg = _recv(ws)
    assert "No workspace" in msg["error"] and "no projects folder is set" in msg["error"]
    assert list(run_dir.iterdir()) == []


# ── particles fact ───────────────────────────────────────────────────────────
@needs_assistant
def test_run_facts_include_particle_count(tmp_path, monkeypatch):
    from hax_gui.routes.assistant import _run_facts
    star = tmp_path / "particles.star"
    star.write_text("data_particles\n\nloop_\n_rlnImageName #1\n_rlnDefocusU #2\n"
                    + "".join(f"{i}@stack.mrcs 10000\n" for i in range(1, 6)))
    facts = _run_facts({"args": {"md": str(star)}}, include_gpus=False)
    assert "particles=5" in facts and "has_ctf=True" in facts
    monkeypatch.setattr(assistant_bridge, "_INSTALLED", False)    # still works without it
    facts = _run_facts({"args": {"md": str(star)}}, include_gpus=False)
    assert "--md:" in facts and "particles=" not in facts


# ── chat ask-and-wait ────────────────────────────────────────────────────────
def _recv(ws, timeout=15.0):
    box: dict = {}
    t = threading.Thread(target=lambda: box.setdefault("m", ws.receive_json()), daemon=True)
    t.start()
    t.join(timeout)
    assert "m" in box, "no frame from the server"
    return box["m"]


@needs_assistant
def test_chat_question_answer_round_trip(monkeypatch):
    from fastapi.testclient import TestClient
    from hax_assistant.interaction import Cancelled, Question

    from hax_gui.app import app
    log: list = []

    def fake_stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        yield {"type": "status", "text": "thinking"}
        try:
            ans = interaction.ask(Question("colour", "Which colour?", kind="choice",
                                           choices=["red", "blue"]))
        except Cancelled:
            log.append("cancelled")
            return
        log.append(ans)
        yield {"type": "content", "delta": f"you chose {ans}"}

    monkeypatch.setattr(assistant_bridge, "stream_events", fake_stream)
    user = [{"role": "user", "content": "hi"}]
    with TestClient(app).websocket_connect(WS + "/ws/assistant/chat") as ws:
        # 1) answers do not cancel the turn; a bad answer is re-asked with the error
        ws.send_json({"messages": user, "turn": 1})
        assert _recv(ws) == {"status": "thinking"}
        q = _recv(ws)["question"]
        assert (q["key"], q["kind"], q["choices"], q["error"]) == ("colour", "choice", ["red", "blue"], None)
        ws.send_json({"answer": {"id": "stale", "value": "red"}})        # ignored
        ws.send_json({"answer": {"id": q["id"], "value": "green"}})
        q2 = _recv(ws)["question"]
        assert q2["id"] != q["id"] and "choose one of" in q2["error"]
        ws.send_json({"answer": {"id": q2["id"], "value": "2"}})
        assert _recv(ws) == {"token": "you chose blue"}
        assert _recv(ws) == {"done": True, "turn": 1}
        assert log == ["blue"]

        # 2) a stop while the question is pending ends the turn cleanly
        ws.send_json({"messages": user, "turn": 2})
        assert _recv(ws) == {"status": "thinking"}
        assert "question" in _recv(ws)
        ws.send_json({"stop": True})
        assert _recv(ws) == {"done": True, "turn": 2}
        assert log[-1] == "cancelled"

        # 3) a new question while one is pending: this turn ends, the new one runs
        ws.send_json({"messages": user, "turn": 3})
        assert _recv(ws) == {"status": "thinking"}
        assert "question" in _recv(ws)
        ws.send_json({"messages": user, "turn": 4})
        assert _recv(ws) == {"done": True, "turn": 3}
        assert _recv(ws) == {"status": "thinking"}
        q4 = _recv(ws)["question"]
        ws.send_json({"answer": {"id": q4["id"], "value": "red"}})
        assert _recv(ws) == {"token": "you chose red"}
        assert _recv(ws) == {"done": True, "turn": 4}
    assert log == ["blue", "cancelled", "cancelled", "red"]


@needs_assistant
def test_chat_interaction_unit():
    from hax_assistant.interaction import Cancelled, Question
    sent, cancel = [], threading.Event()
    inter = assistant_bridge.ChatInteraction(lambda k, v: sent.append((k, v)), cancel)
    inter.notify("working")
    assert sent == [("status", "working")]
    result: dict = {}

    def ask():
        try:
            result["v"] = inter.ask(Question("n", "How many?", kind="number", default="3"))
        except Cancelled:
            result["v"] = "cancelled"
    t = threading.Thread(target=ask)
    t.start()
    while len(sent) < 2:
        threading.Event().wait(0.01)
    assert inter.answer({"id": sent[-1][1]["id"], "value": ""})    # empty -> the default
    t.join(5)
    assert result["v"] == "3"
    t = threading.Thread(target=ask)
    t.start()
    while len(sent) < 3:
        threading.Event().wait(0.01)
    assert inter.answer({"id": sent[-1][1]["id"], "cancel": True})
    t.join(5)
    assert result["v"] == "cancelled" and not inter.answer({"id": sent[-1][1]["id"], "value": "1"})


# ── watch with the assistant's rules ─────────────────────────────────────────
def _write_run(run_dir: Path, loss, config=None, finished=False):
    from tensorboardX import SummaryWriter
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"output_path": str(run_dir), "learning_rate": 1e-4, "batch_size": 8, **(config or {})}
    (run_dir / "run_config.json").write_text(json.dumps(cfg))
    w = SummaryWriter(str(run_dir / "HetSIREN_metrics" / "20260101-000000"))
    for i, v in enumerate(loss):
        w.add_scalar("Training loss (HetSIREN)", float(v), i)
    w.close()
    if finished:
        (run_dir / "HetSIREN").mkdir()
    return run_dir


def _diverged(n=120):
    import numpy as np
    loss = 2 * np.exp(-0.05 * np.arange(n)) + 0.5
    loss[-1] = np.nan
    return loss


def _rising():
    import numpy as np
    return np.concatenate([np.linspace(2, 1, 110), np.linspace(1.0, 1.5, 10)])


class FakeManager:
    """Only `get`; any other manager call fails the test (watch must not act here)."""

    def __init__(self, *jobs):
        self.jobs = {j["id"]: j for j in jobs}

    def get(self, job_id):
        return self.jobs.get(job_id)

    def __getattr__(self, name):
        raise AssertionError(f"unexpected manager.{name}")


def _job(job_id, run_dir, status="running"):
    return {"id": job_id, "program": "hetsiren", "status": status, "workdir": str(run_dir),
            "name": f"run {job_id}", "project_id": "proj", "args": {}}


@pytest.fixture
def watched(monkeypatch, root):
    pytest.importorskip("tensorboardX")
    pytest.importorskip("tbparse")
    published: list = []
    monkeypatch.setattr(watch.events, "publish", published.append)
    monkeypatch.setattr(watch, "_watched", {})

    def setup(*jobs):
        monkeypatch.setattr(watch, "manager", FakeManager(*jobs))
        for j in jobs:
            watch._watched[j["id"]] = {"fired": set()}
    return published, setup


@needs_assistant
def test_watch_uses_assistant_rules_with_proposal(run, root, watched):
    from hax_assistant import autonomy
    published, setup = watched
    autonomy.set_autonomy("ask")
    job = _job("j1", _write_run(root / "proj" / "hetsiren_j1", _diverged()))
    setup(job)
    run(watch._check("j1"))
    by_kind = {e["alert"]: e for e in published}
    e = by_kind["diverged"]
    assert e["kind"] == "watch" and e["severity"] == "critical" and e["calibrated"] is True
    assert "nan" in e["evidence"].lower() and e["level"] == "warn"
    p = e["proposal"]
    assert p["restart"] == "scratch" and p["changes"] == {"learning_rate": pytest.approx(3e-5)}
    assert "Observed" in p["text"] and "learning_rate" in p["text"]
    assert e["decision"] == "confirm" and e["needs_confirmation"] is True    # unowned run at ask
    n = len(published)
    run(watch._check("j1"))
    assert len(published) == n                                              # once per kind

    autonomy.set_autonomy("advisory")
    job2 = _job("j2", _write_run(root / "proj" / "hetsiren_j2", _diverged()))
    setup(job2)
    run(watch._check("j2"))
    e = next(e for e in published if e["job_id"] == "j2" and e["alert"] == "diverged")
    assert e["decision"] == "propose" and "needs_confirmation" not in e and e["proposal"]["text"]


@needs_assistant
def test_watch_finished_is_not_an_alert(run, root, watched):
    import numpy as np
    published, setup = watched
    loss = 2 * np.exp(-0.05 * np.arange(120)) + 0.5
    setup(_job("j3", _write_run(root / "proj" / "hetsiren_j3", loss, finished=True)))
    run(watch._check("j3"))
    assert "finished" not in {e["alert"] for e in published}


@needs_assistant
def test_watch_auto_acts_only_on_calibrated_owned(run, root, watched, monkeypatch):
    import os

    from hax_assistant import autonomy, safety
    published, setup = watched
    autonomy.set_autonomy("auto")
    assert assistant_bridge.ensure_workspace() is None
    acted: list = []

    async def fake_apply(job_id, kind, by="user"):
        acted.append((job_id, kind, by))
        return {"applied": True, "restart": "scratch", "job": {"id": "new1"},
                "message": "— the assistant relaunched from scratch."}
    monkeypatch.setattr(watch, "apply_proposal", fake_apply)

    d = _write_run(root / "proj" / "hetsiren_d", _diverged())
    r = _write_run(root / "proj" / "hetsiren_r", _rising())
    for folder in (d, r):
        safety.register_job(folder, os.getpid(), ["hax_project_manager"])
    u = _write_run(root / "proj" / "hetsiren_u", _diverged())                 # not owned
    setup(_job("d", d), _job("r", r), _job("u", u))
    for jid in ("d", "r", "u"):
        run(watch._check(jid))
    ev = {(e["job_id"], e["alert"]): e for e in published}
    assert acted == [("d", "diverged", "auto")]
    assert ev[("d", "diverged")]["acted"]["new_job_id"] == "new1"
    assert "relaunched" in ev[("d", "diverged")]["title"]
    rising = ev[("r", "rising")]
    assert rising["calibrated"] is False and rising["needs_confirmation"] is True
    assert ev[("u", "diverged")]["needs_confirmation"] is True and "acted" not in ev[("u", "diverged")]


def test_watch_falls_back_without_assistant(run, monkeypatch):
    published: list = []
    monkeypatch.setattr(watch.events, "publish", published.append)
    monkeypatch.setattr(assistant_bridge, "_INSTALLED", False)
    monkeypatch.setattr(watch, "_loss_series", lambda wd: [1.0, 0.9, float("nan")])
    monkeypatch.setattr(watch, "_watched", {"j": {"fired": set()}})
    monkeypatch.setattr(watch, "manager", FakeManager(_job("j", "/nowhere")))
    run(watch._check("j"))
    assert [e["alert"] for e in published] == ["diverged"] and "proposal" not in published[0]


# ── the apply route ──────────────────────────────────────────────────────────
def _tree(folder: Path) -> dict:
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


@pytest.fixture
def finished_diverged_job(client, run, project, md_file):
    pytest.importorskip("tensorboardX")
    pytest.importorskip("tbparse")
    body = {"program": "hetsiren", "launch": True,
            "args": {"md": str(md_file), "sr": 1.0, "ctf_type": "None", "mode": "train"}}
    r = run(client.post(f"/api/projects/{project['id']}/jobs", json=body))
    assert r.status_code == 200, r.text
    job = wait_for_status(client, run, r.json()["id"], {"succeeded"})
    _write_run(Path(job["workdir"]), _diverged())
    return job


@pytest.fixture
def no_destruction(monkeypatch):
    """Any destructive manager/projects call fails the test; rmtree calls are recorded."""
    from hax_gui import projects
    from hax_gui.jobs import manager

    def boom(*a, **k):
        raise AssertionError("destructive call on an assistant path")
    for name in ("rerun", "delete", "delete_for_project"):
        monkeypatch.setattr(manager, name, boom)
    monkeypatch.setattr(projects, "delete", boom)
    removed: list = []
    real = shutil.rmtree
    monkeypatch.setattr(shutil, "rmtree", lambda p, *a, **k: (removed.append(str(p)), real(p, *a, **k)))
    monkeypatch.setattr(watch, "_watched", {})
    monkeypatch.setattr(watch, "_ensure_task", lambda: None)     # no 60 s loop left behind
    return removed


@needs_assistant
def test_apply_scratch_creates_a_new_job_and_keeps_the_old_run(
        client, run, finished_diverged_job, no_destruction):
    from hax_assistant import autonomy, safety
    from hax_assistant.monitor.controller import AUDIT_FILE, count_relaunches
    old = finished_diverged_job
    old_wd = Path(old["workdir"])
    before = _tree(old_wd)
    autonomy.set_autonomy("ask")
    url = f"/api/assistant/watch/{old['id']}/apply"

    r = run(client.post(url, json={"kind": "diverged"}))
    assert r.status_code == 200, r.text
    out = r.json()
    new = out["job"]
    assert out["applied"] is True and out["restart"] == "scratch" and "old run is kept" in out["message"]
    assert new["id"] != old["id"] and new["project_id"] == old["project_id"]
    new_wd = Path(new["workdir"])
    assert new_wd != old_wd and new_wd.name == f"hetsiren_{new['id']}"
    assert new["args"]["learning_rate"] == pytest.approx(3e-5)
    assert new["args"]["md"] == old["args"]["md"] and new["args"]["output_path"] == str(new_wd)
    assert wait_for_status(client, run, new["id"], {"succeeded"})["status"] == "succeeded"

    still = run(client.get(f"/api/jobs/{old['id']}")).json()
    assert still["status"] == "succeeded" and still["workdir"] == old["workdir"]
    after = _tree(old_wd)
    assert {k: after[k] for k in before} == before and set(after) - set(before) == {AUDIT_FILE}
    rec = json.loads((old_wd / AUDIT_FILE).read_text().splitlines()[-1])
    assert rec["decision"] == "relaunched" and rec["new_job_id"] == new["id"]
    assert count_relaunches(old_wd) == 1
    assert not [p for p in no_destruction if Path(p) == old_wd or old_wd in Path(p).parents]
    job_now = run(client.get(f"/api/jobs/{new['id']}")).json()
    assert safety.is_owned(new_wd) or not job_now.get("pid")
    assert new["id"] in watch._watched                                       # watching follows it

    r = run(client.post(url, json={"kind": "diverged"}))                    # a second click
    assert r.status_code == 409 and "already applied" in r.text


@needs_assistant
def test_apply_refusals(client, run, finished_diverged_job, no_destruction, monkeypatch):
    from hax_assistant import autonomy
    old = finished_diverged_job
    url = f"/api/assistant/watch/{old['id']}/apply"
    count = lambda: len(run(client.get("/api/jobs", params={"limit": 1000})).json())   # noqa: E731
    n = count()

    autonomy.set_autonomy("advisory")
    r = run(client.post(url, json={"kind": "diverged"}))
    assert r.status_code == 403 and "advisory" in r.text
    autonomy.set_autonomy("ask")

    r = run(client.post(url, json={"kind": "diverged", "changes": {"learning_rate": 1}}))
    assert r.status_code == 422                       # the client never sends the changes
    r = run(client.post(url, json={"kind": "latent_collapse"}))
    assert r.status_code == 409                       # not a finding of this run

    real = assistant_bridge.remedy_for

    def locked(run_dir, kind, pid_alive=None):
        p = real(run_dir, kind, pid_alive)
        return {**p, "changes": {**p["changes"], "output_path": "/tmp/elsewhere"}}
    monkeypatch.setattr(assistant_bridge, "remedy_for", locked)
    r = run(client.post(url, json={"kind": "diverged"}))
    assert r.status_code == 403 and "output_path" in r.text
    monkeypatch.setattr(assistant_bridge, "remedy_for", real)

    autonomy.set_budget("max_relaunches", 0)
    r = run(client.post(url, json={"kind": "diverged"}))
    assert r.status_code == 409 and "budget" in r.text
    autonomy.set_budget("max_relaunches", 2)

    continued: list = []
    from hax_gui.jobs import manager
    monkeypatch.setattr(manager, "continue_job", lambda *a: continued.append(a))
    monkeypatch.setattr(assistant_bridge, "remedy_for", lambda run_dir, kind, pid_alive=None: {
        **real(run_dir, "diverged", pid_alive), "restart": "resume", "changes": {}})
    r = run(client.post(url, json={"kind": "diverged"}))
    assert r.status_code == 200 and r.json()["applied"] is False and "Continue button" in r.json()["message"]
    assert continued == [] and count() == n


def test_apply_without_assistant(client, run, monkeypatch):
    monkeypatch.setattr(assistant_bridge, "_INSTALLED", False)
    r = run(client.post("/api/assistant/watch/nope/apply", json={"kind": "diverged"}))
    assert r.status_code == 400


# ── guard: no deleting call on the assistant-driven paths ────────────────────
_DESTRUCTIVE = re.compile(
    r"rmtree|os\.remove\b|os\.unlink|\.unlink\(|rmdir|os\.rename|shutil\.move|send2trash|"
    r"(?<![\w.])manager\.(rerun|delete\w*|start)\b|projects\.delete|wf_manager\.|workflows\.\w*delete|"
    r"replace_job_id|restart_running|\.rerun\(|_clear_run_folder|_purge_dataset_cache|recreate_ssd")
# The assistant's workflow module saves, validates, reads and starts workflows; nothing else.
_WF_ALLOWED = {"assistant_workflows.py": re.compile(r"wf_manager\.(save|validate|run|get)\(")}


def test_assistant_paths_never_delete():
    hits = []
    for rel in ("assistant.py", "watch.py", "assistant_actions.py", "assistant_workflows.py", "assistant_tools.py",
                "assistant_monitor.py"):
        allowed = _WF_ALLOWED.get(rel)
        for n, line in enumerate((BACKEND / rel).read_text().splitlines(), 1):
            if _DESTRUCTIVE.search(allowed.sub("", line) if allowed else line):
                hits.append(f"{rel}:{n}: {line.strip()}")
    assert not hits, "deleting call on an assistant-driven path:\n" + "\n".join(hits)
    routes = (BACKEND / "routes" / "assistant.py").read_text()
    assert not re.search(r"(?<![\w.])manager\.(rerun|delete)|wf_manager\.|replace_job_id", routes)
    assert not re.search(r"rmtree|os\.remove\b|\.unlink\(|rmdir", routes)
    assert _DESTRUCTIVE.search("wf_manager.delete(wid)") and _DESTRUCTIVE.search("wf_manager.stop(wid)")
    assert not _DESTRUCTIVE.search(_WF_ALLOWED["assistant_workflows.py"].sub("", "wf_manager.run(wid)"))
