"""The free-tier chat tools in the GUI: a workflow the model launches is a workflow of a GUI project
run by the GUI's engine (in the project the user names, or a new GUI project), never the assistant's
own runner; stop and relaunch go through the job manager, per autonomy and ownership; which project
a request means; and the chat socket handing all this to the session.

The tool loop is driven by a scripted backend; hax is a stub; GPUs are faked.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import GATE, wait_for_status
from test_assistant_wiring import _DESTRUCTIVE, BACKEND
from test_assistant_workflows import (  # noqa: F401  (fixtures)
    _set_autonomy, _settle, ask, data, isolated, no_destruction, schema, stub)

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_tools, events, projects
from hax_gui.app import app
from hax_gui.jobs import manager

needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
pytestmark = needs_assistant


class Backend:
    """Replies in order; each is a tool call (name, args) or the final text."""

    def __init__(self, *replies):
        from hax_assistant.llm.base import Reply, ToolCall
        self.model, self.supports_tools, self.supports_vision = "fake", True, False
        self.replies = [Reply("", [ToolCall("", *r)]) if isinstance(r, tuple) else Reply(r) for r in replies]
        self.calls: list = []

    def parameter_count(self):
        return None

    def chat(self, messages, tools=None, stream=False, **kw):
        self.calls.append(list(messages))
        return self.replies.pop(0)

    def tool_result(self, i=0):
        return json.loads([m for m in self.calls[i + 1] if m.role == "tool"][-1].content)


def _session(backend, schema):
    from hax_assistant.config import Settings
    from hax_assistant.session import AssistantSession
    s = object.__new__(AssistantSession)
    s.settings = Settings(backend="openai", tier="free")
    s.backend, s.schema, s.index, s.embedder = backend, schema, None, None
    return s


def _turn(run, session, text, answers=None, project_id=None, cancelled=False):
    """One free-tier turn with the GUI's tools, the worker in a thread as on the chat socket."""
    from hax_assistant.interaction import ScriptedInteraction
    inter = ScriptedInteraction(answers or {})
    cancel = threading.Event()
    if cancelled:
        cancel.set()

    async def go():
        tools = assistant_tools.chat_tools(asyncio.get_running_loop(), inter, cancel, project_id)
        return await asyncio.to_thread(lambda: list(session.stream_events(
            [{"role": "user", "content": text}], grounded=False, interaction=inter, tools=tools)))
    return run(go()), inter


@pytest.fixture
def no_assistant_runner(monkeypatch):
    """The assistant's own runner and process controller must never be reached from the GUI."""
    import hax_assistant.workflows as wf
    from hax_assistant.monitor.controller import LocalController

    def boom(*a, **k):
        raise AssertionError("the assistant's own runner was used inside the GUI")
    monkeypatch.setattr(wf, "run_workflow", boom)
    monkeypatch.setattr(LocalController, "launch", boom)
    monkeypatch.setattr(LocalController, "stop", boom)


@pytest.fixture
def ready(project, monkeypatch):
    assert assistant_bridge.ensure_workspace() is None           # as the GUI registers its projects root
    return Path(project["path"]).parent


def _new_folders(root, before):
    return sorted(set(p.name for p in root.iterdir()) - before)


# ── which project a request means ───────────────────────────────────────────
def test_resolve_project(ready):
    from hax_assistant.interaction import Cancelled, ScriptedInteraction
    a, b = projects.create("Pkm2 tools"), projects.create("Pkm2 tools")       # same name, two folders
    other = projects.create("adh_tools")
    none = ScriptedInteraction({})
    r = assistant_tools.resolve_project
    assert r(f"run it in my project {other['id']}", none) == other["id"]
    assert r(f"put it in the {other['name']} project", none, default="x") == other["id"]
    assert r("motion of /data/adh_tools/particles.xmd", none, default="x") == "x"   # inside a path: not named
    assert r("continuous motion of /d/p.xmd", none) is None                          # a new one
    assert r("continuous motion, in my project", none, default=a["id"]) == a["id"]     # the chat's
    pick = ScriptedInteraction({"project": f"Pkm2 tools ({b['id']})"})
    assert r("into my project pkm2 TOOLS please", pick) == b["id"]
    assert "several" in pick.asked[0].text and pick.asked[0].choices[-1] == assistant_tools.NEW_PROJECT
    new = ScriptedInteraction({"project": assistant_tools.NEW_PROJECT})
    assert r("in my existing project", new) is None and "mentions" in new.asked[0].text
    assert r("", ScriptedInteraction({"project": other["id"]}), name="nosuch") == other["id"]
    with pytest.raises(Cancelled):
        r("", none, name="nosuch")


def test_place_annotates_proposals(ready):
    from hax_assistant.interaction import ScriptedInteraction
    p = projects.create("place_tools")
    act = {"kind": "workflow", "template": "continuous_motion"}
    assert assistant_tools.place(act, f"in my project {p['id']}", ScriptedInteraction({}))["project_id"] == p["id"]
    assert assistant_tools.place({"kind": "create_job"}, "hi", None, "chat")["project_id"] == "chat"
    skip = assistant_tools.place(act, "in my project", ScriptedInteraction({}), "chat")
    assert skip["project_id"] == "chat"                                # skipped: the chat's project
    motion = {"kind": "prepare_motion"}
    assert assistant_tools.place(motion, "x", None) is motion


# ── launching a workflow ─────────────────────────────────────────────────────
def test_free_tier_launch_in_a_named_project(client, run, project, data, stub, schema, ready,
                                             no_destruction, no_assistant_runner):
    _set_autonomy("auto")
    before = set(p.name for p in ready.iterdir())
    q = events.subscribe()
    be = Backend(("launch_workflow", {"request": ask(data), "project": project["name"]}), "Started.")
    _, inter = _turn(run, _session(be, schema), "run the continuous motion workflow in testproj")
    res = be.tool_result()
    assert res["status"] == "started" and res["project"]["id"] == project["id"], res
    assert [n for n, _ in inter.notifications if "Done automatically" in n]
    assert _new_folders(ready, before) == []                      # no plain folder next to the projects
    seen = []
    while not q.empty():
        seen.append(q.get_nowait())
    events.unsubscribe(q)
    wf_event = next(e for e in seen if e.get("kind") == "workflow")
    assert wf_event["project_id"] == project["id"] and wf_event["new_project"] is False
    w = _settle(client, run, wf_event["workflow_id"])
    assert w["status"] == "succeeded" and w["project_id"] == project["id"], w["run_state"]
    for s in w["run_state"].values():
        job = run(client.get(f"/api/jobs/{s['job_id']}")).json()
        assert job["project_id"] == project["id"] and Path(job["workdir"]).parent == Path(project["path"])


def test_free_tier_launch_without_a_project_makes_a_gui_project(client, run, data, stub, schema, ready,
                                                                no_destruction, no_assistant_runner):
    _set_autonomy("ask")
    before = set(p.name for p in ready.iterdir())
    be = Backend(("launch_workflow", {"request": ask(data)}), "Started.")
    _, inter = _turn(run, _session(be, schema), "launch it", {"confirm_launch_workflow": "yes"})
    res = be.tool_result()
    assert res["status"] == "started", res
    assert "a new project" in inter.asked[-1].text and "Proceed?" in inter.asked[-1].text
    new = _new_folders(ready, before)
    assert len(new) == 1 and (ready / new[0] / projects.META).is_file()
    assert not (ready / new[0] / "assistant_workflow.json").exists()
    assert new[0] in {p["id"] for p in projects.list_projects()} and res["project"]["id"] == new[0]
    jobs = run(client.get(f"/api/projects/{new[0]}/jobs")).json()
    assert jobs and all(j["project_id"] == new[0] for j in jobs)


def test_free_tier_launch_per_autonomy_and_cancel(run, data, schema, ready, no_assistant_runner):
    before = set(p.name for p in ready.iterdir())
    _set_autonomy("advisory")
    be = Backend(("launch_workflow", {"request": ask(data)}), "Here is the plan.")
    _turn(run, _session(be, schema), "launch it")
    res = be.tool_result()
    assert res["status"] == "not_executed" and "a new project" in res["proposal"]
    _set_autonomy("ask")
    be = Backend(("launch_workflow", {"request": ask(data)}), "ok")
    _turn(run, _session(be, schema), "launch it", {"confirm_launch_workflow": "no"})
    assert be.tool_result() == {"status": "declined_by_user"}
    _set_autonomy("auto")
    be = Backend(("launch_workflow", {"request": ask(data)}), "ok")
    _turn(run, _session(be, schema), "launch it", cancelled=True)           # an interrupted turn acts on nothing
    assert be.tool_result() == assistant_tools.STOPPED
    assert _new_folders(ready, before) == []


def test_free_tier_plan_names_the_gui_project(run, project, data, schema, ready, no_assistant_runner):
    be = Backend(("plan_workflow", {"request": ask(data), "project": project["id"]}), "ok")
    _turn(run, _session(be, schema), "plan it")
    res = be.tool_result()
    assert res["ok"] and res["blocking"] == [] and "your project “testproj”" == res["project"]
    assert "hetsiren train" in res["plan"] and res["not_in_the_workflow"]


# ── stop and relaunch through the job manager ────────────────────────────────
def _filter_job(client, run, project, md_file):
    body = {"program": "filter_latents", "args": {"latents": str(md_file)}, "launch": True}
    r = run(client.post(f"/api/projects/{project['id']}/jobs", json=body))
    assert r.status_code == 200, r.text
    return r.json()


def _call(run, name, args, answers=None, level=None):
    from hax_assistant.agent import AgentContext, default_registry
    from hax_assistant.interaction import ScriptedInteraction
    inter = ScriptedInteraction(answers or {})

    async def go():
        reg = default_registry()
        for t in assistant_tools.chat_tools(asyncio.get_running_loop(), inter, threading.Event()):
            reg.register(t)
        return await asyncio.to_thread(reg.call, name, args, AgentContext(interaction=inter, autonomy_level=level))
    return json.loads(run(go())), inter


def test_stop_goes_through_the_job_manager(client, run, project, md_file, ready, no_destruction,
                                           no_assistant_runner):
    GATE.write_text("")
    try:
        job = wait_for_status(client, run, _filter_job(client, run, project, md_file)["id"], {"running"})
        args = {"run_dir": job["workdir"]}
        out, _ = _call(run, "stop_run", args, level="advisory")
        assert out["status"] == "not_executed" and job["workdir"] in out["proposal"]
        out, inter = _call(run, "stop_run", args, level="auto")             # not the assistant's job: asks
        assert out == {"status": "declined_by_user"} and inter.asked[0].key == "confirm_stop_run"
        assert manager.get(job["id"])["status"] == "running"
        out, _ = _call(run, "stop_run", args, {"confirm_stop_run": "yes"}, level="ask")
        assert out["status"] == "stopped"
        assert wait_for_status(client, run, job["id"], {"cancelled", "failed"})["status"] == "cancelled"
    finally:
        GATE.unlink(missing_ok=True)
    out, _ = _call(run, "stop_run", {"run_dir": str(ready)}, {"confirm_stop_run": "yes"}, level="ask")
    assert "not the folder of a job of the GUI" in out["error"]


def test_relaunch_is_a_new_job_and_resume_is_left_to_the_user(client, run, project, md_file, ready,
                                                              no_destruction, no_assistant_runner):
    old = wait_for_status(client, run, _filter_job(client, run, project, md_file)["id"], {"succeeded"})
    args = {"run_dir": old["workdir"], "restart": "scratch", "changes": {}}
    out, inter = _call(run, "relaunch_run", args, {"confirm_relaunch_run": "yes"}, level="ask")
    assert out["status"] == "relaunched", out
    assert "new job in a new folder" in inter.asked[0].text
    new = next(j for j in manager.list_for_project(project["id"]) if j["name"] == out["new_job"])
    assert new["id"] != old["id"] and new["workdir"] != old["workdir"]
    assert wait_for_status(client, run, new["id"], {"succeeded"})["project_id"] == project["id"]
    assert manager.get(old["id"])["status"] == "succeeded" and Path(old["workdir"]).is_dir()
    out, _ = _call(run, "relaunch_run", {**args, "restart": "resume"}, {"confirm_relaunch_run": "yes"}, level="ask")
    assert out["status"] == "not_applied" and "Continue button" in out["message"]
    out, _ = _call(run, "relaunch_run", {**args, "changes": {"output_path": "/tmp/x"}},
                   {"confirm_relaunch_run": "yes"}, level="ask")
    assert out["status"] == "not_relaunched" and "output_path" in out["reason"]


# ── the chat socket hands the session the GUI's tools and places proposals ───
def test_socket_passes_gui_tools_and_places_proposals(project, monkeypatch, ready):
    seen = {}

    def stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        seen["tools"] = [t.spec.name for t in tools]
        yield {"type": "action", "action": {"kind": "create_job", "program": "hetsiren", "input_path": "/d/p.xmd"}}
    monkeypatch.setattr(assistant_bridge, "stream_events", stream)
    with TestClient(app).websocket_connect("ws://127.0.0.1/ws/assistant/chat") as ws:
        ws.send_json({"messages": [{"role": "user", "content": f"run hetsiren in my project {project['id']}"}],
                      "turn": 1})
        assert ws.receive_json()["action"]["project_id"] == project["id"]
        assert ws.receive_json() == {"done": True, "turn": 1}
        ws.send_json({"messages": [{"role": "user", "content": "run hetsiren on /d/p.xmd"}], "turn": 2,
                      "project_id": "the-chats"})
        assert ws.receive_json()["action"]["project_id"] == "the-chats"
        assert ws.receive_json() == {"done": True, "turn": 2}
    assert seen["tools"] == ["list_jobs", "watch_run", "plan_workflow", "launch_workflow", "stop_run", "relaunch_run"]



def test_gui_tools_never_delete():
    hits = [f"assistant_tools.py:{n}: {line.strip()}"
            for n, line in enumerate((BACKEND / "assistant_tools.py").read_text().splitlines(), 1)
            if _DESTRUCTIVE.search(line)]
    assert not hits, "\n".join(hits)
