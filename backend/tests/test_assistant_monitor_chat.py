"""The run monitor in the chat: a linked job's findings are posted into its chat (persisted, unread, one
decision wherever Apply is clicked), a chat can watch any job (the user's too: then every change waits
for the user), and with a tool-calling model the assistant reviews each finding there through the same
gates (scripted backends). Also: an answer frame never goes through detection, the links and the
launched workflows survive a restart, and a relaunched workflow step says what it leaves blocked.

Jobs are the conftest stub held running by the GATE file, with diverged HetSIREN curves written into
their run folders; the assistant's settings are per test.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import GATE, wait_for_status
from test_assistant_gui_tools import Backend, _session
from test_assistant_stop_card import gate  # noqa: F401  (fixture)
from test_assistant_wiring import _diverged, _write_run, no_destruction  # noqa: F401
from test_assistant_workflows import isolated, schema  # noqa: F401  (fixtures)

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_monitor, events, watch
from hax_gui import chats as chat_store
from hax_gui.app import app
from hax_gui.jobs import manager

needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
pytestmark = needs_assistant
APPLY = "/api/assistant/watch/{}/apply"
REVIEW = "/api/assistant/watch/{}/apply-review"


@pytest.fixture
def ready(project, monkeypatch):
    pytest.importorskip("tensorboardX")
    pytest.importorskip("tbparse")
    assert assistant_bridge.ensure_workspace() is None
    monkeypatch.setattr(assistant_bridge, "tier", lambda: "guided")           # no review unless a test asks
    return project


@pytest.fixture
def diverging(client, run, ready, md_file, gate):
    """A running HetSIREN job (the user's) whose loss has just gone NaN."""
    body = {"program": "hetsiren", "launch": True,
            "args": {"md": str(md_file), "sr": 1.0, "ctf_type": "None", "mode": "train"}}
    job = wait_for_status(client, run, run(client.post(f"/api/projects/{ready['id']}/jobs", json=body)).json()["id"],
                          {"running"})
    _write_run(Path(job["workdir"]), _diverged())
    yield job
    GATE.unlink(missing_ok=True)
    for j in manager.list_for_project(ready["id"]):
        if j["status"] in ("pending", "queued", "running", "cancelling"):
            wait_for_status(client, run, j["id"], {"succeeded", "cancelled", "failed"})


def _level(level):
    from hax_assistant import autonomy
    autonomy.set_autonomy(level)


def _chat(title="monitor chat"):
    c = chat_store.manager.create(None, title)
    chat_store.manager.update(c["id"], messages=[{"role": "user", "content": "train it"}])
    return c["id"]


def _alert(run, job):
    """What the watcher does on its next check of the job."""
    report = assistant_bridge.watch_report(job["workdir"], True)
    run(watch._assistant_alerts(manager.get(job["id"]), set(), report))
    return next(f for f in report["findings"] if f["kind"] == "diverged")


def _posted(cid, kind=None):
    return [m for m in chat_store.manager.get(cid)["messages"] if m.get("posted") and (kind is None or m["posted"] == kind)]


def _settle(run):
    async def wait():
        for _ in range(400):
            if not assistant_monitor._tasks:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("the review never finished")
    run(wait())


def _own(job):
    assistant_bridge.register_job(job["workdir"], job["pid"], job.get("command") or [])


# ── part 1 ───────────────────────────────────────────────────────────────────
def test_finding_is_posted_to_its_chat_and_decided_once(client, run, diverging, no_destruction):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _level("auto")                                     # the user's job: asked even under auto
    q = events.subscribe()
    _alert(run, diverging)
    msg = _posted(cid)[0]
    assert msg["watch"]["kind"] == "diverged" and msg["watch"]["needs_confirmation"]
    assert "Nothing changes until you click Apply" in msg["content"] and assistant_monitor.USER_RUN in msg["content"]
    assert next(c for c in chat_store.manager.list() if c["id"] == cid)["unread"]
    seen = []
    while not q.empty():
        seen.append(q.get_nowait())
    assert {e["kind"] for e in seen} >= {"watch", "chat"}
    r = run(client.post(APPLY.format(diverging["id"]), json={"kind": "diverged"}))        # from the chat
    assert r.status_code == 200 and r.json()["applied"], r.text
    assert _posted(cid)[0]["resolved"]["applied"] and any(e.get("kind") == "watch_resolved" for e in _drain(q))
    r = run(client.post(APPLY.format(diverging["id"]), json={"kind": "diverged"}))        # the notification's
    assert r.status_code == 409 and "already" in r.text
    events.unsubscribe(q)
    # the client saves its stale copy: the server's posted message survives, decided, and is now seen
    stale = [{"role": "user", "content": "train it"}, {**msg, "seen": True}, {"role": "user", "content": "thanks"}]
    out = chat_store.manager.update(cid, messages=stale)["messages"]
    assert [m.get("posted") for m in out] == [None, "watch", None] and out[1]["resolved"] and out[1]["seen"]
    assert not next(c for c in chat_store.manager.list() if c["id"] == cid)["unread"]


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_an_unlinked_job_keeps_the_notification_only(run, diverging, no_destruction):
    cid = _chat()
    _level("ask")
    _alert(run, diverging)
    assert _posted(cid) == [] and assistant_monitor._finding(diverging["id"], "diverged") is None


# ── watching from a chat ─────────────────────────────────────────────────────
def test_watch_from_chats_links_moves_and_unlinks(client, run, diverging, no_destruction, monkeypatch):
    a, b = _chat("first"), _chat("second")
    _level("auto")

    def stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        yield {"type": "action", "action": {"kind": "watch_run", "job_id": None,
                                            "on": "stop" not in messages[-1]["content"]}}
    monkeypatch.setattr(assistant_bridge, "stream_events", stream)
    lines = []
    with TestClient(app).websocket_connect("ws://127.0.0.1/ws/assistant/chat") as ws:
        for cid, text in ((a, "watch this run"), (b, "watch this run"), (b, "stop watching it")):
            ws.send_json({"messages": [{"role": "user", "content": text}], "turn": len(lines) + 1, "chat_id": cid,
                          "route_job_id": diverging["id"]})
            lines.append(ws.receive_json()["token"])
            assert ws.receive_json()["done"]
            if len(lines) == 1:
                assert assistant_monitor.chat_of(diverging["id"]) == a and diverging["id"] in watch._watched
    assert lines[0].startswith(f"Watching “{diverging['program']}” (running)") and "you started this run" in lines[0]
    assert "move here from the chat “first”" in lines[1]
    assert lines[2].startswith("Stopped watching") and assistant_monitor.chat_of(diverging["id"]) is None
    assert diverging["id"] not in watch._watched
    assistant_monitor.link(diverging["id"], a)
    GATE.unlink()
    wait_for_status(client, run, diverging["id"], {"succeeded"})
    run(watch._check(diverging["id"]))                    # the watcher sees it ended: the link ends too
    assert assistant_monitor.chat_of(diverging["id"]) is None


def test_an_answer_never_goes_through_detection(monkeypatch):
    from hax_assistant.interaction import Question
    calls = []

    def stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        calls.append(messages[-1]["content"])
        ans = interaction.ask(Question("why", "What is it for?"))
        yield {"type": "content", "delta": f"noted: {ans}"}
    monkeypatch.setattr(assistant_bridge, "stream_events", stream)
    with TestClient(app).websocket_connect("ws://127.0.0.1/ws/assistant/chat") as ws:
        ws.send_json({"messages": [{"role": "user", "content": "prepare a mask"}], "turn": 1})
        q = ws.receive_json()["question"]
        ws.send_json({"answer": {"id": q["id"], "value": "para el entrenamiento"}})
        assert ws.receive_json() == {"token": "noted: para el entrenamiento"}
        assert ws.receive_json()["done"]
    assert calls == ["prepare a mask"]                     # one turn: the answer was never a request


# ── part 2: the assistant reviews it (free tier) ─────────────────────────────
@pytest.fixture
def free(schema, monkeypatch):
    """A tool-calling model, scripted per test."""
    def use(*replies):
        be = Backend(*replies)
        s = _session(be, schema)
        monkeypatch.setattr(assistant_bridge, "_get", lambda: s)
        monkeypatch.setattr(assistant_bridge, "tier", lambda: "free")
        return be
    return use


RULE = {"learning_rate": 3e-05}


def _relaunch(job, **changes):
    return ("relaunch_run", {"job": job["id"], "restart": "scratch", "changes": changes or dict(RULE)})


def _review(cid):
    return _posted(cid, "review")[-1]


def test_review_agrees_and_asks_under_ask(client, run, diverging, no_destruction, free):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _level("ask")
    be = free(_relaunch(diverging), "The loss went NaN: restart with a lower learning rate, as the rule says.")
    _alert(run, diverging)
    _settle(run)
    rev = _review(cid)
    assert "agrees with its proposal" in rev["content"] and rev["watch"]["proposed"]["changes"] == RULE
    assert json.loads([m for m in be.calls[1] if m.role == "tool"][-1].content)["status"] == "awaiting_confirmation"
    jobs_before = len(manager.list_for_project(diverging["project_id"]))
    r = run(client.post(REVIEW.format(diverging["id"]), json={"kind": "diverged"}))
    assert r.status_code == 200 and r.json()["applied"], r.text
    assert len(manager.list_for_project(diverging["project_id"])) == jobs_before + 1
    assert run(client.post(APPLY.format(diverging["id"]), json={"kind": "diverged"})).status_code == 409
    assert all(m["resolved"] for m in _posted(cid))


def test_review_disagrees_side_by_side_and_one_decision(client, run, diverging, no_destruction, free):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _level("ask")
    free(_relaunch(diverging, learning_rate=1e-5, batch_size=4), "Lower it further and halve the batch.")
    _alert(run, diverging)
    _settle(run)
    rev = _review(cid)
    assert "disagrees and proposes a scratch restart with --learning_rate 1e-05, --batch_size 4" in rev["content"]
    assert "instead of a scratch restart with --learning_rate 3e-05" in rev["content"]
    assert rev["watch"]["rule"]["changes"] == RULE                         # both, side by side
    assert run(client.post(APPLY.format(diverging["id"]), json={"kind": "diverged"})).status_code == 200
    assert run(client.post(REVIEW.format(diverging["id"]), json={"kind": "diverged"})).status_code == 409


def test_review_says_leave_it_under_advisory(run, diverging, no_destruction, free):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _level("advisory")
    free("One NaN at the very end; I would leave it and look again in an hour.")
    jobs_before = len(manager.list_for_project(diverging["project_id"]))
    _alert(run, diverging)
    _settle(run)
    rev = _review(cid)
    assert "leaving it as it is, unlike the rule's proposal" in rev["content"] and not rev["watch"]["proposed"]
    assert len(manager.list_for_project(diverging["project_id"])) == jobs_before


def test_auto_acts_only_on_its_own_calibrated_finding(run, diverging, no_destruction, free):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _own(diverging)
    _level("auto")
    free(_relaunch(diverging), "Restarting from scratch with a lower learning rate.")
    finding = _alert(run, diverging)
    assert finding["calibrated"] and _posted(cid, "watch")[0]["watch"]["deferred"]    # the review decides
    _settle(run)
    rev = _review(cid)
    assert rev["watch"]["resolved"]["applied"] and "Done automatically" in rev["content"]
    assert _posted(cid, "watch")[0]["resolved"]["by"] == "assistant"
    assert len(manager.list_for_project(diverging["project_id"])) >= 2


def test_auto_asks_for_a_job_it_did_not_launch(run, diverging, no_destruction, free):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _level("auto")
    free(_relaunch(diverging), "Restart it.")
    jobs_before = len(manager.list_for_project(diverging["project_id"]))
    _alert(run, diverging)
    _settle(run)
    assert _review(cid)["watch"]["proposed"] and not _review(cid)["watch"]["resolved"]
    assert _posted(cid, "watch")[0]["watch"]["needs_confirmation"]
    assert len(manager.list_for_project(diverging["project_id"])) == jobs_before


def test_budget_used_up_is_respected(run, diverging, no_destruction, free):
    from hax_assistant import autonomy
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _own(diverging)
    _level("auto")
    autonomy.set_budget("max_relaunches", 0)
    be = free(_relaunch(diverging), "Tried to restart it.")
    jobs_before = len(manager.list_for_project(diverging["project_id"]))
    _alert(run, diverging)
    _settle(run)
    result = json.loads([m for m in be.calls[1] if m.role == "tool"][-1].content)
    assert result["status"] == "not_relaunched" and "budget" in result["reason"]
    assert len(manager.list_for_project(diverging["project_id"])) == jobs_before and not _review(cid)["watch"]["resolved"]


def test_a_failed_review_keeps_the_finding_and_auto_falls_back(run, diverging, no_destruction, free):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _own(diverging)
    _level("auto")
    be = free()
    be.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model down"))
    _alert(run, diverging)
    _settle(run)
    finding, note = _posted(cid, "watch")[0], _review(cid)
    assert "could not review" in note["content"] and "model down" in note["content"]
    assert finding["resolved"]["by"] == "auto"                     # what the watcher would have done


def test_no_review_while_the_user_has_a_turn_running(run, diverging, no_destruction, free):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    _level("advisory")
    free("Leave it.")
    assistant_monitor.turn_started(cid)
    _alert(run, diverging)
    _settle(run)
    assert _posted(cid, "review") == [] and assistant_monitor._waiting.get(cid)
    async def ended():
        assistant_monitor.turn_ended(cid)                          # on the loop, as the chat socket does
    run(ended())
    _settle(run)
    assert "leaving it" in _review(cid)["content"]
    assert not assistant_monitor.will_review(diverging["id"], "diverged")         # once per finding kind


# ── part 3 and restarts ──────────────────────────────────────────────────────
def test_a_relaunched_workflow_step_says_what_it_leaves_blocked(client, run, diverging, no_destruction):
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid, workflow_id="w1", workflow="motion of spike", step="hetsiren train")
    _level("ask")
    _alert(run, diverging)
    r = run(client.post(APPLY.format(diverging["id"]), json={"kind": "diverged"}))
    assert r.status_code == 200
    msg = r.json()["message"]
    assert "“hetsiren train” step of the workflow “motion of spike”" in msg and "stay blocked" in msg
    assert "--learning_rate 3e-05" in msg and "run that step with the steps after it" in msg
    new = r.json()["job"]["id"]
    assert assistant_monitor.chat_of(new) == cid                  # the relaunched run reports to the same chat


def test_links_and_launched_workflows_survive_a_restart(run, diverging, no_destruction, monkeypatch):
    from hax_gui import assistant_workflows
    cid = _chat()
    assistant_monitor.link(diverging["id"], cid)
    watch._watched.clear()                                         # what a restart forgets
    followed = []
    monkeypatch.setattr(assistant_workflows, "follow", lambda *a: followed.append(a))
    assistant_monitor.remember_workflow("w9", cid, "/tmp/p", [{"node": "n1", "title": "t", "watch": True}])
    run(assistant_monitor.restore())
    assert diverging["id"] in watch._watched
    assert followed == [("w9", [{"node": "n1", "title": "t", "watch": True}], "/tmp/p", cid)]
    assistant_monitor.forget_workflow("w9")
