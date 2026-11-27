"""The chat goes on after a run it launched: a launch's ``then`` note is armed for the chat; when the run
succeeds, at auto the server starts that turn itself, otherwise the chat gets a Continue button; a run
that does not succeed is reported, a chain of automatic turns stops at MAX_CHAIN, and waiting survives a
restart. Scripted model, hax stub."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from conftest import GATE, wait_for_status
from test_assistant_custom_workflow import _design
from test_assistant_gui_tools import Backend, _session, no_assistant_runner, ready  # noqa: F401
from test_assistant_workflows import _set_autonomy, data, isolated, no_destruction, schema, stub  # noqa: F401

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_continue, assistant_tools
from hax_gui import chats as chat_store

pytestmark = pytest.mark.skipif(not assistant_bridge.available(), reason="hax_assistant is not installed")


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(assistant_continue, "POLL_S", 0.05)


@pytest.fixture
def model(schema, monkeypatch):
    """One scripted tool-calling model for the user's turn and the turns the server starts."""
    def use(*replies):
        be = Backend(*replies)
        s = _session(be, schema)
        monkeypatch.setattr(assistant_bridge, "_get", lambda: s)
        monkeypatch.setattr(assistant_bridge, "tier", lambda: "free")
        return be, s
    return use


def _chat():
    c = chat_store.manager.create(None, "continue chat")
    chat_store.manager.update(c["id"], messages=[{"role": "user", "content": "run it"}])
    return c["id"]


def _turn(run, session, text, chat_id, project_id=None):
    from hax_assistant.interaction import ScriptedInteraction
    inter, cancel = ScriptedInteraction({}), threading.Event()

    async def go():
        tools = assistant_tools.chat_tools(asyncio.get_running_loop(), inter, cancel, project_id, chat_id=chat_id)
        return await asyncio.to_thread(lambda: list(session.stream_events(
            [{"role": "user", "content": text}], grounded=False, interaction=inter, tools=tools)))
    return run(go())


def _settle(run, timeout=60.0):
    async def wait():
        for _ in range(int(timeout / 0.05)):
            if not assistant_continue._tasks:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("the chat never went on")
    run(wait())


def _posted(cid):
    return [m for m in chat_store.manager.get(cid)["messages"] if m.get("posted") == "continue"]


def _job(client, run, project, md_file):
    r = run(client.post(f"/api/projects/{project['id']}/jobs",
                        json={"program": "filter_latents", "args": {"latents": str(md_file)}, "launch": True}))
    assert r.status_code == 200, r.text
    return r.json()


def test_job_then_goes_on_by_itself_at_auto(client, run, project, md_file, ready, model, no_assistant_runner):
    _set_autonomy("auto")
    cid = _chat()
    be, s = model(("launch_job", {"program": "filter_latents", "args": {"latents": str(md_file)},
                                  "project": project["name"], "then": "tell the user the filtering is done"}),
                  "Launched.", "The filtering is done.")
    _turn(run, s, "filter the latents, then tell me", cid)
    launched = be.tool_result()
    assert launched["then"].startswith("set"), launched
    wait_for_status(client, run, launched["job"]["id"], {"succeeded"})
    _settle(run)
    went_on = be.calls[-1][-1].content
    assert "has succeeded" in went_on and "tell the user the filtering is done" in went_on
    assert [m["content"] for m in _posted(cid)] == ["The filtering is done."]
    assert assistant_continue._load() == {}


def test_workflow_then_goes_on_by_itself_at_auto(client, run, project, data, stub, ready, model,
                                                  no_destruction, no_assistant_runner):
    _set_autonomy("auto")
    cid = _chat()
    steps, edges = _design(data)
    be, s = model(("launch_custom_workflow", {"name": "chain", "steps": steps[:2], "edges": edges[:1],
                                              "project": project["name"], "then": "select the states and decode"}),
                  "Started.", ("list_jobs", {}), "Both steps succeeded; selecting the states next.")
    _turn(run, s, "train and predict, then decode", cid)
    assert be.tool_result()["then"].startswith("set")
    _settle(run, timeout=90.0)
    assert "the workflow “chain” has succeeded" in be.calls[-2][-1].content
    assert _posted(cid)[-1]["content"].startswith("Both steps succeeded")


def test_continue_button_under_ask(client, run, project, md_file, ready, model, no_assistant_runner):
    _set_autonomy("ask")
    cid = _chat()
    be, _ = model("never called")
    job = _job(client, run, project, md_file)
    run(assistant_continue.arm("job", job["id"], cid, "decode the states", 0, project["id"]))
    _settle(run)
    nxt = _posted(cid)[-1]
    assert "has succeeded" in nxt["content"] and "decode the states" in nxt["content"]
    assert nxt["next"]["shown"] == "Continue: decode the states" and "has succeeded" in nxt["next"]["prompt"]
    assert be.calls == []                                     # nothing ran without the user
    assert run(client.post(f"/api/assistant/continue/{cid}/{nxt['id']}")).status_code == 200
    assert _posted(cid)[-1]["next"]["used"] is True
    assert run(client.post(f"/api/assistant/continue/{cid}/{nxt['id']}")).status_code == 404   # once


def test_a_run_that_fails_is_reported_not_continued(client, run, project, md_file, ready, model, no_assistant_runner):
    _set_autonomy("auto")
    cid = _chat()
    be, _ = model("never called")
    GATE.write_text("")
    try:
        job = _job(client, run, project, md_file)
        wait_for_status(client, run, job["id"], {"running"})
        run(assistant_continue.arm("job", job["id"], cid, "decode the states", 0, project["id"]))
        assert run(client.post(f"/api/jobs/{job['id']}/cancel")).status_code == 200
        _settle(run)
    finally:
        GATE.unlink(missing_ok=True)
    msg = _posted(cid)[-1]
    assert "not started" in msg["content"] and "cancelled" in msg["content"] and "next" not in msg
    assert be.calls == []


def test_chain_stops_after_max_chain(client, run, project, md_file, ready, model, no_assistant_runner):
    _set_autonomy("auto")
    cid = _chat()
    be, _ = model("never called")
    job = _job(client, run, project, md_file)
    run(assistant_continue.arm("job", job["id"], cid, "one more step", assistant_continue.MAX_CHAIN, project["id"]))
    _settle(run)
    msg = _posted(cid)[-1]
    assert f"{assistant_continue.MAX_CHAIN} times in a row" in msg["content"] and msg["next"]["used"] is False
    assert be.calls == []


def test_waiting_survives_a_restart(client, run, project, md_file, ready, model, no_assistant_runner):
    _set_autonomy("ask")
    cid = _chat()
    model("never called")
    job = wait_for_status(client, run, _job(client, run, project, md_file)["id"], {"succeeded"})
    assistant_continue._change(f"job:{job['id']}", {"kind": "job", "id": job["id"], "chat_id": cid,
                                                    "note": "decode", "depth": 0, "project_id": project["id"]})
    run(assistant_continue.restore())
    _settle(run)
    assert _posted(cid)[-1]["next"]["shown"] == "Continue: decode"
    assert json.loads(json.dumps(assistant_continue._load())) == {}
