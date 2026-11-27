"""The free-tier tools name jobs the way a model knows them: launch_workflow reports each step's job,
list_jobs shows a project's jobs, and stop_run/relaunch_run take a job id, job name, run folder or a
project (its running job; several are asked about, never guessed). Jobs of a workflow the assistant
started are registered as its own when they start; jobs the user starts stay the user's.

Scripted tool-calling backend; the conftest stub holds jobs running while the GATE file exists.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import wait_for_status
from test_assistant_gui_tools import Backend, _call, _session, _turn, no_assistant_runner  # noqa: F401
from test_assistant_stop_card import _job, cancels, gate  # noqa: F401  (fixtures)
from test_assistant_workflows import _set_autonomy, ask, data, isolated, no_destruction, schema  # noqa: F401

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_tools, assistant_workflows, projects
from hax_gui.jobs import manager

needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
pytestmark = needs_assistant


@pytest.fixture
def ready(project):
    assert assistant_bridge.ensure_workspace() is None
    return project


def _owned(job):
    from hax_assistant import safety
    return safety.is_owned(job["workdir"])


def _until(run, cond, timeout=15.0):
    async def wait():
        for _ in range(int(timeout / 0.1)):
            if cond():
                return True
            await asyncio.sleep(0.1)
        return False
    assert run(wait()), "timed out"


def test_launch_then_stop_it_by_project_name(client, run, ready, data, schema, gate, cancels,
                                             no_destruction, no_assistant_runner):
    p = projects.create("refs_launch")
    _set_autonomy("auto")
    be = Backend(("launch_workflow", {"request": ask(data), "project": p["name"]}), "Started.")
    _turn(run, _session(be, schema), "run the continuous motion workflow in refs_launch")
    res = be.tool_result()
    first, second = res["steps"]
    assert res["status"] == "started" and res["project"]["id"] == p["id"] and res["workflow"]["id"]
    assert first["job_id"] and Path(first["run_dir"]).parent == Path(p["path"])
    assert (second["step"], second["job_id"], second["run_dir"]) == ("hetsiren predict", None, None)
    job = wait_for_status(client, run, first["job_id"], {"running"})
    _until(run, lambda: _owned(job))                         # registered as the assistant's when it started
    listed = _call(run, "list_jobs", {"project": p["name"]})[0]
    row = next(j for j in listed["jobs"] if j["id"] == job["id"])
    assert row["launched_by_assistant"] and row["run_dir"] == job["workdir"] and row["step"] == "hetsiren train"
    assert row["workflow_id"] == res["workflow"]["id"] and listed["total"] == 1

    be = Backend(("stop_run", {"job": p["name"]}), "Stopped.")
    _, inter = _turn(run, _session(be, schema), "stop it")    # auto, and its own job: no question
    out = be.tool_result()
    assert out["status"] == "stopped" and cancels == [job["id"]] and not inter.asked, (out, inter.asked)
    assert wait_for_status(client, run, job["id"], {"cancelled", "failed"})["status"] == "cancelled"


def test_project_with_several_running_jobs_is_asked_about(client, run, ready, md_file, gate, cancels,
                                                         no_destruction):
    p = projects.create("refs_several")
    jobs = [_job(client, run, p["id"], md_file, wait=("queued", "running")) for _ in range(3)]
    assert not any(_owned(j) for j in jobs)                  # started by hand: the user's
    _set_autonomy("auto")
    out, inter = _call(run, "stop_run", {"job": p["name"]}, {"job": "1", "confirm_stop_run": "yes"})
    q = inter.asked[0]
    assert q.key == "job" and len(q.choices) == 4 and q.choices[-1] == assistant_tools.ALL
    assert inter.asked[1].key == "confirm_stop_run"           # not the assistant's job: asked under auto too
    picked = next(j for j in jobs if Path(j["workdir"]).name in q.choices[0])
    assert out["status"] == "stopped" and cancels == [picked["id"]]
    out, inter = _call(run, "stop_run", {"run_dir": p["path"]}, {"job": assistant_tools.ALL, "confirm_stop_run": "yes"})
    assert out["status"] == "stopped" and len(out["jobs"]) == 2 and sorted(cancels) == sorted(j["id"] for j in jobs)
    for j in jobs:
        wait_for_status(client, run, j["id"], {"cancelled", "failed"})
    out, _ = _call(run, "stop_run", {"job": p["name"]}, {"confirm_stop_run": "yes"})
    assert out["status"] == "not_running" and "nothing is waiting or running" in out["reason"]
    assert "is cancelled" in out["reason"] and len(cancels) == 3


def test_job_name_id_and_list(client, run, ready, md_file, gate, cancels, no_destruction, monkeypatch):
    p = projects.create("refs_named")
    body = {"program": "filter_latents", "args": {"latents": str(md_file)}, "launch": True, "name": "my filter"}
    named = run(client.post(f"/api/projects/{p['id']}/jobs", json=body)).json()
    other = _job(client, run, p["id"], md_file, wait=("queued", "running"))
    wait_for_status(client, run, named["id"], {"queued", "running"})
    out, inter = _call(run, "stop_run", {"job": "My Filter"}, {"confirm_stop_run": "yes"}, level="ask")
    assert out["status"] == "stopped" and cancels == [named["id"]] and [q.key for q in inter.asked] == ["confirm_stop_run"]
    out, _ = _call(run, "stop_run", {"job": other["id"]}, {"confirm_stop_run": "yes"}, level="ask")
    assert out["status"] == "stopped" and cancels == [named["id"], other["id"]]
    out, _ = _call(run, "stop_run", {"job": "no such thing"}, {"confirm_stop_run": "yes"}, level="ask")
    assert "list_jobs" in out["error"]
    monkeypatch.setattr(assistant_tools, "LIST_MAX", 1)
    listed = _call(run, "list_jobs", {"project": p["id"]})[0]
    assert listed["total"] == 2 and len(listed["jobs"]) == 1 and "newest 1 of 2" in listed["truncated"]
    assert listed["jobs"][0]["launched_by_assistant"] is False and "workflow" not in listed["jobs"][0]
    assert "there is no project" in _call(run, "list_jobs", {"project": "nope"})[0]["error"]
    for j in (named, other):
        wait_for_status(client, run, j["id"], {"cancelled", "failed"})


def test_stop_card_names_a_project_like_the_tools(client, run, ready, md_file, gate):
    p = projects.create("refs card")                          # a name with a space
    job = _job(client, run, p["id"], md_file)
    out = assistant_tools.stop_target("please stop the run in refs card", None, None)
    assert [j["id"] for j in out["jobs"]] == [job["id"]]
    empty = projects.create("refs_card_empty")
    assert assistant_tools.stop_target(f"stop the run in {empty['id']}", job["id"], None) == {
        "jobs": [], "named": empty["id"]}                     # nothing running there: not some other job
    run(manager.cancel(job["id"]))
    wait_for_status(client, run, job["id"], {"cancelled", "failed"})
    assert assistant_workflows.launched.get(job["id"]) is None
