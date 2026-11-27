"""Workflows the chat model designs (check_custom_workflow, launch_custom_workflow): wired as a planned
workflow is, guarded step by step like launch_job, run by the GUI's own engine. Scripted model, hax stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_assistant_gui_tools import Backend, _session, _turn, no_assistant_runner, ready  # noqa: F401
from test_assistant_workflows import (  # noqa: F401  (fixtures)
    _set_autonomy, _settle, data, isolated, no_destruction, schema, stub)

from hax_gui import assistant as assistant_bridge
from hax_gui import projects
from hax_gui.jobs import manager

pytestmark = pytest.mark.skipif(not assistant_bridge.available(), reason="hax_assistant is not installed")


def _design(data):
    posed = {"md": str(data["md"]), "sr": 2.0, "ctf_type": "apply"}
    steps = [{"id": "train", "program": "hetsiren", "args": {"mode": "train", **posed, "epochs": 2}},
             {"id": "predict", "program": "hetsiren", "args": {"mode": "predict", **posed}},
             {"id": "fix", "program": "modart", "args": {"reconstruct_halves": True}, "name": "motion-corrected map"}]
    edges = [{"from": "train", "to": "predict", "field": "reload"},
             {"from": "train", "to": "fix", "field": "motion_correction"}]
    return steps, edges


def test_designed_workflow_runs_in_the_gui_engine(client, run, project, data, stub, schema, ready,
                                                  no_destruction, no_assistant_runner):
    _set_autonomy("auto")
    steps, edges = _design(data)
    be = Backend(("check_custom_workflow", {"steps": steps, "edges": edges}),
                 ("launch_custom_workflow", {"name": "my chain", "steps": steps, "edges": edges,
                                             "project": project["name"]}), "Started.")
    _, inter = _turn(run, _session(be, schema), "train, predict and correct the motion")
    checked, launched = be.tool_result(0), be.tool_result(1)
    assert checked["ok"], checked
    fix = next(s for s in checked["steps"] if s["step"] == "motion-corrected map")
    assert set(fix["inherited"]) >= {"md", "sr", "ctf_type"} and fix["wired"] == {"motion_correction": "1. hetsiren train"}
    assert launched["status"] == "started" and launched["project"]["id"] == project["id"], launched
    assert [n for n, _ in inter.notifications if "Done automatically" in n]
    w = _settle(client, run, launched["workflow"]["id"])
    assert w["status"] == "succeeded", w["run_state"]
    trace = [e for e in stub() if e["event"] == "start"]
    tags = [e["tag"] for e in trace[-3:]]
    assert tags[0] == "hetsiren:train" and sorted(tags[1:]) == ["hetsiren:predict", "modart"]
    train_out = Path(trace[-3]["args"]["output_path"])
    assert Path(next(e for e in trace[-2:] if e["tag"] == "hetsiren:predict")["args"]["reload"]) == train_out
    modart = next(e for e in trace[-2:] if e["tag"] == "modart")["args"]
    assert modart["md"] == str(data["md"]) and Path(modart["motion_correction"]).is_relative_to(train_out)
    assert all(Path(e["args"]["output_path"]).parent == Path(project["path"]) for e in trace[-3:])


@pytest.mark.parametrize("change, words", [
    (lambda s, e: (s + [{"id": "decode", "program": "decode_states_from_latents", "args": {}}],
                   e + [{"from": "predict", "to": "decode", "field": "latents_file"},
                        {"from": "train", "to": "decode", "field": "reload"}]), "select_states"),
    (lambda s, e: (s, e + [{"from": "train", "to": "fix", "field": "sr"}]), "cannot take another step's output"),
    (lambda s, e: (s, e + [{"from": "train", "to": "fix", "field": "motion_correction"}]), "wired twice"),
    (lambda s, e: (s, [{"from": "predict", "to": "fix", "field": "motion_correction"}] + e[:1]), "does not produce"),
    (lambda s, e: (s, e + [{"from": "predict", "to": "train", "field": "md"}]), "wait on each other"),
    (lambda s, e: (s, e + [{"from": "ghost", "to": "fix", "field": "md"}]), "not in the workflow"),
    (lambda s, e: ([{**s[0], "args": {**s[0]["args"], "output_path": "/tmp/x"}}] + s[1:], e), "job manager"),
    (lambda s, e: ([{**s[0], "args": {**s[0]["args"], "ctf_type": "nonsense"}}] + s[1:], e), "ctf_type"),
])
def test_designed_workflow_errors(run, project, data, schema, ready, no_assistant_runner, change, words):
    _set_autonomy("auto")
    steps, edges = change(*_design(data))
    before = len(manager.list_for_project(project["id"]))
    be = Backend(("check_custom_workflow", {"steps": steps, "edges": edges}),
                 ("launch_custom_workflow", {"name": "bad", "steps": steps, "edges": edges,
                                             "project": project["name"]}), "ok")
    _turn(run, _session(be, schema), "go")
    checked, launched = be.tool_result(0), be.tool_result(1)
    assert not checked["ok"] and words in json.dumps(checked["errors"]), checked
    assert launched["status"] == "not_created" and words in json.dumps(launched), launched
    assert len(manager.list_for_project(project["id"])) == before


def test_designed_workflow_follows_autonomy(run, project, data, schema, ready, no_assistant_runner):
    steps, edges = _design(data)
    call = ("launch_custom_workflow", {"name": "my chain", "steps": steps, "edges": edges, "project": project["name"]})
    _set_autonomy("advisory")
    be = Backend(call, "ok")
    _turn(run, _session(be, schema), "go")
    assert be.tool_result()["status"] == "not_executed" and "motion_correction ← 1. hetsiren train" in be.tool_result()["proposal"]
    _set_autonomy("ask")
    be = Backend(call, "ok")
    _, inter = _turn(run, _session(be, schema), "go", {"confirm_launch_custom_workflow": "no"})
    assert be.tool_result()["status"] == "declined_by_user" and "Proceed?" in inter.asked[-1].text


def test_designed_workflow_without_a_project_makes_one(client, run, data, stub, schema, ready,
                                                       no_destruction, no_assistant_runner):
    _set_autonomy("auto")
    steps, edges = _design(data)
    be = Backend(("launch_custom_workflow", {"name": "fresh chain", "steps": steps[:2], "edges": edges[:1]}), "ok")
    _turn(run, _session(be, schema), "go")                      # the chat has no project
    res = be.tool_result()
    assert res["status"] == "started", res
    assert projects.get(res["project"]["id"])["name"] == "fresh chain"
    assert _settle(client, run, res["workflow"]["id"])["status"] == "succeeded"
