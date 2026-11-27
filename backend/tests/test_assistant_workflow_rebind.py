"""A relaunched workflow step takes its place in its workflow (rebind_step): the later steps run on the
new job's outputs and the workflow ends well, old run folder intact, only that step's entry changed.
Refused, with the workflow left exactly as it was and the by-hand message, when the old job has not
ended, the run has not settled, the step no longer points at the old job, or the new job is not a fresh
job of the step's program and project. Real workflow engine; the stub hax holds a training while HOLD
exists.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from conftest import wait_for_status
from test_assistant_workflows import (  # noqa: F401  (fixtures)
    STUB, _set_autonomy, _settle, ask, data, isolated, no_destruction, schema)

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_actions, assistant_monitor, assistant_workflows, projects, watch
from hax_gui.jobs import manager
from hax_gui.workflows import WorkflowError, manager as wf

needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
pytestmark = needs_assistant
MOTION, RESOLUTION = "quick continuous motion analysis of", "quick: improve the resolution of"


@pytest.fixture
def held(client, run, tmp_path, monkeypatch, project):
    """The workflow tests' stub hax, which also holds a HetSIREN training while HOLD exists."""
    hold, trace = tmp_path / "hold", tmp_path / "trace.jsonl"
    hold.write_text("")
    script = tmp_path / "hax_stub.py"
    assert STUB.count('note("start")\n') == 1
    script.write_text(f"TRACE = {str(trace)!r}\nHOLD = {str(hold)!r}\n" + STUB.replace(
        'note("start")\n', 'note("start")\nimport os\nwhile tag == "hetsiren:train" and os.path.exists(HOLD):\n'
                           '    time.sleep(0.05)\n'))
    monkeypatch.setattr(assistant_actions.settings, "hax_cli", f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    assert assistant_bridge.ensure_workspace() is None
    _set_autonomy("ask")
    yield hold
    hold.unlink(missing_ok=True)
    for j in manager.list(limit=200):
        if j["status"] in ("pending", "queued", "running", "cancelling"):
            wait_for_status(client, run, j["id"], {"succeeded", "failed", "cancelled"})


@pytest.fixture
def seeds(monkeypatch):
    """Every seed_state call, with the run_state it replaced."""
    calls, real = [], wf.seed_state

    def spy(wid, state):
        calls.append((wf.get(wid)["run_state"], state))
        return real(wid, state)
    monkeypatch.setattr(wf, "seed_state", spy)
    return calls


def _until(run, cond, timeout=30.0):
    async def wait():
        for _ in range(int(timeout / 0.1)):
            if cond():
                return
            await asyncio.sleep(0.1)
        raise AssertionError("timed out")
    run(wait())


def _launch(run, project, data, goal, chat_id=None):
    out = run(assistant_workflows.create(ask(data, goal), project["id"], run=True, chat_id=chat_id))
    wid = out["workflow_id"]
    _until(run, lambda: ((wf.get(wid)["run_state"] or {}).get("n1") or {}).get("status") == "running")
    old = manager.get(wf.get(wid)["run_state"]["n1"]["job_id"])
    _until(run, lambda: assistant_bridge.owned(old["workdir"]))          # the follower registered it
    return wid, manager.get(old["id"])


def _bound(job, field):
    return (job.get("bindings") or {})[field]["job_id"]


def _audit(folder):
    from hax_assistant.monitor.controller import AUDIT_FILE
    return [json.loads(line) for line in (Path(folder) / AUDIT_FILE).read_text().splitlines()]


@pytest.mark.parametrize("goal", [MOTION, RESOLUTION])
def test_a_relaunched_training_takes_its_place(client, run, project, data, held, seeds, no_destruction, goal):
    from hax_gui import chats as chat_store
    cid = chat_store.manager.create(None, "rebind")["id"]
    wid, old = _launch(run, project, data, goal, cid)
    done = run(watch.relaunch(old, "scratch", {"epochs": 3}))
    new = done["job"]
    assert "the workflow continues with the new run, and its later steps (hetsiren predict" in done["message"]
    (before, after), = seeds                                             # one write, of that one step
    assert after["n1"]["job_id"] == new["id"] and {k: v for k, v in after.items() if k != "n1"} == \
        {k: v for k, v in before.items() if k != "n1"}
    held.unlink()
    w = _settle(client, run, wid)
    assert w["status"] == "succeeded" and w["run_state"]["n1"]["job_id"] == new["id"], w["run_state"]
    predict = manager.get(w["run_state"]["n2"]["job_id"])
    assert _bound(predict, "reload") == new["id"]
    if goal == RESOLUTION:
        assert _bound(manager.get(w["run_state"]["n3"]["job_id"]), "motion_correction") == new["id"]
    assert manager.get(old["id"])["status"] == "cancelled" and Path(old["workdir"]).is_dir()
    assert assistant_monitor.chat_of(new["id"]) == cid and assistant_monitor.step_of(new["id"])["step"] == "hetsiren train"
    assert assistant_bridge.owned(new["workdir"]) and new["id"] in watch._watched
    _until(run, lambda: assistant_bridge.owned(predict["workdir"]))     # the later steps are the assistant's too
    rec = [r for r in _audit(old["workdir"]) if r["decision"] == "rebound"]
    assert rec and rec[-1]["new_job_id"] == new["id"] and rec[-1]["workflow_id"] == wid


def test_a_rebound_step_can_be_relaunched_again(client, run, project, data, held, no_destruction):
    wid, old = _launch(run, project, data, MOTION)
    second = run(watch.relaunch(old, "scratch", {"epochs": 3}))["job"]
    _until(run, lambda: manager.get(second["id"])["status"] == "running")
    done = run(watch.relaunch(manager.get(second["id"]), "scratch", {"epochs": 2}))
    assert "the workflow continues with the new run" in done["message"]
    held.unlink()
    w = _settle(client, run, wid)
    assert w["status"] == "succeeded" and w["run_state"]["n1"]["job_id"] == done["job"]["id"]
    assert _bound(manager.get(w["run_state"]["n2"]["job_id"]), "reload") == done["job"]["id"]


# ── refusals: the workflow stays exactly as it was ───────────────────────────
@pytest.fixture
def settled(client, run, project, data, held):
    """A workflow whose training was cancelled and whose run has settled, plus a valid new job for it."""
    wid, old = _launch(run, project, data, MOTION)
    run(manager.cancel(old["id"]))
    wait_for_status(client, run, old["id"], {"cancelled", "failed"})
    _until(run, lambda: wf.get(wid)["status"] != "running")
    args = {k: v for k, v in old["args"].items() if k not in ("output_path", "ssd_scratch_folder")}
    new = run(manager.create("hetsiren", args, old.get("gpu"), project["id"], name="relaunch", launch=True))
    return wid, manager.get(old["id"]), new, args


def _refused(run, wid, old, new, seeds, why):
    before = wf.get(wid)["run_state"]
    out = run(assistant_workflows.rebind_step(old, new, old["workdir"]))
    assert not out["ok"] and why in out["reason"], out
    assert wf.get(wid)["run_state"] == before and not seeds
    assert _audit(old["workdir"])[-1]["decision"] == "rebind_refused" and why in _audit(old["workdir"])[-1]["reason"]


def test_refused_while_the_old_job_runs(client, run, project, data, held, seeds, monkeypatch):
    wid, old = _launch(run, project, data, MOTION)
    monkeypatch.setattr(assistant_workflows, "REBIND_WAIT_S", 0.3)
    _refused(run, wid, old, old, seeds, "the old run has not ended")


def test_refused_while_the_run_has_not_settled(run, settled, seeds, monkeypatch):
    wid, old, new, _ = settled
    real = wf.get
    monkeypatch.setattr(wf, "get", lambda w: {**real(w), "status": "running"} if w == wid else real(w))
    monkeypatch.setattr(assistant_workflows, "REBIND_WAIT_S", 0.3)
    _refused(run, wid, old, new, seeds, "still running other steps")


def test_refused_when_the_step_moved_on(run, settled, seeds):
    wid, old, new, _ = settled
    state = wf.get(wid)["run_state"]
    wf.seed_state(wid, {**state, "n1": {**state["n1"], "job_id": "someone-elses"}})   # re-run by the user meanwhile
    seeds.clear()
    _refused(run, wid, old, new, seeds, "no longer points at the old run")


def test_refused_for_a_job_that_is_not_a_fresh_run_of_the_step(client, run, project, settled, seeds, md_file):
    wid, old, new, args = settled
    other = projects.create("rebind_other")
    elsewhere = run(manager.create("hetsiren", args, old.get("gpu"), other["id"], name="elsewhere", launch=True))
    _refused(run, wid, old, elsewhere, seeds, "in another project")
    filt = run(manager.create("filter_latents", {"latents": str(md_file)}, None, project["id"], launch=True))
    _refused(run, wid, old, filt, seeds, "not the step's program")
    assistant_monitor.link(new["id"], None, workflow_id="w-else", workflow="another", step="hetsiren train")
    _refused(run, wid, old, new, seeds, "already a step of a workflow")
    for j in (elsewhere, filt):
        run(manager.cancel(j["id"]))


def test_a_failed_restart_leaves_the_workflow_as_it_was(run, settled, seeds, monkeypatch):
    wid, old, new, _ = settled
    before = wf.get(wid)["run_state"]

    async def boom(*a, **k):
        raise WorkflowError(["the scheduler said no"])
    monkeypatch.setattr(wf, "run", boom)
    out = run(assistant_workflows.rebind_step(old, new, old["workdir"]))
    assert not out["ok"] and "the scheduler said no" in out["reason"]
    assert wf.get(wid)["run_state"] == before and len(seeds) == 2 and seeds[1][1] == before    # written back
    msg = run(watch._in_workflow(old, new, {"changes": {"epochs": 3}}, old["workdir"]))
    assert "stay blocked" in msg and "--epochs 3" in msg and "run that step with the steps after it" in msg
