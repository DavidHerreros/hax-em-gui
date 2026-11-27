"""The free-tier tools for what a workflow template leaves out: select_states picks the states a decode
reads (CPU, beside the predicted latents), launch_job starts one program as a new job of a GUI project,
per autonomy and the assistant's guards. The tool loop is scripted; hax is a stub."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import wait_for_status
from test_assistant_gui_tools import Backend, _session, _turn, no_assistant_runner, ready  # noqa: F401
from test_assistant_workflows import (  # noqa: F401  (fixtures)
    _set_autonomy, isolated, no_destruction, schema, stub)

from hax_gui import assistant as assistant_bridge
from hax_gui.jobs import manager

pytestmark = pytest.mark.skipif(not assistant_bridge.available(), reason="hax_assistant is not installed")
DECODE = "decode_states_from_latents"


def _predict_run(client, run, project, md_file, latents=True):
    """A finished job standing in for a HetSIREN predict run, and the training run its network is in."""
    r = run(client.post(f"/api/projects/{project['id']}/jobs",
                        json={"program": "filter_latents", "args": {"latents": str(md_file)}, "launch": True}))
    assert r.status_code == 200, r.text
    job = wait_for_status(client, run, r.json()["id"], {"succeeded", "failed"})
    train = Path(project["path"]) / f"train_{job['id']}"
    (train / "HetSIREN").mkdir(parents=True)
    pred = Path(job["workdir"])
    (pred / "run_config.json").write_text(json.dumps({"mode": "predict", "reload": str(train)}))
    if latents:
        rng = np.random.default_rng(0)
        z = np.vstack([rng.normal(-5, 0.3, (30, 3)), rng.normal(1.2, 0.3, (30, 3))])
        rows = "\n".join(f"{i + 1:06d}@s.mrcs " + ",".join(f"{v:.4f}" for v in row) for i, row in enumerate(z))
        (pred / "predicted_latents.star").write_text("data_\nloop_\n_image\n_latent_space\n" + rows + "\n")
    return job, train


def test_select_states_then_decode_as_a_gui_job(client, run, project, md_file, ready, stub, schema,
                                                no_destruction, no_assistant_runner):
    _set_autonomy("auto")
    pred, train = _predict_run(client, run, project, md_file)
    states = Path(pred["workdir"]) / "assistant_states" / "states_k4.npy"
    decode = {"latents_file": str(states), "reload": str(train / "HetSIREN")}
    be = Backend(("select_states", {"job": pred["id"], "n_states": 4}),
                 ("launch_job", {"program": DECODE, "args": decode}), "Decoding.")
    _turn(run, _session(be, schema), "decode the states")       # no project: the one its paths are in
    picked, launched = be.tool_result(0), be.tool_result(1)
    assert picked["status"] == "saved" and picked["file"] == str(states) and picked["n_states"] == 4
    assert sum(picked["particles_per_state"]) == 60 and np.load(states).shape == (4, 3)
    assert picked["decode_with"] == {"program": DECODE, "args": decode}
    assert launched["project"] == project["name"], launched
    done = wait_for_status(client, run, launched["job"]["id"], {"succeeded", "failed"})
    assert done["status"] == "succeeded" and done["project_id"] == project["id"]
    assert Path(done["workdir"]).parent == Path(project["path"])
    start = [e for e in stub() if e["tag"] == DECODE and e["event"] == "start"][-1]
    assert start["args"]["latents_file"] == str(states) and start["args"]["reload"] == decode["reload"]
    assert Path(start["args"]["output_path"]) == Path(done["workdir"])     # the GUI's own new folder
    _turn(run, _session(Backend(("select_states", {"job": pred["id"], "n_states": 4}), "ok"), schema), "again")
    assert (states.parent / "states_k4_2.npy").is_file() and states.is_file()      # never overwritten


def test_launch_job_follows_autonomy(run, project, ready, schema, no_assistant_runner):
    call = ("launch_job", {"program": DECODE, "project": project["name"],
                           "args": {"latents_file": "/x/states.npy", "reload": "/x/HetSIREN"}})
    before = len(manager.list_for_project(project["id"]))
    _set_autonomy("advisory")
    be = Backend(call, "ok")
    _turn(run, _session(be, schema), "decode")
    assert be.tool_result()["status"] == "not_executed" and DECODE in be.tool_result()["proposal"]
    _set_autonomy("ask")
    be = Backend(call, "ok")
    _, inter = _turn(run, _session(be, schema), "decode", {"confirm_launch_job": "no"})
    assert be.tool_result()["status"] == "declined_by_user"
    assert "new job of your project" in inter.asked[-1].text and "--latents_file /x/states.npy" in inter.asked[-1].text
    assert len(manager.list_for_project(project["id"])) == before


@pytest.mark.parametrize("program, args, status, words", [
    (DECODE, {"latents_file": "/x/s.npy", "reload": "/x/H", "output_path": "/tmp/elsewhere"}, "refused", "job manager"),
    (DECODE, {"latents_file": "/x/s.npy", "reload": "--output_path"}, "refused", "plain value"),
    ("update", {}, "refused", "never launches"),
    (DECODE, {"reload": "/x/HetSIREN"}, "invalid", "latents_file"),
])
def test_launch_job_guards(run, project, ready, schema, no_assistant_runner, program, args, status, words):
    _set_autonomy("auto")
    before = len(manager.list_for_project(project["id"]))
    be = Backend(("launch_job", {"program": program, "args": args, "project": project["name"]}), "ok")
    _turn(run, _session(be, schema), "launch it")
    res = be.tool_result()
    assert res["status"] == status and words in json.dumps(res), res
    assert len(manager.list_for_project(project["id"])) == before


def test_select_states_refusals(client, run, project, md_file, ready, schema, tmp_path, no_assistant_runner):
    pred, _ = _predict_run(client, run, project, md_file, latents=False)
    be = Backend(("select_states", {"job": pred["id"]}), "ok")
    _turn(run, _session(be, schema), "pick states")
    assert be.tool_result()["status"] == "refused" and "predict them first" in be.tool_result()["reason"]
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((10, 2)))
    be = Backend(("select_states", {"latents": str(outside)}), "ok")
    _turn(run, _session(be, schema), "pick states")
    assert be.tool_result()["status"] == "refused" and "workspace" in be.tool_result()["reason"]


def test_write_report_builds_the_pdf_and_the_page_fetches_it(client, run, project, md_file, ready, monkeypatch):
    from fastapi.testclient import TestClient
    from hax_gui.app import app
    pred, _ = _predict_run(client, run, project, md_file)
    monkeypatch.setattr(assistant_bridge, "generate_report", lambda out, **kw: Path(out).write_bytes(b"%PDF-1.4 stub"))
    assert run(client.get(f"/api/assistant/run-report/{pred['id']}")).status_code == 404     # none built yet
    results = []

    def stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        tool = next(t for t in tools if t.spec.name == "write_report")
        results.append(tool.fn(None))                           # the attached run
        results.append(tool.fn(None, job=pred["id"]))
        with pytest.raises(ValueError, match="not one job of the GUI"):
            tool.fn(None, job="no-such-job")
        yield {"type": "content", "delta": "Done."}
    monkeypatch.setattr(assistant_bridge, "stream_events", stream)
    with TestClient(app).websocket_connect("ws://127.0.0.1/ws/assistant/chat") as ws:
        ws.send_json({"messages": [{"role": "user", "content": "write the report"}], "turn": 1,
                      "run_job_id": pred["id"]})
        assert ws.receive_json() == {"report": {"job_id": pred["id"]}}
        assert ws.receive_json() == {"report": {"job_id": pred["id"]}}
        assert ws.receive_json() == {"token": "Done."}
        assert ws.receive_json() == {"done": True, "turn": 1}
    assert all(r["status"] == "written" and "_report.pdf" in r["delivery"] for r in results)
    got = run(client.get(f"/api/assistant/run-report/{pred['id']}"))
    assert got.status_code == 200 and got.content == b"%PDF-1.4 stub"
    assert f"{pred['id']}_report.pdf" in got.headers["content-disposition"]
