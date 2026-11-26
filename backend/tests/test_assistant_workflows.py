"""Assistant-planned workflows in the GUI: plan -> graph per template, the preview and create
routes per autonomy level, runs on the GUI's own engine in dependency order, and refusals.

The assistant's settings go to a per-test HAX_ASSISTANT_CONFIG; GPUs are faked; runs use a
stub hax that writes the outputs each program leaves behind.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_actions, assistant_workflows, watch

needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
GPU = {"index": 0, "name": "Fake GPU", "mem_total_mb": 16000, "mem_used_mb": 1000, "mem_free_mb": 15000,
       "util": 0}
PREVIEW, CREATE = "/api/assistant/actions/preview-workflow", "/api/assistant/actions/create-workflow"

STUB = r'''
import json, sys, time
from pathlib import Path
argv = sys.argv[1:]
if argv[:1] == ["--gpu"]:
    argv = argv[2:]
program, rest, args, i = argv[0], argv[1:], {}, 0
while i < len(rest):
    if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
        args[rest[i][2:]], i = rest[i + 1], i + 2
    else:
        args[rest[i][2:]], i = True, i + 1
out = Path(args["output_path"])
tag = program + (":" + args["mode"] if "mode" in args else "")
def note(event):
    with open(TRACE, "a") as f:
        f.write(json.dumps({"event": event, "tag": tag, "args": args, "t": time.time()}) + "\n")
note("start")
time.sleep(0.2)
if tag == "hetsiren:train":
    (out / "HetSIREN" / "checkpoints" / "state").mkdir(parents=True)
    (out / "HetSIREN" / "checkpoints" / "config").write_text("{}")
elif tag == "hetsiren:predict":
    assert (Path(args["reload"]) / "HetSIREN").is_dir()
    (out / "predicted_latents.xmd").write_text("data_\nloop_\n_image\n_latent_space\n1@s.mrcs 0,1\n")
elif program == "modart":
    assert Path(args["motion_correction"]).is_dir()
    (out / "modart_map.mrc").write_bytes(b"map")
elif program == "reconstruct_volume":
    (out / "consensus_reconstruction.mrc").write_bytes(b"map")
note("end")
'''


@pytest.fixture(scope="module")
def schema():
    if not assistant_bridge.available():
        return None
    from hax_assistant.config import _VENDORED_SCHEMA
    from hax_assistant.knowledge.schema import load_schema_index
    return load_schema_index(_VENDORED_SCHEMA, source="vendored")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch, schema):
    monkeypatch.setenv("HAX_ASSISTANT_CONFIG", str(tmp_path / "assistant_cfg" / "settings.json"))
    monkeypatch.delenv("HAX_ASSISTANT_AUTONOMY", raising=False)
    monkeypatch.delenv("HAX_ASSISTANT_WORKSPACE", raising=False)
    monkeypatch.setattr(assistant_actions.settings, "assistant_actions", True)
    monkeypatch.setattr(watch, "_watched", {})
    monkeypatch.setattr(watch, "_ensure_task", lambda: None)       # no 60 s loop left behind
    if schema is not None:
        monkeypatch.setattr("hax_assistant.workflows.plan.list_gpus", lambda: [dict(GPU)])
        monkeypatch.setattr(assistant_bridge, "_schema", lambda: schema)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """A hax stand-in that writes each program's outputs; returns the trace reader."""
    trace = tmp_path / "trace.jsonl"
    script = tmp_path / "hax_stub.py"
    script.write_text(f"TRACE = {str(trace)!r}\n" + STUB)
    monkeypatch.setattr(assistant_actions.settings, "hax_cli",
                        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    return lambda: [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []


@pytest.fixture
def no_destruction(monkeypatch):
    """Any stopping, rerunning or deleting call fails the test."""
    from hax_gui import projects
    from hax_gui.jobs import manager
    from hax_gui.workflows import manager as wf

    def boom(*a, **k):
        raise AssertionError("destructive call on an assistant path")
    for name in ("rerun", "delete", "delete_for_project"):
        monkeypatch.setattr(manager, name, boom)
    for name in ("delete", "delete_for_project", "stop"):
        monkeypatch.setattr(wf, name, boom)
    monkeypatch.setattr(projects, "delete", boom)


def _mrc(path, shape, voxel=2.0):
    import mrcfile
    import numpy as np
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.zeros(shape, np.float32))
        m.voxel_size = voxel


@pytest.fixture
def data(tmp_path):
    pytest.importorskip("mrcfile")
    d = tmp_path / "data"
    d.mkdir()
    _mrc(d / "stack.mrcs", (40, 32, 32))
    cols = ["angleRot", "angleTilt", "anglePsi", "shiftX", "shiftY", "ctfDefocusU", "ctfVoltage", "image"]
    rows = [" ".join(["1.0"] * 7 + [f"{k + 1:06d}@stack.mrcs"]) for k in range(40)]
    (d / "particles.xmd").write_text("data_\nloop_\n" + "".join(f"_{c}\n" for c in cols) + "\n".join(rows) + "\n")
    _mrc(d / "ref.mrc", (32, 32, 32))
    _mrc(d / "ref_mask.mrc", (32, 32, 32))
    return {"md": d / "particles.xmd", "vol": d / "ref.mrc", "mask": d / "ref_mask.mrc", "dir": d}


def ask(data, goal="quick continuous motion analysis of"):
    return f"{goal} {data['md']} with reference {data['vol']} and mask {data['mask']}, pixel size 2 A"


def _set_autonomy(level):
    from hax_assistant import autonomy
    autonomy.set_autonomy(level)


def _workflows(project_id):
    from hax_gui.workflows import manager as wf
    return {w["id"] for w in wf.list(project_id)}


def _settle(client, run, wid, timeout=90.0):
    async def _wait():
        for _ in range(int(timeout / 0.2)):
            w = (await client.get(f"/api/workflows/{wid}")).json()
            if w["status"] not in ("running", "idle"):
                return w
            await asyncio.sleep(0.2)
        raise AssertionError(f"workflow never finished: {w['status']} {w['run_state']}")
    return run(_wait())


# ── plan -> graph ────────────────────────────────────────────────────────────
def _modes(graph):
    return [(n["program"], n["args"].get("mode")) for n in graph["nodes"]]


def _edges(graph):
    return [(e["from"], e["to"], e["field"], e["kind"]) for e in graph["edges"]]


def _assert_clean(out):
    from hax_gui.workflows import manager as wf
    assert out["errors"] == [] and out["ok"], (out["errors"], out["plan"]["describe"])
    assert wf.validate(out["graph"]) == [] and assistant_workflows.check_graph(out["graph"]) == []
    for n in out["graph"]["nodes"]:
        assert not {"output_path", "ssd_scratch_folder"} & set(n["args"])        # filled by the engine
        assert not {e["field"] for e in out["graph"]["edges"] if e["to"] == n["id"]} & set(n["args"])
        assert (n["x"], n["y"]) == (0, 0) and n["gpu"] == "0"                  # tidied when opened


@needs_assistant
def test_graph_continuous_motion(project, data):
    out = assistant_workflows.preview(ask(data), project["id"])
    _assert_clean(out)
    g = out["graph"]
    assert _modes(g) == [("hetsiren", "train"), ("hetsiren", "predict")]
    assert _edges(g) == [("n1", "n2", "reload", "hetsiren_model")]
    train, predict = (n["args"] for n in g["nodes"])
    assert train["transport_mass"] is True and train["md"] == str(data["md"])
    assert (train["vol"], train["mask"], train["sr"], train["ctf_type"]) == (
        str(data["vol"]), str(data["mask"]), 2.0, "apply")
    assert {"epochs", "batch_size", "lat_dim", "load_images_to_ram"} <= set(train)
    assert "reload" not in predict and predict["md"] == str(data["md"])
    assert [o["title"] for o in out["omitted"]] == ["select_states", "decode_states_from_latents", "analysis"]
    assert "every particle" in out["omitted"][1]["reason"]
    assert "inside the assistant" in out["omitted"][2]["reason"] and "report" in out["omitted"][2]["reason"]
    assert out["steps"][0]["reasons"]["epochs"] and out["steps"][1]["wired"] == {"reload": "1. hetsiren train"}
    assert out["steps"][0]["watch"] and not out["steps"][1]["watch"]
    assert out["autonomy"] == {"level": "advisory", "decision": "propose"}
    assert out["project"] == {"id": project["id"], "name": project["name"], "new": False}


@needs_assistant
def test_graph_improve_resolution(project, data):
    out = assistant_workflows.preview(ask(data, "quick: improve the resolution of"), project["id"])
    _assert_clean(out)
    g = out["graph"]
    assert out["plan"]["template"] == "improve_resolution"
    assert _modes(g) == [("hetsiren", "train"), ("hetsiren", "predict"), ("modart", None)]
    assert _edges(g) == [("n1", "n2", "reload", "hetsiren_model"),
                         ("n1", "n3", "motion_correction", "motion_model")]
    train, modart = g["nodes"][0]["args"], g["nodes"][2]["args"]
    assert {k: modart[k] for k in ("md", "vol", "mask", "sr", "ctf_type")} == \
        {k: train[k] for k in ("md", "vol", "mask", "sr", "ctf_type")}             # inherited, as on the canvas
    assert modart["reconstruct_halves"] is True and "motion_correction" not in modart
    assert set(out["steps"][2]["inherited"]) == {"md", "vol", "mask", "sr", "ctf_type"}
    assert "sr" not in out["steps"][2]["reasons"] and out["steps"][2]["reasons"]["reconstruct_halves"]
    assert [o["title"] for o in out["omitted"]] == ["select_states", "decode_states_from_latents", "analysis"]


@needs_assistant
def test_graph_improve_resolution_of_a_run(project, data, tmp_path):
    run_dir = tmp_path / "old_run"
    (run_dir / "HetSIREN").mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(
        {"md": str(data["md"]), "sr": 2.0, "ctf_type": "apply", "transport_mass": True}))
    out = assistant_workflows.preview(f"improve the resolution of my run {run_dir}", project["id"])
    _assert_clean(out)
    assert out["plan"]["template"] == "improve_resolution_run" and out["omitted"] == []
    assert _modes(out["graph"]) == [("modart", None)] and out["graph"]["edges"] == []
    args = out["graph"]["nodes"][0]["args"]
    assert args["motion_correction"] == str(run_dir / "HetSIREN") and args["md"] == str(data["md"])
    assert args["sr"] == 2.0 and args["reconstruct_halves"] is True


@needs_assistant
def test_graph_reconstruction(project, data):
    out = assistant_workflows.preview(f"reconstruct a map from {data['md']}, pixel size 2 A", project["id"])
    _assert_clean(out)
    assert _modes(out["graph"]) == [("reconstruct_volume", None)] and out["omitted"] == []
    assert out["graph"]["nodes"][0]["args"] == {"md": str(data["md"]), "sr": 2.0, "ctf_type": "apply",
                                                 "write_mask": True}


# ── routes ───────────────────────────────────────────────────────────────────
@needs_assistant
def test_preview_route_is_read_only(client, run, project, data):
    root = Path(project["path"]).parent
    before, wfs = sorted(p.name for p in root.iterdir()), _workflows(None)
    r = run(client.post(PREVIEW, json={"request": ask(data)}))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] and out["plan"]["template"] == "continuous_motion" and out["graph"]["nodes"]
    assert out["project"]["new"] and out["project"]["name"] == out["plan"]["name"]
    assert any(c["name"] == "project folder" and c["status"] == "pass" for c in out["plan"]["checks"])
    assert out["autonomy"]["decision"] == "propose"
    assert sorted(p.name for p in root.iterdir()) == before and _workflows(None) == wfs
    r = run(client.post(PREVIEW, json={"request": "hello there"}))
    assert r.status_code == 422 and "no goal" in r.text


@needs_assistant
def test_create_draft_runs_nothing(client, run, project, data, no_destruction):
    jobs_before = len(run(client.get(f"/api/projects/{project['id']}/jobs")).json())
    body = {"request": ask(data), "project_id": project["id"], "name": "My draft"}
    shown = run(client.post(PREVIEW, json=body)).json()["fingerprint"]
    wfs = _workflows(project["id"])
    r = run(client.post(CREATE, json={**body, "run": False, "fingerprint": "0" * 16}))
    assert r.status_code == 409 and "plan changed" in r.text and _workflows(project["id"]) == wfs
    r = run(client.post(CREATE, json={**body, "run": False, "fingerprint": shown}))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["started"] is False and out["project_id"] == project["id"] and out["decision"] == "propose"
    w = run(client.get(f"/api/workflows/{out['workflow_id']}")).json()
    assert w["name"] == "My draft" and w["project_id"] == project["id"] and w["kind"] == "standalone"
    assert w["status"] == "idle" and w["run_state"] is None and len(w["graph"]["nodes"]) == 2
    assert len(run(client.get(f"/api/projects/{project['id']}/jobs")).json()) == jobs_before


@needs_assistant
def test_run_refused_when_not_allowed(client, run, project, data):
    wfs = _workflows(project["id"])
    r = run(client.post(CREATE, json={"request": ask(data), "project_id": project["id"], "run": True}))
    assert r.status_code == 403 and "advisory" in r.text
    _set_autonomy("ask")
    r = run(client.post(CREATE, json={"request": ask(data), "project_id": project["id"], "run": True,
                                      "auto": True}))
    assert r.status_code == 403 and "no longer auto" in r.text
    assert _workflows(project["id"]) == wfs


@needs_assistant
@pytest.mark.parametrize("level", ["ask", "auto"])
def test_create_and_run_in_dependency_order(client, run, project, data, stub, no_destruction, level):
    _set_autonomy(level)
    body = {"request": ask(data, "quick: improve the resolution of"), "project_id": project["id"],
            "run": True, "auto": level == "auto"}
    r = run(client.post(CREATE, json=body))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["started"] is True and out["decision"] == ("confirm" if level == "ask" else "proceed")
    w = _settle(client, run, out["workflow_id"])
    assert w["status"] == "succeeded", w["run_state"]
    jobs = {nid: run(client.get(f"/api/jobs/{s['job_id']}")).json() for nid, s in w["run_state"].items()}
    assert all(j["project_id"] == project["id"] for j in jobs.values())
    train = Path(jobs["n1"]["workdir"])
    assert train.parent == Path(project["path"]) and train.name == f"hetsiren_{jobs['n1']['id']}"
    events = stub()
    starts = {e["tag"]: e for e in events if e["event"] == "start"}
    ends = {e["tag"]: e for e in events if e["event"] == "end"}
    assert [e["tag"] for e in events if e["event"] == "start"][0] == "hetsiren:train"
    assert ends["hetsiren:train"]["t"] <= min(starts["hetsiren:predict"]["t"], starts["modart"]["t"])
    assert starts["hetsiren:predict"]["args"]["reload"] == str(train)
    modart = starts["modart"]["args"]
    assert modart["motion_correction"] == str(train / "HetSIREN")
    assert (modart["md"], modart["vol"], modart["mask"]) == (str(data["md"]), str(data["vol"]), str(data["mask"]))
    assert starts["hetsiren:train"]["args"]["transport_mass"] is True

    async def _watched():
        for _ in range(50):
            if jobs["n1"]["id"] in watch._watched:
                return True
            await asyncio.sleep(0.1)
        return False
    assert run(_watched()) and jobs["n2"]["id"] not in watch._watched     # only the training step


@needs_assistant
def test_failed_preflight_refuses_creation(client, run, project, data, tmp_path):
    root = Path(project["path"]).parent
    before, wfs = sorted(p.name for p in root.iterdir()), _workflows(None)
    _set_autonomy("auto")
    text = f"quick continuous motion analysis of {tmp_path / 'missing.xmd'}, pixel size 2 A"
    for project_id in (project["id"], None):
        r = run(client.post(CREATE, json={"request": text, "project_id": project_id, "run": True}))
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert any("not found" in c["message"] for c in detail["failed_checks"])
    assert sorted(p.name for p in root.iterdir()) == before and _workflows(None) == wfs


@needs_assistant
def test_new_project_never_reuses_a_folder(client, run, project, data):
    root = Path(project["path"]).parent
    taken = root / "assistant_wf_test"
    taken.mkdir(exist_ok=True)
    (taken / "keep.txt").write_text("user data")
    before = sorted(p.name for p in root.iterdir())
    r = run(client.post(CREATE, json={"request": ask(data), "name": "assistant_wf_test", "run": False}))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["project_id"] != "assistant_wf_test" and out["project_id"].startswith("assistant_wf_test")
    assert sorted(p.name for p in root.iterdir()) == sorted(before + [out["project_id"]])
    assert [p.name for p in taken.iterdir()] == ["keep.txt"] and (taken / "keep.txt").read_text() == "user data"
    w = run(client.get(f"/api/workflows/{out['workflow_id']}")).json()
    assert w["project_id"] == out["project_id"] and w["status"] == "idle"
    assert run(client.get(f"/api/projects/{out['project_id']}/jobs")).json() == []


@needs_assistant
def test_bad_bodies_are_rejected(client, run, project, data):
    base = {"request": ask(data), "project_id": project["id"]}
    for extra in ({"graph": {"nodes": [], "edges": []}}, {"args": {"epochs": 1}}, {"gpu": "0; ls"}):
        assert run(client.post(CREATE, json={**base, **extra})).status_code == 422
    assert run(client.post(PREVIEW, json={**base, "run": True})).status_code == 422
    r = run(client.post(CREATE, json={**base, "project_id": "nope"}))
    assert r.status_code == 404


def test_disabled_without_actions_or_assistant(client, run, monkeypatch):
    monkeypatch.setattr(assistant_actions.settings, "assistant_actions", False)
    body = {"request": "continuous motion of /x.xmd"}
    expected = 403 if assistant_bridge.available() else 400
    assert run(client.post(PREVIEW, json=body)).status_code == expected
    monkeypatch.setattr(assistant_bridge, "_INSTALLED", False)
    assert run(client.post(CREATE, json=body)).status_code == 400
