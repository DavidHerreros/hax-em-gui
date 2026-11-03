"""Cascade invalidation over the binding graph, and workflow graph logic
(validation, cycle detection, downstream closure)."""

from __future__ import annotations

import json

from conftest import wait_for_status


def _create(client, run, project, md_file):
    r = run(client.post(f"/api/projects/{project['id']}/jobs", json={
        "program": "estimate_latent_covariances", "args": {"md": str(md_file)}, "launch": True}))
    assert r.status_code == 200, r.text
    return r.json()


def test_cascade_marks_dependents_stale(client, run, project, md_file):
    from hax_gui import cascade
    from hax_gui.jobs import manager

    a = _create(client, run, project, md_file)
    b = _create(client, run, project, md_file)
    c = _create(client, run, project, md_file)
    for j in (a, b, c):
        wait_for_status(client, run, j["id"], {"succeeded"})
    # Wire B to consume A and C to consume B (bindings are what the graph is
    # derived from; written directly so no adapter materialisation is needed).
    manager._exec("UPDATE jobs SET bindings=? WHERE id=?",
                  (json.dumps({"md": {"job_id": a["id"], "kind": "particles"}}), b["id"]))
    manager._exec("UPDATE jobs SET bindings=? WHERE id=?",
                  (json.dumps({"md": {"job_id": b["id"], "kind": "particles"}}), c["id"]))

    deps = cascade.dependents_of([a["id"]])
    ids = [d["id"] for d in deps]
    assert ids == [b["id"], c["id"]]          # transitive, oldest first

    run(cascade.invalidate(deps))
    for j in (b, c):
        r = run(client.get(f"/api/jobs/{j['id']}"))
        assert r.json()["stale"] is True
    r = run(client.get(f"/api/jobs/{a['id']}"))
    assert r.json()["stale"] is False          # the source itself is not stale

    # rerunning a stale job clears the flag again (bindings dropped first: the
    # fabricated kind has no real artifact behind it for produce() to materialise)
    manager._exec("UPDATE jobs SET bindings=NULL WHERE id=?", (b["id"],))
    r = run(client.post(f"/api/jobs/{b['id']}/rerun", json={}))
    assert r.status_code == 200, r.text
    assert wait_for_status(client, run, b["id"], {"succeeded"})["stale"] is False


def test_rerun_refused_while_dependent_runs(client, run, project, md_file):
    from conftest import GATE
    from hax_gui.jobs import manager

    a = _create(client, run, project, md_file)
    wait_for_status(client, run, a["id"], {"succeeded"})
    GATE.write_text("")
    try:
        b = _create(client, run, project, md_file)
        wait_for_status(client, run, b["id"], {"running"})
        manager._exec("UPDATE jobs SET bindings=? WHERE id=?",
                      (json.dumps({"md": {"job_id": a["id"], "kind": "particles"}}), b["id"]))
        # A rerun of A must refuse (409) while B is reading A's folder …
        r = run(client.post(f"/api/jobs/{a['id']}/rerun", json={}))
        assert r.status_code == 409
        assert any(d["id"] == b["id"] for d in r.json()["detail"]["running"])
        # … and go through once the client re-submits with cancel_running.
        r = run(client.post(f"/api/jobs/{a['id']}/rerun", json={"cancel_running": True}))
        assert r.status_code == 200, r.text
        wait_for_status(client, run, b["id"], {"cancelled"})
        GATE.unlink(missing_ok=True)   # A's relaunched stub blocks on the gate too
        wait_for_status(client, run, a["id"], {"succeeded"})
    finally:
        GATE.unlink(missing_ok=True)


# ---- workflows -----------------------------------------------------------
def test_workflow_validate_and_topo(client, run, project):
    from hax_gui.workflows import manager as wf

    assert wf.validate({"nodes": [], "edges": []})            # empty graph is an error

    nodes = [{"id": "n1", "program": "estimate_latent_covariances"},
             {"id": "n2", "program": "estimate_latent_covariances"},
             {"id": "n3", "program": "no_such_program"}]
    errors = wf.validate({"nodes": nodes, "edges": []})
    assert any("no_such_program" in e for e in errors)

    # a cycle must be reported, whatever else is wrong with the edges
    cyc = {"nodes": nodes[:2],
           "edges": [{"from": "n1", "to": "n2", "field": "md", "kind": "particles"},
                     {"from": "n2", "to": "n1", "field": "md", "kind": "particles"}]}
    assert wf.validate(cyc)

    saved = wf.save(project["id"], "t", {
        "nodes": [{"id": "a", "program": "x"}, {"id": "b", "program": "x"}, {"id": "c", "program": "x"}],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
    })
    assert wf.downstream_nodes(saved["id"], ["a"]) == ["b", "c"]   # transitive closure
    wf.delete(saved["id"])


def test_workflow_crud_roundtrip(client, run, project):
    graph = {"nodes": [{"id": "a", "program": "estimate_latent_covariances"}], "edges": []}
    r = run(client.post("/api/workflows", json={"project_id": project["id"], "name": "wf1", "graph": graph}))
    assert r.status_code == 200, r.text
    wid = r.json()["id"]
    r = run(client.get(f"/api/workflows/{wid}"))
    assert r.json()["graph"]["nodes"][0]["id"] == "a"
    r = run(client.delete(f"/api/workflows/{wid}"))
    assert r.status_code == 200
    assert run(client.get(f"/api/workflows/{wid}")).status_code == 404


def test_project_export_import_roundtrip(client, run, project, md_file):
    """Export the project, import it back. Importing onto the same server keeps
    the folder (fresh name) but skips job rows whose ids already exist; once the
    originals are deleted, a second import brings the rows through with their
    paths remapped into the new project folder."""
    r = run(client.post(f"/api/projects/{project['id']}/jobs", json={
        "program": "estimate_latent_covariances", "args": {"md": str(md_file)}, "launch": True}))
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    wait_for_status(client, run, jid, {"succeeded"})
    job = run(client.get(f"/api/jobs/{jid}")).json()
    assert job["provenance"] and "packages" in job["provenance"]   # provenance stamped

    r = run(client.get(f"/api/projects/{project['id']}/export"))
    assert r.status_code == 200
    blob = r.content
    assert blob[:2] == b"\x1f\x8b"          # gzip magic

    r = run(client.post("/api/projects/import", content=blob))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["project"]["id"] != project["id"]      # fresh folder name
    assert jid in out["skipped"]                       # same server: rows already exist

    # delete the original rows, import again: now the jobs come through, remapped
    for j in run(client.get(f"/api/projects/{project['id']}/jobs")).json():
        run(client.delete(f"/api/jobs/{j['id']}"))
    r = run(client.post("/api/projects/import", content=blob))
    assert r.status_code == 200, r.text
    out2 = r.json()
    assert out2["imported"] >= 1 and not out2["skipped"]
    imported_jobs = run(client.get(f"/api/projects/{out2['project']['id']}/jobs")).json()
    assert any(j["id"] == jid for j in imported_jobs)
    the_job = next(j for j in imported_jobs if j["id"] == jid)
    assert the_job["workdir"].startswith(out2["project"]["path"])   # paths remapped
    # cleanup the imported copies
    for p2 in (out["project"], out2["project"]):
        for j in run(client.get(f"/api/projects/{p2['id']}/jobs")).json():
            run(client.delete(f"/api/jobs/{j['id']}"))
        run(client.delete(f"/api/projects/{p2['id']}"))


def test_workflow_step_waiting_for_device_is_a_queued_job(client, run, project, md_file):
    """Two steps pinned to the same device with one slot: the second must exist as a
    real job parked 'queued' (visible in the project's run list) while it waits, then
    start once the first frees the slot. It used to be nothing at all — no row, and a
    grey 'pending' on the canvas that reconcile later flattened to 'idle'."""
    from conftest import GATE
    import time

    graph = {"nodes": [
        {"id": "a", "program": "estimate_latent_covariances", "gpu": "0", "args": {"md": str(md_file)}},
        {"id": "b", "program": "estimate_latent_covariances", "gpu": "0", "args": {"md": str(md_file)}},
    ], "edges": []}
    r = run(client.post("/api/workflows", json={"project_id": project["id"], "name": "gate", "graph": graph}))
    assert r.status_code == 200, r.text
    wid = r.json()["id"]
    run(client.put(f"/api/workflows/{wid}", json={"max_per_device": 1}))

    GATE.write_text("")                       # the stub blocks: 'a' will hold the slot
    try:
        r = run(client.post(f"/api/workflows/{wid}/run", json={}))
        assert r.status_code == 200, r.text

        def state():
            return run(client.get(f"/api/workflows/{wid}")).json()["run_state"]

        deadline = time.time() + 30
        while time.time() < deadline:
            s = state()
            if s["a"]["status"] == "running" and s["b"]["status"] == "queued" and s["b"]["job_id"]:
                break
            time.sleep(0.3)
        else:
            raise AssertionError(f"expected a running / b queued with a job, got {state()}")

        # The waiting step is a real, listed job, parked as queued...
        jobs = run(client.get(f"/api/projects/{project['id']}/jobs")).json()
        by_id = {j["id"]: j for j in jobs}
        assert by_id[s["b"]["job_id"]]["status"] == "queued"
        # ...that the admission dispatcher does not launch behind the scheduler's back.
        time.sleep(3)
        assert run(client.get(f"/api/jobs/{s['b']['job_id']}")).json()["status"] == "queued"
    finally:
        GATE.unlink(missing_ok=True)

    deadline = time.time() + 40
    while time.time() < deadline:
        w = run(client.get(f"/api/workflows/{wid}")).json()
        s = w["run_state"]
        if s["a"]["status"] == "succeeded" and s["b"]["status"] == "succeeded":
            break
        time.sleep(0.3)
    else:
        raise AssertionError(f"steps did not both finish: {w['run_state']}")
    assert w["status"] == "succeeded"
    run(client.delete(f"/api/workflows/{wid}"))
